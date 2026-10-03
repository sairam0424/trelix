import { execFile } from "node:child_process";
import { mkdir, mkdtemp, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import { isCommitId } from "./commit-id.js";
import { buildGitChildEnv } from "./child-env.js";

const execFileAsync = promisify(execFile);

// The checkout lives in `trelix-review-<random>`; the askpass helper and git's
// empty HOME live in `trelix-review-aux-<random>`, outside the checkout, so a PR
// cannot overwrite or collide with them. Deriving the second prefix from the first
// keeps `sweepStaleWorkspaces` covering both; review-outcome.ts derives the prefix of
// its own directory (`trelix-review-outcome-<random>`) the same way.
export const TEMP_DIR_PREFIX = "trelix-review-";
const AUX_DIR_PREFIX = `${TEMP_DIR_PREFIX}aux-`;

// Command-scope config for EVERY git child (`-c` outranks any config file):
// - safe.bareRepository=explicit: a bare repository a PR embeds in its tree is never
//   adopted just because git runs with its cwd inside it, which is how an embedded
//   repo's `core.fsmonitor` (or similar) gets run as a command.
// - protocol.file.allow=never: a `file://` or path remote never reads the host. A second
//   guard: GIT_ALLOW_PROTOCOL=https already says so, and overrides this setting.
// - credential.helper=: no configured helper, so our askpass is always used and
//   never a stale cached credential.
const GIT_COMMAND_CONFIG: readonly string[] = [
    "-c",
    "safe.bareRepository=explicit",
    "-c",
    "protocol.file.allow=never",
    "-c",
    "credential.helper=",
];

// ~2 minutes — generous for a shallow, single-ref fetch of one PR, short
// enough that a genuinely hung clone (dead remote, network partition)
// doesn't tie up a webhook-delivery worker indefinitely.
const CHECKOUT_TIMEOUT_MS = 2 * 60 * 1000;

/**
 * Declared here (not imported from review-runner.ts) so review-runner.ts
 * can import from this module without a circular dependency — a narrower
 * shape than ReviewRequest is all a checkout needs.
 */
export interface CheckoutTarget {
    owner: string;
    repo: string;
    prNumber: number;
}

/** How a git child is run; the default is Node's `execFile`. */
export type GitExecFile = (
    file: string,
    args: string[],
    options: { cwd: string; timeout: number; env: NodeJS.ProcessEnv },
) => Promise<unknown>;

export interface CheckoutOptions {
    timeoutMs?: number;
    /** Overridable so tests can point at a local bare repo instead of a real GitHub remote. */
    remoteUrl?: string;
    /**
     * Tests only: the git protocols allowed (GIT_ALLOW_PROTOCOL); a local bare
     * repo needs "file". Deliberately an explicit parameter and never read from the
     * environment, so a host variable cannot widen it in production.
     */
    allowProtocol?: string;
    /** Tests only: substitute the runner to observe the env each git child gets. */
    execFile?: GitExecFile;
}

export interface Workspace {
    path: string;
    /** The commit the checkout is at (`git rev-parse HEAD`): a full lowercase hex id. */
    headSha: string;
    cleanup(): Promise<void>;
}

/**
 * Static, secret-free askpass script: it never contains the token itself
 * — only a reference to the `TRELIX_GIT_TOKEN` env var, which git's
 * askpass protocol reads at fetch time via this script's stdout. The
 * token therefore never appears in this file, in `argv` for any `git`
 * subprocess, or in `.git/config` (no credential helper caches it, and
 * the remote URL never embeds it). Written to the aux directory, not the
 * checkout.
 */
const GIT_ASKPASS_SCRIPT = `#!/bin/sh
# Invoked once per credential prompt by git's askpass protocol ("Username
# for ...", "Password for ...") — the requested value goes on stdout.
case "$1" in
  Username*) echo "x-access-token" ;;
  Password*) echo "$TRELIX_GIT_TOKEN" ;;
esac
`;

/** The `stdout` a `GitExecFile` result carries, or "" when it has none (a test double). */
function stdoutOf(result: unknown): string {
    if (typeof result !== "object" || result === null) {
        return "";
    }
    const { stdout } = result as { stdout?: unknown };
    return typeof stdout === "string" ? stdout : "";
}

/**
 * Clones a single pull request's head into a fresh, per-request temp
 * workspace via an installation token, and returns a handle whose
 * `cleanup()` removes it. Callers MUST call `cleanup()` (typically in a
 * `finally` block) once done with the workspace.
 *
 * Fetches `refs/pull/<prNumber>/head` against the base repo rather than
 * cloning with `--branch=<head.ref>` — the head branch only exists on the
 * *fork's* repo for an external-contributor PR (the normal case for a
 * public review bot), while `refs/pull/<n>/head` always exists on the
 * base repo regardless of fork status, with no second remote needed.
 */
export async function checkoutPullRequest(
    token: string,
    target: CheckoutTarget,
    options: CheckoutOptions = {},
): Promise<Workspace> {
    const timeoutMs = options.timeoutMs ?? CHECKOUT_TIMEOUT_MS;
    const remoteUrl =
        options.remoteUrl ??
        `https://github.com/${target.owner}/${target.repo}.git`;

    const run = options.execFile ?? execFileAsync;

    const dir = await mkdtemp(join(tmpdir(), TEMP_DIR_PREFIX));
    let auxDir: string | undefined;
    // Both removals are always attempted: a checkout that will not go away (a file the
    // PR made unremovable, EPERM) must not also leave the aux directory, which holds the
    // askpass helper, behind. Rejects once both have been tried if either failed.
    const removeTempDirs = async () => {
        const targets = auxDir === undefined ? [dir] : [dir, auxDir];
        const results = await Promise.allSettled(
            targets.map((target) =>
                rm(target, { recursive: true, force: true }),
            ),
        );
        const failures = results.flatMap((result) =>
            result.status === "rejected" ? [result.reason] : [],
        );
        if (failures.length > 0) {
            throw new AggregateError(
                failures,
                `could not remove ${failures.length} of ${targets.length} temp directories`,
            );
        }
    };

    try {
        auxDir = await mkdtemp(join(tmpdir(), AUX_DIR_PREFIX));
        // The askpass script and git's HOME live outside the checkout: a PR that
        // tracks a file of the same name cannot clash with them, and nothing the
        // PR carries can replace them. `cleanup()` removes both directories.
        const askpassPath = join(auxDir, "askpass.sh");
        const homeDir = join(auxDir, "home");
        await mkdir(homeDir);
        await writeFile(askpassPath, GIT_ASKPASS_SCRIPT, { mode: 0o700 });

        // execFile's `env` option REPLACES the inherited environment entirely, so
        // every git child gets this allow-listed env and nothing from the host.
        const env = buildGitChildEnv({
            token,
            askpassPath,
            homeDir,
            allowProtocol: options.allowProtocol,
        });
        const git = (args: string[]) =>
            run("git", [...GIT_COMMAND_CONFIG, ...args], {
                cwd: dir,
                timeout: timeoutMs,
                env,
            });

        // An empty template keeps git from copying a hooks directory (or anything
        // else from the host's template dir) into the fresh repository.
        await git(["init", "--quiet", "--template="]);
        // Never embeds the token in the remote URL — credentials pass through
        // GIT_ASKPASS/env only (see the `fetch` call below).
        await git(["remote", "add", "origin", remoteUrl]);
        await git([
            "fetch",
            "--depth",
            "1",
            "--no-tags",
            // Deliberately no --recurse-submodules/--shallow-submodules: this
            // service clones content from PRs opened by arbitrary external
            // contributors, and an attacker-controlled `.gitmodules` is a real
            // risk. Diff-level review doesn't need submodule contents, so this
            // is an intentional security boundary, not an oversight.
            "origin",
            `refs/pull/${target.prNumber}/head`,
        ]);
        // core.symlinks=false materialises every symlink the PR committed as a
        // plain file holding the link text, so nothing in the workspace can
        // point outside it (`trelix index` would otherwise read the target).
        await git([
            "-c",
            "core.symlinks=false",
            "checkout",
            "--quiet",
            "FETCH_HEAD",
        ]);
        // What was actually checked out: refs/pull/<n>/head moves with every push, so
        // it can be newer than the delivery that asked for this review (runReview
        // compares the two).
        const headSha = stdoutOf(await git(["rev-parse", "HEAD"])).trim();
        if (!isCommitId(headSha)) {
            throw new Error("git rev-parse HEAD did not print a commit id");
        }

        // A PR must not supply its own trelix data directory: `trelix index`
        // would adopt a committed `.trelix/index.db` (or a `.trelix` link) as the
        // index. `rm` with `recursive`/`force` removes a link without following it.
        await rm(join(dir, ".trelix"), { recursive: true, force: true });

        return { path: dir, headSha, cleanup: removeTempDirs };
    } catch (err) {
        // A `Workspace` (and thus its `cleanup()`) only exists once this
        // function returns one — if any step above throws, this catch is the
        // only thing that can remove the temp dirs before rethrowing. The step's own
        // error is what the caller needs, so a removal failure is logged, not thrown
        // over it; the leftover is swept at the next boot.
        try {
            await removeTempDirs();
        } catch (cleanupErr) {
            console.warn(
                "[repo-checkout] checkoutPullRequest: could not remove the temp directories after a failed checkout:",
                cleanupErr,
            );
        }
        throw err;
    }
}

/**
 * Deletes any leftover `trelix-review-*` directories under the OS temp dir
 * (which includes the `trelix-review-aux-*` askpass/HOME directories)
 * from a previous instance — `checkoutPullRequest`'s own `try`/`catch`/
 * `finally`-based cleanup can't run if the process is killed outright
 * (SIGKILL, e.g. an OOM kill during `trelix index` under a constrained
 * memory limit), which would otherwise leak disk space indefinitely.
 * Intended to be called once at boot, before any webhook is served.
 *
 * Largely redundant on Render specifically (its free-tier filesystem is
 * already wiped on every restart), but cheap and correct for any other
 * deployment target where the filesystem persists across restarts.
 *
 * Best-effort: a removal failure for one entry (e.g. a permissions issue,
 * or the directory disappearing between listing and removal because its
 * own in-flight `cleanup()` won the race) is logged and skipped, never
 * thrown — a sweep that fails should not prevent the server from starting.
 */
export async function sweepStaleWorkspaces(): Promise<void> {
    const root = tmpdir();
    let entries: string[];
    try {
        entries = await readdir(root);
    } catch (err) {
        console.warn(
            `[repo-checkout] sweepStaleWorkspaces: could not list ${root}:`,
            err,
        );
        return;
    }

    const stale = entries.filter((name) => name.startsWith(TEMP_DIR_PREFIX));
    for (const name of stale) {
        try {
            await rm(join(root, name), { recursive: true, force: true });
        } catch (err) {
            console.warn(
                `[repo-checkout] sweepStaleWorkspaces: failed to remove ${name}:`,
                err,
            );
        }
    }
}
