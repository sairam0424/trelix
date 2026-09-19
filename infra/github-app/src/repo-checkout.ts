import { execFile } from "node:child_process";
import { mkdtemp, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

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

export interface CheckoutOptions {
    timeoutMs?: number;
    /** Overridable so tests can point at a local bare repo instead of a real GitHub remote. */
    remoteUrl?: string;
}

export interface Workspace {
    path: string;
    cleanup(): Promise<void>;
}

/**
 * Static, secret-free askpass script: it never contains the token itself
 * — only a reference to the `TRELIX_GIT_TOKEN` env var, which git's
 * askpass protocol reads at fetch time via this script's stdout. The
 * token therefore never appears in this file, in `argv` for any `git`
 * subprocess, or in `.git/config` (no credential helper caches it, and
 * the remote URL never embeds it).
 */
const GIT_ASKPASS_SCRIPT = `#!/bin/sh
# Invoked once per credential prompt by git's askpass protocol ("Username
# for ...", "Password for ...") — the requested value goes on stdout.
case "$1" in
  Username*) echo "x-access-token" ;;
  Password*) echo "$TRELIX_GIT_TOKEN" ;;
esac
`;

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

    const dir = await mkdtemp(join(tmpdir(), "trelix-review-"));

    try {
        const askpassPath = join(dir, ".git-askpass.sh");
        // The askpass script lives inside the workspace itself, so the same
        // `rm(dir, ...)` that removes the clone also removes it — no separate
        // cleanup step needed.
        await writeFile(askpassPath, GIT_ASKPASS_SCRIPT, { mode: 0o700 });

        await execFileAsync("git", ["init", "--quiet"], {
            cwd: dir,
            timeout: timeoutMs,
        });
        // Never embeds the token in the remote URL — credentials pass through
        // GIT_ASKPASS/env only (see the `fetch` call below).
        await execFileAsync("git", ["remote", "add", "origin", remoteUrl], {
            cwd: dir,
            timeout: timeoutMs,
        });

        await execFileAsync(
            "git",
            [
                "-c",
                "credential.helper=", // disable any configured helper so our askpass is always used, never a stale cached credential
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
            ],
            {
                cwd: dir,
                timeout: timeoutMs,
                // execFile's `env` option REPLACES the inherited environment
                // entirely if set at all — spreading `...process.env` first is
                // required, or PATH resolution for `git` itself breaks.
                env: {
                    ...process.env,
                    GIT_ASKPASS: askpassPath,
                    GIT_TERMINAL_PROMPT: "0",
                    TRELIX_GIT_TOKEN: token,
                },
            },
        );

        await execFileAsync("git", ["checkout", "--quiet", "FETCH_HEAD"], {
            cwd: dir,
            timeout: timeoutMs,
        });

        return {
            path: dir,
            cleanup: async () => {
                await rm(dir, { recursive: true, force: true });
            },
        };
    } catch (err) {
        // A `Workspace` (and thus its `cleanup()`) only exists once this
        // function returns one — if any step above throws, this catch is the
        // only thing that can remove the temp dir before rethrowing.
        await rm(dir, { recursive: true, force: true });
        throw err;
    }
}

/**
 * Deletes any leftover `trelix-review-*` directories under the OS temp dir
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

    const stale = entries.filter((name) => name.startsWith("trelix-review-"));
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
