import { execFile } from "node:child_process";
import {
    existsSync,
    lstatSync,
    mkdirSync,
    mkdtempSync,
    readFileSync,
    readdirSync,
    rmSync,
    statSync,
    symlinkSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { promisify } from "node:util";
import { rm } from "node:fs/promises";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { buildIndexChildEnv, buildReviewChildEnv } from "../src/child-env.js";
import {
    assertPrivateTmpdir,
    enterPrivateTmpdir,
    type PrivateTmpdir,
    withPrivateTmpdir,
} from "./support/private-tmpdir.js";
import {
    checkoutPullRequest,
    sweepStaleWorkspaces,
    type GitExecFile,
} from "../src/repo-checkout.js";

// Wraps the module's `rm` so a test can make one removal fail; every other call, in the
// module and in these tests, reaches the real one.
vi.mock("node:fs/promises", async (importOriginal) => {
    const actual = await importOriginal<typeof import("node:fs/promises")>();
    return { ...actual, rm: vi.fn(actual.rm) };
});

const execFileAsync = promisify(execFile);

// A distinctive marker string standing in for a real installation
// credential — never a value that's ever been valid against GitHub. Its
// only job is to be something the no-leak assertion below can grep for.
const CANARY_CREDENTIAL = "repo-checkout-test-canary-9f2c8b41ad";

// The module only lets git speak https. These tests fetch from a local bare repo,
// which is the "file" transport, so each one opts in explicitly.
const LOCAL_REMOTE_PROTOCOL = "file";

/** Recursively lists every regular file under `rootDir`. */
function listFilesRecursively(rootDir: string): string[] {
    const files: string[] = [];
    const stack = [rootDir];
    while (stack.length > 0) {
        const current = stack.pop() as string;
        for (const entry of readdirSync(current)) {
            const full = join(current, entry);
            const stat = statSync(full);
            if (stat.isDirectory()) {
                stack.push(full);
            } else if (stat.isFile()) {
                files.push(full);
            }
        }
    }
    return files;
}

/**
 * `trelix-review-*` dirs currently sitting under the OS temp dir. Only meaningful in a
 * private one (tests/support/private-tmpdir.ts): the shared one also holds what other
 * test files and other copies of the suite create while this runs.
 */
function listTrelixReviewDirs(): string[] {
    assertPrivateTmpdir();
    return readdirSync(tmpdir()).filter((name) =>
        name.startsWith("trelix-review-"),
    );
}

/** One git child, as the module's execFile seam saw it. */
interface RecordedGitCall {
    args: string[];
    cwd: string;
    env: NodeJS.ProcessEnv;
    /** What HOME held at the moment git ran. */
    homeEntries: string[];
}

/**
 * A `GitExecFile` that records every git child. With `realGit` it then runs the
 * real git; without, it does nothing, so the test needs no repository at all.
 */
function recordGitCalls(realGit: boolean): {
    run: GitExecFile;
    calls: RecordedGitCall[];
} {
    const calls: RecordedGitCall[] = [];
    const run: GitExecFile = async (file, args, options) => {
        calls.push({
            args,
            cwd: options.cwd,
            env: options.env,
            homeEntries: readdirSync(options.env.HOME as string),
        });
        return realGit ? execFileAsync(file, args, options) : undefined;
    };
    return { run, calls };
}

/** The `-c key=value` pairs git is given before its subcommand. */
function leadingConfigArgs(args: string[]): string[] {
    let end = 0;
    while (args[end] === "-c") end += 2;
    return args.slice(0, end);
}

function subcommandOf(args: string[]): string {
    return args[leadingConfigArgs(args).length];
}

/**
 * Makes the module's `rm` reject with EPERM for exactly `failingPath`, as it would for a
 * checkout holding something the process may not remove. Every other path is really
 * removed. Undo with `vi.mocked(rm).mockReset()`.
 */
async function makeRmFailFor(failingPath: string): Promise<void> {
    const actual =
        await vi.importActual<typeof import("node:fs/promises")>(
            "node:fs/promises",
        );
    vi.mocked(rm).mockImplementation(async (path, options) => {
        if (path === failingPath) {
            throw Object.assign(
                new Error("EPERM: operation not permitted, rm"),
                {
                    code: "EPERM",
                },
            );
        }
        return actual.rm(path, options);
    });
}

/**
 * A bare repo standing in for the base repo GitHub hosts, where
 * `refs/pull/<n>/head` is a commit holding exactly `files`.
 */
async function makePullRequestOrigin(
    files: Record<string, string>,
    prNumber: number,
): Promise<{ sourceDir: string; originDir: string }> {
    const sourceDir = mkdtempSync(join(tmpdir(), "trelix-checkout-pr-src-"));
    const git = (...args: string[]) =>
        execFileAsync("git", args, { cwd: sourceDir });
    await git("init", "--quiet");
    await git("config", "user.email", "test@example.com");
    await git("config", "user.name", "Test");
    for (const [path, content] of Object.entries(files)) {
        mkdirSync(dirname(join(sourceDir, path)), { recursive: true });
        writeFileSync(join(sourceDir, path), content);
    }
    await git("add", "-f", ".");
    await git("commit", "--quiet", "-m", "pull request head");
    const { stdout } = await git("rev-parse", "HEAD");

    const originDir = mkdtempSync(join(tmpdir(), "trelix-checkout-pr-origin-"));
    await execFileAsync("git", [
        "clone",
        "--quiet",
        "--bare",
        sourceDir,
        originDir,
    ]);
    await execFileAsync(
        "git",
        ["update-ref", `refs/pull/${prNumber}/head`, stdout.trim()],
        { cwd: originDir },
    );
    return { sourceDir, originDir };
}

describe("checkoutPullRequest", () => {
    let sourceDir: string;
    let originDir: string;
    let headSha: string;

    beforeEach(async () => {
        // A real local git repo, standing in for the PR author's actual
        // commit history.
        sourceDir = mkdtempSync(join(tmpdir(), "trelix-checkout-src-"));
        await execFileAsync("git", ["init", "--quiet"], { cwd: sourceDir });
        await execFileAsync(
            "git",
            ["config", "user.email", "test@example.com"],
            { cwd: sourceDir },
        );
        await execFileAsync("git", ["config", "user.name", "Test"], {
            cwd: sourceDir,
        });
        writeFileSync(join(sourceDir, "README.md"), "hello\n");
        await execFileAsync("git", ["add", "."], { cwd: sourceDir });
        await execFileAsync("git", ["commit", "--quiet", "-m", "initial"], {
            cwd: sourceDir,
        });
        const { stdout } = await execFileAsync("git", ["rev-parse", "HEAD"], {
            cwd: sourceDir,
        });
        headSha = stdout.trim();

        // A real bare repo standing in for the base repo GitHub hosts —
        // `refs/pull/<n>/head` exists here regardless of whether the PR came
        // from a branch on this repo or a fork; checkoutPullRequest must
        // never assume the head branch itself exists on this remote.
        originDir = mkdtempSync(join(tmpdir(), "trelix-checkout-origin-"));
        await execFileAsync("git", [
            "clone",
            "--quiet",
            "--bare",
            sourceDir,
            originDir,
        ]);
        await execFileAsync(
            "git",
            ["update-ref", "refs/pull/7/head", headSha],
            { cwd: originDir },
        );
    });

    afterEach(() => {
        rmSync(sourceDir, { recursive: true, force: true });
        rmSync(originDir, { recursive: true, force: true });
    });

    it("checks out the PR ref's HEAD from the origin", async () => {
        const workspace = await checkoutPullRequest(
            CANARY_CREDENTIAL,
            { owner: "o", repo: "r", prNumber: 7 },
            { remoteUrl: originDir, allowProtocol: LOCAL_REMOTE_PROTOCOL },
        );

        try {
            const { stdout } = await execFileAsync(
                "git",
                ["rev-parse", "HEAD"],
                {
                    cwd: workspace.path,
                },
            );
            expect(stdout.trim()).toBe(headSha);
            expect(
                readFileSync(join(workspace.path, "README.md"), "utf8"),
            ).toBe("hello\n");
        } finally {
            await workspace.cleanup();
        }
    });

    it("cleanup() removes the workspace directory", async () => {
        const workspace = await checkoutPullRequest(
            CANARY_CREDENTIAL,
            { owner: "o", repo: "r", prNumber: 7 },
            { remoteUrl: originDir, allowProtocol: LOCAL_REMOTE_PROTOCOL },
        );

        await workspace.cleanup();

        expect(existsSync(workspace.path)).toBe(false);
    });

    it("never writes the credential to any file in the workspace, including .git/config", async () => {
        const workspace = await checkoutPullRequest(
            CANARY_CREDENTIAL,
            { owner: "o", repo: "r", prNumber: 7 },
            { remoteUrl: originDir, allowProtocol: LOCAL_REMOTE_PROTOCOL },
        );

        try {
            // The concrete, falsifiable security property: grep every file
            // under the workspace (git objects, refs, config — everything) for
            // the literal canary string, byte for byte.
            const offenders = listFilesRecursively(workspace.path).filter(
                (file) =>
                    readFileSync(file, "latin1").includes(CANARY_CREDENTIAL),
            );
            expect(offenders).toEqual([]);
        } finally {
            await workspace.cleanup();
        }
    });

    it("cleans up the temp workspace even when the checkout fails (unreachable remote)", async () => {
        // A private temp dir: other test files create trelix-review-* directories in the
        // shared one (one per review, for the outcome record) while this runs.
        await withPrivateTmpdir(async () => {
            const before = listTrelixReviewDirs();
            const nonexistentRemote = join(
                tmpdir(),
                `trelix-checkout-nonexistent-${Date.now()}`,
            );

            await expect(
                checkoutPullRequest(
                    CANARY_CREDENTIAL,
                    { owner: "o", repo: "r", prNumber: 7 },
                    {
                        remoteUrl: nonexistentRemote,
                        allowProtocol: LOCAL_REMOTE_PROTOCOL,
                        timeoutMs: 15_000,
                    },
                ),
            ).rejects.toThrow();

            const after = listTrelixReviewDirs();
            expect(after).toEqual(before);
        });
    }, 20_000);
});

describe("checkoutPullRequest with a hostile PR head", () => {
    let sourceDir: string;
    let originDir: string;
    let outsideDir: string;
    const outsideContent = "outside-content-that-must-not-be-indexed\n";

    beforeEach(async () => {
        outsideDir = mkdtempSync(join(tmpdir(), "trelix-checkout-outside-"));
        writeFileSync(join(outsideDir, "target.txt"), outsideContent);

        sourceDir = mkdtempSync(join(tmpdir(), "trelix-checkout-hostile-src-"));
        const git = (...args: string[]) =>
            execFileAsync("git", args, { cwd: sourceDir });
        await git("init", "--quiet");
        await git("config", "user.email", "test@example.com");
        await git("config", "user.name", "Test");
        // Committed as a mode 120000 entry: a symlink to a file outside the
        // repo, plus a `.trelix` directory that would otherwise hand
        // `trelix index` an attacker-supplied data directory.
        symlinkSync(join(outsideDir, "target.txt"), join(sourceDir, "leak.md"));
        mkdirSync(join(sourceDir, ".trelix"));
        writeFileSync(join(sourceDir, ".trelix", "index.db"), "not a database");
        writeFileSync(join(sourceDir, "README.md"), "hello\n");
        await git("add", "-f", ".");
        await git("commit", "--quiet", "-m", "hostile");
        const { stdout } = await git("rev-parse", "HEAD");

        originDir = mkdtempSync(
            join(tmpdir(), "trelix-checkout-hostile-origin-"),
        );
        await execFileAsync("git", [
            "clone",
            "--quiet",
            "--bare",
            sourceDir,
            originDir,
        ]);
        await execFileAsync(
            "git",
            ["update-ref", "refs/pull/9/head", stdout.trim()],
            { cwd: originDir },
        );
    });

    afterEach(() => {
        rmSync(sourceDir, { recursive: true, force: true });
        rmSync(originDir, { recursive: true, force: true });
        rmSync(outsideDir, { recursive: true, force: true });
    });

    it("materialises a committed out-of-repo symlink as a plain file holding the link text", async () => {
        const workspace = await checkoutPullRequest(
            CANARY_CREDENTIAL,
            { owner: "o", repo: "r", prNumber: 9 },
            { remoteUrl: originDir, allowProtocol: LOCAL_REMOTE_PROTOCOL },
        );

        try {
            const leak = join(workspace.path, "leak.md");
            expect(lstatSync(leak).isSymbolicLink()).toBe(false);
            expect(lstatSync(leak).isFile()).toBe(true);
            // The file holds the link TARGET text, not the target's bytes.
            expect(readFileSync(leak, "utf8")).toBe(
                join(outsideDir, "target.txt"),
            );
            expect(readFileSync(leak, "utf8")).not.toContain(outsideContent);
        } finally {
            await workspace.cleanup();
        }
    });

    it("removes a committed .trelix entry so `trelix index` starts from a clean data dir", async () => {
        const workspace = await checkoutPullRequest(
            CANARY_CREDENTIAL,
            { owner: "o", repo: "r", prNumber: 9 },
            { remoteUrl: originDir, allowProtocol: LOCAL_REMOTE_PROTOCOL },
        );

        try {
            expect(existsSync(join(workspace.path, ".trelix"))).toBe(false);
            // The rest of the tree is still checked out.
            expect(
                readFileSync(join(workspace.path, "README.md"), "utf8"),
            ).toBe("hello\n");
        } finally {
            await workspace.cleanup();
        }
    });
});

// Names the App's process may hold that no git child has any use for: its own
// credentials, the platform's tokens, the operator's LLM settings, and an
// arbitrary name nobody listed.
const HOST_ONLY_NAMES = [
    "GITHUB_APP_PRIVATE_KEY",
    "GITHUB_WEBHOOK_SECRET",
    "GITHUB_TOKEN",
    "RAILWAY_TOKEN",
    "RAILWAY_PROJECT_ID",
    "OPENAI_API_KEY",
    "AZURE_API_KEY",
    "TRELIX_LLM_PROVIDER",
    "SOME_UNLISTED_SECRET",
];

describe("checkoutPullRequest git children", () => {
    afterEach(() => {
        vi.unstubAllEnvs();
    });

    const target = { owner: "o", repo: "r", prNumber: 7 };
    const httpsRemote = "https://example.invalid/o/r.git";

    it("runs init, remote add, fetch and checkout, each with the isolated allow-listed env and nothing from the host", async () => {
        for (const name of HOST_ONLY_NAMES) {
            vi.stubEnv(name, `host-value-of-${name}`);
        }
        vi.stubEnv("LANG", "en_US.UTF-8");
        // A host that allows more than https must not widen what git may speak.
        vi.stubEnv("GIT_ALLOW_PROTOCOL", "file:ssh:http");
        const { run, calls } = recordGitCalls(false);

        const workspace = await checkoutPullRequest(CANARY_CREDENTIAL, target, {
            remoteUrl: httpsRemote,
            execFile: run,
        });

        try {
            expect(calls.map((c) => subcommandOf(c.args))).toEqual([
                "init",
                "remote",
                "fetch",
                "checkout",
            ]);
            for (const call of calls) {
                // Exactly these names, so a new host variable cannot slip in.
                expect(Object.keys(call.env).sort()).toEqual([
                    "GIT_ALLOW_PROTOCOL",
                    "GIT_ASKPASS",
                    "GIT_CONFIG_GLOBAL",
                    "GIT_CONFIG_NOSYSTEM",
                    "GIT_CONFIG_SYSTEM",
                    "GIT_TERMINAL_PROMPT",
                    "HOME",
                    "LANG",
                    "PATH",
                    "TRELIX_GIT_TOKEN",
                    "XDG_CONFIG_HOME",
                ]);
                for (const name of HOST_ONLY_NAMES) {
                    expect(call.env).not.toHaveProperty(name);
                }
                expect(call.env).toMatchObject({
                    GIT_ALLOW_PROTOCOL: "https",
                    GIT_CONFIG_GLOBAL: "/dev/null",
                    GIT_CONFIG_SYSTEM: "/dev/null",
                    GIT_CONFIG_NOSYSTEM: "1",
                    GIT_TERMINAL_PROMPT: "0",
                    LANG: "en_US.UTF-8",
                    TRELIX_GIT_TOKEN: CANARY_CREDENTIAL,
                });
                // git reads no user config: HOME is an empty directory.
                expect(call.env.XDG_CONFIG_HOME).toBe(call.env.HOME);
                expect(call.homeEntries).toEqual([]);
                // The token travels by env only, never on a command line.
                expect(call.args.join(" ")).not.toContain(CANARY_CREDENTIAL);
                // Every child carries the command-scope safety config.
                expect(call.args.slice(0, 6)).toEqual([
                    "-c",
                    "safe.bareRepository=explicit",
                    "-c",
                    "protocol.file.allow=never",
                    "-c",
                    "credential.helper=",
                ]);
            }
        } finally {
            await workspace.cleanup();
        }
    });

    it("builds the repository from an empty template and keeps the fetch shallow, tagless and submodule-free", async () => {
        const { run, calls } = recordGitCalls(false);

        const workspace = await checkoutPullRequest(CANARY_CREDENTIAL, target, {
            remoteUrl: httpsRemote,
            execFile: run,
        });
        await workspace.cleanup();

        const afterConfig = (i: number) => calls[i].args.slice(6);
        expect(afterConfig(0)).toEqual(["init", "--quiet", "--template="]);
        expect(afterConfig(1)).toEqual([
            "remote",
            "add",
            "origin",
            httpsRemote,
        ]);
        expect(afterConfig(2)).toEqual([
            "fetch",
            "--depth",
            "1",
            "--no-tags",
            "origin",
            "refs/pull/7/head",
        ]);
        expect(afterConfig(3)).toEqual([
            "-c",
            "core.symlinks=false",
            "checkout",
            "--quiet",
            "FETCH_HEAD",
        ]);
    });

    it("keeps the askpass helper and git's HOME outside the checkout, in a trelix-review-aux-* directory that cleanup() removes", async () => {
        const { run, calls } = recordGitCalls(false);

        const workspace = await checkoutPullRequest(CANARY_CREDENTIAL, target, {
            remoteUrl: httpsRemote,
            execFile: run,
        });
        const askpass = calls[0].env.GIT_ASKPASS as string;
        const auxDir = dirname(askpass);

        expect(existsSync(askpass)).toBe(true);
        expect(auxDir.startsWith(workspace.path)).toBe(false);
        expect(dirname(auxDir)).toBe(tmpdir());
        expect(auxDir.split("/").pop()).toMatch(/^trelix-review-aux-/);
        expect(dirname(calls[0].env.HOME as string)).toBe(auxDir);

        await workspace.cleanup();

        expect(existsSync(workspace.path)).toBe(false);
        expect(existsSync(auxDir)).toBe(false);
    });

    it("answers git's credential prompts with x-access-token and the token from TRELIX_GIT_TOKEN", async () => {
        const { run, calls } = recordGitCalls(false);
        const workspace = await checkoutPullRequest(CANARY_CREDENTIAL, target, {
            remoteUrl: httpsRemote,
            execFile: run,
        });

        try {
            const askpass = calls[0].env.GIT_ASKPASS as string;
            const answer = async (prompt: string) =>
                (
                    await execFileAsync(askpass, [prompt], {
                        env: { TRELIX_GIT_TOKEN: CANARY_CREDENTIAL },
                    })
                ).stdout.trim();

            expect(await answer("Username for 'https://github.com': ")).toBe(
                "x-access-token",
            );
            expect(await answer("Password for 'https://github.com': ")).toBe(
                CANARY_CREDENTIAL,
            );
        } finally {
            await workspace.cleanup();
        }
    });

    it("removes the checkout and the aux directory when a git step fails", async () => {
        const seen: RecordedGitCall[] = [];
        const failingFetch: GitExecFile = async (_file, args, options) => {
            seen.push({
                args,
                cwd: options.cwd,
                env: options.env,
                homeEntries: [],
            });
            if (subcommandOf(args) === "fetch") {
                throw new Error("simulated fetch failure");
            }
            return undefined;
        };

        await expect(
            checkoutPullRequest(CANARY_CREDENTIAL, target, {
                remoteUrl: httpsRemote,
                execFile: failingFetch,
            }),
        ).rejects.toThrow("simulated fetch failure");

        const { cwd, env } = seen[seen.length - 1];
        expect(existsSync(cwd)).toBe(false);
        expect(existsSync(dirname(env.GIT_ASKPASS as string))).toBe(false);
    });

    describe("when the checkout cannot be removed", () => {
        const leftovers: string[] = [];

        afterEach(() => {
            vi.mocked(rm).mockReset();
            vi.restoreAllMocks();
            for (const path of leftovers) {
                rmSync(path, { recursive: true, force: true });
            }
            leftovers.length = 0;
        });

        it("cleanup() still removes the aux directory, then rejects with the removal failure", async () => {
            const { run, calls } = recordGitCalls(false);
            const workspace = await checkoutPullRequest(
                CANARY_CREDENTIAL,
                target,
                { remoteUrl: httpsRemote, execFile: run },
            );
            const auxDir = dirname(calls[0].env.GIT_ASKPASS as string);
            leftovers.push(workspace.path, auxDir);
            await makeRmFailFor(workspace.path);

            const cleanup = workspace.cleanup();

            await expect(cleanup).rejects.toBeInstanceOf(AggregateError);
            await expect(cleanup).rejects.toMatchObject({
                errors: [expect.objectContaining({ code: "EPERM" })],
            });
            expect(existsSync(auxDir)).toBe(false);
            expect(existsSync(workspace.path)).toBe(true);
        });

        it("a failed git step still rejects with the git error, warns about the leftover and removes the aux directory", async () => {
            const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
            const seen: RecordedGitCall[] = [];
            const failingFetch: GitExecFile = async (_file, args, options) => {
                seen.push({
                    args,
                    cwd: options.cwd,
                    env: options.env,
                    homeEntries: [],
                });
                if (subcommandOf(args) === "fetch") {
                    await makeRmFailFor(options.cwd);
                    throw new Error("simulated fetch failure");
                }
                return undefined;
            };

            await expect(
                checkoutPullRequest(CANARY_CREDENTIAL, target, {
                    remoteUrl: httpsRemote,
                    execFile: failingFetch,
                }),
            ).rejects.toThrow("simulated fetch failure");

            const { cwd, env } = seen[seen.length - 1];
            const auxDir = dirname(env.GIT_ASKPASS as string);
            leftovers.push(cwd, auxDir);
            expect(existsSync(auxDir)).toBe(false);
            expect(existsSync(cwd)).toBe(true);
            expect(warn).toHaveBeenCalledWith(
                expect.stringContaining(
                    "could not remove the temp directories",
                ),
                expect.any(AggregateError),
            );
        });
    });
});

describe("checkoutPullRequest with an embedded bare repository in the PR", () => {
    let sourceDir: string;
    let originDir: string;
    let markerDir: string;
    let markerPath: string;

    beforeEach(async () => {
        markerDir = mkdtempSync(join(tmpdir(), "trelix-checkout-marker-"));
        markerPath = join(markerDir, "fsmonitor-ran");
        // A bare repository committed into the tree as ordinary files (git refuses
        // only a nested `.git`). Its config points core.fsmonitor at a command;
        // bare = false plus a worktree makes `git status` and `git diff` consult
        // it. Run from inside `evil/`, git adopts this repository and runs the
        // command, unless it was told not to discover bare repositories.
        ({ sourceDir, originDir } = await makePullRequestOrigin(
            {
                "README.md": "hello\n",
                "evil/HEAD": "ref: refs/heads/main\n",
                "evil/config": [
                    "[core]",
                    "\trepositoryformatversion = 0",
                    "\tbare = false",
                    "\tworktree = ..",
                    `\tfsmonitor = "touch '${markerPath}'"`,
                    "",
                ].join("\n"),
                "evil/objects/info/.keep": "",
                "evil/refs/heads/.keep": "",
            },
            11,
        ));
    });

    afterEach(() => {
        rmSync(sourceDir, { recursive: true, force: true });
        rmSync(originDir, { recursive: true, force: true });
        rmSync(markerDir, { recursive: true, force: true });
    });

    const target = { owner: "o", repo: "r", prNumber: 11 };

    it("control: plain git run inside the embedded repository does execute its fsmonitor command", async () => {
        const workspace = await checkoutPullRequest(CANARY_CREDENTIAL, target, {
            remoteUrl: originDir,
            allowProtocol: LOCAL_REMOTE_PROTOCOL,
        });

        try {
            // Neutral config only, so a developer's own safe.bareRepository
            // setting cannot switch the fixture off.
            await execFileAsync("git", ["status"], {
                cwd: join(workspace.path, "evil"),
                env: {
                    PATH: process.env.PATH,
                    HOME: markerDir,
                    GIT_CONFIG_GLOBAL: "/dev/null",
                    GIT_CONFIG_NOSYSTEM: "1",
                },
            });

            expect(existsSync(markerPath)).toBe(true);
        } finally {
            await workspace.cleanup();
        }
    });

    it("never runs the embedded fsmonitor when git is used inside that directory with the module's own env and config", async () => {
        const { run, calls } = recordGitCalls(true);
        const workspace = await checkoutPullRequest(CANARY_CREDENTIAL, target, {
            remoteUrl: originDir,
            allowProtocol: LOCAL_REMOTE_PROTOCOL,
            execFile: run,
        });

        try {
            // The exact env and -c config the module gave its own checkout child.
            const checkout = calls[calls.length - 1];
            expect(subcommandOf(checkout.args)).toBe("checkout");

            for (const subcommand of ["status", "diff"]) {
                await expect(
                    execFileAsync(
                        "git",
                        [...leadingConfigArgs(checkout.args), subcommand],
                        {
                            cwd: join(workspace.path, "evil"),
                            env: checkout.env,
                        },
                    ),
                ).rejects.toThrow();
                expect(existsSync(markerPath)).toBe(false);
            }
        } finally {
            await workspace.cleanup();
        }
    });

    it.each([
        ["index", (base: NodeJS.ProcessEnv) => buildIndexChildEnv(base)],
        [
            "review",
            (base: NodeJS.ProcessEnv) => buildReviewChildEnv("token", base),
        ],
    ])(
        "never runs the embedded fsmonitor for the git calls a `trelix %s` child makes itself, which get no -c config",
        async (_kind, build) => {
            const workspace = await checkoutPullRequest(
                CANARY_CREDENTIAL,
                target,
                { remoteUrl: originDir, allowProtocol: LOCAL_REMOTE_PROTOCOL },
            );

            try {
                // The trelix child's own env. Only the two config-file switches are
                // added, so a developer's gitconfig cannot change the outcome; the
                // defence under test is what build() put in the env.
                const env = {
                    ...build({ PATH: process.env.PATH, HOME: markerDir }),
                    GIT_CONFIG_GLOBAL: "/dev/null",
                    GIT_CONFIG_NOSYSTEM: "1",
                };

                for (const subcommand of ["status", "diff"]) {
                    await expect(
                        execFileAsync("git", [subcommand], {
                            cwd: join(workspace.path, "evil"),
                            env,
                        }),
                    ).rejects.toThrow();
                    expect(existsSync(markerPath)).toBe(false);
                }
                // The same env still works where trelix runs git: the checkout root.
                for (const args of [
                    ["rev-parse", "HEAD"],
                    ["log", "--oneline"],
                    ["diff", "--stat", "HEAD"],
                ]) {
                    await execFileAsync("git", args, {
                        cwd: workspace.path,
                        env,
                    });
                }
            } finally {
                await workspace.cleanup();
            }
        },
    );

    it("still checks the rest of the tree out", async () => {
        const workspace = await checkoutPullRequest(CANARY_CREDENTIAL, target, {
            remoteUrl: originDir,
            allowProtocol: LOCAL_REMOTE_PROTOCOL,
        });

        try {
            expect(
                readFileSync(join(workspace.path, "README.md"), "utf8"),
            ).toBe("hello\n");
            // An empty template: git copied no hooks directory from the host.
            expect(existsSync(join(workspace.path, ".git", "hooks"))).toBe(
                false,
            );
        } finally {
            await workspace.cleanup();
        }
    });
});

describe("checkoutPullRequest with a PR that tracks .git-askpass.sh", () => {
    let sourceDir: string;
    let originDir: string;
    const trackedContent = "#!/bin/sh\necho tracked-by-the-pull-request\n";

    beforeEach(async () => {
        ({ sourceDir, originDir } = await makePullRequestOrigin(
            { "README.md": "hello\n", ".git-askpass.sh": trackedContent },
            12,
        ));
    });

    afterEach(() => {
        rmSync(sourceDir, { recursive: true, force: true });
        rmSync(originDir, { recursive: true, force: true });
    });

    it("checks the file out as the PR wrote it, because the askpass helper lives elsewhere", async () => {
        const { run, calls } = recordGitCalls(true);
        const workspace = await checkoutPullRequest(
            CANARY_CREDENTIAL,
            { owner: "o", repo: "r", prNumber: 12 },
            {
                remoteUrl: originDir,
                allowProtocol: LOCAL_REMOTE_PROTOCOL,
                execFile: run,
            },
        );

        try {
            expect(
                readFileSync(join(workspace.path, ".git-askpass.sh"), "utf8"),
            ).toBe(trackedContent);
            expect(
                (calls[0].env.GIT_ASKPASS as string).startsWith(workspace.path),
            ).toBe(false);
        } finally {
            await workspace.cleanup();
        }
    });
});

describe("sweepStaleWorkspaces", () => {
    const leaked: string[] = [];
    let privateTmpdir: PrivateTmpdir;

    // sweepStaleWorkspaces removes every trelix-review-* entry of the temp dir, so these
    // tests run in a private one: in the shared one it would remove the workspaces and
    // outcome directories of reviews that other test files are running.
    beforeEach(() => {
        privateTmpdir = enterPrivateTmpdir();
        assertPrivateTmpdir();
    });

    afterEach(() => {
        for (const dir of leaked) {
            rmSync(dir, { recursive: true, force: true });
        }
        leaked.length = 0;
        privateTmpdir.leave();
    });

    it("removes leftover trelix-review-* directories from a previous crashed instance", async () => {
        const dir = mkdtempSync(join(tmpdir(), "trelix-review-"));
        leaked.push(dir);
        writeFileSync(join(dir, "leftover.txt"), "from a crashed instance\n");

        await sweepStaleWorkspaces();

        expect(existsSync(dir)).toBe(false);
    });

    it("removes leftover trelix-review-aux-* askpass/HOME directories too", async () => {
        const auxDir = mkdtempSync(join(tmpdir(), "trelix-review-aux-"));
        leaked.push(auxDir);
        writeFileSync(join(auxDir, "askpass.sh"), "#!/bin/sh\n");

        await sweepStaleWorkspaces();

        expect(existsSync(auxDir)).toBe(false);
    });

    it("never touches directories that don't match the trelix-review- prefix", async () => {
        const unrelated = mkdtempSync(join(tmpdir(), "trelix-checkout-src-"));
        leaked.push(unrelated);

        await sweepStaleWorkspaces();

        expect(existsSync(unrelated)).toBe(true);
    });

    it("does not throw when a matching entry disappears between listing and removal", async () => {
        // sweepStaleWorkspaces must tolerate a dir being removed concurrently
        // (e.g. by that workspace's own in-flight cleanup()) -- rm's `force`
        // option already makes a single removal idempotent; this proves the
        // sweep as a whole doesn't throw even if nothing is actually there.
        await expect(sweepStaleWorkspaces()).resolves.toBeUndefined();
    });
});
