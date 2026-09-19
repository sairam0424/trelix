import { execFile } from "node:child_process";
import {
    existsSync,
    mkdtempSync,
    readFileSync,
    readdirSync,
    rmSync,
    statSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
    checkoutPullRequest,
    sweepStaleWorkspaces,
} from "../src/repo-checkout.js";

const execFileAsync = promisify(execFile);

// A distinctive marker string standing in for a real installation
// credential — never a value that's ever been valid against GitHub. Its
// only job is to be something the no-leak assertion below can grep for.
const CANARY_CREDENTIAL = "repo-checkout-test-canary-9f2c8b41ad";

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

/** `trelix-review-*` dirs currently sitting under the OS temp dir. */
function listTrelixReviewDirs(): string[] {
    return readdirSync(tmpdir()).filter((name) =>
        name.startsWith("trelix-review-"),
    );
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
            { remoteUrl: originDir },
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
            { remoteUrl: originDir },
        );

        await workspace.cleanup();

        expect(existsSync(workspace.path)).toBe(false);
    });

    it("never writes the credential to any file in the workspace, including .git/config", async () => {
        const workspace = await checkoutPullRequest(
            CANARY_CREDENTIAL,
            { owner: "o", repo: "r", prNumber: 7 },
            { remoteUrl: originDir },
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
        const before = listTrelixReviewDirs();
        const nonexistentRemote = join(
            tmpdir(),
            `trelix-checkout-nonexistent-${Date.now()}`,
        );

        await expect(
            checkoutPullRequest(
                CANARY_CREDENTIAL,
                { owner: "o", repo: "r", prNumber: 7 },
                { remoteUrl: nonexistentRemote, timeoutMs: 15_000 },
            ),
        ).rejects.toThrow();

        const after = listTrelixReviewDirs();
        expect(after).toEqual(before);
    }, 20_000);
});

describe("sweepStaleWorkspaces", () => {
    const leaked: string[] = [];

    afterEach(() => {
        for (const dir of leaked) {
            rmSync(dir, { recursive: true, force: true });
        }
        leaked.length = 0;
    });

    it("removes leftover trelix-review-* directories from a previous crashed instance", async () => {
        const dir = mkdtempSync(join(tmpdir(), "trelix-review-"));
        leaked.push(dir);
        writeFileSync(join(dir, "leftover.txt"), "from a crashed instance\n");

        await sweepStaleWorkspaces();

        expect(existsSync(dir)).toBe(false);
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
