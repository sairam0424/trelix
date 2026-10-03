/**
 * A temp directory of a test's own, for tests whose subject is the contents of the
 * OS temp directory.
 *
 * repo-checkout.test.ts checks that a failed checkout leaves no `trelix-review-*`
 * entry behind, and tests sweepStaleWorkspaces, which removes every such entry.
 * Both look at the shared OS temp directory, where other test files (vitest runs
 * them in parallel) and other copies of the suite create `trelix-review-*`
 * directories of their own: a listing taken twice then differed, and a sweep removed
 * a workspace from under a test that was using it. Pointing TMPDIR at a fresh
 * directory for the length of the test makes `os.tmpdir()`, which the code under
 * test calls each time it needs a directory, return a place nobody else writes to.
 */
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";

/** Not a `trelix-review-` name: sweepStaleWorkspaces in another process must not be able to remove it. */
const PRIVATE_TMPDIR_PREFIX = "trelix-private-tmp-";

export interface PrivateTmpdir {
    /** The directory `os.tmpdir()` returns until `leave()`. */
    dir: string;
    /** Puts TMPDIR back as it was and removes the directory. Safe to call twice. */
    leave(): void;
}

export function enterPrivateTmpdir(): PrivateTmpdir {
    const original = process.env.TMPDIR;
    const dir = mkdtempSync(join(tmpdir(), PRIVATE_TMPDIR_PREFIX));
    process.env.TMPDIR = dir;
    let left = false;
    return {
        dir,
        leave() {
            if (left) {
                return;
            }
            left = true;
            if (original === undefined) {
                delete process.env.TMPDIR;
            } else {
                process.env.TMPDIR = original;
            }
            rmSync(dir, { recursive: true, force: true });
        },
    };
}

/** Runs `work` with a private temp directory, and leaves it however `work` ends. */
export async function withPrivateTmpdir<T>(
    work: (dir: string) => Promise<T> | T,
): Promise<T> {
    const entered = enterPrivateTmpdir();
    try {
        return await work(entered.dir);
    } finally {
        entered.leave();
    }
}

/**
 * Throws unless `os.tmpdir()` is a private directory from enterPrivateTmpdir: a test
 * that lists or sweeps the temp directory calls this first, so that forgetting to
 * enter one fails at once instead of making the test depend on what else is running.
 */
export function assertPrivateTmpdir(): void {
    if (!basename(tmpdir()).startsWith(PRIVATE_TMPDIR_PREFIX)) {
        throw new Error(
            `this test looks at the temp directory and must run in a private one, but os.tmpdir() is ${tmpdir()}`,
        );
    }
}
