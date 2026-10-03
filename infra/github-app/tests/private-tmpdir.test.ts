import { existsSync, mkdtempSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { sweepStaleWorkspaces } from "../src/repo-checkout.js";
import {
    assertPrivateTmpdir,
    enterPrivateTmpdir,
    withPrivateTmpdir,
} from "./support/private-tmpdir.js";

describe("withPrivateTmpdir", () => {
    const madeInTheSharedDir: string[] = [];

    afterEach(() => {
        for (const dir of madeInTheSharedDir.splice(0)) {
            rmSync(dir, { recursive: true, force: true });
        }
    });

    it("makes os.tmpdir() a fresh empty directory of its own, and puts it back", async () => {
        const shared = tmpdir();

        await withPrivateTmpdir((dir) => {
            expect(tmpdir()).toBe(dir);
            expect(dir).not.toBe(shared);
            expect(basename(dir)).toMatch(
                /^trelix-private-tmp-[A-Za-z0-9]{6}$/,
            );
            expect(readdirSync(dir)).toEqual([]);
        });

        expect(tmpdir()).toBe(shared);
    });

    it("names it so that sweepStaleWorkspaces in another process, which removes every trelix-review-* entry, leaves it alone", async () => {
        await withPrivateTmpdir((dir) => {
            expect(basename(dir).startsWith("trelix-review-")).toBe(false);
        });
    });

    it("removes the directory when the work ends, and when it throws", async () => {
        let seen = "";
        await withPrivateTmpdir((dir) => {
            seen = dir;
            expect(existsSync(dir)).toBe(true);
        });
        expect(existsSync(seen)).toBe(false);

        let thrownIn = "";
        await expect(
            withPrivateTmpdir((dir) => {
                thrownIn = dir;
                throw new Error("work failed (simulated)");
            }),
        ).rejects.toThrow("work failed (simulated)");
        expect(existsSync(thrownIn)).toBe(false);
    });

    it("gives back TMPDIR as it was: a value, or none at all", async () => {
        const original = process.env.TMPDIR;
        try {
            process.env.TMPDIR = "/tmp";
            await withPrivateTmpdir(() => undefined);
            expect(process.env.TMPDIR).toBe("/tmp");

            delete process.env.TMPDIR;
            await withPrivateTmpdir(() => undefined);
            expect("TMPDIR" in process.env).toBe(false);
        } finally {
            if (original === undefined) {
                delete process.env.TMPDIR;
            } else {
                process.env.TMPDIR = original;
            }
        }
    });

    it("can be left twice", () => {
        const entered = enterPrivateTmpdir();

        entered.leave();

        expect(() => entered.leave()).not.toThrow();
    });

    it("does not see what another test file leaves in the shared temp directory", async () => {
        const shared = tmpdir();
        const workspace = mkdtempSync(join(shared, "trelix-review-"));
        const outcome = mkdtempSync(join(shared, "trelix-review-outcome-"));
        madeInTheSharedDir.push(workspace, outcome);

        await withPrivateTmpdir(() => {
            expect(readdirSync(tmpdir())).toEqual([]);
        });
    });

    it("keeps sweepStaleWorkspaces away from a directory in the shared temp directory", async () => {
        const shared = tmpdir();
        const inFlight = mkdtempSync(join(shared, "trelix-review-"));
        madeInTheSharedDir.push(inFlight);

        await withPrivateTmpdir(async () => {
            await sweepStaleWorkspaces();
        });

        expect(existsSync(inFlight)).toBe(true);
    });
});

describe("assertPrivateTmpdir", () => {
    it("fails in the shared temp directory, which a test that lists or sweeps must not use", () => {
        expect(() => assertPrivateTmpdir()).toThrow(
            "must run in a private one",
        );
    });

    it("passes in a private one", async () => {
        await withPrivateTmpdir(() => {
            expect(() => assertPrivateTmpdir()).not.toThrow();
        });
    });
});
