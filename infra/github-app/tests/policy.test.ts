import { describe, expect, it } from "vitest";
import {
    isInstallationAllowed,
    parseAccountList,
    parseInstallationList,
    parseInstallPolicyMode,
    type InstallPolicy,
} from "../src/policy.js";

const allowlist = (
    accounts: string[],
    installations: number[],
): InstallPolicy => ({
    mode: "allowlist",
    accounts: new Set(accounts),
    installations: new Set(installations),
});

/** Quadratic code takes seconds on this; linear code takes a millisecond. */
const HOSTILE_INPUT_BUDGET_MS = 500;
const HOSTILE_INPUT_CHARS = 50_000;

function elapsedMs(work: () => void): number {
    const started = performance.now();
    work();
    return performance.now() - started;
}

describe("parseInstallPolicyMode", () => {
    it.each([
        ["open", "open"],
        ["allowlist", "allowlist"],
        ["ALLOWLIST", "allowlist"],
        ["  Open  ", "open"],
        ["\tallowlist\n", "allowlist"],
    ])("reads %j as %s", (raw, mode) => {
        expect(parseInstallPolicyMode(raw)).toBe(mode);
    });

    it.each([undefined, "", "   "])(
        "reads %j (unset or blank) as open, the default",
        (raw) => {
            expect(parseInstallPolicyMode(raw)).toBe("open");
        },
    );

    it.each(["allow", "allow-list", "deny", "closed", "true", "1", "open,"])(
        "refuses the unknown policy %j",
        (raw) => {
            expect(() => parseInstallPolicyMode(raw)).toThrow(
                'Invalid TRELIX_APP_INSTALL_POLICY: expected "open" or "allowlist"',
            );
        },
    );

    it("does not quote the rejected value in the error", () => {
        let message = "";
        try {
            parseInstallPolicyMode("canary-not-a-policy");
        } catch (err) {
            message = String((err as Error).message);
        }

        expect(message).not.toContain("canary-not-a-policy");
        expect(message).toContain("TRELIX_APP_INSTALL_POLICY");
    });
});

describe("parseAccountList", () => {
    it("splits on commas and trims whitespace around each entry", () => {
        expect([
            ...parseAccountList("  sairam0424 ,\n acme-corp\t, octo-cat "),
        ]).toEqual(["sairam0424", "acme-corp", "octo-cat"]);
    });

    it("lowercases, because GitHub logins are case-insensitive", () => {
        expect([...parseAccountList("SaIrAm0424")]).toEqual(["sairam0424"]);
    });

    it("merges duplicates, including ones that differ only in case or spacing", () => {
        expect([...parseAccountList("a, A , a,b,B")]).toEqual(["a", "b"]);
    });

    it.each([undefined, "", "   ", ",", " , ,, "])(
        "reads %j as an empty list",
        (raw) => {
            expect(parseAccountList(raw).size).toBe(0);
        },
    );

    it("does not strip an @ or split on spaces: what is listed is matched as written", () => {
        expect([...parseAccountList("@someone, two words")]).toEqual([
            "@someone",
            "two words",
        ]);
    });
});

describe("parseInstallationList", () => {
    it("reads decimal installation ids, trimmed", () => {
        const { ids, ignored } = parseInstallationList(" 123 ,456,\n789 ");

        expect([...ids]).toEqual([123, 456, 789]);
        expect(ignored).toBe(0);
    });

    it("merges duplicates", () => {
        const { ids, ignored } = parseInstallationList("5,5, 5 ,6");

        expect([...ids]).toEqual([5, 6]);
        expect(ignored).toBe(0);
    });

    it.each([
        ["a word", "octocat"],
        ["a sign", "-5"],
        ["a plus sign", "+5"],
        ["zero", "0"],
        ["a leading zero", "007"],
        ["a fraction", "1.5"],
        ["an exponent", "1e3"],
        ["hexadecimal", "0x10"],
        ["two numbers in one entry", "1 2"],
        ["more digits than a safe integer", "99999999999999999"],
        ["a number past the safe integer range", "9007199254740993"],
        ["unicode digits", "١٢٣"],
    ])("counts %s as ignored, allowing nothing", (_name, entry) => {
        const { ids, ignored } = parseInstallationList(`${entry},42`);

        expect([...ids]).toEqual([42]);
        expect(ignored).toBe(1);
    });

    it("does not count blank entries as garbage", () => {
        const { ids, ignored } = parseInstallationList(",, 7 ,");

        expect([...ids]).toEqual([7]);
        expect(ignored).toBe(0);
    });

    it.each([undefined, "", "  "])("reads %j as an empty list", (raw) => {
        const { ids, ignored } = parseInstallationList(raw);

        expect(ids.size).toBe(0);
        expect(ignored).toBe(0);
    });

    it("accepts the largest safe integer", () => {
        const { ids } = parseInstallationList("9007199254740991");

        expect([...ids]).toEqual([9007199254740991]);
    });
});

describe("isInstallationAllowed", () => {
    const target = { account: "SairAm0424", installationId: 777 };

    it("serves everyone in open mode, even with lists set", () => {
        const policy: InstallPolicy = {
            mode: "open",
            accounts: new Set(["someone-else"]),
            installations: new Set([1]),
        };

        expect(isInstallationAllowed(policy, target)).toBe(true);
    });

    it("serves nobody in allowlist mode when both lists are empty", () => {
        expect(isInstallationAllowed(allowlist([], []), target)).toBe(false);
    });

    it("serves a listed account, whatever the case of the login", () => {
        expect(
            isInstallationAllowed(allowlist(["sairam0424"], []), target),
        ).toBe(true);
    });

    it("serves a listed installation id", () => {
        expect(isInstallationAllowed(allowlist([], [777]), target)).toBe(true);
    });

    it("serves an installation if either its account or its id is listed", () => {
        expect(isInstallationAllowed(allowlist(["other"], [777]), target)).toBe(
            true,
        );
        expect(
            isInstallationAllowed(allowlist(["sairam0424"], [1]), target),
        ).toBe(true);
    });

    it("refuses an installation that is on neither list", () => {
        expect(
            isInstallationAllowed(allowlist(["other"], [1, 2]), target),
        ).toBe(false);
    });

    it("does not match a login that merely contains or starts with a listed one", () => {
        const policy = allowlist(["sairam"], []);

        expect(
            isInstallationAllowed(policy, {
                account: "sairam0424",
                installationId: 1,
            }),
        ).toBe(false);
        expect(
            isInstallationAllowed(policy, {
                account: "xsairam",
                installationId: 1,
            }),
        ).toBe(false);
    });

    it("refuses when the mode is one it does not know", () => {
        const policy = {
            mode: "everyone",
            accounts: new Set<string>(),
            installations: new Set<number>(),
        } as unknown as InstallPolicy;

        expect(isInstallationAllowed(policy, target)).toBe(false);
    });

    it("does not treat a wildcard entry as a wildcard", () => {
        expect(isInstallationAllowed(allowlist(["*"], []), target)).toBe(false);
    });
});

describe("hostile input stays linear", () => {
    it("parses a 50,000-character list whose entries are mostly whitespace", () => {
        const raw = `a,${" ".repeat(HOSTILE_INPUT_CHARS)}x`;
        let accounts = new Set<string>();

        const ms = elapsedMs(() => {
            accounts = parseAccountList(raw);
        });

        expect([...accounts]).toEqual(["a", "x"]);
        expect(ms).toBeLessThan(HOSTILE_INPUT_BUDGET_MS);
    });

    it("parses a 50,000-character installation list with whitespace-padded garbage", () => {
        const raw = `1,${" ".repeat(HOSTILE_INPUT_CHARS)}x`;
        let parsed = parseInstallationList("");

        const ms = elapsedMs(() => {
            parsed = parseInstallationList(raw);
        });

        expect([...parsed.ids]).toEqual([1]);
        expect(parsed.ignored).toBe(1);
        expect(ms).toBeLessThan(HOSTILE_INPUT_BUDGET_MS);
    });

    it("parses 50,000 commas", () => {
        const raw = ",".repeat(HOSTILE_INPUT_CHARS);
        let accounts = new Set<string>();

        const ms = elapsedMs(() => {
            accounts = parseAccountList(raw);
        });

        expect(accounts.size).toBe(0);
        expect(ms).toBeLessThan(HOSTILE_INPUT_BUDGET_MS);
    });

    it("parses a list of thousands of distinct ids", () => {
        const raw = Array.from({ length: 9_000 }, (_, i) => 1_000_000 + i).join(
            ",",
        );
        let count = 0;

        const ms = elapsedMs(() => {
            count = parseInstallationList(raw).ids.size;
        });

        expect(count).toBe(9_000);
        expect(ms).toBeLessThan(HOSTILE_INPUT_BUDGET_MS);
    });

    it("checks a 50,000-character account name against a list", () => {
        const policy = allowlist(["sairam0424"], []);
        let allowed = true;

        const ms = elapsedMs(() => {
            allowed = isInstallationAllowed(policy, {
                account: "A".repeat(HOSTILE_INPUT_CHARS),
                installationId: 1,
            });
        });

        expect(allowed).toBe(false);
        expect(ms).toBeLessThan(HOSTILE_INPUT_BUDGET_MS);
    });
});
