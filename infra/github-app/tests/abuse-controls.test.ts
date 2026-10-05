import { describe, expect, it, vi } from "vitest";
import {
    DEFAULT_ABUSE_CONTROLS,
    loadAbuseControls,
} from "../src/abuse-controls.js";

/** Loads from a fixed env and collects what was warned. */
function load(env: Record<string, string | undefined>) {
    const warnings: string[] = [];
    const controls = loadAbuseControls(env, (line) => warnings.push(line));
    return { controls, warnings };
}

const OPEN_WARNING =
    "[config] WARNING: TRELIX_APP_INSTALL_POLICY is open: any account that installs this App can spend this deployment's LLM quota. Set it to allowlist, with TRELIX_APP_ALLOWED_ACCOUNTS and/or TRELIX_APP_ALLOWED_INSTALLATIONS, to restrict who is served";

describe("loadAbuseControls defaults", () => {
    it("with nothing set: reviews on, open policy, queue 20 waiting (10 per installation), 2 running (1 per installation), 17 minutes a job", () => {
        const { controls } = load({});

        expect(controls).toEqual({
            reviewsEnabled: true,
            installPolicy: {
                mode: "open",
                accounts: new Set(),
                installations: new Set(),
            },
            queue: {
                capacity: 20,
                perGroupCapacity: 10,
                concurrency: 2,
                perGroupConcurrency: 1,
                jobTimeoutMs: 1_020_000,
            },
        });
    });

    it("the exported default is the same thing", () => {
        expect(DEFAULT_ABUSE_CONTROLS).toEqual(load({}).controls);
    });

    it("warns loudly at startup that the default policy is open", () => {
        const { warnings } = load({});

        expect(warnings).toEqual([OPEN_WARNING]);
    });

    it("warns about an explicit open policy too", () => {
        const { warnings } = load({ TRELIX_APP_INSTALL_POLICY: "open" });

        expect(warnings).toEqual([OPEN_WARNING]);
    });

    it("logs to the console when no logger is passed", () => {
        const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
        try {
            loadAbuseControls({});

            expect(warn.mock.calls).toEqual([[OPEN_WARNING]]);
        } finally {
            warn.mockRestore();
        }
    });

    it("reads process.env when no env is passed", () => {
        vi.stubEnv("TRELIX_APP_REVIEWS_ENABLED", "false");
        try {
            const controls = loadAbuseControls(undefined, () => {});

            expect(controls.reviewsEnabled).toBe(false);
        } finally {
            vi.unstubAllEnvs();
        }
    });
});

describe("the kill switch (TRELIX_APP_REVIEWS_ENABLED)", () => {
    it.each(["true", "TRUE", " True ", "1", "yes", "on"])(
        "reads %j as on",
        (value) => {
            const { controls } = load({ TRELIX_APP_REVIEWS_ENABLED: value });

            expect(controls.reviewsEnabled).toBe(true);
        },
    );

    it.each([undefined, "", "   "])(
        "reads %j (unset or blank) as on",
        (value) => {
            const { controls, warnings } = load({
                TRELIX_APP_REVIEWS_ENABLED: value,
            });

            expect(controls.reviewsEnabled).toBe(true);
            expect(warnings).toEqual([OPEN_WARNING]);
        },
    );

    it.each(["false", "FALSE", " False ", "0", "no", "off"])(
        "reads %j as off and warns once that reviews are disabled",
        (value) => {
            const { controls, warnings } = load({
                TRELIX_APP_REVIEWS_ENABLED: value,
            });

            expect(controls.reviewsEnabled).toBe(false);
            expect(warnings).toEqual([
                "[config] WARNING: reviews are disabled (TRELIX_APP_REVIEWS_ENABLED); every pull_request delivery will be acknowledged and ignored",
                OPEN_WARNING,
            ]);
        },
    );

    it.each(["flase", "disabled", "2", "maybe", "true false", "null"])(
        "reads the unreadable %j as off, and says why",
        (value) => {
            const { controls, warnings } = load({
                TRELIX_APP_REVIEWS_ENABLED: value,
            });

            expect(controls.reviewsEnabled).toBe(false);
            expect(warnings[0]).toBe(
                "[config] WARNING: TRELIX_APP_REVIEWS_ENABLED is not true or false; reviews are OFF until it is fixed",
            );
        },
    );
});

describe("the installation policy", () => {
    it("reads allowlist with accounts and installations", () => {
        const { controls, warnings } = load({
            TRELIX_APP_INSTALL_POLICY: "allowlist",
            TRELIX_APP_ALLOWED_ACCOUNTS: " Sairam0424 , acme ",
            TRELIX_APP_ALLOWED_INSTALLATIONS: "11, 22",
        });

        expect(controls.installPolicy).toEqual({
            mode: "allowlist",
            accounts: new Set(["sairam0424", "acme"]),
            installations: new Set([11, 22]),
        });
        expect(warnings).toEqual([]);
    });

    it("warns, and allows nobody, for allowlist with both lists empty", () => {
        const { controls, warnings } = load({
            TRELIX_APP_INSTALL_POLICY: "allowlist",
        });

        expect(controls.installPolicy.mode).toBe("allowlist");
        expect(controls.installPolicy.accounts.size).toBe(0);
        expect(controls.installPolicy.installations.size).toBe(0);
        expect(warnings).toEqual([
            "[config] WARNING: TRELIX_APP_INSTALL_POLICY is allowlist but no account or installation is listed; every installation will be ignored",
        ]);
    });

    it("warns, and allows nobody, when the only entries are not installation ids", () => {
        const { controls, warnings } = load({
            TRELIX_APP_INSTALL_POLICY: "allowlist",
            TRELIX_APP_ALLOWED_INSTALLATIONS: "octocat, 12abc",
        });

        expect(controls.installPolicy.installations.size).toBe(0);
        expect(warnings).toEqual([
            "[config] 2 value(s) in TRELIX_APP_ALLOWED_INSTALLATIONS are not an installation id and allow nothing",
            "[config] WARNING: TRELIX_APP_INSTALL_POLICY is allowlist but no account or installation is listed; every installation will be ignored",
        ]);
    });

    it("keeps the readable entries next to unreadable ones", () => {
        const { controls, warnings } = load({
            TRELIX_APP_INSTALL_POLICY: "allowlist",
            TRELIX_APP_ALLOWED_INSTALLATIONS: "5, nonsense, 6",
        });

        expect([...controls.installPolicy.installations]).toEqual([5, 6]);
        expect(warnings).toEqual([
            "[config] 1 value(s) in TRELIX_APP_ALLOWED_INSTALLATIONS are not an installation id and allow nothing",
        ]);
    });

    it("warns that a list is not applied while the policy is open, and serves everyone", () => {
        const { controls, warnings } = load({
            TRELIX_APP_ALLOWED_ACCOUNTS: "sairam0424",
        });

        expect(controls.installPolicy.mode).toBe("open");
        expect(warnings).toEqual([
            OPEN_WARNING,
            "[config] WARNING: an allow-list is set but TRELIX_APP_INSTALL_POLICY is not allowlist, so the list is NOT applied",
        ]);
    });

    it("refuses to start under an unknown policy, without quoting it", () => {
        const env = { TRELIX_APP_INSTALL_POLICY: "canary-policy" };
        const warnings: string[] = [];
        let message = "";

        try {
            loadAbuseControls(env, (line) => warnings.push(line));
        } catch (err) {
            message = (err as Error).message;
        }

        expect(message).toBe(
            'Invalid TRELIX_APP_INSTALL_POLICY: expected "open" or "allowlist"',
        );
        expect(message).not.toContain("canary-policy");
    });
});

describe("the queue caps", () => {
    it("reads all five", () => {
        const { controls } = load({
            TRELIX_APP_QUEUE_CAPACITY: "50",
            TRELIX_APP_QUEUE_CAPACITY_PER_INSTALLATION: "7",
            TRELIX_APP_CONCURRENCY: "4",
            TRELIX_APP_CONCURRENCY_PER_INSTALLATION: "2",
            TRELIX_APP_JOB_TIMEOUT_MINUTES: "30",
        });

        expect(controls.queue).toEqual({
            capacity: 50,
            perGroupCapacity: 7,
            concurrency: 4,
            perGroupConcurrency: 2,
            jobTimeoutMs: 1_800_000,
        });
    });

    it("accepts the smallest and the largest value", () => {
        const { controls, warnings } = load({
            TRELIX_APP_QUEUE_CAPACITY: "1000",
            TRELIX_APP_QUEUE_CAPACITY_PER_INSTALLATION: "1000",
            TRELIX_APP_CONCURRENCY: "16",
            TRELIX_APP_CONCURRENCY_PER_INSTALLATION: "1",
            TRELIX_APP_JOB_TIMEOUT_MINUTES: "240",
        });

        expect(controls.queue).toEqual({
            capacity: 1000,
            perGroupCapacity: 1000,
            concurrency: 16,
            perGroupConcurrency: 1,
            jobTimeoutMs: 14_400_000,
        });
        expect(warnings).toEqual([OPEN_WARNING]);
    });

    it("accepts a share of 1", () => {
        const { controls, warnings } = load({
            TRELIX_APP_QUEUE_CAPACITY_PER_INSTALLATION: "1",
        });

        expect(controls.queue.perGroupCapacity).toBe(1);
        expect(warnings).toEqual([OPEN_WARNING]);
    });

    describe("the share of the wait line one installation may hold, when unset", () => {
        it.each([
            [1, 1],
            [2, 1],
            [3, 2],
            [7, 4],
            [20, 10],
            [21, 11],
            [1000, 500],
        ])("is half of a capacity of %i, rounded up: %i", (capacity, share) => {
            const { controls } = load({
                TRELIX_APP_QUEUE_CAPACITY: String(capacity),
            });

            expect(controls.queue.perGroupCapacity).toBe(share);
        });

        it("follows a capacity that could not be read back to the default one", () => {
            const { controls } = load({ TRELIX_APP_QUEUE_CAPACITY: "abc" });

            expect(controls.queue.capacity).toBe(20);
            expect(controls.queue.perGroupCapacity).toBe(10);
        });
    });

    const CAPS = [
        {
            name: "TRELIX_APP_QUEUE_CAPACITY",
            field: "capacity",
            range: "1 to 1000",
            fallback: 20,
            tooBig: "1001",
        },
        {
            name: "TRELIX_APP_QUEUE_CAPACITY_PER_INSTALLATION",
            field: "perGroupCapacity",
            range: "1 to 1000",
            fallback: 10,
            tooBig: "1001",
        },
        {
            name: "TRELIX_APP_CONCURRENCY",
            field: "concurrency",
            range: "1 to 16",
            fallback: 2,
            tooBig: "17",
        },
        {
            name: "TRELIX_APP_CONCURRENCY_PER_INSTALLATION",
            field: "perGroupConcurrency",
            range: "1 to 16",
            fallback: 1,
            tooBig: "17",
        },
    ] as const;
    const UNREADABLE = [
        "abc",
        "0",
        "-3",
        "2.5",
        "1e3",
        "0x10",
        "10 cores",
        "99999999999999999999",
        "NaN",
        "Infinity",
    ];

    describe.each(CAPS)("$name", ({ name, field, range, fallback, tooBig }) => {
        it.each([...UNREADABLE, tooBig])(
            "falls back to the default %j, with a warning that names the variable",
            (bad) => {
                const { controls, warnings } = load({ [name]: bad });

                expect(controls.queue[field]).toBe(fallback);
                expect(warnings).toContain(
                    `[config] ${name} is not a whole number from ${range}; using the default ${fallback}`,
                );
            },
        );
    });

    it.each([
        "TRELIX_APP_QUEUE_CAPACITY",
        "TRELIX_APP_QUEUE_CAPACITY_PER_INSTALLATION",
        "TRELIX_APP_CONCURRENCY",
        "TRELIX_APP_CONCURRENCY_PER_INSTALLATION",
        "TRELIX_APP_JOB_TIMEOUT_MINUTES",
    ])("%s unset or blank is the default and silent", (name) => {
        for (const value of [undefined, "", "  "]) {
            const { warnings } = load({ [name]: value });

            expect(warnings).toEqual([OPEN_WARNING]);
        }
    });

    it("logs the name of a variable it could not read, never its value", () => {
        const { warnings } = load({
            TRELIX_APP_QUEUE_CAPACITY: "canary-value-12345",
            TRELIX_APP_INSTALL_POLICY: "allowlist",
            TRELIX_APP_ALLOWED_INSTALLATIONS: "canary-id-67890",
        });

        expect(warnings.join("\n")).not.toContain("canary");
    });
});

describe("the per-job deadline", () => {
    const NAME = "TRELIX_APP_JOB_TIMEOUT_MINUTES";
    const WARNING =
        "[config] TRELIX_APP_JOB_TIMEOUT_MINUTES is not a whole number from 13 to 240; using the default 17";

    it("is 17 minutes with nothing set: the 2, 5 and 5 minutes of a review's stages, and 5 more", () => {
        expect(load({}).controls.queue.jobTimeoutMs).toBe(
            (2 + 5 + 5 + 5) * 60 * 1000,
        );
    });

    it("is read in whole minutes and used in milliseconds", () => {
        const { controls, warnings } = load({ [NAME]: " 45 " });

        expect(controls.queue.jobTimeoutMs).toBe(2_700_000);
        expect(warnings).toEqual([OPEN_WARNING]);
    });

    it("accepts 13 minutes, the least that is more than the stages one after the other", () => {
        const { controls, warnings } = load({ [NAME]: "13" });

        expect(controls.queue.jobTimeoutMs).toBe(780_000);
        expect(warnings).toEqual([OPEN_WARNING]);
    });

    it.each(["12", "1", "0", "241", "99999", "abc", "-3", "2.5", "1e3", "1h"])(
        "falls back to the default for %j, with a warning that names the variable",
        (bad) => {
            const { controls, warnings } = load({ [NAME]: bad });

            expect(controls.queue.jobTimeoutMs).toBe(1_020_000);
            expect(warnings).toContain(WARNING);
        },
    );
});
