/**
 * The README's environment table is the operator's only list of what the App reads,
 * and its account of the redelivery backstop is what the dedupe design leans on.
 * Both drift unnoticed, so this ties them to the code: a variable read in src/ but
 * missing from the table fails here, and so does a change to the redelivery
 * script that makes the README's description wrong.
 */
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const APP_DIR = fileURLToPath(new URL("..", import.meta.url));
const REPO_ROOT = join(APP_DIR, "..", "..");
const readme = readFileSync(join(APP_DIR, "README.md"), "utf8");

/** The literal list: adding a variable to the table means adding it here, on purpose. */
const DOCUMENTED_VARIABLES = [
    "GITHUB_APP_ID",
    "GITHUB_APP_PRIVATE_KEY",
    "GITHUB_WEBHOOK_SECRET",
    "NODE_ENV",
    "PORT",
    "TMPDIR",
    "TRELIX_APP_ALLOWED_ACCOUNTS",
    "TRELIX_APP_ALLOWED_INSTALLATIONS",
    "TRELIX_APP_CONCURRENCY",
    "TRELIX_APP_CONCURRENCY_PER_INSTALLATION",
    "TRELIX_APP_INSTALL_POLICY",
    "TRELIX_APP_QUEUE_CAPACITY",
    "TRELIX_APP_REVIEWS_ENABLED",
];

function sourceText(): string {
    const dir = join(APP_DIR, "src");
    return readdirSync(dir, { recursive: true, encoding: "utf8" })
        .filter((name) => name.endsWith(".ts"))
        .map((name) => readFileSync(join(dir, name), "utf8"))
        .join("\n");
}

/** The rows of the "Environment variables" table, by variable name. */
function tableRows(): Map<string, string> {
    const start = readme.indexOf("### Environment variables");
    const end = readme.indexOf("\n### ", start + 1);
    expect(start).toBeGreaterThan(-1);
    const rows = new Map<string, string>();
    for (const line of readme.slice(start, end).split("\n")) {
        const match = /^\| `([A-Z][A-Z0-9_]*)` \|/.exec(line);
        if (match?.[1] !== undefined) {
            rows.set(match[1], line);
        }
    }
    return rows;
}

describe("the README's environment table", () => {
    it("lists exactly the variables the App reads", () => {
        expect([...tableRows().keys()].sort()).toEqual(
            [...DOCUMENTED_VARIABLES].sort(),
        );
    });

    it("lists every TRELIX_APP_ variable the source names", () => {
        const named = new Set(
            sourceText().match(/TRELIX_APP_[A-Z_]+[A-Z]/g) ?? [],
        );

        expect(named.size).toBeGreaterThanOrEqual(7);
        for (const name of named) {
            expect(tableRows().has(name), name).toBe(true);
        }
    });

    it("lists every variable the source reads from process.env or requireEnv", () => {
        const source = sourceText();
        const read = new Set([
            ...[...source.matchAll(/process\.env\.([A-Z][A-Z0-9_]*)/g)].map(
                (m) => m[1] ?? "",
            ),
            ...[...source.matchAll(/requireEnv\("([A-Z][A-Z0-9_]*)"\)/g)].map(
                (m) => m[1] ?? "",
            ),
        ]);

        expect([...read].sort()).toEqual(
            [
                "GITHUB_APP_ID",
                "GITHUB_APP_PRIVATE_KEY",
                "GITHUB_WEBHOOK_SECRET",
                "PORT",
            ].sort(),
        );
        for (const name of read) {
            expect(tableRows().has(name), name).toBe(true);
        }
    });

    it("states the same defaults and ranges the code enforces", () => {
        const rows = tableRows();

        expect(rows.get("TRELIX_APP_QUEUE_CAPACITY")).toContain(
            "`20` (1 to 1000)",
        );
        expect(rows.get("TRELIX_APP_CONCURRENCY")).toContain("`2` (1 to 16)");
        expect(rows.get("TRELIX_APP_CONCURRENCY_PER_INSTALLATION")).toContain(
            "`1` (1 to 16)",
        );
        expect(rows.get("TRELIX_APP_INSTALL_POLICY")).toContain("`open`");
        expect(rows.get("TRELIX_APP_REVIEWS_ENABLED")).toContain("`true`");
        expect(rows.get("PORT")).toContain("`3000`");
    });

    it("is what the NODE_ENV paragraph points to", () => {
        const start = readme.indexOf("- **`NODE_ENV` and error responses**");

        expect(start).toBeGreaterThan(-1);
        expect(readme.slice(start, start + 200)).toContain(
            '"Environment variables"',
        );
    });
});

describe("the README's account of the redelivery backstop", () => {
    const script = readFileSync(
        join(APP_DIR, "src", "scripts", "redeliver-webhook-deliveries.ts"),
        "utf8",
    );
    const workflow = readFileSync(
        join(
            REPO_ROOT,
            ".github",
            "workflows",
            "redeliver-failed-webhooks.yml",
        ),
        "utf8",
    );

    it("still matches the script: one page of 100 deliveries, 15 minutes old, 5xx and 4xx failed", () => {
        expect(script).toContain(
            "listWebhookDeliveries({\n        per_page: 100,",
        );
        expect(script).not.toMatch(/paginate|\bpage:\s*\d/);
        expect(script).toContain(
            "MIN_AGE_BEFORE_REDELIVER_MS = 15 * 60 * 1000",
        );
        expect(script).toContain("statusCode < 200 || statusCode >= 400");
        // The README says the script keeps no record of what it redelivered.
        expect(script).not.toMatch(/\.redelivery\b/);
        expect(readme).toContain("**100 most recent deliveries**");
        expect(readme).toContain("younger than 15 minutes");
    });

    it("still matches the workflow: every 6 hours", () => {
        expect(workflow).toContain('cron: "0 */6 * * *"');
        expect(readme).toContain("runs every 6 hours");
    });
});
