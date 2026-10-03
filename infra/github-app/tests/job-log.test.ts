import { afterEach, describe, expect, it, vi } from "vitest";
import { createJobFailureLogger } from "../src/job-log.js";

const HOOK = "canary-hook-s";
const PEM = "canary-pem-body";
const PROTECTED = [HOOK, PEM];
const PREFIX = "[webhook] review failed";

afterEach(() => {
    vi.restoreAllMocks();
});

function capture() {
    const lines: string[] = [];
    const log = createJobFailureLogger(PROTECTED, PREFIX, (line) =>
        lines.push(line),
    );
    return { log, lines };
}

describe("createJobFailureLogger", () => {
    it("writes one line: the prefix and a JSON record of the job, the error's name and its text", () => {
        const { log, lines } = capture();
        const err = new RangeError("out of range");
        err.stack = "RangeError: out of range\n    at somewhere (file.ts:1:1)";

        log(err, "owner/repo#42");

        expect(lines).toEqual([
            `${PREFIX} {"job":"owner/repo#42","error":"RangeError","detail":"RangeError: out of range\\n    at somewhere (file.ts:1:1)"}`,
        ]);
        expect(lines[0]).not.toContain("\n");
    });

    it("removes every protected value from the text, each occurrence, and from the error name", () => {
        const { log, lines } = capture();
        const err = new Error(`a ${HOOK} b ${PEM} c ${HOOK}`);
        err.name = `Failure-${HOOK}`;

        log(err, "o/r#1");

        expect(lines[0]).not.toContain(HOOK);
        expect(lines[0]).not.toContain(PEM);
        expect(lines[0]).toContain("a [redacted] b [redacted] c [redacted]");
        expect(lines[0]).toContain('"error":"Failure-[redacted]"');
    });

    it("removes a protected value that straddles the cut at 4,000 characters", () => {
        const { log, lines } = capture();
        const err = new Error("x");
        err.stack = `${"a".repeat(3_995)}${HOOK}${"b".repeat(50)}`;

        log(err, "o/r#1");

        expect(lines[0]).not.toContain("canary-hook");
        expect(lines[0]).not.toContain("hook-s");
        expect(lines[0]).toContain("…[truncated]");
    });

    it("cuts a very long error text", () => {
        const { log, lines } = capture();
        const err = new Error("x");
        err.stack = "s".repeat(500_000);

        log(err, "o/r#1");

        expect(lines[0]?.length).toBeLessThan(4_200);
    });

    it("describes a thrown string by its text and a thrown object by its type", () => {
        const { log, lines } = capture();

        log(`plain ${HOOK} text`, "o/r#1");
        log({ note: HOOK }, "o/r#2");
        log(null, "o/r#3");
        log(Object.create(null), "o/r#4");

        expect(lines).toEqual([
            `${PREFIX} {"job":"o/r#1","error":"string","detail":"plain [redacted] text"}`,
            `${PREFIX} {"job":"o/r#2","error":"object","detail":"[object Object]"}`,
            `${PREFIX} {"job":"o/r#3","error":"object","detail":"[object Null]"}`,
            `${PREFIX} {"job":"o/r#4","error":"object","detail":"[object Object]"}`,
        ]);
    });

    it("writes a fixed line, quoting nothing, when the error throws as it is read", () => {
        const consoleError = vi
            .spyOn(console, "error")
            .mockImplementation(() => {});
        const { log, lines } = capture();
        const hostile = new Error("x");
        Object.defineProperty(hostile, "stack", {
            get() {
                throw new Error(`getter failed: ${HOOK}`);
            },
        });

        log(hostile, "o/r#1");

        expect(lines).toEqual([]);
        expect(consoleError.mock.calls).toEqual([
            ["[queue] a job failed; the error could not be logged"],
        ]);
    });

    it("writes the same fixed line when the writer throws, and never throws itself", () => {
        const consoleError = vi
            .spyOn(console, "error")
            .mockImplementation(() => {});
        const log = createJobFailureLogger(PROTECTED, PREFIX, () => {
            throw new Error(`disk full: ${HOOK}`);
        });

        expect(() => log(new Error("boom"), "o/r#1")).not.toThrow();
        expect(consoleError.mock.calls).toEqual([
            ["[queue] a job failed; the error could not be logged"],
        ]);
    });

    it("does not throw when even the console throws", () => {
        vi.spyOn(console, "error").mockImplementation(() => {
            throw new Error("console closed");
        });
        const log = createJobFailureLogger(PROTECTED, PREFIX, () => {
            throw new Error("writer closed");
        });

        expect(() => log(new Error("boom"), "o/r#1")).not.toThrow();
    });

    it("writes to the console when no writer is given", () => {
        const consoleError = vi
            .spyOn(console, "error")
            .mockImplementation(() => {});
        const log = createJobFailureLogger(PROTECTED, PREFIX);

        log(new Error("boom"), "o/r#1");

        expect(consoleError).toHaveBeenCalledTimes(1);
        expect(String(consoleError.mock.calls[0]?.[0])).toContain(
            `${PREFIX} {"job":"o/r#1","error":"Error"`,
        );
    });
});
