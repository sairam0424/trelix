import { describe, expect, it } from "vitest";
import type { CheckAnnotation } from "../src/review-runner.js";
import {
    sanitizeAnnotation,
    sanitizeAnnotationPath,
    sanitizeAnnotations,
    sanitizeCheckOutput,
    sanitizeText,
} from "../src/sanitize.js";

/** A cap large enough that no case below is truncated. */
const ROOMY = 100_000;
const AT = "\uFF20"; // fullwidth commercial at
const LT = "\uFF1C"; // fullwidth less-than sign
const GT = "\uFF1E"; // fullwidth greater-than sign

describe("length caps", () => {
    it("cuts a title at 140 characters with an ellipsis", () => {
        const title = sanitizeText("x".repeat(500), 140, { singleLine: true });
        expect(title).toBe(`${"x".repeat(139)}\u2026`);
        expect(title).toHaveLength(140);
    });

    it("leaves text that fits exactly", () => {
        expect(sanitizeText("abcdef", 6)).toBe("abcdef");
        expect(sanitizeText("abcdefg", 6)).toBe("abcde\u2026");
    });

    it("drops the whitespace before the ellipsis", () => {
        expect(sanitizeText("ab   cdefgh", 6)).toBe("ab\u2026");
    });

    it("never cuts a surrogate pair in half", () => {
        expect(sanitizeText("aaaaaaaa\u{1F600}b", 10)).toBe("aaaaaaaa\u2026");
    });

    it("never cuts a character reference it wrote in half", () => {
        // "aaaaa&#64;b" is "aaaaa&amp;#64;b" once written out.
        expect(sanitizeText("aaaaa&#64;b", 7)).toBe("aaaaa\u2026");
        expect(sanitizeText("aaaaa&#64;b", 9)).toBe("aaaaa\u2026");
        expect(sanitizeText("aaaaa&#64;b", 10)).toBe("aaaaa\u2026");
        expect(sanitizeText("aaaa&#64;b", 8)).toBe("aaaa\u2026");
    });

    it("counts a fullwidth bracket as one character, so there is nothing of it to cut in half", () => {
        expect(sanitizeText("aaaa<<<<", 6)).toBe(`aaaa${LT}\u2026`);
        expect(sanitizeText("<aaaa>", 6)).toBe(`${LT}aaaa${GT}`);
    });

    it("cuts after a character reference that is whole, and after a lone ampersand", () => {
        expect(sanitizeText("aaaaa&#64;b", 11)).toBe("aaaaa&amp;\u2026");
        expect(sanitizeText("aaaaa&#64;b", 12)).toBe("aaaaa&amp;#\u2026");
        expect(sanitizeText("aaaa&bcdef", 7)).toBe("aaaa&b\u2026");
    });

    it("returns an empty string for a cap below one", () => {
        expect(sanitizeText("abc", 0)).toBe("");
        expect(sanitizeText("abc", -1)).toBe("");
    });

    it("returns an empty string for anything that is not a string", () => {
        for (const value of [undefined, null, 42, true, {}, ["a"]]) {
            expect(sanitizeText(value, ROOMY)).toBe("");
        }
    });

    it("turns line breaks and tabs in a single-line field into spaces", () => {
        expect(sanitizeText("a\nb\tc", ROOMY, { singleLine: true })).toBe(
            "a b c",
        );
    });

    it("caps the title, summary and message of a check output", () => {
        const output = sanitizeCheckOutput({
            title: "t".repeat(300),
            summary: "s".repeat(9000),
            annotations: [
                annotation({
                    message: "m".repeat(5000),
                    title: "n".repeat(300),
                }),
            ],
        });
        expect(output.title).toHaveLength(140);
        expect(output.summary).toHaveLength(4000);
        expect(output.annotations?.[0].message).toHaveLength(2000);
        expect(output.annotations?.[0].title).toHaveLength(140);
    });

    it("keeps at most 50 annotations", () => {
        const many = Array.from({ length: 60 }, (_, i) =>
            annotation({ path: `f${i}.py` }),
        );
        expect(sanitizeAnnotations(many)).toHaveLength(50);
        expect(sanitizeAnnotations(many, 100)).toHaveLength(50);
        expect(sanitizeAnnotations(many, 3)).toHaveLength(3);
        expect(sanitizeAnnotations(many, 0)).toHaveLength(0);
        expect(
            sanitizeCheckOutput({ title: "t", summary: "s", annotations: many })
                .annotations,
        ).toHaveLength(50);
    });
});

function annotation(overrides: Partial<CheckAnnotation> = {}): CheckAnnotation {
    return {
        path: "src/a.py",
        start_line: 3,
        end_line: 4,
        annotation_level: "warning",
        message: "note",
        title: "trelix review",
        ...overrides,
    };
}

describe("annotation paths", () => {
    const kept: Array<[string, string, string]> = [
        ["a relative path", "src/app.py", "src/app.py"],
        [
            "a scoped package",
            "packages/@scope/pkg/index.ts",
            "packages/@scope/pkg/index.ts",
        ],
        ["brackets", "app/posts/[id]/page.tsx", "app/posts/[id]/page.tsx"],
        ["a space", "my docs/a b.md", "my docs/a b.md"],
        ["dots that are not a parent segment", "a/..b/c../d", "a/..b/c../d"],
        ["a leading ./", "./a.py", "./a.py"],
        ["an override character", "src/\u202Eevil.ts", "src/evil.ts"],
        ["a zero-width character", "src/a\u200B.py", "src/a.py"],
        ["exactly 1024 characters", "a".repeat(1024), "a".repeat(1024)],
    ];
    it.each(kept)("keeps %s", (_name, input, expected) => {
        expect(sanitizeAnnotationPath(input)).toBe(expected);
    });

    const dropped: Array<[string, unknown]> = [
        ["an absolute path", "/etc/passwd"],
        ["a backslash root", "\\windows\\x"],
        ["a drive path", "C:\\x\\y"],
        ["a drive path with a slash", "c:/x"],
        ["a leading ..", "../secret"],
        ["a middle ..", "a/../b"],
        ["a trailing ..", "a/.."],
        ["a .. after a backslash", "a\\..\\b"],
        ["a .. hidden by a zero-width character", ".\u200B."],
        ["an empty path", ""],
        ["a path that is empty once hidden characters go", "\u200B\u202E"],
        ["a path with a newline", "a\nb"],
        ["a path with a tab", "a\tb"],
        ["a path with a line separator", "a\u2028b"],
        ["a path over 1024 characters", "a".repeat(1025)],
        ["a URL", "https://attacker.example/x"],
        ["a file URL", "file:///etc/passwd"],
        ["an HTML tag", "<img>"],
        ["a path with only an opening angle bracket", "a<b"],
        ["a path with only a closing angle bracket", "a>b"],
        ["a Markdown link", "a](b"],
        ["a number", 5],
        ["undefined", undefined],
        ["null", null],
    ];
    it.each(dropped)("drops %s", (_name, input) => {
        expect(sanitizeAnnotationPath(input)).toBeNull();
    });
});

describe("annotations", () => {
    it("sanitises the message and title and keeps the rest", () => {
        const result = sanitizeAnnotation(
            annotation({
                path: "src/\u202Ea.py",
                message: "see [x](https://attacker.example) @octocat",
                title: "trelix\nreview <b>",
            }),
        );
        expect(result).toEqual({
            path: "src/a.py",
            start_line: 3,
            end_line: 4,
            annotation_level: "warning",
            message: `see [x] (https[:]//attacker.example) ${AT}octocat`,
            title: `trelix review ${LT}b${GT}`,
        });
    });

    it("drops an annotation whose path is not acceptable", () => {
        expect(sanitizeAnnotation(annotation({ path: "../x" }))).toBeNull();
        expect(
            sanitizeAnnotations([
                annotation({ path: "/etc/passwd" }),
                annotation({ path: "ok.py" }),
            ]).map((a) => a.path),
        ).toEqual(["ok.py"]);
    });

    it("posts a placeholder for a message that is empty once sanitised", () => {
        for (const message of ["", "\u200B\u202E", "<!-- only a comment -->"]) {
            expect(sanitizeAnnotation(annotation({ message }))?.message).toBe(
                "(no details provided)",
            );
        }
    });

    it("falls back to the fixed title when the title sanitises to nothing", () => {
        expect(sanitizeAnnotation(annotation({ title: "\u200B" }))?.title).toBe(
            "trelix review",
        );
    });

    it("turns an unknown level into a notice", () => {
        const odd = annotation({
            annotation_level: "critical" as CheckAnnotation["annotation_level"],
        });
        expect(sanitizeAnnotation(odd)?.annotation_level).toBe("notice");
    });

    it("copies only the known fields", () => {
        const extra = {
            ...annotation(),
            raw_details: "https://attacker.example",
            blob_href: "x",
        };
        expect(Object.keys(sanitizeAnnotation(extra) ?? {}).sort()).toEqual([
            "annotation_level",
            "end_line",
            "message",
            "path",
            "start_line",
            "title",
        ]);
    });
});

describe("check output", () => {
    it("sanitises the title, the summary and every annotation", () => {
        const output = sanitizeCheckOutput({
            title: "found <b>1</b>\nissue",
            summary: "![](https://attacker.example/p.png)done @octocat",
            annotations: [annotation({ message: "<!-- x -->ok" })],
        });
        expect(output).toEqual({
            title: `found ${LT}b${GT}1${LT}/b${GT} issue`,
            summary: `done ${AT}octocat`,
            annotations: [annotation({ message: "ok" })],
        });
    });

    it("has no annotations key when it was given none", () => {
        const output = sanitizeCheckOutput({ title: "t", summary: "s" });
        expect(output).toEqual({ title: "t", summary: "s" });
        expect("annotations" in output).toBe(false);
    });

    it("copies nothing but the known fields", () => {
        const output = sanitizeCheckOutput({
            title: "t",
            summary: "s",
            text: "https://attacker.example",
            images: [{ alt: "x", image_url: "https://attacker.example/p.png" }],
        } as never);
        expect(Object.keys(output).sort()).toEqual(["summary", "title"]);
    });
});

describe("idempotence", () => {
    it("is a fixed point of a cut, which keeps its ellipsis", () => {
        const once = sanitizeText("<b>".repeat(100), 50);
        expect(once).toHaveLength(50);
        expect(sanitizeText(once, 50)).toBe(once);
    });

    it("is a fixed point of a whole check output", () => {
        const input = {
            title: "a <b> [x](https://attacker.example)\n",
            summary: "![](https://attacker.example/p.png)x @octocat <!-- y",
            annotations: [
                annotation({
                    message: "www.a.example &#64;x <i>",
                    path: "a/b",
                }),
                annotation({ path: "../x" }),
            ],
        };
        const once = sanitizeCheckOutput(input);
        expect(sanitizeCheckOutput(once)).toEqual(once);
    });
});
