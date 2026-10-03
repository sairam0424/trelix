/**
 * Sanitiser for every string the App posts to GitHub Checks.
 *
 * The text comes from an LLM that reads attacker-written pull requests, so it
 * may be prompt-injected. A Check is rendered to maintainers, so what we post
 * must not be able to show them an image (a tracking pixel), a link or bare
 * URL, an @mention, raw HTML or hidden text.
 *
 * Design rules:
 *
 * - Pure, deterministic and linear time. There is no Markdown parser here: a
 *   parser can be desynchronised from GitHub's by crafted input, so every
 *   rule is a context-free rewrite. Each regular expression has at most one
 *   unbounded quantifier, and the few repetitions it needs are bounded.
 * - Idempotent: sanitizeText(sanitizeText(x)) equals sanitizeText(x). Each
 *   rule's output is a fixed point of every rule, and the rules that remove
 *   text run before the rules that look for a pattern the removal could have
 *   created (see the order in sanitizeText).
 * - Context free also means code is treated like prose: every @ becomes a
 *   fullwidth @ and `List<String>` becomes `List＜String＞` (fullwidth angle
 *   brackets), even inside backticks. Lookalikes rather than entities, because
 *   Markdown shows an entity literally inside a code span. Readable, but not
 *   byte for byte.
 *
 * Line numbers are not text and are not touched here.
 */

export interface CheckAnnotation {
    path: string;
    start_line: number;
    end_line: number;
    annotation_level: "failure" | "warning" | "notice";
    message: string;
    title: string;
}

/** Caps, in UTF-16 code units of the final output. GitHub allows far more. */
export const MAX_TITLE_LENGTH = 140;
export const MAX_SUMMARY_LENGTH = 4000;
export const MAX_MESSAGE_LENGTH = 2000;
export const MAX_PATH_LENGTH = 1024;
/** GitHub accepts at most 50 annotations in one create-check-run request. */
export const MAX_ANNOTATIONS = 50;

/** Combining marks allowed in one run. Longer runs are "zalgo" text that bleeds over neighbouring lines. */
const MAX_COMBINING_MARKS = 4;

/** Posted when a message is empty once sanitised: an annotation needs a message, and one bad annotation would fail the whole request. */
export const EMPTY_MESSAGE_PLACEHOLDER = "(no details provided)";
const ANNOTATION_TITLE_FALLBACK = "trelix review";

const ELLIPSIS = "\u2026";
/** Fullwidth commercial at. GitHub does not treat it as a mention or an email. */
const FULLWIDTH_AT = "\uFF20";
/** Fullwidth less-than and greater-than signs. They cannot open or close a tag, and (unlike an entity) read the same inside a code span. */
const FULLWIDTH_LESS_THAN = "\uFF1C";
const FULLWIDTH_GREATER_THAN = "\uFF1E";

// Characters removed before anything else, one fragment per class so that each
// class is covered by exactly one entry (and can be tested on its own).
const CONTROL_CHARS =
    "\\u0000-\\u0008\\u000B\\u000C\\u000E-\\u001F\\u007F-\\u009F";
const ZERO_WIDTH_CHARS =
    "\\u00AD\\u034F\\u180B-\\u180F\\u200B-\\u200D\\u2060-\\u2065\\u206A-\\u206F\\uFEFF";
const BIDI_CHARS = "\\u061C\\u200E\\u200F\\u202A-\\u202E\\u2066-\\u2069";
const TAG_CHARS = "\\u{E0000}-\\u{E007F}";
// Variation selectors (incl. the supplement), invisible fillers, object/annotation
// anchors, non-characters and format controls that render as nothing.
const OTHER_HIDDEN_CHARS =
    "\\uFE00-\\uFE0F\\u{E0080}-\\u{E0FFF}\\u115F\\u1160\\u17B4\\u17B5\\u3164\\uFFA0\\uFFF0-\\uFFFC\\uFFFE\\uFFFF\\u{1BCA0}-\\u{1BCA3}\\u{1D173}-\\u{1D17A}";
// With the `u` flag a surrogate class only matches an unpaired surrogate.
const LONE_SURROGATES = "\\uD800-\\uDFFF";

const INVISIBLE_CHARS = new RegExp(
    `[${CONTROL_CHARS}${ZERO_WIDTH_CHARS}${BIDI_CHARS}${TAG_CHARS}${OTHER_HIDDEN_CHARS}${LONE_SURROGATES}]`,
    "gu",
);

const LINE_BREAKS = /\r\n?|[\u2028\u2029]/g;
const WHITESPACE_CONTROLS = /[\n\t]/g;

const COMMENT_OPENER = "<!--";

// `![alt](destination)`. Bounded so an unterminated opener costs a fixed
// amount, however many there are. Anything this misses (an alt or a
// destination over the bound) is still made inert by breakLinkSyntax.
const MARKDOWN_INLINE_IMAGE = /!\[[^\]\n]{0,200}\]\([^)\n]{0,500}\)/g;

const COMBINING_RUN = new RegExp(`\\p{M}{${MAX_COMBINING_MARKS + 1},}`, "gu");

// A scheme separator, with or without a Markdown backslash in front of any of
// its characters (`https\://x` and `https:\/\/x` render as `https://x`).
const SCHEME_SEPARATOR = /\\?:\\?\/\\?\//g;
const WWW_PREFIX = /www\\?\./gi;
// `](` starts an inline link or image destination and `]:` a reference
// definition (which renders as nothing at all).
const LINK_SYNTAX = /\]([(:])/g;

// Every @, not only one before a letter or digit. GitHub turns @user and
// @org/team into mentions and name@domain into a mailto: link, and the
// domain may start with `-`, `_` or `.`, also as a backslash escape
// (`a@-b.example`, `a@\.b.example` both link). Which @ is harmless depends on
// how the renderer reads what follows, so none is kept.
const AT_SIGN = /@/g;

const ANGLE_BRACKET = /[<>]/g;
const FULLWIDTH_ANGLE_BRACKETS: Readonly<Record<string, string>> = {
    "<": FULLWIDTH_LESS_THAN,
    ">": FULLWIDTH_GREATER_THAN,
};

// `&` only where it would start a character reference (`&name;`, `&#64;`,
// `&#x40;`), which the renderer would decode (`&#64;octocat` is a mention).
// `&amp;`, `&lt;` and `&gt;` are left as written: `&amp;` is what this rule
// writes, so the output must be a fixed point of it, and the other two decode
// to characters that are plain text here.
const CHARACTER_REFERENCE = /&(?=[#A-Za-z0-9]{1,32};)(?!(?:amp|lt|gt);)/g;
const WRITTEN_AMPERSAND = "&amp;";

export interface SanitizeOptions {
    /** Titles are a single line: line breaks and tabs become spaces. */
    singleLine?: boolean;
}

/** `<!-- ... -->`, including the HTML5 forms `<!-->`, `<!--->` and `--!>`. An unterminated opener removes the rest of the text. */
function stripHtmlComments(text: string): string {
    let start = text.indexOf(COMMENT_OPENER);
    if (start === -1) {
        return text;
    }
    const parts: string[] = [];
    let position = 0;
    const closer = /--!?>/g;
    while (start !== -1) {
        parts.push(text.slice(position, start));
        // Start inside the opener so `<!-->` and `<!--->` close at once.
        closer.lastIndex = start + 2;
        const match = closer.exec(text);
        if (match === null) {
            return parts.join("");
        }
        position = match.index + match[0].length;
        start = text.indexOf(COMMENT_OPENER, position);
    }
    parts.push(text.slice(position));
    return parts.join("");
}

function normaliseCharacters(text: string, singleLine: boolean): string {
    const visible = text
        .replace(LINE_BREAKS, "\n")
        .replace(INVISIBLE_CHARS, "");
    return singleLine ? visible.replace(WHITESPACE_CONTROLS, " ") : visible;
}

function capCombiningMarks(text: string): string {
    return text.replace(COMBINING_RUN, (run) =>
        Array.from(run).slice(0, MAX_COMBINING_MARKS).join(""),
    );
}

function defangLinks(text: string): string {
    return text
        .replace(SCHEME_SEPARATOR, "[:]//")
        .replace(WWW_PREFIX, (prefix) => `${prefix.slice(0, 3)}[.]`);
}

function breakLinkSyntax(text: string): string {
    return text.replace(LINK_SYNTAX, "] $1");
}

function neutraliseAtSigns(text: string): string {
    return text.replace(AT_SIGN, FULLWIDTH_AT);
}

function neutraliseHtml(text: string): string {
    return text
        .replace(ANGLE_BRACKET, (bracket) => FULLWIDTH_ANGLE_BRACKETS[bracket])
        .replace(CHARACTER_REFERENCE, WRITTEN_AMPERSAND);
}

/** Where to cut so that nothing we emit is split: a surrogate pair or the `&amp;` we write. */
function safeCutPoint(text: string, cut: number): number {
    const previous = text.charCodeAt(cut - 1);
    if (previous >= 0xd800 && previous <= 0xdbff) {
        return cut - 1;
    }
    const windowStart = Math.max(0, cut - WRITTEN_AMPERSAND.length);
    for (let index = cut - 1; index >= windowStart; index -= 1) {
        if (text[index] !== "&") {
            continue;
        }
        const splitsReference =
            text.startsWith(WRITTEN_AMPERSAND, index) &&
            index + WRITTEN_AMPERSAND.length > cut;
        return splitsReference ? index : cut;
    }
    return cut;
}

function truncate(text: string, maxLength: number): string {
    if (text.length <= maxLength) {
        return text;
    }
    const cut = safeCutPoint(text, maxLength - ELLIPSIS.length);
    return text.slice(0, cut).trimEnd() + ELLIPSIS;
}

/**
 * Returns `input` made safe to show in a Check. Anything that is not a string
 * becomes "". The result is at most `maxLength` code units long.
 *
 * The order matters: every rule that removes text runs before the rules that
 * look for patterns the removal could have created.
 */
export function sanitizeText(
    input: unknown,
    maxLength: number,
    options: SanitizeOptions = {},
): string {
    if (typeof input !== "string" || maxLength < 1) {
        return "";
    }
    let text = normaliseCharacters(input, options.singleLine === true);
    text = stripHtmlComments(text);
    text = text.replace(MARKDOWN_INLINE_IMAGE, "");
    text = capCombiningMarks(text);
    text = defangLinks(text);
    text = breakLinkSyntax(text);
    text = neutraliseAtSigns(text);
    text = neutraliseHtml(text);
    return truncate(text.trim(), maxLength);
}

const PATH_SEPARATORS = /[\\/]/;
const DRIVE_PREFIX = /^[A-Za-z]:[\\/]/;
const PATH_LINE_BREAKS = /[\r\n\t\u2028\u2029]/;

function isAbsolutePath(path: string): boolean {
    return (
        path.startsWith("/") || path.startsWith("\\") || DRIVE_PREFIX.test(path)
    );
}

function hasParentSegment(path: string): boolean {
    return path.split(PATH_SEPARATORS).some((segment) => segment === "..");
}

function looksLikeMarkup(path: string): boolean {
    return (
        path.includes("<") ||
        path.includes(">") ||
        path.includes("://") ||
        path.includes("](")
    );
}

/**
 * A file path for an annotation, or null when the annotation must be dropped:
 * not a string, empty or over MAX_PATH_LENGTH, multi-line, absolute, with a
 * `..` segment, or shaped like markup or a URL. Hidden characters are removed
 * first, so a zero-width character inside `..` cannot hide the segment.
 *
 * The path is otherwise left as it is: it is a label that GitHub matches
 * against the repository's files, and `@scope/pkg` or `[id]` are real names.
 */
export function sanitizeAnnotationPath(input: unknown): string | null {
    if (typeof input !== "string") {
        return null;
    }
    const path = input.replace(INVISIBLE_CHARS, "");
    if (path === "" || path.length > MAX_PATH_LENGTH) {
        return null;
    }
    if (PATH_LINE_BREAKS.test(path) || isAbsolutePath(path)) {
        return null;
    }
    if (hasParentSegment(path) || looksLikeMarkup(path)) {
        return null;
    }
    return path;
}

const ANNOTATION_LEVELS: ReadonlySet<string> = new Set([
    "failure",
    "warning",
    "notice",
]);

/** The sanitised annotation, or null when its path is not acceptable. Only the known fields are copied. */
export function sanitizeAnnotation(
    annotation: CheckAnnotation,
): CheckAnnotation | null {
    const path = sanitizeAnnotationPath(annotation.path);
    if (path === null) {
        return null;
    }
    return {
        path,
        start_line: annotation.start_line,
        end_line: annotation.end_line,
        annotation_level: ANNOTATION_LEVELS.has(annotation.annotation_level)
            ? annotation.annotation_level
            : "notice",
        message:
            sanitizeText(annotation.message, MAX_MESSAGE_LENGTH) ||
            EMPTY_MESSAGE_PLACEHOLDER,
        title:
            sanitizeText(annotation.title, MAX_TITLE_LENGTH, {
                singleLine: true,
            }) || ANNOTATION_TITLE_FALLBACK,
    };
}

/** Sanitises each annotation, drops the unacceptable ones and keeps at most `limit` (never more than MAX_ANNOTATIONS). */
export function sanitizeAnnotations(
    annotations: readonly CheckAnnotation[],
    limit: number = MAX_ANNOTATIONS,
): CheckAnnotation[] {
    const cap = Math.min(limit, MAX_ANNOTATIONS);
    const kept: CheckAnnotation[] = [];
    for (const annotation of annotations) {
        if (kept.length >= cap) {
            break;
        }
        const sanitised = sanitizeAnnotation(annotation);
        if (sanitised !== null) {
            kept.push(sanitised);
        }
    }
    return kept;
}

export interface CheckOutput {
    title: string;
    summary: string;
    annotations?: CheckAnnotation[];
}

/** Everything in the `output` of a create-check-run request. Nothing else is copied through. */
export function sanitizeCheckOutput(output: CheckOutput): CheckOutput {
    const sanitised: CheckOutput = {
        title: sanitizeText(output.title, MAX_TITLE_LENGTH, {
            singleLine: true,
        }),
        summary: sanitizeText(output.summary, MAX_SUMMARY_LENGTH),
    };
    if (output.annotations === undefined) {
        return sanitised;
    }
    return {
        ...sanitised,
        annotations: sanitizeAnnotations(output.annotations),
    };
}
