/**
 * An oracle that is independent of src/sanitize.ts: what must never be left in
 * text that maintainers see in a Check, whatever the input was. It is written
 * from the requirements (no image, link, bare URL, @mention, raw HTML, hidden
 * text), not from the implementation.
 *
 * It asks for no "@" at all, not only for none that starts a mention: GitHub
 * also links name@domain, and a domain may start with "-", "_" or "." (a
 * backslash escape included), so a rule that predicts which "@" is harmless
 * is a guess about the renderer.
 */

const HIDDEN_CHARACTER =
    /[\u0000-\u0008\u000B-\u001F\u007F-\u009F\u2028\u2029\p{Default_Ignorable_Code_Point}\p{Bidi_Control}\uD800-\uDFFF]/u;

/** A character reference other than the three the sanitiser itself writes. */
const FOREIGN_CHARACTER_REFERENCE =
    /&(?=[#A-Za-z0-9]{1,32};)(?!(?:amp|lt|gt);)/;

/** A run of combining marks longer than four. */
const LONG_COMBINING_RUN = /\p{M}{5,}/u;

/** Returns the first rule the text breaks, or null when none is broken. */
export function activeMarkupProblem(text: string): string | null {
    const checks: Array<[string, boolean]> = [
        ["raw angle bracket", /[<>]/.test(text)],
        ["URL scheme separator", text.includes("://")],
        ["www autolink", /www\\?\./i.test(text)],
        ["at sign (mention or email)", text.includes("@")],
        ["inline link or image", text.includes("](")],
        ["link reference definition", text.includes("]:")],
        ["hidden character", HIDDEN_CHARACTER.test(text)],
        ["character reference", FOREIGN_CHARACTER_REFERENCE.test(text)],
        ["long combining run", LONG_COMBINING_RUN.test(text)],
        ["untrimmed", text !== text.trim()],
    ];
    const broken = checks.find(([, isBroken]) => isBroken);
    return broken === undefined ? null : broken[0];
}
