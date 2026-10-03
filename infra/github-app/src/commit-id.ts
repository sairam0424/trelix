/**
 * Commit ids, as the App meets them: the `pull_request.head.sha` of a webhook payload
 * and what `git rev-parse HEAD` prints in a fresh checkout.
 *
 * Both are compared and then sent to GitHub as a Check's `head_sha`, so anything that
 * is not a full lowercase hex object id is refused rather than cleaned up.
 */

// 40 hex digits for SHA-1 repositories, 64 for SHA-256 ones.
const SHA1_LENGTH = 40;
const SHA256_LENGTH = 64;
const HEX_DIGITS = /^[0-9a-f]+$/;

/**
 * True for a full-length lowercase hex commit id. The length is checked first, so an
 * input of any size costs one comparison; the pattern only ever sees 40 or 64
 * characters.
 */
export function isCommitId(value: unknown): value is string {
    if (typeof value !== "string") {
        return false;
    }
    if (value.length !== SHA1_LENGTH && value.length !== SHA256_LENGTH) {
        return false;
    }
    return HEX_DIGITS.test(value);
}
