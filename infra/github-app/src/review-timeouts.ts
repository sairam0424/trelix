/**
 * The time limits of the three stages of one review that have a timer of their
 * own, in one place so the queue's per-job deadline (abuse-controls.ts) is
 * sized from them and not from a number of its own. No imports, so the config
 * code can read them without loading the review machinery.
 */

// ~2 minutes — generous for a shallow, single-ref fetch of one PR, short
// enough that a genuinely hung clone (dead remote, network partition)
// doesn't tie up a webhook-delivery worker indefinitely. It applies to each
// git command of the checkout, but only the fetch talks to the network.
export const CHECKOUT_TIMEOUT_MS = 2 * 60 * 1000;

// Indexing a huge/pathological repo shouldn't hang the whole review either
// — same ceiling as the review step itself.
export const INDEX_TIMEOUT_MS = 5 * 60 * 1000;

// A slow/hung `trelix review` (LLM synthesis latency, a huge diff, a stuck
// index) would otherwise tie up this process indefinitely per webhook
// delivery — Node kills the child and execFileAsync rejects once this
// elapses.
export const REVIEW_TIMEOUT_MS = 5 * 60 * 1000;

/**
 * What one review may spend in its stages, one after the other: the checkout,
 * the index and the review child. A healthy review stays below each stage's
 * own limit, so it finishes within this sum; the queue's deadline must be
 * larger (see `DEFAULT_QUEUE_LIMITS.jobTimeoutMs`).
 */
export const REVIEW_STAGES_TIMEOUT_MS =
    CHECKOUT_TIMEOUT_MS + INDEX_TIMEOUT_MS + REVIEW_TIMEOUT_MS;
