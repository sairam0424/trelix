const SLASH_CHAR_CODE = 47;

/**
 * Removes every trailing `/` from a base URL in one pass.
 *
 * `url.replace(/\/+$/, "")` does the same job but retries the run from every slash in it, so a
 * base URL holding a long run of slashes that does not end it (before a query string, say)
 * takes quadratic time. A run at the very end is fast with the regex too.
 */
export function trimTrailingSlashes(url: string): string {
  let end = url.length;
  while (end > 0 && url.charCodeAt(end - 1) === SLASH_CHAR_CODE) end--;
  return url.slice(0, end);
}
