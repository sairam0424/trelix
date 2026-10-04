import type { ErrorRequestHandler, Request, Response } from "express";

/** Body of every response to an error the server did not expect. */
const INTERNAL_ERROR_BODY = { error: "internal server error" } as const;

/**
 * The only client errors the handler reports with their own status: the ones
 * `express.json()` raises for a webhook body it cannot read (malformed JSON,
 * a body over the size cap, an unsupported charset or encoding). The text
 * for each is fixed here, so nothing the sender wrote can come back in it.
 */
const CLIENT_ERROR_MESSAGES: ReadonlyMap<number, string> = new Map([
    [400, "bad request"],
    [413, "payload too large"],
    [415, "unsupported media type"],
]);

const REDACTED = "[redacted]";

/** Longest error text written to the log; a hostile message cannot flood it. */
const MAX_LOGGED_DETAIL_CHARS = 4000;

/** Longest request path written to the log; the sender chooses the path. */
const MAX_LOGGED_PATH_CHARS = 200;

/**
 * Written to the console when the configured logger throws or its promise
 * rejects, or when the line cannot be built because the error object throws
 * when it is read. It quotes nothing: what was thrown is not known to be free
 * of the secrets.
 */
const LOG_FAILURE_LINE = "[app] request failed; the error could not be logged";

export interface ErrorHandlerOptions {
    /** Values that must never reach the log, such as the webhook secret and the private key. */
    readonly secrets: readonly string[];
    /**
     * Called once for every error the handler takes. It may be async (a
     * transport that sends the line elsewhere): the returned promise is not
     * awaited, and a rejection is handled like a throw.
     */
    readonly logError: (line: string) => void;
}

/**
 * Replaces every occurrence of each secret in `text`. The longest secret goes
 * first, so a secret that contains another one is not left half-visible.
 * `split`/`join` make one linear pass per secret; nothing is re-scanned.
 */
export function redactSecrets(
    text: string,
    secrets: readonly string[],
): string {
    // filter() returns a new array, so the sort below never reorders the caller's.
    return secrets
        .filter((secret) => secret.length > 0)
        .sort((a, b) => b.length - a.length)
        .reduce((safe, secret) => safe.split(secret).join(REDACTED), text);
}

/**
 * The status to answer with. Only an error that marks itself safe to show
 * (`expose === true`, which the body parser's errors do and an Octokit
 * `RequestError` does not) and carries one of the known client statuses keeps
 * its own status. Everything else is a 500: a `status` on an arbitrary error
 * describes a call the server made, not the request it is answering.
 */
function statusFor(err: unknown): number {
    if (typeof err !== "object" || err === null) {
        return 500;
    }
    try {
        const { status, expose } = err as {
            status?: unknown;
            expose?: unknown;
        };
        if (expose !== true || typeof status !== "number") {
            return 500;
        }
        return CLIENT_ERROR_MESSAGES.has(status) ? status : 500;
    } catch {
        // A getter on the error threw. Answer as for any unexpected error; the
        // error itself still goes to the log below.
        return 500;
    }
}

export function errorName(err: unknown): string {
    return err instanceof Error ? err.name : typeof err;
}

function errorDetail(err: unknown): string {
    if (err instanceof Error) {
        return err.stack ?? `${err.name}: ${err.message}`;
    }
    if (typeof err === "string") {
        return err;
    }
    // Never throws, whatever the value is (a null-prototype object breaks String()).
    return Object.prototype.toString.call(err);
}

function truncate(text: string, limit: number): string {
    return text.length > limit ? `${text.slice(0, limit)}…[truncated]` : text;
}

/**
 * The text of an error for a log line: its stack (else its name and message)
 * with every secret removed, then cut to MAX_LOGGED_DETAIL_CHARS. The secrets
 * come out before the cut, so one that straddles the limit is not left as two
 * halves that match nothing. May throw if the error object does when it is read.
 */
export function describeErrorForLog(
    err: unknown,
    secrets: readonly string[],
): string {
    return truncate(
        redactSecrets(errorDetail(err), secrets),
        MAX_LOGGED_DETAIL_CHARS,
    );
}

/**
 * The one log line for an error. The secrets come out before the cut: a
 * secret that straddles the limit would otherwise be cut in two, and neither
 * half would match it. May throw if the error object does when it is read.
 */
function buildLogLine(
    err: unknown,
    req: Request,
    res: Response,
    status: number,
    secrets: readonly string[],
): string {
    const alreadySent = res.headersSent;
    const record = {
        method: req.method,
        path: truncate(req.path, MAX_LOGGED_PATH_CHARS),
        // Once the response has started, the client has its status, not the
        // one this error would have been answered with.
        status: alreadySent ? res.statusCode : status,
        alreadySent: alreadySent ? true : undefined,
        error: errorName(err),
        // JSON.stringify leaves out an undefined field: a 4xx has no detail.
        detail: status >= 500 ? describeErrorForLog(err, secrets) : undefined,
    };
    return `[app] request failed ${JSON.stringify(record)}`;
}

/**
 * Hands a logger's result to `report` if it is a promise that rejects. A
 * logger is typed to return nothing, but an async one returns a promise, and
 * nobody awaits it: without this its rejection would be unhandled. It is not
 * awaited here either, so the caller never waits for a log.
 */
export function reportIfRejected(result: unknown, report: () => void): void {
    void Promise.resolve(result).catch(report);
}

/**
 * Writes the line, and never throws: the response below has to go out even
 * when the log cannot be written. The failure itself, a throw or a rejected
 * promise, is reported on the console with a fixed line.
 */
function writeLogLine(
    logError: (line: string) => void,
    buildLine: () => string,
): void {
    try {
        reportIfRejected(logError(buildLine()), reportLogFailure);
    } catch {
        reportLogFailure();
    }
}

function reportLogFailure(): void {
    try {
        console.error(LOG_FAILURE_LINE);
    } catch {
        // The console was the last place left to report to.
    }
}

/**
 * The last middleware of the app. Whatever reaches it, the client gets a fixed
 * body and a status, never the error's message or stack. The detail goes to
 * the log, once, with the configured secrets removed. The request's body,
 * headers and query string are never logged: the signature header and the
 * webhook payload are not the error's business. The response does not depend
 * on the log: if the logger throws, or returns a promise that rejects, a fixed
 * line goes to the console and the response is sent all the same.
 *
 * A message is only logged for a 5xx. A 4xx is the sender's fault and its
 * message can quote the request body (a JSON syntax error does), so the line
 * carries the error's name and the status instead.
 */
export function createErrorHandler(
    options: ErrorHandlerOptions,
): ErrorRequestHandler {
    // The four parameters are required: Express treats a function of any other
    // arity as an ordinary middleware and never calls it with an error.
    return (err, req, res, _next) => {
        const status = statusFor(err);
        writeLogLine(options.logError, () =>
            buildLogLine(err, req, res, status, options.secrets),
        );

        if (res.headersSent) {
            // Too late for a status or a body. Close a half-written response so
            // the client is not left waiting; leave a finished one alone.
            if (!res.writableEnded) {
                res.destroy();
            }
            return;
        }

        const body =
            status >= 500
                ? INTERNAL_ERROR_BODY
                : { error: CLIENT_ERROR_MESSAGES.get(status) };
        res.status(status).json(body);
    };
}
