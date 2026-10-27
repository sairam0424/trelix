import {
    describeErrorForLog,
    errorName,
    redactSecrets,
    reportIfRejected,
} from "./error-handler.js";

/**
 * Written to the console when the failure of a job cannot be logged (the writer
 * throws or its promise rejects), or when the error object throws as it is
 * read. It quotes nothing: what was thrown is not known to be free of the
 * secrets.
 */
const LOG_FAILURE_LINE = "[queue] a job failed; the error could not be logged";

// Looks up console.error on each call, so a test can spy on it.
function logToConsole(line: string): void {
    console.error(line);
}

/**
 * Builds the reporter the review queue calls for a job that threw or
 * rejected. A job runs after its webhook was answered, so there is no HTTP
 * error handler to take the error: this writes the one line instead, in the
 * shape of the error handler's (`[app] request failed {...}`): a JSON record
 * on a single line, with the configured secrets removed from the error text
 * before it is cut, and never the request, its headers or its body.
 *
 * `prefix` starts the line. `write` may be async: its promise is not awaited,
 * and a rejection is reported like a throw. The reporter never throws.
 */
export function createJobFailureLogger(
    secrets: readonly string[],
    prefix: string,
    write: (line: string) => void = logToConsole,
): (err: unknown, label: string) => void {
    return (err, label) => {
        try {
            const record = {
                job: label,
                error: redactSecrets(errorName(err), secrets),
                detail: describeErrorForLog(err, secrets),
            };
            reportIfRejected(
                write(`${prefix} ${JSON.stringify(record)}`),
                reportLogFailure,
            );
        } catch {
            reportLogFailure();
        }
    };
}

function reportLogFailure(): void {
    try {
        console.error(LOG_FAILURE_LINE);
    } catch {
        // The console was the last place left to report to.
    }
}
