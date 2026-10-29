import type { Server } from "node:http";
import { MAX_WEBHOOK_PAYLOAD_BYTES } from "./webhook.js";

/**
 * What the HTTP server holds for requests it has not finished reading. These
 * limits bound the memory and the time one request can take: a sender that
 * cannot send a whole request in time is cut off, and once the cap is reached
 * every further connection is refused. They check nobody's identity and are not
 * a rate limiter, which would not bound a slow upload.
 *
 * The cap is one pool shared by every sender, not a limit per client. A sender
 * that holds all of it with slow uploads makes the server refuse every other
 * connection, GitHub's deliveries and /health included, and can go on doing so
 * by opening a new connection each time the request timeout cuts one. The cap
 * trades a possible out-of-memory kill, which loses the in-memory review queue
 * and dedupe memory, for a possible refusal of service; the redelivery backstop
 * sends refused deliveries again.
 *
 * Node's defaults (60 s for the headers, 300 s for the whole request, no cap on
 * connections) let one slow sender hold a body of up to 5 MiB for five minutes.
 * A hundred of them hold 500 MiB, which is the whole 512 MB container, and an
 * out-of-memory kill loses the in-memory review queue and dedupe memory.
 *
 * The values follow from GitHub's side of the exchange: it gives up on a
 * delivery after 10 seconds (docs.github.com, "Best practices for using
 * webhooks", "Respond within 10 seconds"), and a delivery is at most 5 MiB
 * here and mostly under 1 MiB. Sending 5 MiB takes about 0.4 s at 100 Mbit/s
 * and 8 s at 5 Mbit/s, so a delivery that is still unfinished after the
 * request timeout is one GitHub has already given up on.
 */
export interface ServerLimits {
    /** Longest a request may take to send its headers (about 1 KB). */
    readonly headersTimeoutMs: number;
    /** Longest a request may take to arrive whole, body included. */
    readonly requestTimeoutMs: number;
    /** How often Node compares open requests with the two timeouts. */
    readonly connectionsCheckingIntervalMs: number;
    /** Most connections open at once; Node closes the socket of any above it. */
    readonly maxConnections: number;
}

/** Half of GitHub's own 10 s; the headers of a delivery are about 1 KB. */
export const HEADERS_TIMEOUT_MS = 5_000;

/**
 * GitHub's 10 s plus 5 s for the hop through the platform's proxy and for the
 * two clocks. It must not be below HEADERS_TIMEOUT_MS: http.createServer()
 * refuses that pair, but assigning the properties, as applyServerLimits does,
 * checks nothing.
 */
export const REQUEST_TIMEOUT_MS = 15_000;

/**
 * Node checks the two timeouts only on this interval, 30 s by default, so with
 * the default a request could outlive REQUEST_TIMEOUT_MS by up to 30 s. A check
 * is a walk over at most MAX_CONNECTIONS connections.
 */
export const CONNECTIONS_CHECKING_INTERVAL_MS = 1_000;

/** The memory this service is sized for: the Dockerfile's 512 MB, the Railway Free plan's 0.5 GB. */
const CONTAINER_MEMORY_BYTES = 512 * 1024 * 1024;

/**
 * The share of the container that the bodies of unfinished requests may fill
 * when every connection is carrying a full-size one. The rest is for the
 * running reviews and for the process itself.
 */
const REQUEST_BODY_MEMORY_SHARE = 0.4;

/**
 * 40 with the 5 MiB cap: the bodies alone come to 40 x 5 MiB = 200 MiB, 39
 * percent of 512 MiB. raw-body, under express.raw, keeps one Buffer per chunk
 * it is given, so a sender of tiny chunks costs more than its byte count;
 * REQUEST_TIMEOUT_MS bounds that, not the 5 MiB cap. A genuine delivery is
 * answered within milliseconds, so in normal traffic 40 open connections serve
 * far more than the 20 reviews the queue holds by default, with room for
 * /health. Under attack there is no such room: see the comment at the top of
 * this file.
 */
export const MAX_CONNECTIONS = Math.floor(
    (CONTAINER_MEMORY_BYTES * REQUEST_BODY_MEMORY_SHARE) /
        MAX_WEBHOOK_PAYLOAD_BYTES,
);

export const SERVER_LIMITS: ServerLimits = {
    headersTimeoutMs: HEADERS_TIMEOUT_MS,
    requestTimeoutMs: REQUEST_TIMEOUT_MS,
    connectionsCheckingIntervalMs: CONNECTIONS_CHECKING_INTERVAL_MS,
    maxConnections: MAX_CONNECTIONS,
};

/**
 * `connectionsCheckingInterval` is an option of `http.createServer` that Node
 * also keeps as a property, read when the server starts listening. @types/node
 * declares only the option.
 */
type CheckedServer = Server & { connectionsCheckingInterval: number };

/**
 * Sets the limits on a server. Call it right after `app.listen()`: the
 * timeouts and the cap are read per connection, and the checking interval is
 * read when the server emits "listening", which is after the current tick.
 */
export function applyServerLimits(
    server: Server,
    limits: ServerLimits = SERVER_LIMITS,
): void {
    server.headersTimeout = limits.headersTimeoutMs;
    server.requestTimeout = limits.requestTimeoutMs;
    (server as CheckedServer).connectionsCheckingInterval =
        limits.connectionsCheckingIntervalMs;
    server.maxConnections = limits.maxConnections;
}
