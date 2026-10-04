import { describe, expect, it } from "vitest";
import type { Job, JobOutcome } from "../src/queue.js";
import { DAY_MS, makeHarness, settle } from "./support/queue-harness.js";

describe("JobQueue", () => {
    describe("running jobs", () => {
        it("does not start a job inside submit: the caller answers first", async () => {
            const h = makeHarness();
            const { job } = h.hold("a");

            expect(h.queue.submit(job)).toBe("accepted");
            expect(h.started).toEqual([]);

            await settle();
            expect(h.started).toEqual(["a"]);
        });

        it("runs at most `concurrency` jobs at once and starts the rest in submission order", async () => {
            const h = makeHarness({ capacity: 10, concurrency: 2 });
            const jobs = ["j1", "j2", "j3", "j4"].map((key) => h.hold(key));
            for (const { job } of jobs) {
                h.queue.submit(job);
            }

            await settle();
            expect(h.started).toEqual(["j1", "j2"]);

            jobs[0]?.gate.resolve();
            await settle();
            expect(h.started).toEqual(["j1", "j2", "j3"]);

            jobs[1]?.gate.resolve();
            await settle();
            expect(h.started).toEqual(["j1", "j2", "j3", "j4"]);
        });

        it("runs jobs of one group one at a time even when global slots are free", async () => {
            const h = makeHarness({
                capacity: 10,
                concurrency: 4,
                perGroupConcurrency: 1,
            });
            const first = h.hold("a1", "installation-1");
            const second = h.hold("a2", "installation-1");
            h.queue.submit(first.job);
            h.queue.submit(second.job);

            await settle();
            expect(h.started).toEqual(["a1"]);

            first.gate.resolve();
            await settle();
            expect(h.started).toEqual(["a1", "a2"]);
        });

        it("lets a group run as many jobs as perGroupConcurrency allows", async () => {
            const h = makeHarness({
                capacity: 10,
                concurrency: 4,
                perGroupConcurrency: 2,
            });
            for (const key of ["a1", "a2", "a3"]) {
                h.queue.submit(h.hold(key, "installation-1").job);
            }

            await settle();

            expect(h.started).toEqual(["a1", "a2"]);
        });

        it("does not let a busy group hold up another group's job behind it", async () => {
            const h = makeHarness({
                capacity: 10,
                concurrency: 2,
                perGroupConcurrency: 1,
            });
            h.queue.submit(h.hold("a1", "A").job);
            h.queue.submit(h.hold("a2", "A").job); // waits for a1
            h.queue.submit(h.hold("b1", "B").job); // behind a2, but B is free

            await settle();

            expect(h.started).toEqual(["a1", "b1"]);
        });

        it("keeps first-in-first-out order inside a group", async () => {
            const h = makeHarness({ capacity: 10, concurrency: 3 });
            const a1 = h.hold("a1", "A");
            h.queue.submit(a1.job);
            h.queue.submit(h.hold("a2", "A").job);
            h.queue.submit(h.hold("a3", "A").job);
            await settle();

            a1.gate.resolve();
            await settle();

            expect(h.started).toEqual(["a1", "a2"]);
        });
    });

    describe("capacity", () => {
        it("refuses a job when `capacity` jobs are already waiting", async () => {
            const h = makeHarness({ capacity: 2, concurrency: 1 });
            const results = ["j1", "j2", "j3", "j4"].map((key) =>
                h.queue.submit(h.hold(key).job),
            );

            expect(results).toEqual([
                "accepted",
                "accepted",
                "accepted",
                "full",
            ]);
            await settle();
            expect(h.queue.counts()).toEqual({ waiting: 2, running: 1 });
        });

        it("counts only the jobs that wait, not the ones that run", async () => {
            const h = makeHarness({ capacity: 1, concurrency: 1 });

            expect(h.queue.submit(h.hold("j1").job)).toBe("accepted"); // runs
            expect(h.queue.submit(h.hold("j2").job)).toBe("accepted"); // waits
            expect(h.queue.submit(h.hold("j3").job)).toBe("full");
        });

        it("gives the claim back when it refuses, so the same key is accepted once there is room", async () => {
            const h = makeHarness({ capacity: 1, concurrency: 1 });
            const first = h.hold("j1");
            h.queue.submit(first.job);
            h.queue.submit(h.hold("j2").job);
            expect(h.queue.submit(h.hold("j3").job)).toBe("full");
            expect(h.claims.counts().inFlight).toBe(2);

            first.gate.resolve();
            await settle(); // j2 starts; the wait line is empty

            expect(h.queue.submit(h.hold("j3").job)).toBe("accepted");
        });

        it("does not refuse a job that can start at once because the wait line is full of blocked jobs", async () => {
            const h = makeHarness({
                capacity: 1,
                concurrency: 2,
                perGroupConcurrency: 1,
            });
            h.queue.submit(h.hold("a1", "A").job); // runs
            expect(h.queue.submit(h.hold("a2", "A").job)).toBe("accepted"); // waits: fills the line
            expect(h.queue.submit(h.hold("a3", "A").job)).toBe("full"); // would wait: refused

            expect(h.queue.submit(h.hold("b1", "B").job)).toBe("accepted"); // starts at once

            await settle();
            expect(h.started).toEqual(["a1", "b1"]);
        });

        it("stays within capacity + concurrency under a flood of distinct jobs", async () => {
            const h = makeHarness({
                capacity: 20,
                concurrency: 2,
                perGroupConcurrency: 1,
            });
            const results: string[] = [];
            for (let i = 0; i < 1_000; i += 1) {
                results.push(h.queue.submit(h.hold(`job-${i}`, `g-${i}`).job));
            }
            await settle();

            expect(results.filter((r) => r === "accepted")).toHaveLength(22);
            expect(results.filter((r) => r === "full")).toHaveLength(978);
            expect(h.queue.counts()).toEqual({ waiting: 20, running: 2 });
            expect(h.claims.counts()).toEqual({ inFlight: 22, kept: 0 });
        });

        it("holds nothing once every job has finished", async () => {
            const h = makeHarness({ capacity: 5, concurrency: 2 });
            const held = ["j1", "j2", "j3", "j4"].map((key) => h.hold(key));
            for (const { job } of held) {
                h.queue.submit(job);
            }

            for (const { gate } of held) {
                gate.resolve();
                await settle();
            }

            expect(h.queue.counts()).toEqual({ waiting: 0, running: 0 });
            expect(h.claims.counts().inFlight).toBe(0);
        });
    });

    describe("the share of the wait line one group may hold", () => {
        it("refuses a group's job once perGroupCapacity of its jobs are waiting, and still takes another group's", async () => {
            const h = makeHarness({
                capacity: 10,
                perGroupCapacity: 2,
                concurrency: 1,
            });
            const results = ["a1", "a2", "a3", "a4"].map((key) =>
                h.queue.submit(h.hold(key, "A").job),
            );
            const other = h.queue.submit(h.hold("b1", "B").job);

            expect(results).toEqual([
                "accepted", // runs
                "accepted", // waits
                "accepted", // waits
                "group_full",
            ]);
            expect(other).toBe("accepted");
            await settle();
            expect(h.queue.counts()).toEqual({ waiting: 3, running: 1 });
        });

        it("lets one group alone wait up to its share, no further, and leaves the rest of the line to the others", async () => {
            const h = makeHarness({
                capacity: 6,
                perGroupCapacity: 3,
                concurrency: 1,
            });
            const own = ["a1", "a2", "a3", "a4", "a5"].map((key) =>
                h.queue.submit(h.hold(key, "A").job),
            );
            const others = ["b1", "b2", "b3", "b4"].map((key) =>
                h.queue.submit(h.hold(key, "B").job),
            );

            expect(own).toEqual([
                "accepted", // runs
                "accepted",
                "accepted",
                "accepted",
                "group_full",
            ]);
            expect(others).toEqual([
                "accepted",
                "accepted",
                "accepted",
                "full",
            ]);
        });

        it("says 'full' when the whole line is full, even for a group that is also at its share", async () => {
            const h = makeHarness({
                capacity: 2,
                perGroupCapacity: 2,
                concurrency: 1,
            });
            h.queue.submit(h.hold("a1", "A").job); // runs
            h.queue.submit(h.hold("a2", "A").job);
            h.queue.submit(h.hold("a3", "A").job);

            expect(h.queue.submit(h.hold("a4", "A").job)).toBe("full");
        });

        it("counts only the jobs that wait: running ones do not use the share", async () => {
            const h = makeHarness({
                capacity: 10,
                perGroupCapacity: 1,
                concurrency: 4,
                perGroupConcurrency: 2,
            });
            const results = ["a1", "a2", "a3", "a4"].map((key) =>
                h.queue.submit(h.hold(key, "A").job),
            );

            expect(results).toEqual([
                "accepted", // runs
                "accepted", // runs
                "accepted", // waits
                "group_full",
            ]);
            await settle();
            expect(h.queue.counts()).toEqual({ waiting: 1, running: 2 });
        });

        it("frees the share when a waiting job starts", async () => {
            const h = makeHarness({
                capacity: 10,
                perGroupCapacity: 1,
                concurrency: 2,
                perGroupConcurrency: 1,
            });
            const first = h.hold("a1", "A");
            h.queue.submit(first.job); // runs
            h.queue.submit(h.hold("a2", "A").job); // waits: the whole share
            expect(h.queue.submit(h.hold("a3", "A").job)).toBe("group_full");

            first.gate.resolve(); // a2 starts, so nothing of A is waiting
            await settle();

            expect(h.started).toEqual(["a1", "a2"]);
            expect(h.queue.submit(h.hold("a4", "A").job)).toBe("accepted");
            expect(h.queue.submit(h.hold("a5", "A").job)).toBe("group_full");
        });

        it("gives the claim back when it refuses, so the same key is accepted once the share is free", async () => {
            const h = makeHarness({
                capacity: 10,
                perGroupCapacity: 1,
                concurrency: 1,
            });
            const first = h.hold("a1", "A");
            h.queue.submit(first.job);
            h.queue.submit(h.hold("a2", "A").job);
            expect(h.queue.submit(h.hold("a3", "A").job)).toBe("group_full");
            expect(h.claims.counts().inFlight).toBe(2);

            first.gate.resolve();
            await settle(); // a2 starts; nothing of A is waiting

            expect(h.queue.submit(h.hold("a3", "A").job)).toBe("accepted");
        });

        it("still answers 'duplicate' for a key that is claimed, from a group with no share left", async () => {
            const h = makeHarness({
                capacity: 10,
                perGroupCapacity: 1,
                concurrency: 1,
            });
            h.queue.submit(h.hold("a1", "A").job); // runs
            h.queue.submit(h.hold("a2", "A").job); // waits: the whole share

            expect(h.queue.submit(h.hold("a2", "A").job)).toBe("duplicate");
            expect(h.queue.submit(h.hold("a1", "A").job)).toBe("duplicate");
            expect(h.claims.counts().inFlight).toBe(2);
        });

        it("holds a flood from one group to its share", async () => {
            const h = makeHarness({
                capacity: 20,
                perGroupCapacity: 10,
                concurrency: 2,
                perGroupConcurrency: 1,
            });
            const results: string[] = [];
            for (let i = 0; i < 1_000; i += 1) {
                results.push(h.queue.submit(h.hold(`job-${i}`, "A").job));
            }
            await settle();

            expect(results.filter((r) => r === "accepted")).toHaveLength(11);
            expect(results.filter((r) => r === "group_full")).toHaveLength(989);
            expect(h.queue.counts()).toEqual({ waiting: 10, running: 1 });
            expect(h.claims.counts()).toEqual({ inFlight: 11, kept: 0 });
        });
    });

    describe("dedupe", () => {
        it("refuses a second job with the same key and runs the first once", async () => {
            const h = makeHarness();
            const first = h.hold("same-key");
            const second = h.hold("same-key");

            expect(h.queue.submit(first.job)).toBe("accepted");
            expect(h.queue.submit(second.job)).toBe("duplicate");

            await settle();
            expect(h.started).toEqual(["same-key"]);
        });

        it("never queues a refused duplicate: it does not wait, and does not run after the first finishes", async () => {
            const h = makeHarness();
            const first = h.hold("same-key");
            const second = h.hold("same-key");
            h.queue.submit(first.job);
            h.queue.submit(second.job);
            await settle();

            expect(h.queue.counts()).toEqual({ waiting: 0, running: 1 });

            first.gate.resolve("retry"); // releases the key, so a queued copy would now start
            await settle();

            expect(h.started).toEqual(["same-key"]);
            expect(h.queue.counts()).toEqual({ waiting: 0, running: 0 });
        });

        it("refuses a job whose key is kept after a successful run, until the keep period ends", async () => {
            const h = makeHarness();
            const first = h.hold("k");
            h.queue.submit(first.job);
            await settle();
            first.gate.resolve("done");
            await settle();

            expect(h.queue.submit(h.hold("k").job)).toBe("duplicate");

            h.clock.now += DAY_MS - 1;
            expect(h.queue.submit(h.hold("k").job)).toBe("duplicate");

            h.clock.now += 1;
            expect(h.queue.submit(h.hold("k").job)).toBe("accepted");
        });

        it("accepts the key again at once when the job ends with 'retry'", async () => {
            const h = makeHarness();
            const first = h.hold("k");
            h.queue.submit(first.job);
            await settle();

            first.gate.resolve("retry");
            await settle();

            expect(h.queue.submit(h.hold("k").job)).toBe("accepted");
        });

        it("accepts the key again at once when the job rejects", async () => {
            const h = makeHarness();
            const first = h.hold("k");
            h.queue.submit(first.job);
            await settle();

            first.gate.reject(new Error("review failed"));
            await settle();

            expect(h.queue.submit(h.hold("k").job)).toBe("accepted");
        });

        it("accepts the key again at once when the job throws before it returns a promise", async () => {
            const h = makeHarness();
            const throwing: Job = {
                key: "k",
                group: "g",
                label: "k",
                run: () => {
                    throw new Error("sync failure");
                },
            };
            h.queue.submit(throwing);
            await settle();

            expect(h.queue.submit(h.hold("k").job)).toBe("accepted");
        });

        it("releases the claim of a job that resolved with something that is not an outcome", async () => {
            const h = makeHarness();
            const odd: Job = {
                key: "k",
                group: "g",
                label: "k",
                run: () => Promise.resolve(undefined as unknown as JobOutcome),
            };
            h.queue.submit(odd);
            await settle();

            expect(h.queue.submit(h.hold("k").job)).toBe("accepted");
        });

        it("counts a running job's key as claimed", async () => {
            const h = makeHarness();
            h.queue.submit(h.hold("k").job);
            await settle();

            expect(h.queue.submit(h.hold("k").job)).toBe("duplicate");
        });
    });
});
