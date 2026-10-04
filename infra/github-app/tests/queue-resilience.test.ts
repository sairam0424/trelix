import { afterEach, describe, expect, it, vi } from "vitest";
import { makeHarness, settle } from "./support/queue-harness.js";

afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
});

describe("JobQueue, when a job fails and when it shuts down", () => {
    describe("the worker survives a bad job", () => {
        it("reports a rejected job once, by label, and runs the next job", async () => {
            const h = makeHarness({ concurrency: 1, capacity: 5 });
            const bad = h.hold("bad");
            const good = h.hold("good");
            h.queue.submit(bad.job);
            h.queue.submit(good.job);
            await settle();

            const failure = new Error("boom");
            bad.gate.reject(failure);
            await settle();

            expect(h.errors).toEqual([{ err: failure, label: "bad" }]);
            expect(h.started).toEqual(["bad", "good"]);
        });

        it("reports a job that throws before it returns a promise, and runs the next job", async () => {
            const h = makeHarness({ concurrency: 1, capacity: 5 });
            const failure = new Error("sync failure");
            h.queue.submit({
                key: "bad",
                group: "g1",
                label: "bad",
                run: () => {
                    throw failure;
                },
            });
            h.queue.submit(h.hold("good", "g2").job);

            await settle();

            expect(h.errors).toEqual([{ err: failure, label: "bad" }]);
            expect(h.started).toEqual(["good"]);
        });

        it("reports a rejection that is not an Error", async () => {
            const h = makeHarness();
            const bad = h.hold("bad");
            h.queue.submit(bad.job);
            await settle();

            bad.gate.reject("a plain string");
            await settle();

            expect(h.errors).toEqual([{ err: "a plain string", label: "bad" }]);
        });

        it("does not report a job that finished normally", async () => {
            const h = makeHarness();
            const fine = h.hold("fine");
            h.queue.submit(fine.job);
            await settle();

            fine.gate.resolve("retry");
            await settle();

            expect(h.errors).toEqual([]);
        });

        it("survives an error reporter that throws, saying so on the console with a fixed line", async () => {
            const consoleError = vi
                .spyOn(console, "error")
                .mockImplementation(() => {});
            const h = makeHarness({ concurrency: 1, capacity: 5 }, () => {
                throw new Error("logger is down: canary-secret");
            });
            const bad = h.hold("bad");
            h.queue.submit(bad.job);
            h.queue.submit(h.hold("good", "g2").job);
            await settle();

            bad.gate.reject(new Error("boom"));
            await settle();

            expect(h.started).toEqual(["bad", "good"]);
            expect(consoleError.mock.calls).toEqual([
                ["[queue] a job failed; the error could not be logged"],
            ]);
        });

        it("produces no unhandled rejection for a failing job", async () => {
            const unhandled = vi.fn();
            process.on("unhandledRejection", unhandled);
            try {
                const h = makeHarness();
                const bad = h.hold("bad");
                h.queue.submit(bad.job);
                await settle();

                bad.gate.reject(new Error("boom"));
                await settle();
                await new Promise((resolve) => setTimeout(resolve, 0));

                expect(unhandled).not.toHaveBeenCalled();
            } finally {
                process.off("unhandledRejection", unhandled);
            }
        });
    });

    describe("timers", () => {
        it("arms none while it accepts, runs and finishes jobs", async () => {
            vi.useFakeTimers();
            const h = makeHarness();
            const held = ["j1", "j2", "j3"].map((key) => h.hold(key));
            for (const { job } of held) {
                h.queue.submit(job);
            }
            await settle();
            expect(vi.getTimerCount()).toBe(0);

            for (const { gate } of held) {
                gate.resolve();
                await settle();
            }

            expect(vi.getTimerCount()).toBe(0);
        });
    });

    describe("shutdown", () => {
        it("drops the jobs that are waiting, releases their claims and refuses new jobs", async () => {
            const h = makeHarness({ concurrency: 1, capacity: 5 });
            const running = h.hold("running");
            h.queue.submit(running.job);
            h.queue.submit(h.hold("waiting-1").job);
            h.queue.submit(h.hold("waiting-2").job);
            await settle();

            const done = h.queue.shutdown(10_000);
            expect(h.queue.counts().waiting).toBe(0);
            expect(h.claims.counts().inFlight).toBe(1);
            expect(h.queue.submit(h.hold("late").job)).toBe("closed");

            running.gate.resolve();
            await expect(done).resolves.toEqual({
                abandoned: 2,
                stillRunning: 0,
            });
            await settle();
            expect(h.started).toEqual(["running"]);
        });

        it("does not claim the key of a job it refuses", async () => {
            const h = makeHarness();
            await h.queue.shutdown(1_000);

            expect(h.queue.submit(h.hold("late").job)).toBe("closed");
            expect(h.claims.counts().inFlight).toBe(0);
        });

        it("returns at once, with no timer, when nothing is running", async () => {
            vi.useFakeTimers();
            const h = makeHarness();

            const report = await h.queue.shutdown(60_000);

            expect(report).toEqual({ abandoned: 0, stillRunning: 0 });
            expect(vi.getTimerCount()).toBe(0);
        });

        it("gives up after the grace period and reports the jobs still running", async () => {
            vi.useFakeTimers();
            const h = makeHarness();
            h.queue.submit(h.hold("stuck").job);
            await vi.advanceTimersByTimeAsync(0);

            const done = h.queue.shutdown(5_000);
            await vi.advanceTimersByTimeAsync(4_999);
            let settledEarly = false;
            void done.then(() => {
                settledEarly = true;
            });
            await vi.advanceTimersByTimeAsync(0);
            expect(settledEarly).toBe(false);

            await vi.advanceTimersByTimeAsync(1);
            await expect(done).resolves.toEqual({
                abandoned: 0,
                stillRunning: 1,
            });
            expect(vi.getTimerCount()).toBe(0);
        });

        it("clears its timer when the running jobs finish before the grace period ends", async () => {
            vi.useFakeTimers();
            const h = makeHarness();
            const running = h.hold("running");
            h.queue.submit(running.job);
            await vi.advanceTimersByTimeAsync(0);

            const done = h.queue.shutdown(60_000);
            expect(vi.getTimerCount()).toBe(1);
            running.gate.resolve();
            await vi.advanceTimersByTimeAsync(0);

            await expect(done).resolves.toEqual({
                abandoned: 0,
                stillRunning: 0,
            });
            expect(vi.getTimerCount()).toBe(0);
        });

        it("can be called twice", async () => {
            const h = makeHarness();
            const running = h.hold("running");
            h.queue.submit(running.job);
            await settle();

            const first = h.queue.shutdown(10_000);
            const second = h.queue.shutdown(10_000);
            running.gate.resolve();

            await expect(first).resolves.toMatchObject({ stillRunning: 0 });
            await expect(second).resolves.toMatchObject({ stillRunning: 0 });
        });
    });
});
