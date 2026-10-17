import { describe, it, expect } from "vitest";
import { useLatestRequest } from "../src/composables/useLatestRequest.js";

describe("useLatestRequest", () => {
    it("aborts the request in flight as the next starts, only the latest finishing", () => {
        const requests = useLatestRequest();
        const first = requests.start();
        const second = requests.start();
        expect(first.aborted).toBe(true);
        expect(second.aborted).toBe(false);
        expect(requests.finish(first)).toBe(false);
        expect(requests.finish(second)).toBe(true);
        expect(requests.abort()).toBe(false); // none in flight
        expect(second.aborted).toBe(false);
    });

    it("shares one signal between the requests of a key, a new key aborting it", () => {
        const requests = useLatestRequest();
        const total = requests.start({ q: "liberty" });
        const stats = requests.start({ q: "liberty" });
        expect(stats).toBe(total);
        const other = requests.start({ q: "freedom" });
        expect(total.aborted).toBe(true);
        expect(other.aborted).toBe(false);
    });

    it("ends the request in flight, saying there was one", () => {
        const requests = useLatestRequest();
        const signal = requests.start();
        expect(requests.abort()).toBe(true);
        expect(signal.aborted).toBe(true);
        expect(requests.finish(signal)).toBe(false);
    });
});
