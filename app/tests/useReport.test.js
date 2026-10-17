import { describe, it, expect, vi, afterEach } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { defineComponent, h } from "vue";
import { createTestPinia } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import { useReport } from "../src/composables/useReport.js";

// A page of the report, whose search fetches with request(signal) and shows what it answers
function mountReport({ report = "concordance", request, watchUrl } = {}) {
    const pinia = createTestPinia();
    const shown = [];
    let run;
    const Page = defineComponent({
        setup() {
            ({ run } = useReport(report, async (signal) => shown.push(await request(signal)), { watchUrl }));
            return () => h("div");
        },
    });
    const wrapper = mount(Page, { global: { plugins: [pinia] } });
    return { wrapper, shown, run, store: useMainStore() };
}

// Requests that stay pending until answered, in any order, or aborted, rejecting as axios does
function pendingRequests() {
    const requests = [];
    const request = vi.fn((signal) => new Promise((resolve, reject) => {
        requests.push({ signal, answer: resolve });
        signal.addEventListener("abort", () => reject(Object.assign(new Error("canceled"), { name: "CanceledError" })));
    }));
    return { requests, request };
}

describe("useReport", () => {
    afterEach(() => vi.restoreAllMocks());

    it("says the report in the form and as the current one", () => {
        const { store } = mountReport({ report: "kwic", request: () => Promise.resolve() });
        expect(store.formData.report).toBe("kwic");
        expect(store.currentReport).toBe("kwic");
    });

    it("shows the newest search, aborting the one in flight, whose answer comes too late", async () => {
        const { requests, request } = pendingRequests();
        const { run, shown, store } = mountReport({ request });
        run();
        run();
        expect(requests[0].signal.aborted).toBe(true);
        expect(store.searching).toBe(true); // the aborted search didn't end the new one's

        requests[1].answer("page 3");
        requests[0].answer("page 2");
        await flushPromises();
        expect(shown).toEqual(["page 3"]);
        expect(store.searching).toBe(false);
    });

    it("ends the search in flight when the page is left", async () => {
        const { requests, request } = pendingRequests();
        const { run, wrapper, store } = mountReport({ request });
        run();
        wrapper.unmount();
        expect(requests[0].signal.aborted).toBe(true);
        expect(store.searching).toBe(false);

        store.searching = true; // the next page's search
        await flushPromises();
        expect(store.searching).toBe(true); // the aborted one, ending later, leaves it alone
    });

    it("logs an error, ending the search", async () => {
        const log = vi.spyOn(console, "log").mockImplementation(() => {});
        const { run, store } = mountReport({ request: () => Promise.reject(new Error("500")) });
        await run();
        expect(store.searching).toBe(false);
        expect(log).toHaveBeenCalledWith("MESSAGE FROM concordance-report:", expect.any(Error));
    });

    it("runs again when the URL changes the search, not for the facets alone, nor for another report", async () => {
        const request = vi.fn(() => Promise.resolve());
        const { store } = mountReport({ request });
        store.urlUpdate = { q: "liberty" };
        await flushPromises();
        expect(request).toHaveBeenCalledTimes(1);

        store.urlUpdate = { q: "liberty", facet: "author" };
        await flushPromises();
        expect(request).toHaveBeenCalledTimes(1);

        store.formData.report = "kwic"; // the URL of the KWIC this page leaves for
        store.urlUpdate = { q: "liberty", report: "kwic" };
        await flushPromises();
        expect(request).toHaveBeenCalledTimes(1);
    });

    it("leaves the URL to the page when asked", async () => {
        const request = vi.fn(() => Promise.resolve());
        const { store } = mountReport({ report: "collocation", request, watchUrl: false });
        store.urlUpdate = { q: "liberty" };
        await flushPromises();
        expect(request).not.toHaveBeenCalled();
    });
});
