import { describe, it, expect, vi } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { nextTick } from "vue";
import { createGlobalConfig, createMockHttp } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import concordanceFixture from "./fixtures/concordance.json";
import Concordance from "../src/components/Concordance.vue";

function mountConcordance(overrides = {}) {
    const http = overrides.http || createMockHttp({ "concordance.py": concordanceFixture });
    const global = createGlobalConfig({
        http,
        route: { name: "concordance", path: "/concordance", query: { q: "liberty", report: "concordance" } },
        stubs: {
            ResultsSummary: { template: "<div class='results-summary-stub' />" },
            Facets: { template: "<div class='facets-stub' />" },
            Pages: { template: "<div class='pages-stub' />" },
            Citations: { template: "<span class='citations-stub' />" },
        },
        ...overrides,
    });

    const store = useMainStore();
    store.formData = { ...store.formData, q: "liberty", report: "concordance", results_per_page: 25 };

    return mount(Concordance, { global });
}

describe("Concordance", () => {
    it("fetches and renders concordance results", async () => {
        const wrapper = mountConcordance();
        await flushPromises();
        await nextTick();

        const results = wrapper.findAll(".philologic-occurrence");
        expect(results.length).toBe(concordanceFixture.results.length);
    });

    it("renders context with highlighted term", async () => {
        const wrapper = mountConcordance();
        await flushPromises();
        await nextTick();

        expect(wrapper.html()).toContain("highlight");
        expect(wrapper.html()).toContain("liberty");
    });

    it("renders citation for each result", async () => {
        const wrapper = mountConcordance();
        await flushPromises();
        await nextTick();

        const citations = wrapper.findAll(".citations-stub");
        expect(citations.length).toBe(concordanceFixture.results.length);
    });

    it("renders 'more context' button for each result", async () => {
        const wrapper = mountConcordance();
        await flushPromises();
        await nextTick();

        const moreButtons = wrapper.findAll(".more-context");
        expect(moreButtons.length).toBe(concordanceFixture.results.length);
    });

    it("makes HTTP request for concordance results", async () => {
        const http = createMockHttp({ "concordance.py": concordanceFixture });
        mountConcordance({ http });
        await flushPromises();

        expect(http.get).toHaveBeenCalled();
        expect(http.get.mock.calls[0][0]).toContain("concordance.py");
    });

    // @click="moreContext(index, $event)"
    it("more context button exists and is clickable for each result", async () => {
        const wrapper = mountConcordance();
        await flushPromises();
        await nextTick();

        const moreButtons = wrapper.findAll(".more-context");
        expect(moreButtons.length).toBe(concordanceFixture.results.length);
        // Each button has an aria-label for accessibility
        for (const btn of moreButtons) {
            expect(btn.attributes("aria-label")).toBeTruthy();
            expect(btn.attributes("type")).toBe("button");
        }
    });

    // @keyup="dicoLookup($event, result.metadata_fields.year)"
    it("handles dictionary lookup keypress on concordance text", async () => {
        const wrapper = mountConcordance();
        await flushPromises();
        await nextTick();

        const textDiv = wrapper.find(".concordance-text");
        // Pressing 'd' triggers dictionary lookup (opens window)
        const windowSpy = vi.spyOn(window, "open").mockImplementation(() => {});
        await textDiv.trigger("keyup", { key: "d" });
        // Should not throw — dictionaryLookup handles gracefully even without selection
        windowSpy.mockRestore();
    });
});
