import { describe, it, expect, vi } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { nextTick } from "vue";
import { createGlobalConfig, createMockHttp } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import biblioFixture from "./fixtures/bibliography.json";
import Bibliography from "../src/components/Bibliography.vue";

function mountBibliography(overrides = {}) {
    const http = overrides.http || createMockHttp({ "bibliography.py": biblioFixture });
    const global = createGlobalConfig({
        http,
        route: { name: "bibliography", path: "/bibliography", query: { author: "Brown", report: "bibliography" } },
        stubs: {
            ResultsSummary: { template: "<div class='results-summary-stub' />" },
            Facets: { template: "<div class='facets-stub' />" },
            Pages: { template: "<div class='pages-stub' />" },
            Citations: { template: "<span class='citations-stub' />" },
            ProgressSpinner: { template: "<div class='spinner-stub' />" },
        },
        ...overrides,
    });

    const store = useMainStore();
    store.formData = { ...store.formData, report: "bibliography", author: "Brown" };

    return mount(Bibliography, { global });
}

describe("Bibliography", () => {
    it("makes HTTP request for bibliography results", async () => {
        const http = createMockHttp({ "bibliography.py": biblioFixture });
        mountBibliography({ http });
        await flushPromises();
        expect(http.get).toHaveBeenCalled();
        expect(http.get.mock.calls[0][0]).toContain("bibliography.py");
    });

    it("renders citations for each result", async () => {
        const wrapper = mountBibliography();
        await flushPromises();
        await nextTick();
        const citations = wrapper.findAll(".citations-stub");
        expect(citations.length).toBe(biblioFixture.results.length);
    });

    it("sets results length in store", async () => {
        mountBibliography();
        await flushPromises();
        const store = useMainStore();
        expect(store.resultsLength).toBe(biblioFixture.results_length);
    });

    it("renders facets sidebar when showFacets is true", async () => {
        const wrapper = mountBibliography();
        const store = useMainStore();
        store.showFacets = true;
        await flushPromises();
        await nextTick();
        // Bibliography's template gates the .facets-column wrapper on showFacets.
        // We assert on the wrapper's existence rather than the inner Facets stub
        // class because <script setup> imports are bound at compile time and
        // bypass the test-utils stub registry — the real Facets component
        // mounts here, not the stub.
        expect(wrapper.find(".facets-column").exists()).toBe(true);
    });
});
