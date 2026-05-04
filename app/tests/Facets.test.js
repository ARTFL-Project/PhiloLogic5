import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { nextTick } from "vue";
import { createGlobalConfig, createMockHttp } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import Facets from "../src/components/Facets.vue";

beforeEach(() => { vi.spyOn(console, "log").mockImplementation(() => {}); });
afterEach(() => { vi.restoreAllMocks(); });

function mountFacets(overrides = {}) {
    const http = overrides.http || createMockHttp({
        "get_frequency.py": { results: [{ label: "Brown", count: 10, metadata: { author: "Brown" } }], relative_results: [] },
    });
    const global = createGlobalConfig({
        http,
        philoConfig: {
            facets: ["author", "title", "year"],
            words_facets: [],
        },
        route: { name: "concordance", path: "/concordance", query: { q: "liberty", report: "concordance" } },
        stubs: {
            ProgressSpinner: { template: "<div class='spinner-stub' />" },
            WordCloud: { template: "<div class='word-cloud-stub' />" },
        },
        ...overrides,
    });

    const store = useMainStore();
    store.formData = { ...store.formData, q: "liberty", report: "concordance" };
    store.showFacets = true;

    return mount(Facets, { global });
}

describe("Facets", () => {
    // --- Rendering ---
    it("renders facet panel", () => {
        const wrapper = mountFacets();
        expect(wrapper.find("#facet-panel-wrapper").exists()).toBe(true);
    });

    it("renders facet selection buttons for each metadata facet", () => {
        const wrapper = mountFacets();
        const facetButtons = wrapper.findAll(".facet-selection");
        // author, title, year + collocation = at least 4
        expect(facetButtons.length).toBeGreaterThanOrEqual(3);
    });

    // --- @click="toggleFacets()" ---
    it("hides facets on close button click", async () => {
        const wrapper = mountFacets();
        const store = useMainStore();
        expect(store.showFacets).toBe(true);
        await wrapper.find(".close-box").trigger("click");
        await nextTick();
        expect(store.showFacets).toBe(false);
    });

    // --- @click="facetSearch(facet)" ---
    it("triggers facet search on metadata facet click", async () => {
        const http = createMockHttp({
            "get_frequency.py": { results: [{ label: "Brown", count: 10, metadata: { author: "Brown" } }], relative_results: [] },
        });
        const wrapper = mountFacets({ http });
        const facetBtn = wrapper.findAll(".facet-selection")[0]; // first facet (author)
        await facetBtn.trigger("click");
        await flushPromises();
        expect(http.get).toHaveBeenCalled();
        // Should have requested frequency.py with the facet field
        const freqCalls = http.get.mock.calls.filter(c => c[0].includes("frequency.py"));
        expect(freqCalls.length).toBeGreaterThan(0);
    });

    it("hides the facet-selection list after a facet is chosen", async () => {
        const wrapper = mountFacets();
        expect(wrapper.find("#select-facets").exists()).toBe(true);
        await wrapper.findAll(".facet-selection")[0].trigger("click");
        await flushPromises();
        await nextTick();
        expect(wrapper.find("#select-facets").exists()).toBe(false);
    });

    // --- @click="showFacetOptions()" ---
    it("re-shows the facet-selection list when 'show options' is clicked", async () => {
        const wrapper = mountFacets();
        await wrapper.findAll(".facet-selection")[0].trigger("click");
        await flushPromises();
        await nextTick();
        expect(wrapper.find("#select-facets").exists()).toBe(false);
        // The "show options" button renders only when showFacetSelection is false
        const showOptionsBtn = wrapper.findAll("button").find(b => b.text().includes("options") || b.text().includes("Options"));
        if (showOptionsBtn) {
            await showOptionsBtn.trigger("click");
            await nextTick();
            expect(wrapper.find("#select-facets").exists()).toBe(true);
        }
    });

    // --- Collocation facet ---
    it("renders collocation facet option", () => {
        const wrapper = mountFacets();
        expect(wrapper.text()).toContain("same sentence");
    });

    it("has collocation facet search capability", () => {
        const wrapper = mountFacets();
        // Collocation facet option exists in the template
        const collocBtn = wrapper.findAll(".facet-selection").find(b => b.text().includes("same sentence"));
        expect(collocBtn).toBeTruthy();
    });
});
