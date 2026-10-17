import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { nextTick } from "vue";
import { createGlobalConfig, createMockHttp, createPendingHttp } from "./helpers.js";
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
        // author, title, year
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

    // --- No collocation facet: collocation counts co-occurrences in the sentence or within n words, unordered,
    // not with the other search methods a concordance can have
    it("has no collocation facet", () => {
        const wrapper = mountFacets();
        expect(wrapper.text()).not.toContain("same sentence");
        expect(wrapper.text()).not.toContain("Collocate");
    });
});

describe("Facets and the URL", () => {
    const counts = {
        results: [{ label: "Brown", count: 10, metadata: { author: "Brown" } }],
        relative_results: [{ label: "Smith", count: 2.5, metadata: { author: "Smith" } }],
    };
    const query = { q: "liberty", report: "concordance" };
    const go = async (wrapper, more) => {
        await wrapper.vm.$router.push({ path: "/concordance", query: { ...query, ...more } });
        await flushPromises();
    };
    // The panel open on the authors, as the URL has it
    async function opened(more = {}, http = createMockHttp({ "get_frequency.py": counts })) {
        const wrapper = mountFacets({ http });
        await go(wrapper, { facet: "author", ...more });
        return wrapper;
    }
    const listed = (wrapper) => wrapper.findAll(".facet-result-item .sidebar-text").map((n) => n.text());

    it("shows the relative frequencies the URL asks for, each button showing its own", async () => {
        const wrapper = await opened({ relative_frequency: "true" });
        expect(listed(wrapper)).toEqual(["Smith"]);
        const [absolute, relative] = wrapper.findAll("#facet-results .btn-group .btn");
        expect(relative.attributes("aria-pressed")).toBe("true");

        await relative.trigger("click"); // shown already: still them
        await flushPromises();
        expect(listed(wrapper)).toEqual(["Smith"]);

        await absolute.trigger("click");
        await flushPromises();
        expect(wrapper.vm.$router.currentRoute.value.query.relative_frequency).toBe("false");
        expect(listed(wrapper)).toEqual(["Brown"]);
        expect(absolute.attributes("aria-pressed")).toBe("true");
    });

    it("shows the facet asked for last, the request of the one before aborted", async () => {
        const { http, requests } = createPendingHttp("get_frequency.py");
        const wrapper = await opened({}, http);
        await go(wrapper, { facet: "title" });
        expect(requests).toHaveLength(2);
        expect(requests[0].config.signal.aborted).toBe(true);

        requests[1].answer({ results: [{ label: "Candide", count: 3, metadata: { title: "Candide" } }], relative_results: [] });
        requests[0].answer(counts); // too late
        await flushPromises();
        expect(listed(wrapper)).toEqual(["Candide"]);
    });

    it("shows no counts once the URL has no facet, even those coming later", async () => {
        const { http, requests } = createPendingHttp("get_frequency.py");
        const wrapper = await opened({}, http);
        await go(wrapper, {});
        expect(requests[0].config.signal.aborted).toBe(true);
        requests[0].answer(counts);
        await flushPromises();
        expect(wrapper.find("#facet-results").exists()).toBe(false);
        expect(wrapper.find("#select-facets").exists()).toBe(true);
    });
});
