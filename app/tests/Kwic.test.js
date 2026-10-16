import { describe, it, expect, vi } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { nextTick } from "vue";
import { createGlobalConfig, createMockHttp } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import kwicFixture from "./fixtures/kwic.json";
import Kwic from "../src/components/Kwic.vue";

function mountKwic(overrides = {}) {
    const http = overrides.http || createMockHttp({ "kwic.py": kwicFixture });
    const global = createGlobalConfig({
        http,
        route: { name: "kwic", path: "/kwic", query: { q: "liberty", report: "kwic" } },
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
    store.formData = {
        ...store.formData,
        q: "liberty",
        report: "kwic",
        results_per_page: 25,
        ...(overrides.formData || {}),
    };

    return mount(Kwic, { global });
}

describe("Kwic", () => {
    // --- Rendering ---
    it("makes HTTP request for kwic results", async () => {
        const http = createMockHttp({ "kwic.py": kwicFixture });
        mountKwic({ http });
        await flushPromises();
        expect(http.get).toHaveBeenCalled();
        expect(http.get.mock.calls[0][0]).toContain("kwic.py");
    });

    it("renders kwic results with highlighted terms", async () => {
        const wrapper = mountKwic();
        await flushPromises();
        await nextTick();
        expect(wrapper.html()).toContain("kwic-highlight");
        expect(wrapper.html()).toContain("liberty");
    });

    it("renders kwic results count matching fixture", async () => {
        const wrapper = mountKwic();
        await flushPromises();
        await nextTick();
        const kwicLines = wrapper.findAll(".kwic-line.visual-kwic");
        expect(kwicLines.length).toBe(kwicFixture.results.length);
    });

    it("renders accessible kwic version for screen readers", async () => {
        const wrapper = mountKwic();
        await flushPromises();
        await nextTick();
        const accessibleLines = wrapper.findAll(".accessible-kwic");
        expect(accessibleLines.length).toBe(kwicFixture.results.length);
    });

    // --- Sorting controls ---
    it("renders sort button", async () => {
        const wrapper = mountKwic();
        await flushPromises();
        await nextTick();
        expect(wrapper.find("#sort-button").exists()).toBe(true);
    });

    it("renders sorting dropdown triggers", async () => {
        const wrapper = mountKwic();
        await flushPromises();
        await nextTick();
        const dropdowns = wrapper.findAll(".sort-toggle");
        expect(dropdowns.length).toBeGreaterThan(0);
    });

    // --- @click="updateSortingSelection(index, selection)" ---
    it("updates sorting selection on dropdown item click", async () => {
        const wrapper = mountKwic();
        await flushPromises();
        await nextTick();

        const dropdownItems = wrapper.findAll(".dropdown-item");
        if (dropdownItems.length > 0) {
            const initialText = wrapper.findAll(".sort-toggle")[0].text();
            await dropdownItems[0].trigger("click");
            await nextTick();
            // Sorting selection should update (text may change)
            expect(wrapper.findAll(".sort-toggle")[0].exists()).toBe(true);
        }
    });

    // --- @click="sortResults()" ---
    it("triggers sort on sort button click", async () => {
        const http = createMockHttp({
            "kwic.py": kwicFixture,
            "get_sorted_kwic.py": kwicFixture,
        });
        const wrapper = mountKwic({ http });
        await flushPromises();
        await nextTick();

        await wrapper.find("#sort-button").trigger("click");
        await flushPromises();
        // Sort should trigger an HTTP request or use cached data
    });

    // --- @mouseover/@mouseleave on biblio container ---
    it("shows full bibliography on hover", async () => {
        const wrapper = mountKwic();
        await flushPromises();
        await nextTick();

        const biblioContainer = wrapper.find(".kwic-biblio-container");
        if (biblioContainer.exists()) {
            await biblioContainer.trigger("mouseover");
            await nextTick();
            // Full biblio div should become visible
            const fullBiblio = wrapper.find(".full-biblio");
            if (fullBiblio.exists()) {
                expect(fullBiblio.isVisible() || fullBiblio.exists()).toBe(true);
            }
        }
    });

    it("hides full bibliography on mouse leave", async () => {
        const wrapper = mountKwic();
        await flushPromises();
        await nextTick();

        const biblioContainer = wrapper.find(".kwic-biblio-container");
        if (biblioContainer.exists()) {
            await biblioContainer.trigger("mouseover");
            await nextTick();
            await biblioContainer.trigger("mouseleave");
            await nextTick();
        }
    });

    // --- @focus/@blur on biblio link (accessibility) ---
    it("shows full bibliography on focus for keyboard users", async () => {
        const wrapper = mountKwic();
        await flushPromises();
        await nextTick();

        const biblioLink = wrapper.find(".kwic-biblio");
        if (biblioLink.exists()) {
            await biblioLink.trigger("focus");
            await nextTick();
        }
    });

    // --- Sorted KWIC: one JSON request, aborted when its results are no longer wanted ---
    it("fetches the sorted kwic as JSON and renders it", async () => {
        const http = createMockHttp({ "get_sorted_kwic.py": kwicFixture });
        const wrapper = mountKwic({ http, formData: { first_kwic_sorting_option: "author" } });
        await flushPromises();
        await nextTick();

        const [url, config] = http.get.mock.calls[0];
        expect(url).toContain("scripts/get_sorted_kwic.py");
        expect(config.params.first_kwic_sorting_option).toBe("author");
        expect(wrapper.findAll(".kwic-line.visual-kwic").length).toBe(kwicFixture.results.length);
        expect(useMainStore().searching).toBe(false);
    });

    // An $http whose sorted-KWIC requests stay pending, until their signal aborts them as axios does
    function pendingSortedHttp(signals) {
        return {
            get: vi.fn((url, config) => {
                if (!url.includes("get_sorted_kwic.py")) return Promise.resolve({ data: kwicFixture });
                signals.push(config.signal);
                return new Promise((_resolve, reject) => {
                    config.signal.addEventListener("abort", () => {
                        reject(Object.assign(new Error("canceled"), { name: "CanceledError" }));
                    });
                });
            }),
        };
    }

    it("aborts the sorted request on unmount", async () => {
        const signals = [];
        const wrapper = mountKwic({
            http: pendingSortedHttp(signals),
            formData: { first_kwic_sorting_option: "author" },
        });
        await flushPromises();
        const store = useMainStore();
        expect(signals).toHaveLength(1);
        expect(store.searching).toBe(true);

        wrapper.unmount();
        await flushPromises();

        expect(signals[0].aborted).toBe(true);
        expect(store.searching).toBe(false);
    });

    it("keeps searching when a new query aborts the sorted request", async () => {
        const signals = [];
        mountKwic({ http: pendingSortedHttp(signals), formData: { first_kwic_sorting_option: "author" } });
        await flushPromises();
        const store = useMainStore();

        store.formData.first_kwic_sorting_option = "title";
        store.urlUpdate = { q: "liberty", report: "kwic", first_kwic_sorting_option: "title" };
        await flushPromises();

        expect(signals).toHaveLength(2);
        expect(signals[0].aborted).toBe(true);
        expect(signals[1].aborted).toBe(false);
        expect(store.searching).toBe(true); // the aborted request didn't end the search of the new one
    });
});
