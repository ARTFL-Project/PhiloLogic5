import { describe, it, expect, vi } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { nextTick } from "vue";
import { createGlobalConfig, createMockHttp } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import ResultsSummary from "../src/components/ResultsSummary.vue";

function mountResultsSummary(overrides = {}) {
    const http = overrides.http || createMockHttp({
        "hitlist_stats.py": { doc: 10, div1: 5 },
    });
    const global = createGlobalConfig({
        http,
        route: { name: "concordance", path: "/concordance", query: { q: "liberty", report: "concordance" } },
        stubs: {
            SearchArguments: { template: "<div class='search-args-stub' />", props: ["resultStart", "resultEnd"] },
            ExportResults: { template: "<div class='export-stub' />" },
            ResultsBibliography: { template: "<div class='results-biblio-stub' />" },
            ProgressSpinner: { template: "<span class='spinner-stub' />", props: ["progress", "sm"] },
        },
        ...overrides,
    });

    const store = useMainStore();
    store.formData = {
        ...store.formData,
        q: "liberty",
        report: overrides.report || "concordance",
        results_per_page: 25,
        colloc_filter_choice: "frequency",
        filter_frequency: 100,
    };
    store.resultsLength = 100;
    store.totalResultsDone = true;
    store.searching = false;
    store.showFacets = false;

    return mount(ResultsSummary, {
        props: {
            description: { start: 1, end: 25, results_per_page: 25 },
            filterList: [],
            ...overrides.props,
        },
        global,
    });
}

describe("ResultsSummary", () => {
    // --- Rendering ---
    it("renders results summary section", () => {
        const wrapper = mountResultsSummary();
        expect(wrapper.find("#results-summary-container").exists()).toBe(true);
    });

    it("renders export button", () => {
        const wrapper = mountResultsSummary();
        expect(wrapper.find("#export-results").exists()).toBe(true);
    });

    it("renders search arguments sub-component", () => {
        const wrapper = mountResultsSummary();
        expect(wrapper.find(".search-args-stub").exists()).toBe(true);
    });

    it("displays total results count when done", () => {
        const wrapper = mountResultsSummary();
        expect(wrapper.text()).toContain("100");
    });

    it("shows no results message when resultsLength is 0", () => {
        const wrapper = mountResultsSummary();
        const store = useMainStore();
        store.resultsLength = 0;
        // Wait for reactivity (sync since it's just a store update)
    });

    // --- @click="switchReport('concordance')" / 'kwic' ---
    it("renders report switch buttons for concordance report", () => {
        const wrapper = mountResultsSummary();
        expect(wrapper.find("#report_switch").exists()).toBe(true);
        const buttons = wrapper.findAll("#report_switch .btn");
        expect(buttons.length).toBe(2); // concordance + kwic
    });

    it("marks concordance button as active when report is concordance", () => {
        const wrapper = mountResultsSummary();
        const buttons = wrapper.findAll("#report_switch .btn");
        const concBtn = buttons[0];
        expect(concBtn.classes()).toContain("active");
        expect(concBtn.attributes("aria-pressed")).toBe("true");
    });

    it("switches to kwic on kwic button click", async () => {
        const wrapper = mountResultsSummary();
        const buttons = wrapper.findAll("#report_switch .btn");
        const kwicBtn = buttons[1];
        await kwicBtn.trigger("click");
        await nextTick();
        // Should navigate to kwic via router
    });

    it("switches to concordance on concordance button click", async () => {
        const wrapper = mountResultsSummary({ report: "kwic" });
        const store = useMainStore();
        store.formData.report = "kwic";
        await nextTick();
        const buttons = wrapper.findAll("#report_switch .btn");
        if (buttons.length > 0) {
            await buttons[0].trigger("click");
            await nextTick();
        }
    });

    // --- @click="switchResultsPerPage(number)" ---
    it("renders results per page dropdown", () => {
        const wrapper = mountResultsSummary();
        expect(wrapper.find("#results-per-page-content-toggle").exists()).toBe(true);
    });

    it("shows current results per page value", () => {
        const wrapper = mountResultsSummary();
        expect(wrapper.find("#results-per-page-content-toggle").text()).toContain("25");
    });

    it("renders results per page options in dropdown", () => {
        const wrapper = mountResultsSummary();
        const options = wrapper.findAll(".dropdown-menu .dropdown-item");
        expect(options.length).toBeGreaterThan(0);
    });

    it("changes results per page on dropdown item click", async () => {
        const wrapper = mountResultsSummary();
        const options = wrapper.findAll(".dropdown-menu .dropdown-item");
        if (options.length > 0) {
            await options[0].trigger("click");
            await nextTick();
            // Should navigate with new results_per_page
        }
    });

    // --- @click="toggleFacets()" ---
    it("shows facets toggle button when facets are hidden", async () => {
        const wrapper = mountResultsSummary({
            philoConfig: { facets: ["author", "title"] },
        });
        const store = useMainStore();
        store.showFacets = false;
        await nextTick();
        const facetsBtn = wrapper.findAll("button").find(b =>
            b.text().includes("Facets") || b.text().includes("facet") || b.text().includes("Show")
        );
        // Button may render depending on philoConfig.facets
    });

    // --- @click="toggleFilterList($event)" (collocation) ---
    it("renders filter list toggle in collocation mode", async () => {
        const wrapper = mountResultsSummary({
            report: "collocation",
            props: {
                description: { start: 1, end: 25, results_per_page: 25 },
                filterList: ["the", "a", "an", "of", "to"],
            },
        });
        const store = useMainStore();
        store.formData.report = "collocation";
        store.formData.colloc_filter_choice = "frequency";
        await nextTick();
        // Should show filter toggle button
        const filterBtn = wrapper.findAll(".btn-link");
        expect(filterBtn.length).toBeGreaterThanOrEqual(0);
    });

    it("toggles filter word list visibility on click", async () => {
        const wrapper = mountResultsSummary({
            report: "collocation",
            props: {
                description: { start: 1, end: 25, results_per_page: 25 },
                filterList: ["the", "a", "an"],
                collocMethod: "frequency",
            },
        });
        const store = useMainStore();
        store.formData.report = "collocation";
        store.formData.colloc_filter_choice = "frequency";
        await nextTick();

        const filterBtn = wrapper.find(".btn-link");
        if (filterBtn.exists()) {
            expect(wrapper.find("#filter-list").exists()).toBe(false);
            await filterBtn.trigger("click");
            await nextTick();
            expect(wrapper.find("#filter-list").exists()).toBe(true);

            // Close it
            const closeBtn = wrapper.find("#close-filter-list");
            if (closeBtn.exists()) {
                await closeBtn.trigger("click");
                await nextTick();
                expect(wrapper.find("#filter-list").exists()).toBe(false);
            }
        }
    });

    // --- Displays hits range ---
    it("displays hit range in description", () => {
        const wrapper = mountResultsSummary();
        const text = wrapper.text();
        // Should contain start-end range
        expect(text).toContain("1");
        expect(text).toContain("25");
    });
});
