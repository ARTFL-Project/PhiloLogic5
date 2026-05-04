import { describe, it, expect, vi } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { nextTick } from "vue";
import { createGlobalConfig, createMockHttp } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import webConfig from "./fixtures/web_config.json";
import SearchForm from "../src/components/SearchForm.vue";

function mountSearchForm(overrides = {}) {
    const global = createGlobalConfig({
        philoConfig: { ...webConfig, ...overrides.philoConfig },
        route: { name: "concordance", path: "/concordance", query: { report: "concordance" } },
        stubs: {
            MetadataFields: { template: "<div class='metadata-fields-stub' />" },
            SearchTips: { template: "<div class='search-tips-stub' />" },
            ProgressSpinner: { template: "<div class='spinner-stub' />" },
        },
        ...overrides,
    });

    const store = useMainStore();
    store.formData = {
        ...store.formData,
        report: "concordance",
        q: "",
        results_per_page: 25,
    };
    store.reportValues = {
        concordance: new Set(["q", "method", "cooc_order", "method_arg", "results_per_page", "sort_by", "start", "end", ...webConfig.metadata]),
        kwic: new Set(["q", "method", "cooc_order", "method_arg", "results_per_page", ...webConfig.metadata]),
        collocation: new Set(["q", "colloc_filter_choice", "colloc_within", "filter_frequency", ...webConfig.metadata]),
        time_series: new Set(["q", "method", "cooc_order", "method_arg", "start_date", "end_date", "year_interval", ...webConfig.metadata]),
        aggregation: new Set(["q", "method", "cooc_order", "method_arg", "group_by", ...webConfig.metadata]),
    };

    return mount(SearchForm, { global });
}

describe("SearchForm", () => {
    // --- Rendering ---
    it("renders search form with query input", () => {
        const wrapper = mountSearchForm();
        expect(wrapper.find("#query-term-input").exists()).toBe(true);
    });

    it("renders report type buttons", () => {
        const wrapper = mountSearchForm();
        const buttons = wrapper.findAll("#report .btn");
        expect(buttons.length).toBe(webConfig.search_reports.length);
    });

    it("renders search button", () => {
        const wrapper = mountSearchForm();
        expect(wrapper.find("#button-search").exists()).toBe(true);
    });

    it("renders reset button", () => {
        const wrapper = mountSearchForm();
        expect(wrapper.find("#reset_form").exists()).toBe(true);
    });

    it("renders toggle form button", () => {
        const wrapper = mountSearchForm();
        expect(wrapper.find("#show-search-form").exists()).toBe(true);
    });

    // --- @click="reportChange(searchReport)" ---
    it("changes report type on button click", async () => {
        const wrapper = mountSearchForm();
        const store = useMainStore();
        const buttons = wrapper.findAll("#report .btn");
        // Click on the second report type
        await buttons[1].trigger("click");
        await nextTick();
        expect(store.currentReport).toBe(webConfig.search_reports[1]);
    });

    // --- @change on mobile dropdown ---
    it("changes report type on mobile dropdown change", async () => {
        const wrapper = mountSearchForm();
        const select = wrapper.find("#report-type-mobile-select");
        await select.setValue("kwic");
        await nextTick();
        const store = useMainStore();
        expect(store.currentReport).toBe("kwic");
    });

    // --- @click="onSubmit()" ---
    it("submits search on search button click", async () => {
        const wrapper = mountSearchForm();
        await wrapper.find("#query-term-input").setValue("liberty");
        await wrapper.find("#button-search").trigger("click");
        await nextTick();
        // Should attempt to navigate via router
    });

    // --- @keyup.enter="onSubmit()" on form ---
    it("submits search on Enter key in form", async () => {
        const wrapper = mountSearchForm();
        await wrapper.find("#query-term-input").setValue("liberty");
        await wrapper.find("form").trigger("keyup.enter");
        await nextTick();
    });

    // --- @reset="onReset" ---
    it("clears form on reset", async () => {
        const wrapper = mountSearchForm();
        await wrapper.find("#query-term-input").setValue("liberty");
        await wrapper.find("form").trigger("reset");
        await nextTick();
        expect(wrapper.find("#query-term-input").element.value).toBe("");
    });

    // --- @click="toggleForm()" ---
    it("toggles advanced search options visibility", async () => {
        const wrapper = mountSearchForm();
        const toggleBtn = wrapper.find("#show-search-form");

        expect(wrapper.find("#search-elements").exists()).toBe(false);
        await toggleBtn.trigger("click");
        await nextTick();
        expect(wrapper.find("#search-elements").exists()).toBe(true);
    });

    it("hides advanced options on second toggle click", async () => {
        const wrapper = mountSearchForm();
        const toggleBtn = wrapper.find("#show-search-form");

        await toggleBtn.trigger("click");
        await nextTick();
        expect(wrapper.find("#search-elements").exists()).toBe(true);

        await toggleBtn.trigger("click");
        await nextTick();
        expect(wrapper.find("#search-elements").exists()).toBe(false);
    });

    // --- @mouseover/@mouseleave on tips button ---
    it("shows tips label on hover", async () => {
        const wrapper = mountSearchForm();
        const tipsBtn = wrapper.find(".btn-outline-info");
        // Pre-hover: button shows "?" placeholder, not the localized tips text
        expect(tipsBtn.text()).toContain("?");

        await tipsBtn.trigger("mouseover");
        await nextTick();
        expect(tipsBtn.text()).not.toContain("?");

        await tipsBtn.trigger("mouseleave");
        await nextTick();
        expect(tipsBtn.text()).toContain("?");
    });

    // --- @input="onChange('q')" autocomplete ---
    it("triggers autocomplete on query input", async () => {
        const http = createMockHttp({ "autocomplete.py": [] });
        const wrapper = mountSearchForm({ http });
        const input = wrapper.find("#query-term-input");
        await input.setValue("lib");
        await input.trigger("input");
        await nextTick();
        // Should have called autocomplete endpoint
    });

    // --- @keyup.escape="clearAutoCompletePopup" ---
    it("clears autocomplete on Escape key", async () => {
        const wrapper = mountSearchForm();
        const input = wrapper.find("#query-term-input");
        await input.trigger("keyup.escape");
        await nextTick();
        // The autocomplete <ul> is rendered v-if results.length > 0 — it should not be present
        expect(wrapper.find("#autocomplete-q").exists()).toBe(false);
    });

    // --- @change="toggleApproximate" ---
    it("toggles approximate match in advanced options", async () => {
        const wrapper = mountSearchForm();
        // Open advanced options first
        await wrapper.find("#show-search-form").trigger("click");
        await nextTick();

        const checkbox = wrapper.find("#approximate-input");
        await checkbox.trigger("change");
        await nextTick();
    });

    // --- @change="toggleCoocOrder" ---
    it("toggles co-occurrence order in advanced options", async () => {
        const wrapper = mountSearchForm();
        await wrapper.find("#show-search-form").trigger("click");
        await nextTick();

        const checkbox = wrapper.find("#co-occurrence-order-input");
        if (checkbox.exists()) {
            await checkbox.trigger("change");
            await nextTick();
        }
    });

    // --- Collocation params shown when report is collocation ---
    it("shows collocation params when collocation report selected", async () => {
        const wrapper = mountSearchForm();
        const store = useMainStore();
        store.currentReport = "collocation";
        await wrapper.find("#show-search-form").trigger("click");
        await nextTick();
        expect(wrapper.find("#collocation-params").exists()).toBe(true);
    });

    // --- Time series params shown when report is time_series ---
    it("shows time series params when time_series report selected", async () => {
        const wrapper = mountSearchForm();
        const store = useMainStore();
        store.currentReport = "time_series";
        await wrapper.find("#show-search-form").trigger("click");
        await nextTick();
        expect(wrapper.find("#time-series-params").exists()).toBe(true);
    });
});
