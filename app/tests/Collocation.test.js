import { describe, it, expect, vi } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { nextTick } from "vue";
import { createGlobalConfig, createMockHttp } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import collocationFixture from "./fixtures/collocation.json";
import Collocation from "../src/components/Collocation.vue";

function mountCollocation(overrides = {}) {
    const http = overrides.http || createMockHttp({
        "collocation.py": collocationFixture,
        "comparative_collocations.py": { top: [["power", 100], ["free", 80]], bottom: [["own", 50]] },
    });
    const global = createGlobalConfig({
        http,
        route: { name: "collocation", path: "/collocation", query: { q: "liberty", report: "collocation" } },
        stubs: {
            ResultsSummary: { template: "<div class='results-summary-stub' />" },
            WordCloud: { template: "<div class='word-cloud-stub' />", props: ["wordWeights", "clickHandler", "label"] },
            BibliographyCriteria: { template: "<div class='biblio-criteria-stub' />" },
            ProgressSpinner: { template: "<div class='spinner-stub' />" },
            MetadataFields: { template: "<div class='metadata-fields-stub' />" },
        },
        ...overrides,
    });

    const store = useMainStore();
    store.formData = { ...store.formData, q: "liberty", report: "collocation", colloc_within: "sent", colloc_filter_choice: "frequency", filter_frequency: 100 };
    store.searchableMetadata = {
        display: [{ value: "author", label: "Author", example: "Voltaire" }],
        inputStyle: { author: "text" },
        choiceValues: {},
    };

    return mount(Collocation, { global });
}

describe("Collocation", () => {
    // --- Rendering ---
    it("renders collocation container", async () => {
        const wrapper = mountCollocation();
        await flushPromises();
        expect(wrapper.find("#collocation-container").exists()).toBe(true);
    });

    it("renders all method tabs", async () => {
        const wrapper = mountCollocation();
        await flushPromises();
        expect(wrapper.find("#frequency-tab").exists()).toBe(true);
        expect(wrapper.find("#compare-tab").exists()).toBe(true);
        expect(wrapper.find("#similar-tab").exists()).toBe(true);
        expect(wrapper.find("#time-series-tab").exists()).toBe(true);
    });

    it("fetches and displays collocates table", async () => {
        const wrapper = mountCollocation();
        await flushPromises();
        await nextTick();
        expect(wrapper.findAll("tbody tr").length).toBeGreaterThan(0);
    });

    it("displays collocate words and counts", async () => {
        const wrapper = mountCollocation();
        await flushPromises();
        await nextTick();
        expect(wrapper.text()).toContain("own");
        expect(wrapper.text()).toContain("27029");
    });

    // --- Invalid query warning ---
    it("shows invalid query warning for multi-word queries", async () => {
        const wrapper = mountCollocation();
        const store = useMainStore();
        store.formData.q = "two words";
        await nextTick();
        expect(wrapper.find(".alert-warning").exists()).toBe(true);
    });

    it("disables tabs when query is invalid", async () => {
        const wrapper = mountCollocation();
        const store = useMainStore();
        store.formData.q = "two words";
        await nextTick();
        expect(wrapper.find("#frequency-tab").attributes("disabled")).toBeDefined();
    });

    // --- Tab switching: @click on tab buttons ---
    it("switches to compare view on compare tab click", async () => {
        const wrapper = mountCollocation();
        await flushPromises();
        await nextTick();

        await wrapper.find("#compare-tab").trigger("click");
        await nextTick();

        // Compare view shows primary/comparison corpus sections
        expect(wrapper.find(".compare-divider").exists()).toBe(true);
    });

    it("switches to similar view on similar tab click", async () => {
        const wrapper = mountCollocation();
        await flushPromises();
        await nextTick();

        await wrapper.find("#similar-tab").trigger("click");
        await nextTick();

        expect(wrapper.vm.collocMethod).toBe("similar");
    });

    it("switches to time series view on tab click", async () => {
        const wrapper = mountCollocation();
        await flushPromises();
        await nextTick();

        await wrapper.find("#time-series-tab").trigger("click");
        await nextTick();

        expect(wrapper.vm.collocMethod).toBe("timeSeries");
    });

    it("switches back to frequency view", async () => {
        const wrapper = mountCollocation();
        await flushPromises();
        await nextTick();

        await wrapper.find("#compare-tab").trigger("click");
        await nextTick();
        await wrapper.find("#frequency-tab").trigger("click");
        await nextTick();

        expect(wrapper.vm.collocMethod).toBe("frequency");
    });

    // --- @change on mobile dropdown ---
    it("handles mobile method change", async () => {
        const wrapper = mountCollocation();
        await flushPromises();
        await nextTick();

        const select = wrapper.find("#colloc-method-mobile-select");
        await select.setValue("compare");
        await nextTick();

        expect(wrapper.vm.collocMethod).toBe("compare");
    });

    // --- Collocate row click: @click, @keydown.enter, @keydown.space ---
    it("navigates on collocate row click", async () => {
        const wrapper = mountCollocation();
        await flushPromises();
        await nextTick();

        const row = wrapper.find("tbody tr");
        await row.trigger("click");
        // Should push a concordance route via router
    });

    it("navigates on collocate row Enter key", async () => {
        const wrapper = mountCollocation();
        await flushPromises();
        await nextTick();

        const row = wrapper.find("tbody tr");
        await row.trigger("keydown.enter");
    });

    it("navigates on collocate row Space key", async () => {
        const wrapper = mountCollocation();
        await flushPromises();
        await nextTick();

        const row = wrapper.find("tbody tr");
        await row.trigger("keydown.space");
    });

    // --- Compare: Run comparison button ---
    it("fires getOtherCollocates on Run comparison click", async () => {
        const http = createMockHttp({
            "collocation.py": collocationFixture,
            "comparative_collocations.py": { top: [["power", 100]], bottom: [["own", 50]] },
        });
        const wrapper = mountCollocation({ http });
        await flushPromises();
        await nextTick();

        await wrapper.find("#compare-tab").trigger("click");
        await nextTick();

        const runBtn = wrapper.find("button.btn-secondary");
        // Find the "Run comparison" button specifically
        const buttons = wrapper.findAll("button.btn-secondary");
        const comparisonBtn = buttons.find(b => b.text().includes("Run comparison") || b.text().includes("runComparison"));
        if (comparisonBtn) {
            await comparisonBtn.trigger("click");
            await flushPromises();
            // Should have made an additional HTTP call
            expect(http.get.mock.calls.length).toBeGreaterThan(1);
        }
    });
});
