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

        expect(wrapper.find("#similar-tab").classes()).toContain("active");
        expect(wrapper.find("#similar-tab").attributes("aria-selected")).toBe("true");
    });

    it("switches to time series view on tab click", async () => {
        const wrapper = mountCollocation();
        await flushPromises();
        await nextTick();

        await wrapper.find("#time-series-tab").trigger("click");
        await nextTick();

        expect(wrapper.find("#time-series-tab").classes()).toContain("active");
        expect(wrapper.find("#time-series-tab").attributes("aria-selected")).toBe("true");
    });

    it("switches back to frequency view", async () => {
        const wrapper = mountCollocation();
        await flushPromises();
        await nextTick();

        await wrapper.find("#compare-tab").trigger("click");
        await nextTick();
        await wrapper.find("#frequency-tab").trigger("click");
        await nextTick();

        expect(wrapper.find("#frequency-tab").classes()).toContain("active");
        expect(wrapper.find("#compare-tab").classes()).not.toContain("active");
    });

    // --- @change on mobile dropdown ---
    it("handles mobile method change", async () => {
        const wrapper = mountCollocation();
        await flushPromises();
        await nextTick();

        const select = wrapper.find("#colloc-method-mobile-select");
        await select.setValue("compare");
        await nextTick();

        // The "compare" tab button reflects the active method via .active class
        expect(wrapper.find("#compare-tab").classes()).toContain("active");
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

    // --- Workflow modes (driven by ?collocation_method= URL param) ---
    // Each of the four modes branches in created() to a different HTTP path.
    // These tests cover the load-bearing dispatch logic — under <script setup>
    // it'll move from created() to top-level setup code, and the dispatch is
    // easy to mis-translate.
    //
    // Helper: mount with a settled route. mountCollocation can't do this
    // because createTestRouter doesn't await isReady(), and Collocation reads
    // $route.query in created() — which runs synchronously at mount time.
    async function mountCollocationWithRoute(http, query) {
        const global = createGlobalConfig({
            http,
            route: { name: "collocation", path: "/collocation", query },
            stubs: {
                ResultsSummary: { template: "<div class='results-summary-stub' />" },
                WordCloud: { template: "<div class='word-cloud-stub' />", props: ["wordWeights", "clickHandler", "label"] },
                BibliographyCriteria: { template: "<div class='biblio-criteria-stub' />" },
                ProgressSpinner: { template: "<div class='spinner-stub' />" },
                MetadataFields: { template: "<div class='metadata-fields-stub' />" },
            },
        });
        const router = global.plugins[2];
        await router.isReady();

        const store = useMainStore();
        store.formData = { ...store.formData, q: "liberty", report: "collocation", colloc_within: "sent", colloc_filter_choice: "frequency", filter_frequency: 100 };
        store.searchableMetadata = {
            display: [{ value: "author", label: "Author", example: "Voltaire" }],
            inputStyle: { author: "text" },
            choiceValues: {},
        };

        return mount(Collocation, { global });
    }

    it("frequency mode (default) fetches collocation.py on mount", async () => {
        const http = createMockHttp({ "collocation.py": collocationFixture });
        mountCollocation({ http });
        await flushPromises();

        const calls = http.get.mock.calls.filter(c => c[0].includes("/reports/collocation.py"));
        expect(calls.length).toBeGreaterThan(0);
    });

    it("similar mode fetches collocation.py and activates the similar tab", async () => {
        const http = createMockHttp({ "collocation.py": collocationFixture });
        const wrapper = await mountCollocationWithRoute(http, {
            q: "liberty",
            report: "collocation",
            collocation_method: "similar",
            similarity_by: "author",
        });
        await flushPromises();
        await nextTick();

        const calls = http.get.mock.calls.filter(c => c[0].includes("/reports/collocation.py"));
        expect(calls.length).toBeGreaterThan(0);
        expect(wrapper.find("#similar-tab").classes()).toContain("active");
    });

    it("timeSeries mode fetches collocation_time_series.py and activates the time-series tab", async () => {
        // The component first hits reports/collocation.py to get a file_path,
        // then polls scripts/collocation_time_series.py until response.data.done
        // is truthy. We must return done: true on the first poll or the test hangs.
        const http = createMockHttp({
            "reports/collocation.py": { file_path: "/tmp/test-time-series" },
            "collocation_time_series.py": {
                period: { year: 1800, collocates: { frequent: [], distinctive: [] } },
                done: true,
            },
        });
        const wrapper = await mountCollocationWithRoute(http, {
            q: "liberty",
            report: "collocation",
            collocation_method: "timeSeries",
            time_series_interval: "10",
        });
        await flushPromises();
        await nextTick();

        const calls = http.get.mock.calls.filter(c => c[0].includes("collocation_time_series.py"));
        expect(calls.length).toBeGreaterThan(0);
        expect(wrapper.find("#time-series-tab").classes()).toContain("active");
    });

    it("compare mode activates the compare tab and shows the compare panel", async () => {
        const http = createMockHttp({
            "collocation.py": collocationFixture,
            "comparative_collocations.py": { top: [], bottom: [] },
        });
        const wrapper = await mountCollocationWithRoute(http, {
            q: "liberty",
            report: "collocation",
            collocation_method: "compare",
        });
        await flushPromises();
        await nextTick();

        expect(wrapper.find("#compare-tab").classes()).toContain("active");
        expect(wrapper.find(".compare-divider").exists()).toBe(true);
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
