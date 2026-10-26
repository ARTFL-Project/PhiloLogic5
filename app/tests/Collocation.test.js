import { describe, it, expect } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { nextTick } from "vue";
import { createGlobalConfig, createMockHttp } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import collocationFixture from "./fixtures/collocation.json";
import Collocation from "../src/components/Collocation.vue";
import Compare from "../src/components/collocation/Compare.vue";

function mountCollocation(overrides = {}) {
    const http = overrides.http || createMockHttp({
        "collocation.py": collocationFixture,
        "comparative_collocations.py": { top: [["power", 100], ["free", 80]], bottom: [["own", 50]] },
        // Default time-series mock terminates the recursive poll on first call.
        // Tests that need multi-period results should pass their own http override.
        "collocation_time_series.py": { period: { year: 1800, collocates: { frequent: [], distinctive: [] } }, done: true },
    });
    const global = createGlobalConfig({
        http,
        route: { name: "collocation", path: "/collocation", query: { q: "liberty", report: "collocation" } },
        stubs: {
            ResultsSummary: { template: "<div class='results-summary-stub' />" },
            WordCloud: { template: "<div class='word-cloud-stub' />", props: ["wordWeights", "clickHandler"] },
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
        expect(wrapper.find("#word-senses-tab").exists()).toBe(true);
        expect(wrapper.find("#sense-overview-tab").exists()).toBe(true);
        expect(wrapper.find("#time-series-tab").exists()).toBe(false); // usage patterns over time: replaced by the senses
        expect(wrapper.find("#word-map-tab").exists()).toBe(false);
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

    // --- Multi-word queries: phrases and co-occurrences, which the server counts collocates of ---
    it("shows no warning for multi-word queries", async () => {
        const wrapper = mountCollocation();
        const store = useMainStore();
        store.formData.q = "two words";
        await nextTick();
        expect(wrapper.find(".alert-warning").exists()).toBe(false);
    });

    it("keeps tabs enabled for multi-word queries", async () => {
        const wrapper = mountCollocation();
        const store = useMainStore();
        store.formData.q = "two words";
        await nextTick();
        expect(wrapper.find("#frequency-tab").attributes("disabled")).toBeUndefined();
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

    it("takes a tab clicked on a short link's page for the same search", async () => {
        const http = createMockHttp({ "collocation.py": collocationFixture });
        const wrapper = mountCollocation({ http });
        const store = useMainStore();
        // the defaults App fills the form with, which the URL a tab click makes has
        store.defaultFields = { ...store.formData, q: "", report: "home", colloc_filter_choice: "" };
        store.formData = { ...store.formData, ...store.defaultFields, q: "liberty", report: "collocation" };
        await wrapper.vm.$router.push({ path: "/collocation", query: { q: "liberty" } });
        await flushPromises();
        const searches = () => http.get.mock.calls.filter(([url, config]) => url.includes("/collocation.py") && !config.params.map_field).length;
        const before = searches();

        await wrapper.find("#similar-tab").trigger("click");
        await flushPromises();
        expect(wrapper.vm.$route.query.colloc_within).toBe("sent"); // the URL now has the defaults
        expect(searches()).toBe(before);
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

    it("switches to the senses at a glance on tab click", async () => {
        const wrapper = mountCollocation();
        await flushPromises();
        await nextTick();

        await wrapper.find("#sense-overview-tab").trigger("click");
        await nextTick();

        expect(wrapper.find("#sense-overview-tab").classes()).toContain("active");
        expect(wrapper.find("#sense-overview-tab").attributes("aria-selected")).toBe("true");
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
    // asApp: the store's form follows the URL, as App sets it on each route
    async function mountCollocationWithRoute(http, query, { asApp = false } = {}) {
        const global = createGlobalConfig({
            http,
            route: { name: "collocation", path: "/collocation", query },
            stubs: {
                ResultsSummary: { template: "<div class='results-summary-stub' />" },
                WordCloud: { template: "<div class='word-cloud-stub' />", props: ["wordWeights", "clickHandler"] },
                BibliographyCriteria: { template: "<div class='biblio-criteria-stub' />" },
                ProgressSpinner: { template: "<div class='spinner-stub' />" },
                MetadataFields: { template: "<div class='metadata-fields-stub' />" },
            },
        });
        const router = global.plugins[2];
        await router.isReady();

        const store = useMainStore();
        store.formData = { ...store.formData, q: "liberty", report: "collocation", colloc_within: "sent", colloc_filter_choice: "frequency", filter_frequency: 100 };
        if (asApp) {
            store.formData = { ...store.formData, ...query };
            router.afterEach((to) => { store.formData = { ...store.formData, ...to.query }; });
        }
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

    it("links to the usage patterns over time open the senses, which replaced them", async () => {
        const http = createMockHttp({
            "get_word_senses.py": { n_hits: 0, n_sentences: 0, n_works: 0, nodes: [], time: null, filter_list: [] },
        });
        const wrapper = await mountCollocationWithRoute(http, {
            q: "liberty",
            report: "collocation",
            collocation_method: "timeSeries",
        });
        await flushPromises();
        await nextTick();

        expect(http.get.mock.calls.some((c) => c[0].includes("get_word_senses.py"))).toBe(true);
        expect(wrapper.find("#word-senses-tab").classes()).toContain("active");
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

    function compareHttp() {
        return createMockHttp({
            "collocation.py": collocationFixture,
            "comparative_collocations.py": { top: [["power", 1.5]], bottom: [["own", 2.5]] },
        });
    }
    const searchesOf = (http) => http.get.mock.calls
        .filter((c) => c[0].includes("/reports/collocation.py")).map((c) => c[1].params);
    const comparisonsOf = (http) => http.get.mock.calls.filter((c) => c[0].includes("comparative_collocations.py"));
    const runButton = (wrapper) => wrapper.findAll("button").find((b) => b.text().includes("Run comparison"));

    it("runs the comparison the form gives from the URL, searching the primary subcorpus anew", async () => {
        const http = compareHttp();
        const wrapper = await mountCollocationWithRoute(http, {
            q: "liberty", report: "collocation", collocation_method: "compare", compare_title: "Candide",
        }, { asApp: true });
        await flushPromises();
        const compare = wrapper.findComponent(Compare);
        compare.props("primaryMetadataValues").author = '"Hugo"';
        compare.props("comparedMetadataValues").author = '"Zola"';
        compare.props("comparedMetadataValues").title = "";  // erased: no longer compared
        http.get.mockClear();
        await runButton(wrapper).trigger("click");
        await flushPromises();

        const query = wrapper.vm.$router.currentRoute.value.query;
        expect(query).toMatchObject({ author: '"Hugo"', compare_author: '"Zola"', collocation_method: "compare" });
        expect(query).not.toHaveProperty("compare_title");
        // the primary subcorpus, then the one compared with, which keeps none of the comparison's parameters
        expect(searchesOf(http).map((p) => p.author)).toEqual(['"Hugo"', '"Zola"']);
        expect(searchesOf(http)[1]).not.toHaveProperty("compare_title");
        expect(comparisonsOf(http).length).toBe(1);
    });

    it("compares with other metadata without searching the primary subcorpus again", async () => {
        const http = compareHttp();
        const wrapper = await mountCollocationWithRoute(http, {
            q: "liberty", report: "collocation", collocation_method: "compare", author: '"Hugo"', compare_author: '"Sand"',
        }, { asApp: true });
        await flushPromises();
        await runButton(wrapper).trigger("click");  // the URL as the form makes it
        await flushPromises();
        wrapper.findComponent(Compare).props("comparedMetadataValues").author = '"Zola"';
        http.get.mockClear();
        await runButton(wrapper).trigger("click");
        await flushPromises();

        expect(wrapper.vm.$router.currentRoute.value.query).toMatchObject({ author: '"Hugo"', compare_author: '"Zola"' });
        expect(searchesOf(http).map((p) => p.author)).toEqual(['"Zola"']);
        expect(comparisonsOf(http).length).toBe(1);
    });

    it("restores both subcorpora from the URL, as on a reload, and compares them", async () => {
        const http = compareHttp();
        const wrapper = await mountCollocationWithRoute(http, {
            q: "liberty", report: "collocation", collocation_method: "compare", author: '"Hugo"', compare_author: '"Zola"',
        }, { asApp: true });
        await flushPromises();
        const compare = wrapper.findComponent(Compare);

        expect(compare.props("primaryMetadataValues").author).toBe('"Hugo"');
        expect(compare.props("comparedMetadataValues").author).toBe('"Zola"');
        expect(searchesOf(http).map((p) => p.author)).toEqual(['"Hugo"', '"Zola"']);
        expect(comparisonsOf(http).length).toBe(1);
    });

    it("says there is nothing to compare when both subcorpora are the whole corpus", async () => {
        const http = compareHttp();
        const wrapper = await mountCollocationWithRoute(http, {
            q: "liberty", report: "collocation", collocation_method: "compare",
        }, { asApp: true });
        await flushPromises();

        expect(comparisonsOf(http).length).toBe(0);
        expect(wrapper.text()).toContain("Nothing to compare");
    });

    // --- The tabs shown are kept, with their results, until the search changes ---
    const sensesResult = { n_hits: 0, n_sentences: 0, n_works: 0, nodes: [], time: null, filter_list: ["the"] };
    const sensesOf = (http) => http.get.mock.calls.filter((c) => c[0].includes("get_word_senses.py"));
    const keptTabsHttp = () => createMockHttp({ "collocation.py": collocationFixture, "get_word_senses.py": sensesResult });
    const showTab = async (wrapper, id) => {
        await wrapper.find(id).trigger("click");
        await flushPromises();
    };
    const newSearch = async (wrapper, q) => {
        const router = wrapper.vm.$router;
        await router.push({ path: "/collocation", query: { ...router.currentRoute.value.query, q } });
        await flushPromises();
    };
    // The page as a search opens it: its URL with the parameters of the tabs' own (a shorter one is another search)
    async function searched(http) {
        const wrapper = await mountCollocationWithRoute(http, { q: "liberty", report: "collocation" }, { asApp: true });
        await flushPromises();
        await showTab(wrapper, "#frequency-tab");
        return wrapper;
    }

    it("fetches nothing again when coming back to a tab", async () => {
        const http = keptTabsHttp();
        const wrapper = await searched(http);
        await showTab(wrapper, "#word-senses-tab");
        const fetched = http.get.mock.calls.length;

        await showTab(wrapper, "#frequency-tab");  // its outliers, and the senses, as they were
        await showTab(wrapper, "#word-senses-tab");

        expect(http.get.mock.calls.length).toBe(fetched);
        expect(sensesOf(http).length).toBe(1);
        expect(wrapper.find("#word-senses-tab").classes()).toContain("active");
    });

    it("drops the tabs kept when the search changes, keeping the one shown", async () => {
        const http = keptTabsHttp();
        const wrapper = await searched(http);
        await showTab(wrapper, "#word-senses-tab");
        await newSearch(wrapper, "freedom");

        // the tab shown, its senses fetched anew, then the frequency's, of the new search
        expect(sensesOf(http).map((c) => c[1].params.q)).toEqual(["liberty", "freedom"]);
        http.get.mockClear();
        await showTab(wrapper, "#frequency-tab");
        expect(searchesOf(http).length).toBeGreaterThan(0);
        expect(searchesOf(http).every((p) => p.q === "freedom")).toBe(true);
    });

    it("fetches a tab loaded on demand once, shown again after the search changed", async () => {
        await import("../src/components/collocation/SenseOverview.vue");  // loaded, as once its tab was shown
        const http = keptTabsHttp();
        const wrapper = await searched(http);
        await showTab(wrapper, "#sense-overview-tab");
        await showTab(wrapper, "#frequency-tab");
        await newSearch(wrapper, "freedom");
        await showTab(wrapper, "#sense-overview-tab");  // it fetches as it mounts, and the page asks too

        expect(sensesOf(http).map((c) => c[1].params.q)).toEqual(["liberty", "freedom"]);
    });

    it("fetches the senses once when a link's shorter URL opens a tab", async () => {
        await import("../src/components/collocation/SenseOverview.vue");  // loaded, as once its tab was shown
        const http = keptTabsHttp();
        const wrapper = await mountCollocationWithRoute(http, { q: "liberty", report: "collocation" }, { asApp: true });
        await flushPromises();
        await showTab(wrapper, "#sense-overview-tab");  // the tab's URL, longer, is another search
        await showTab(wrapper, "#frequency-tab");
        await showTab(wrapper, "#sense-overview-tab");

        expect(sensesOf(http).length).toBe(1);
    });

    it("fetches Frequency's outliers once, shown after a search made on a senses tab", async () => {
        const http = keptTabsHttp();
        const wrapper = await searched(http);
        await showTab(wrapper, "#word-senses-tab");
        await newSearch(wrapper, "freedom");
        http.get.mockClear();
        await showTab(wrapper, "#frequency-tab");

        const outliers = http.get.mock.calls.filter((c) => c[0].includes("get_outlier_groups.py"));
        expect(outliers.length).toBe(2); // one for each field compared (author, title)
        expect(searchesOf(http).every((p) => p.q === "freedom")).toBe(true);
    });

    it("gives the summary a kept tab's words left out when it is shown, not when its result comes later", async () => {
        const http = keptTabsHttp();
        const respond = http.get.getMockImplementation();
        let release;
        http.get.mockImplementation((url, options) => url.includes("get_word_senses.py")
            ? new Promise((resolve) => { release = () => resolve({ data: sensesResult }); })
            : respond(url, options));
        const wrapper = await searched(http);
        const filterList = () => wrapper.find(".results-summary-stub").attributes("filter-list");
        const frequencyList = filterList();

        await showTab(wrapper, "#word-senses-tab");
        await showTab(wrapper, "#frequency-tab");
        release();  // the senses come while the frequency is shown
        await flushPromises();
        expect(filterList()).toBe(frequencyList);

        await showTab(wrapper, "#word-senses-tab");
        expect(filterList()).toBe("the");
        expect(sensesOf(http).length).toBe(1);
    });
});
