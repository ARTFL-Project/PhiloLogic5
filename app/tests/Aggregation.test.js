import { describe, it, expect, vi } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { nextTick } from "vue";
import { createTestPinia, createTestI18n, createTestConfig, createTestRouter, createMockHttp } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import {
    paramsFilter, paramsToRoute, paramsToUrlString, copyObject, saveToLocalStorage,
    mergeResults, sortResults, deepEqual, dictionaryLookup, dateRangeHandler,
    buildBiblioCriteria, extractSurfaceFromCollocate, debug, isOnlyFacetChange, buildTocTree,
} from "../src/mixins.js";
import aggregationFixture from "./fixtures/aggregation.json";
import Aggregation from "../src/components/Aggregation.vue";

const mixinMethods = {
    paramsFilter, paramsToRoute, paramsToUrlString, copyObject, saveToLocalStorage,
    mergeResults, sortResults, deepEqual, dictionaryLookup, dateRangeHandler,
    buildBiblioCriteria, extractSurfaceFromCollocate, debug, isOnlyFacetChange, buildTocTree,
};

async function mountAggregation(overrides = {}) {
    const http = overrides.http || createMockHttp({ "aggregation.py": aggregationFixture });
    const pinia = createTestPinia();
    const i18n = createTestI18n();
    const config = createTestConfig({
        aggregation_config: [{
            field: "author",
            object_level: "doc",
            field_citation: [{ field: "author", object_level: "doc", prefix: "", suffix: "", link: true, style: { "font-variant": "small-caps" } }],
            break_up_field: "title",
            break_up_field_citation: [{ field: "title", object_level: "doc", prefix: "", suffix: "", link: true, style: {} }],
        }],
        ...overrides.philoConfig,
    });
    const router = createTestRouter({
        name: "aggregation", path: "/aggregation",
        query: { q: "liberty", group_by: "author", author: "Brown" },
    });
    await router.isReady();

    const store = useMainStore();
    store.formData = { ...store.formData, q: "liberty", report: "aggregation", group_by: "author", author: "Brown" };
    store.aggregationCache = { results: [], query: {} };

    return mount(Aggregation, {
        global: {
            plugins: [pinia, i18n, router],
            provide: { $http: http, $dbUrl: "/testdb", $philoConfig: config },
            mixins: [{ methods: mixinMethods }],
            stubs: {
                ResultsSummary: { template: "<div class='results-summary-stub' />", props: ["groupLength"] },
                Citations: { template: "<span class='citations-stub' />", props: ["citation", "resultNumber"] },
            },
            mocks: { $philoConfig: config, $dbUrl: "/testdb", $scrollTo: vi.fn() },
            directives: { scroll: { mounted() {}, unmounted() {} } },
        },
    });
}

describe("Aggregation", () => {
    // --- Rendering ---
    it("makes HTTP request for aggregation results", async () => {
        const http = createMockHttp({ "aggregation.py": aggregationFixture });
        await mountAggregation({ http });
        await flushPromises();
        expect(http.get).toHaveBeenCalled();
    });

    it("renders result items after fetch", async () => {
        const wrapper = await mountAggregation();
        await flushPromises();
        await nextTick();
        expect(wrapper.findAll(".list-group-item").length).toBeGreaterThan(0);
    });

    it("displays count badges", async () => {
        const wrapper = await mountAggregation();
        await flushPromises();
        await nextTick();
        expect(wrapper.findAll(".badge").length).toBeGreaterThan(0);
    });

    // --- @click="toggleBreakUp(resultIndex)" ---
    it("expands breakdown on button click", async () => {
        const wrapper = await mountAggregation();
        await flushPromises();
        await nextTick();

        const expandBtn = wrapper.find("[aria-expanded]");
        if (expandBtn.exists()) {
            expect(expandBtn.attributes("aria-expanded")).toBe("false");
            await expandBtn.trigger("click");
            await nextTick();
            expect(expandBtn.attributes("aria-expanded")).toBe("true");
        }
    });

    it("collapses breakdown on second click", async () => {
        const wrapper = await mountAggregation();
        await flushPromises();
        await nextTick();

        const expandBtn = wrapper.find("[aria-expanded]");
        if (expandBtn.exists()) {
            await expandBtn.trigger("click");
            await nextTick();
            await expandBtn.trigger("click");
            await nextTick();
            expect(expandBtn.attributes("aria-expanded")).toBe("false");
        }
    });

    it("shows breakdown items when expanded", async () => {
        const wrapper = await mountAggregation();
        await flushPromises();
        await nextTick();

        const expandBtn = wrapper.find("[aria-expanded]");
        if (expandBtn.exists()) {
            await expandBtn.trigger("click");
            await nextTick();
            expect(wrapper.find(".breakdown-container").exists()).toBe(true);
        }
    });
});
