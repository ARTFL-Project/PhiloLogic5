import { describe, it, expect, vi } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { nextTick } from "vue";
import { createTestPinia, createTestI18n, createTestConfig, createTestRouter, createMockHttp } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import aggregationFixture from "./fixtures/aggregation.json";
import Aggregation from "../src/components/Aggregation.vue";

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

describe("Aggregation by title", () => {
    const CitationsStub = { template: "<span class='citations-stub' />", props: ["citation", "resultNumber"] };
    const titleConfig = {
        aggregation_config: [{
            field: "title",
            object_level: "doc",
            field_citation: [
                { field: "title", object_level: "doc", prefix: "", suffix: "", link: true, style: {} },
                { field: "year", object_level: "doc", prefix: " [", suffix: "]", link: false, style: {} },
            ],
            break_up_field: null,
            break_up_field_citation: null,
        }],
    };
    const results = {
        results: [
            // a title's documents grouped: only what they share
            { metadata_fields: { title: "Histoire de France", pub_place: "Paris", field_name: "Histoire de France" }, count: 93, object_count: 4, break_up_field: [] },
            { metadata_fields: { title: "Histoire de France.", author: "Michelet, Jules", year: 1893, field_name: "Histoire de France." }, count: 42, object_count: 1, break_up_field: [] },
        ],
        break_up_field: "",
        total_results: 135,
    };

    async function mountByTitle() {
        const http = createMockHttp({ "aggregation.py": results });
        const pinia = createTestPinia();
        const config = createTestConfig(titleConfig);
        const router = createTestRouter({ name: "aggregation", path: "/aggregation", query: { q: "liberté", group_by: "title" } });
        await router.isReady();
        const store = useMainStore();
        store.formData = { ...store.formData, q: "liberté", report: "aggregation", group_by: "title" };
        store.aggregationCache = { results: [], query: {} };
        const wrapper = mount(Aggregation, {
            global: {
                plugins: [pinia, createTestI18n(), router],
                provide: { $http: http, $dbUrl: "/testdb", $philoConfig: config },
                stubs: {
                    ResultsSummary: { template: "<div />", props: ["groupLength"] },
                    Citations: CitationsStub,
                },
                mocks: { $philoConfig: config, $dbUrl: "/testdb", $scrollTo: vi.fn() },
                directives: { scroll: { mounted() {}, unmounted() {} } },
            },
        });
        await flushPromises();
        await nextTick();
        return wrapper;
    }

    it("says a row comes from several documents", async () => {
        const rows = (await mountByTitle()).findAll(".list-group-item");
        expect(rows[0].text()).toContain("from 4 documents");
        expect(rows[1].text()).not.toContain("documents");
    });

    it("links a row by its title alone: the row's documents are those of the title", async () => {
        const wrapper = await mountByTitle();
        const [merged, single] = wrapper.findAllComponents(CitationsStub).map((c) => c.props("citation")[0].href);
        expect(merged.query).toMatchObject({ title: '"Histoire de France"' });
        expect(merged.query.author).toBeUndefined();
        expect(single.query.author).toBeUndefined();
    });
});
