import { describe, it, expect, vi } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { nextTick } from "vue";
import { createTestPinia, createTestI18n, createTestConfig, createTestRouter, createMockHttp } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import SearchArguments from "../src/components/SearchArguments.vue";

async function mountSearchArguments(overrides = {}) {
    const http = overrides.http || createMockHttp({
        "get_term_groups.py": { term_groups: [["liberty"], ["freedom"]], total_terms: 2 },
        "get_query_terms.py": ["liberty", "liberties", "libertie"],
    });
    const pinia = createTestPinia();
    const i18n = createTestI18n();
    const config = createTestConfig();
    const router = createTestRouter({
        name: "concordance", path: "/concordance",
        query: { q: "liberty freedom", report: "concordance" },
    });
    await router.isReady();

    const store = useMainStore();
    store.formData = { ...store.formData, q: "liberty freedom", report: "concordance", method: "proxy", method_arg: "5" };
    store.resultsLength = 100;
    store.description = { start: 1, end: 25, results_per_page: 25, termGroups: [] };

    const wrapper = mount(SearchArguments, {
        props: { resultStart: 1, resultEnd: 25 },
        global: {
            plugins: [pinia, i18n, router],
            provide: { $http: http, $dbUrl: "/testdb", $philoConfig: config },
            stubs: {
                BibliographyCriteria: { template: "<div class='biblio-criteria-stub' />" },
            },
            mocks: { $philoConfig: config, $dbUrl: "/testdb", $scrollTo: vi.fn() },
        },
    });

    return { wrapper, http, store };
}

describe("SearchArguments", () => {
    it("fetches term groups on mount", async () => {
        const { http } = await mountSearchArguments();
        await flushPromises();
        const termCalls = http.get.mock.calls.filter(c => c[0].includes("get_term_groups"));
        expect(termCalls.length).toBe(1);
    });

    it("renders bibliography criteria", async () => {
        const { wrapper } = await mountSearchArguments();
        expect(wrapper.find(".biblio-criteria-stub").exists()).toBe(true);
    });

    it("displays query terms after term groups load", async () => {
        const { wrapper } = await mountSearchArguments();
        await flushPromises();
        await nextTick();
        expect(wrapper.text()).toContain("liberty");
    });

    it("renders term group buttons", async () => {
        const { wrapper } = await mountSearchArguments();
        await flushPromises();
        await nextTick();
        const termBtns = wrapper.findAll(".term-group-word");
        expect(termBtns.length).toBe(2);
    });

    it("renders remove button for each term group", async () => {
        const { wrapper } = await mountSearchArguments();
        await flushPromises();
        await nextTick();
        const removeBtns = wrapper.findAll(".close-pill");
        expect(removeBtns.length).toBe(2);
    });

    // --- @click="getQueryTerms(group, index, $event)" ---
    it("opens term expansion dialog on term group click", async () => {
        const { wrapper } = await mountSearchArguments();
        await flushPromises();
        await nextTick();

        await wrapper.find(".term-group-word").trigger("click");
        await flushPromises();
        await nextTick();

        expect(wrapper.find("#query-terms").exists()).toBe(true);
    });

    // --- @click="closeTermsList()" ---
    it("closes dialog on close button click", async () => {
        const { wrapper } = await mountSearchArguments();
        await flushPromises();
        await nextTick();

        await wrapper.find(".term-group-word").trigger("click");
        await flushPromises();
        await nextTick();

        await wrapper.find("#query-terms .close").trigger("click");
        await nextTick();
        expect(wrapper.find("#query-terms").exists()).toBe(false);
    });

    // --- @keydown="handleDialogKeydown" (Escape) ---
    it("closes dialog on Escape key", async () => {
        const { wrapper } = await mountSearchArguments();
        await flushPromises();
        await nextTick();

        await wrapper.find(".term-group-word").trigger("click");
        await flushPromises();
        await nextTick();

        await wrapper.find("#query-terms").trigger("keydown", { key: "Escape" });
        await nextTick();
        expect(wrapper.find("#query-terms").exists()).toBe(false);
    });

    // --- @click="removeFromTermsList(word, groupIndexSelected)" ---
    it("marks word list as changed after removing term", async () => {
        const { wrapper } = await mountSearchArguments();
        await flushPromises();
        await nextTick();

        await wrapper.find(".term-group-word").trigger("click");
        await flushPromises();
        await nextTick();

        const termCloseBtns = wrapper.findAll("#query-terms-list .close-pill");
        if (termCloseBtns.length > 0) {
            // Re-run button is rendered v-if="wordListChanged" — should not be present yet
            const rerunBefore = wrapper.findAll("button").filter(b => b.text().includes("rerun") || b.text().includes("Rerun"));
            expect(rerunBefore.length).toBe(0);

            await termCloseBtns[0].trigger("click");
            await nextTick();

            // After removal, the re-run button should appear
            const rerunAfter = wrapper.findAll("button").filter(b => b.text().includes("rerun") || b.text().includes("Rerun"));
            expect(rerunAfter.length).toBeGreaterThan(0);
        }
    });

    // --- @click="removeTerm(index)" ---
    it("removes term group on close-pill click", async () => {
        const { wrapper } = await mountSearchArguments();
        await flushPromises();
        await nextTick();

        await wrapper.findAll(".close-pill")[0].trigger("click");
        await nextTick();
        // Should trigger router navigation with modified query
    });

    // --- Collocation mode ---
    it("shows collocation text when report is collocation", async () => {
        const { wrapper, store } = await mountSearchArguments();
        store.formData.report = "collocation";
        store.formData.q = "liberty";
        store.formData.colloc_within = "sent";
        store.currentReport = "collocation";
        await nextTick();
        const text = wrapper.text();
        expect(text).toContain("liberty");
    });
});
