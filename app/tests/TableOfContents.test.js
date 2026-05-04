import { describe, it, expect, vi } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { nextTick } from "vue";
import { createTestPinia, createTestI18n, createTestConfig, createTestRouter, createMockHttp } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import TableOfContents from "../src/components/TableOfContents.vue";

const tocFixture = {
    toc: [
        { philo_type: "div1", label: "Chapter 1", philo_id: "1 1 0 0 0 0 0", head: "Chapter 1", n: "1", byte: "0" },
        { philo_type: "div1", label: "Chapter 2", philo_id: "1 2 0 0 0 0 0", head: "Chapter 2", n: "2", byte: "1000" },
    ],
    citation: [{ label: "Brown, Wieland", href: "/navigate/1/table-of-contents", style: {}, prefix: "", suffix: "" }],
    tei_header: "<div><p>TEI Header</p></div>",
};

async function mountTableOfContents(overrides = {}) {
    const http = overrides.http || createMockHttp({ "table_of_contents.py": tocFixture });
    const pinia = createTestPinia();
    const i18n = createTestI18n();
    const config = createTestConfig(overrides.philoConfig);
    const router = createTestRouter({
        name: "tableOfContents",
        path: "/navigate/1/table-of-contents",
        query: {},
    });
    await router.isReady();

    const store = useMainStore();
    store.formData = { ...store.formData, report: "tableOfContents" };

    // Suppress recursive update warnings from the TOC component
    vi.spyOn(console, "warn").mockImplementation(() => {});

    return mount(TableOfContents, {
        global: {
            plugins: [pinia, i18n, router],
            provide: { $http: http, $dbUrl: "/testdb", $philoConfig: config },
            stubs: {
                Citations: { template: "<span class='citations-stub' />" },
                ProgressSpinner: { template: "<div class='spinner-stub' />" },
            },
            mocks: { $philoConfig: config, $dbUrl: "/testdb", $scrollTo: vi.fn() },
            directives: { scroll: { mounted() {}, unmounted() {} } },
        },
    });
}

describe("TableOfContents", () => {
    it("makes HTTP request for TOC data", async () => {
        const http = createMockHttp({ "table_of_contents.py": tocFixture });
        await mountTableOfContents({ http });
        await flushPromises();
        expect(http.get).toHaveBeenCalled();
    });

    it("renders TOC entries after fetch", async () => {
        const wrapper = await mountTableOfContents();
        await flushPromises();
        await nextTick();
        expect(wrapper.text()).toContain("Chapter 1");
    });

    it("renders citations", async () => {
        const wrapper = await mountTableOfContents();
        await flushPromises();
        await nextTick();
        expect(wrapper.find(".citations-stub").exists()).toBe(true);
    });

    // --- @click="toggleHeader()" ---
    it("toggles the TEI header section on button click", async () => {
        const http = createMockHttp({
            "table_of_contents.py": tocFixture,
            "get_header.py": { header: "<div><p>TEI Header</p></div>" },
        });
        const wrapper = await mountTableOfContents({ http, philoConfig: { header_in_toc: true } });
        await flushPromises();
        await nextTick();

        // Initially closed
        expect(wrapper.find("#tei-header").exists()).toBe(false);

        await wrapper.find("#show-header").trigger("click");
        await flushPromises();
        await nextTick();
        expect(wrapper.find("#tei-header").exists()).toBe(true);

        await wrapper.find("#show-header").trigger("click");
        await nextTick();
        expect(wrapper.find("#tei-header").exists()).toBe(false);
    });
});
