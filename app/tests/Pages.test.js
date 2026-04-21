import { describe, it, expect } from "vitest";
import { mount } from "@vue/test-utils";
import { createI18n } from "vue-i18n";
import { setActivePinia, createPinia } from "pinia";
import { useMainStore } from "../src/stores/main.js";
import Pages from "../src/components/Pages.vue";

const i18n = createI18n({
    legacy: false,
    locale: "en",
    messages: { en: { pages: { searchResultsPages: "Pages" } } },
});

function mountPages(storeOverrides = {}) {
    const pinia = createPinia();
    setActivePinia(pinia);
    const store = useMainStore();
    Object.assign(store, storeOverrides);

    return mount(Pages, {
        global: {
            plugins: [pinia, i18n],
            stubs: { "router-link": { template: '<a class="page-link"><slot /></a>', props: ["to"] } },
            mocks: {
                $route: {
                    query: { results_per_page: "25", start: "1", end: "25" },
                    name: "concordance",
                },
                $router: { push: () => {} },
            },
        },
    });
}

describe("Pages", () => {
    it("renders nothing when no results", () => {
        const wrapper = mountPages({ resultsLength: 0 });
        expect(wrapper.findAll(".page-link").length).toBe(0);
    });

    it("renders nothing when results fit in one page", () => {
        const wrapper = mountPages({
            resultsLength: 10,
            totalResultsDone: true,
            formData: { results_per_page: 25, start: "1", end: "25", report: "concordance" },
        });
        // 10 results with 25 per page = 1 page, no pagination needed
        expect(wrapper.findAll(".page-link").length).toBe(0);
    });
});
