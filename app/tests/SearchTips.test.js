import { describe, it, expect } from "vitest";
import { mount } from "@vue/test-utils";
import { createI18n } from "vue-i18n";
import SearchTips from "../src/components/SearchTips.vue";
import en from "../src/locales/en.json";

const i18n = createI18n({
    legacy: false,
    locale: "en",
    messages: { en },
});

describe("SearchTips", () => {
    it("renders search tips content", () => {
        const wrapper = mount(SearchTips, { global: { plugins: [i18n] } });
        expect(wrapper.text()).toContain("Basic Operators");
    });

    it("shows word search section", () => {
        const wrapper = mount(SearchTips, { global: { plugins: [i18n] } });
        expect(wrapper.text()).toContain("Word Searches");
    });

    it("shows metadata search section", () => {
        const wrapper = mount(SearchTips, { global: { plugins: [i18n] } });
        expect(wrapper.text()).toContain("Metadata Searches");
    });

    it("shows regex syntax section", () => {
        const wrapper = mount(SearchTips, { global: { plugins: [i18n] } });
        expect(wrapper.text()).toContain("Regular Expression Syntax");
    });
});
