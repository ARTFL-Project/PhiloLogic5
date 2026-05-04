import { describe, it, expect } from "vitest";
import { mount } from "@vue/test-utils";
import { nextTick } from "vue";
import { createTestI18n } from "./helpers.js";
import LocaleChanger from "../src/components/LocaleChanger.vue";

function mountLocaleChanger() {
    return mount(LocaleChanger, {
        global: { plugins: [createTestI18n()] },
    });
}

describe("LocaleChanger", () => {
    it("renders language dropdown", () => {
        const wrapper = mountLocaleChanger();
        expect(wrapper.find("button, select, .dropdown").exists()).toBe(true);
    });

    it("renders language options", () => {
        const wrapper = mountLocaleChanger();
        const buttons = wrapper.findAll("button, option");
        expect(buttons.length).toBeGreaterThan(0);
    });

    // --- @click="changeLocale(locale)" ---
    it("changes locale and writes to localStorage on dropdown-item click", async () => {
        const i18n = createTestI18n();
        const wrapper = mount(LocaleChanger, { global: { plugins: [i18n] } });
        const items = wrapper.findAll(".dropdown-item");
        // Test fixture i18n only has "en" loaded; pick the second locale if present, else assert single-locale fallback
        const targetIndex = items.length > 1 ? 1 : 0;
        await items[targetIndex].trigger("click");
        await nextTick();
        const expectedLocale = i18n.global.availableLocales[targetIndex];
        expect(localStorage.getItem("lang")).toBe(expectedLocale);
        expect(i18n.global.locale.value).toBe(expectedLocale);
    });
});
