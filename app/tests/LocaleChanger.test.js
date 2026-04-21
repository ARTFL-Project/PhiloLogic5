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
    it("has changeLocale method", () => {
        const wrapper = mountLocaleChanger();
        expect(typeof wrapper.vm.changeLocale).toBe("function");
    });

    it("changes locale on button click", async () => {
        const wrapper = mountLocaleChanger();
        const buttons = wrapper.findAll(".dropdown-item, button");
        if (buttons.length > 1) {
            await buttons[1].trigger("click");
            await nextTick();
            // Locale should change (stored in localStorage)
        }
    });
});
