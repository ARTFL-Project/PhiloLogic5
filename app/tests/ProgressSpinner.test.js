import { describe, it, expect, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { nextTick } from "vue";
import { createTestI18n } from "./helpers.js";
import ProgressSpinner from "../src/components/ProgressSpinner.vue";

const i18n = createTestI18n();

function mountSpinner(props = {}) {
    return mount(ProgressSpinner, {
        props,
        global: { plugins: [i18n] },
    });
}

describe("ProgressSpinner", () => {
    it("renders spinner element", () => {
        const wrapper = mountSpinner();
        expect(wrapper.find(".spinner-border").exists()).toBe(true);
    });

    it("shows progress percentage when progress > 0", () => {
        const wrapper = mountSpinner({ progress: 75 });
        expect(wrapper.find(".spinner-text").text()).toContain("75%");
    });

    it("does not show percentage when progress is 0", () => {
        const wrapper = mountSpinner({ progress: 0 });
        expect(wrapper.find(".spinner-text").exists()).toBe(false);
    });

    it("shows text prop when no progress", () => {
        const wrapper = mountSpinner({ text: "Loading..." });
        expect(wrapper.find(".spinner-text").text()).toBe("Loading...");
    });

    it("prefers progress over text when both provided", () => {
        const wrapper = mountSpinner({ progress: 50, text: "Loading" });
        expect(wrapper.find(".spinner-text").text()).toContain("50%");
    });

    it("applies small class when sm prop is true", () => {
        const wrapper = mountSpinner({ sm: true });
        expect(wrapper.find(".spinner-border-sm").exists()).toBe(true);
    });

    it("applies large class when lg prop is true", () => {
        const wrapper = mountSpinner({ lg: true });
        expect(wrapper.find(".spinner-large").exists()).toBe(true);
    });

    it("applies extra-large class when xl prop is true", () => {
        const wrapper = mountSpinner({ xl: true });
        expect(wrapper.find(".spinner-xl").exists()).toBe(true);
    });

    it("announces status to screen readers after delay", async () => {
        vi.useFakeTimers();
        const wrapper = mountSpinner();
        expect(wrapper.find("[role='status']").text()).toBe("");
        vi.advanceTimersByTime(150);
        await nextTick();
        expect(wrapper.find("[role='status']").text()).toContain("Loading");
        vi.useRealTimers();
    });

    it("uses custom message prop for announcement", async () => {
        vi.useFakeTimers();
        const wrapper = mountSpinner({ message: "Fetching data" });
        vi.advanceTimersByTime(150);
        await nextTick();
        expect(wrapper.find("[role='status']").text()).toContain("Fetching data");
        vi.useRealTimers();
    });

    it("clears timeout on unmount", () => {
        vi.useFakeTimers();
        const clearSpy = vi.spyOn(global, "clearTimeout");
        const wrapper = mountSpinner();
        wrapper.unmount();
        expect(clearSpy).toHaveBeenCalled();
        clearSpy.mockRestore();
        vi.useRealTimers();
    });
});
