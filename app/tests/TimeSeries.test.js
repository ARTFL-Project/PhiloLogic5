import { describe, it, expect, vi } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { nextTick } from "vue";
import { createGlobalConfig, createMockHttp } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import timeSeriesFixture from "./fixtures/time_series.json";
import TimeSeries from "../src/components/TimeSeries.vue";

function mountTimeSeries(overrides = {}) {
    const http = overrides.http || createMockHttp({ "time_series.py": timeSeriesFixture });
    const global = createGlobalConfig({
        http,
        route: { name: "time_series", path: "/time_series", query: { q: "liberty", report: "time_series", start_date: "1500", end_date: "1800", year_interval: "50" } },
        stubs: {
            ResultsSummary: { template: "<div class='results-summary-stub' />" },
            Bar: { template: "<canvas class='chart-stub' />", props: ["data", "options"] },
        },
        ...overrides,
    });

    const store = useMainStore();
    store.formData = {
        ...store.formData,
        q: "liberty", report: "time_series",
        start_date: "1500", end_date: "1800", year_interval: "50",
    };

    return mount(TimeSeries, { global });
}

describe("TimeSeries", () => {
    // --- Rendering ---
    it("renders time series container", async () => {
        const wrapper = mountTimeSeries();
        await flushPromises();
        expect(wrapper.find("#time-series-container").exists()).toBe(true);
    });

    it("renders frequency toggle buttons", async () => {
        const wrapper = mountTimeSeries();
        await flushPromises();
        const buttons = wrapper.findAll(".btn-group .btn");
        expect(buttons.length).toBe(2);
    });

    it("makes HTTP request for time series data", async () => {
        const http = createMockHttp({ "time_series.py": timeSeriesFixture });
        mountTimeSeries({ http });
        await flushPromises();
        expect(http.get).toHaveBeenCalled();
        expect(http.get.mock.calls[0][0]).toContain("time_series.py");
    });

    // --- @click="toggleFrequency('absolute_time')" / 'relative_time' ---
    it("toggles to relative frequency on button click", async () => {
        const wrapper = mountTimeSeries();
        await flushPromises();
        await nextTick();

        const buttons = wrapper.findAll(".btn-group .btn");
        const relativeBtn = buttons[1]; // second button is relative
        await relativeBtn.trigger("click");
        await nextTick();

        expect(wrapper.vm.frequencyType).toBe("relative_time");
    });

    it("toggles back to absolute frequency on button click", async () => {
        const wrapper = mountTimeSeries();
        await flushPromises();
        await nextTick();

        const buttons = wrapper.findAll(".btn-group .btn");
        // Click relative then absolute
        await buttons[1].trigger("click");
        await nextTick();
        await buttons[0].trigger("click");
        await nextTick();

        expect(wrapper.vm.frequencyType).toBe("absolute_time");
    });

    it("sets active class on selected frequency button", async () => {
        const wrapper = mountTimeSeries();
        await flushPromises();
        await nextTick();

        const buttons = wrapper.findAll(".btn-group .btn");
        await buttons[1].trigger("click");
        await nextTick();

        expect(buttons[1].classes()).toContain("active");
    });
});
