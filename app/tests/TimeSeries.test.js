import { describe, it, expect } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { nextTick } from "vue";
import { createGlobalConfig, createMockHttp, createPendingHttp } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import timeSeriesFixture from "./fixtures/time_series.json";
import TimeSeries from "../src/components/TimeSeries.vue";
import TimeSeriesChart from "../src/components/TimeSeriesChart.vue";

function mountTimeSeries(overrides = {}) {
    const http = overrides.http || createMockHttp({ "time_series.py": timeSeriesFixture });
    const global = createGlobalConfig({
        http,
        route: { name: "time_series", path: "/time_series", query: { q: "liberty", report: "time_series", start_date: "1500", end_date: "1800", year_interval: "50" } },
        stubs: {
            ResultsSummary: { template: "<div class='results-summary-stub' />" },
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

        expect(buttons[1].classes()).toContain("active");
        expect(buttons[0].classes()).not.toContain("active");
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

        expect(buttons[0].classes()).toContain("active");
        expect(buttons[1].classes()).not.toContain("active");
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

    // --- The chart ---
    it("charts each period, named and counted", async () => {
        const wrapper = mountTimeSeries();
        await flushPromises();
        const bars = wrapper.findAll(".ts-column");
        expect(bars).toHaveLength(7); // 1500 to 1800 by 50
        expect(bars[0].attributes("aria-label")).toBe("1500-1549: 165 occurrences");
        expect(bars[6].attributes("aria-label")).toMatch(/^1800: /); // the last period stops at the end date
        expect(wrapper.findAll("table tbody tr td")[1].text()).toBe("165 occurrences");
        expect(wrapper.find("table tbody tr td").text()).toBe("1500-1549");
        expect(wrapper.find(".ts-hint").text()).toContain("move between periods");
    });

    it("charts the relative frequencies, and says so", async () => {
        const wrapper = mountTimeSeries();
        await flushPromises();
        await wrapper.findAll(".btn-group .btn")[1].trigger("click");
        const per10000 = Math.round((165 / 29174035) * 10000 * 100) / 100;
        expect(wrapper.find(".ts-column").attributes("aria-label")).toBe(`1500-1549: ${per10000} per 10,000 words`);
        expect(wrapper.find("[role='status']").text()).toBe("Chart updated to show Relative Frequency");
    });

    it("shows the concordance of a period chosen", async () => {
        const wrapper = mountTimeSeries();
        await flushPromises();
        await wrapper.findAll(".ts-column")[1].trigger("click");
        await flushPromises();
        const route = wrapper.vm.$router.currentRoute.value;
        expect(route.path).toBe("/concordance");
        expect(route.query.year).toBe("1550-1599");
        expect(useMainStore().formData.year).toBe("1550-1599");
    });

    // --- A search at a time ---
    it("shows the newest search, the request of the one it replaced aborted", async () => {
        const { http, requests } = createPendingHttp("time_series.py");
        const wrapper = mountTimeSeries({ http });
        await flushPromises();
        const store = useMainStore();
        store.urlUpdate = { q: "freedom", report: "time_series", start_date: "1500", end_date: "1800" };
        await flushPromises();
        expect(requests).toHaveLength(2);
        expect(requests[0].config.signal.aborted).toBe(true);

        // the newer one's counts twice the older one's
        const doubled = structuredClone(timeSeriesFixture);
        for (const date of Object.values(doubled.results.absolute_count)) date.count *= 2;
        requests[1].answer(doubled);
        requests[0].answer(timeSeriesFixture); // too late
        await flushPromises();
        const first = timeSeriesFixture.results.absolute_count["1500"].count;
        expect(wrapper.findComponent(TimeSeriesChart).props("values")[0]).toBe(2 * first);
        expect(store.searching).toBe(false);
    });
});
