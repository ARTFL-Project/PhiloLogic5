import { afterEach, describe, it, expect, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { nextTick } from "vue";
import { createTestI18n } from "./helpers.js";
import TimeSeriesChart from "../src/components/TimeSeriesChart.vue";

let wrapper = null;
function mountChart(props = {}) {
    wrapper = mount(TimeSeriesChart, {
        props: {
            values: [3, 0, 17, 8],
            labels: [1500, 1550, 1600, 1650],
            ranges: ["1500-1549", "1550-1599", "1600-1649", "1650"],
            valueText: (value) => `${value} occurrence${value === 1 ? "" : "s"}`,
            xTitle: "Year",
            yTitle: "Absolute Frequency",
            label: "The chart",
            describedby: "instructions",
            hint: "The keys",
            ...props,
        },
        global: { plugins: [createTestI18n()] },
        attachTo: document.body, // for the focus
    });
    return wrapper;
}
afterEach(() => {
    wrapper?.unmount();
    wrapper = null;
    vi.useRealTimers();
});

const columns = () => wrapper.findAll(".ts-column");
const tooltip = () => wrapper.find(".ts-tooltip");

describe("TimeSeriesChart", () => {
    it("names its group and each period's button", () => {
        mountChart();
        const group = wrapper.find("[role='group']");
        expect(group.attributes("aria-label")).toBe("The chart");
        expect(group.attributes("aria-describedby")).toBe("instructions");
        expect(columns().map((c) => c.attributes("aria-label"))).toEqual([
            "1500-1549: 3 occurrences",
            "1550-1599: 0 occurrences",
            "1600-1649: 17 occurrences",
            "1650: 8 occurrences",
        ]);
    });

    it("draws the bars to a scale of round steps from 0, the highest under the top", () => {
        mountChart();
        const ticks = wrapper.findAll(".ts-y-tick").map((t) => Number(t.text()));
        expect(ticks[0]).toBe(0);
        const step = ticks[1];
        expect([1, 2, 5, 10, 20]).toContain(step);
        ticks.forEach((t, i) => expect(t).toBe(i * step));
        expect(ticks.at(-1)).toBeGreaterThanOrEqual(17);
        expect(ticks.at(-1) - step).toBeLessThan(17);
        const height = (i) => parseFloat(wrapper.findAll(".ts-bar")[i].element.style.height);
        expect(height(2)).toBeCloseTo((17 / ticks.at(-1)) * 100);
        expect(height(1)).toBe(0);
    });

    it("labels as many periods as fit", () => {
        const n = 400;
        mountChart({
            values: Array(n).fill(1),
            labels: Array.from({ length: n }, (_, i) => 1000 + i),
            ranges: Array.from({ length: n }, (_, i) => `${1000 + i}`),
        });
        const shown = wrapper.findAll(".ts-x-tick:not(.ts-probe)").map((t) => Number(t.text()));
        expect(shown[0]).toBe(1000);
        expect(shown.length).toBeLessThan(n / 2);
        const every = shown[1] - shown[0];
        shown.forEach((year, i) => expect(year).toBe(1000 + i * every));
    });

    it("leaves gaps between the bars, unless under a pixel", () => {
        const barWidth = () => wrapper.find(".ts-columns").element.style.getPropertyValue("--bar-width");
        mountChart();
        expect(barWidth()).toBe("72%");
        wrapper.unmount();
        const n = 400; // 2 px each, before the chart is laid out
        mountChart({ values: Array(n).fill(1), labels: Array(n).fill(1), ranges: Array(n).fill("1") });
        expect(barWidth()).toBe("100%");
    });

    it("is one tab stop, the arrow keys, Home and End moving the focus", async () => {
        mountChart();
        expect(columns().map((c) => c.attributes("tabindex"))).toEqual(["0", "-1", "-1", "-1"]);
        columns()[0].element.focus();
        await columns()[0].trigger("keydown", { key: "ArrowRight" });
        expect(document.activeElement).toBe(columns()[1].element);
        await columns()[1].trigger("keydown", { key: "End" });
        expect(document.activeElement).toBe(columns()[3].element);
        await columns()[3].trigger("keydown", { key: "ArrowRight" }); // the last stays
        expect(document.activeElement).toBe(columns()[3].element);
        await columns()[3].trigger("keydown", { key: "Home" });
        await columns()[0].trigger("keydown", { key: "ArrowLeft" }); // and so does the first
        expect(document.activeElement).toBe(columns()[0].element);
        await columns()[0].trigger("keydown", { key: "ArrowRight" });
        await nextTick();
        expect(columns().map((c) => c.attributes("tabindex"))).toEqual(["-1", "0", "-1", "-1"]);
    });

    it("shows the keys while a period has the focus", async () => {
        mountChart();
        const hint = () => wrapper.find(".ts-hint");
        expect(hint().text()).toBe("The keys");
        expect(hint().classes()).not.toContain("shown");
        columns()[0].element.focus();
        await nextTick();
        expect(hint().classes()).toContain("shown");
        await columns()[0].trigger("keydown", { key: "ArrowRight" });
        expect(hint().classes()).toContain("shown");
        columns()[1].element.blur();
        await nextTick();
        expect(hint().classes()).not.toContain("shown");
    });

    it("shows the focused period's tooltip, which Escape dismisses until the focus moves", async () => {
        mountChart();
        columns()[2].element.focus();
        await nextTick();
        expect(tooltip().text()).toContain("1600-1649");
        expect(tooltip().text()).toContain("17 occurrences");
        expect(tooltip().attributes("aria-hidden")).toBe("true");
        expect(columns()[2].classes()).toContain("active");

        document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
        await nextTick();
        expect(tooltip().exists()).toBe(false);

        await columns()[2].trigger("keydown", { key: "ArrowRight" });
        expect(tooltip().text()).toContain("1650");
        columns()[3].element.blur();
        await nextTick();
        expect(tooltip().exists()).toBe(false);
    });

    it("keeps the tooltip of the period pointed at while the pointer moves onto it", async () => {
        vi.useFakeTimers();
        mountChart();
        await columns()[0].trigger("pointerenter");
        expect(tooltip().text()).toContain("1500-1549");
        await columns()[0].trigger("pointerleave");
        await tooltip().trigger("pointerenter");
        vi.advanceTimersByTime(1000);
        await nextTick();
        expect(tooltip().text()).toContain("1500-1549");

        await tooltip().trigger("pointerleave");
        vi.advanceTimersByTime(1000);
        await nextTick();
        expect(tooltip().exists()).toBe(false);
    });

    it("dismisses the tooltip of the period pointed at with Escape", async () => {
        mountChart();
        await columns()[1].trigger("pointerenter");
        document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
        await nextTick();
        expect(tooltip().exists()).toBe(false);
        await columns()[2].trigger("pointerenter");
        expect(tooltip().text()).toContain("1600-1649");
    });

    it("emits the period chosen, by its bar or its tooltip", async () => {
        mountChart();
        await columns()[3].trigger("click");
        await columns()[1].trigger("pointerenter");
        await tooltip().trigger("click");
        expect(wrapper.emitted("select")).toEqual([[3], [1]]);
    });
});
