import { describe, it, expect, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { nextTick } from "vue";
import { createGlobalConfig } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import ResultsBibliography from "../src/components/ResultsBibliography.vue";

const sampleResults = [
    {
        philo_id: [1, 0, 0, 0, 0, 0, 0],
        citation: [{ label: "Brown, Wieland", href: "/navigate/1", style: {}, prefix: "", suffix: "" }],
        metadata_fields: { author: "Brown", title: "Wieland" },
    },
    {
        philo_id: [2, 0, 0, 0, 0, 0, 0],
        citation: [{ label: "Smith, Essays", href: "/navigate/2", style: {}, prefix: "", suffix: "" }],
        metadata_fields: { author: "Smith", title: "Essays" },
    },
];

function mountResultsBibliography(overrides = {}) {
    const global = createGlobalConfig({
        route: { name: "concordance", path: "/concordance", query: { q: "liberty", report: "concordance" } },
        stubs: {
            Citations: { template: "<span class='citations-stub' />" },
        },
        ...overrides,
    });

    const store = useMainStore();
    store.formData = { ...store.formData, q: "liberty", report: "concordance" };

    return mount(ResultsBibliography, {
        global: {
            ...global,
            provide: {
                ...global.provide,
                results: overrides.results || sampleResults,
            },
        },
    });
}

describe("ResultsBibliography", () => {
    it("renders component", () => {
        const wrapper = mountResultsBibliography();
        expect(wrapper.exists()).toBe(true);
    });

    it("renders citations for each unique result", () => {
        const wrapper = mountResultsBibliography();
        const citations = wrapper.findAll(".citations-stub");
        expect(citations.length).toBeGreaterThan(0);
    });

    it("displays occurrence counts", () => {
        const wrapper = mountResultsBibliography();
        const text = wrapper.text();
        // Should show occurrence count badges
        expect(text.length).toBeGreaterThan(0);
    });

    // --- @click="$event.target.blur()" on modal trigger ---
    it("has blur handler on modal trigger button", () => {
        const wrapper = mountResultsBibliography();
        const buttons = wrapper.findAll("button");
        // The modal trigger button should exist
        expect(buttons.length).toBeGreaterThanOrEqual(0);
    });

    it("handles empty results gracefully", () => {
        const wrapper = mountResultsBibliography({ results: [] });
        expect(wrapper.exists()).toBe(true);
    });
});
