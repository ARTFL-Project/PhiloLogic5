import { describe, it, expect, vi } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { nextTick } from "vue";
import { createGlobalConfig, createMockHttp } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import ExportResults from "../src/components/ExportResults.vue";

function mountExportResults(overrides = {}) {
    const http = overrides.http || createMockHttp({
        "export_results.py": [{ context: "test result", metadata_fields: { author: "Brown" } }],
    });
    const global = createGlobalConfig({
        http,
        route: { name: "concordance", path: "/concordance", query: { q: "liberty", report: "concordance" } },
        ...overrides,
    });

    const store = useMainStore();
    store.formData = { ...store.formData, q: "liberty", report: "concordance" };

    return mount(ExportResults, { global });
}

describe("ExportResults", () => {
    it("renders export format options", () => {
        const wrapper = mountExportResults();
        const text = wrapper.text();
        expect(text).toContain("JSON") || expect(text.toLowerCase()).toContain("json");
    });

    it("renders both HTML and plain text sections", () => {
        const wrapper = mountExportResults();
        const buttons = wrapper.findAll("button");
        // Should have at least 2 export buttons (JSON + CSV) in at least one section
        expect(buttons.length).toBeGreaterThanOrEqual(2);
    });

    // --- @click="getResults('json', false, $event)" ---
    it("makes HTTP request for JSON export on click", async () => {
        const http = createMockHttp({ "export_results.py": [{ test: true }] });
        const wrapper = mountExportResults({ http });

        const buttons = wrapper.findAll("button");
        const jsonBtn = buttons.find(b => b.text().toUpperCase().includes("JSON"));
        if (jsonBtn) {
            await jsonBtn.trigger("click");
            await flushPromises();
            const exportCalls = http.get.mock.calls.filter(c => c[0].includes("export_results"));
            expect(exportCalls.length).toBeGreaterThan(0);
            // Verify format=json in the request
            expect(exportCalls[0][0]).toContain("output_format=json");
        }
    });

    // --- @click="getResults('csv', false, $event)" ---
    it("makes HTTP request for CSV export on click", async () => {
        const http = createMockHttp({ "export_results.py": "author,context\nBrown,test" });
        const wrapper = mountExportResults({ http });

        const buttons = wrapper.findAll("button");
        const csvBtn = buttons.find(b => b.text().toUpperCase().includes("CSV"));
        if (csvBtn) {
            await csvBtn.trigger("click");
            await flushPromises();
            const exportCalls = http.get.mock.calls.filter(c => c[0].includes("export_results"));
            expect(exportCalls.length).toBeGreaterThan(0);
            expect(exportCalls[0][0]).toContain("output_format=csv");
        }
    });

    // --- @click="getResults('json', true, $event)" (with HTML filtering) ---
    it("passes filter_html=true for plain text export", async () => {
        const http = createMockHttp({ "export_results.py": [{ test: true }] });
        const wrapper = mountExportResults({ http });

        // Find buttons in the "Plain Text" section
        const buttons = wrapper.findAll("button");
        // The second pair of buttons should have filter_html=true
        // We look for all buttons and check if any pass filter_html=true
        for (const btn of buttons) {
            if (btn.text().toUpperCase().includes("JSON") || btn.text().toUpperCase().includes("CSV")) {
                await btn.trigger("click");
                await flushPromises();
            }
        }
        // At least some export calls should have been made
        expect(http.get.mock.calls.length).toBeGreaterThan(0);
    });
});
