import { describe, it, expect, beforeEach } from "vitest";
import { setActivePinia, createPinia } from "pinia";
import { useMainStore } from "../src/stores/main.js";

describe("Main Store", () => {
    let store;

    beforeEach(() => {
        setActivePinia(createPinia());
        store = useMainStore();
    });

    it("has correct default formData", () => {
        expect(store.formData.report).toBe("home");
        expect(store.formData.q).toBe("");
        expect(store.formData.method).toBe("proxy");
        expect(store.formData.results_per_page).toBe(25);
        expect(store.formData.approximate).toBe("no");
    });

    it("updateFormData replaces formData entirely", () => {
        store.updateFormData({ report: "concordance", q: "test" });
        expect(store.formData.report).toBe("concordance");
        expect(store.formData.q).toBe("test");
    });

    it("setDefaultFields merges into formData", () => {
        store.setDefaultFields({ q: "default", report: "home" });
        expect(store.formData.q).toBe("default");
        expect(store.formData.method).toBe("proxy"); // preserved
    });

    it("updateFormDataField updates a single field", () => {
        store.updateFormDataField({ key: "q", value: "new query" });
        expect(store.formData.q).toBe("new query");
    });

    it("updateAllMetadata merges metadata into formData", () => {
        store.updateAllMetadata({ author: "Voltaire", title: "Candide" });
        expect(store.formData.author).toBe("Voltaire");
        expect(store.formData.title).toBe("Candide");
        expect(store.formData.q).toBe(""); // preserved
    });

    it("setReportValues stores report field sets", () => {
        const reportValues = {
            concordance: new Set(["q", "method"]),
        };
        store.setReportValues(reportValues);
        expect(store.reportValues.concordance.has("q")).toBe(true);
    });

    it("updateStartEndDate updates both date fields", () => {
        store.updateStartEndDate({ startDate: "1800", endDate: "1900" });
        expect(store.formData.start_date).toBe("1800");
        expect(store.formData.end_date).toBe("1900");
    });

    it("updateDescription sets description", () => {
        store.updateDescription({ start: 1, end: 25, results_per_page: 25 });
        expect(store.description.start).toBe(1);
        expect(store.description.end).toBe(25);
    });

    it("updateResultsLength sets results count", () => {
        store.updateResultsLength(500);
        expect(store.resultsLength).toBe(500);
    });

    it("searching flag defaults to false", () => {
        expect(store.searching).toBe(false);
    });
});
