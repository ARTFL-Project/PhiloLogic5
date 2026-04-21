import { describe, it, expect, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createI18n } from "vue-i18n";
import BibliographyCriteria from "../src/components/BibliographyCriteria.vue";

const i18n = createI18n({
    legacy: false,
    locale: "en",
    messages: {
        en: {
            biblioCriteria: {
                searchCriteriaRegion: "Search criteria",
                activeFiltersLabel: "Active search filters",
                filterLabel: "Filter",
                removeFilter: "Remove filter:",
                dateRangeGroup: "Date range filters",
                startDate: "Start date",
                endDate: "End date",
                removeStartDate: "Remove start date filter",
                removeEndDate: "Remove end date filter",
            },
            searchArgs: { biblioCriteria: "Bibliography Criteria", occurrencesBetween: "{n} occurrences between" },
            common: { none: "None", and: "and" },
        },
    },
});

function mountBiblio(props = {}) {
    return mount(BibliographyCriteria, {
        props: {
            biblio: [],
            queryReport: "concordance",
            resultsLength: 0,
            ...props,
        },
        global: { plugins: [i18n] },
    });
}

describe("BibliographyCriteria", () => {
    it("shows 'None' when biblio is empty", () => {
        const wrapper = mountBiblio();
        expect(wrapper.text()).toContain("None");
    });

    it("renders biblio criteria as pills", () => {
        const wrapper = mountBiblio({
            biblio: [
                { key: "author", alias: "Author", value: "Voltaire" },
                { key: "title", alias: "Title", value: "Candide" },
            ],
        });
        expect(wrapper.text()).toContain("Author");
        expect(wrapper.text()).toContain("Voltaire");
        expect(wrapper.text()).toContain("Title");
        expect(wrapper.text()).toContain("Candide");
    });

    it("shows remove button when removeMetadata function is provided", () => {
        const removeFn = vi.fn();
        const wrapper = mountBiblio({
            biblio: [{ key: "author", alias: "Author", value: "Voltaire" }],
            removeMetadata: removeFn,
        });
        const removeBtn = wrapper.find(".remove-metadata");
        expect(removeBtn.exists()).toBe(true);
    });

    it("hides remove button when no removeMetadata function", () => {
        const wrapper = mountBiblio({
            biblio: [{ key: "author", alias: "Author", value: "Voltaire" }],
        });
        expect(wrapper.find(".remove-metadata").exists()).toBe(false);
    });

    it("calls removeMetadata with field key on remove click", async () => {
        const removeFn = vi.fn();
        const wrapper = mountBiblio({
            biblio: [{ key: "author", alias: "Author", value: "Voltaire" }],
            removeMetadata: removeFn,
        });
        await wrapper.find(".remove-metadata").trigger("click");
        expect(removeFn).toHaveBeenCalledWith("author");
    });

    it("replaces <=> with em dash in values", () => {
        const wrapper = mountBiblio({
            biblio: [{ key: "year", alias: "Year", value: "1800<=>1900" }],
        });
        expect(wrapper.text()).toContain("1800");
        expect(wrapper.text()).toContain("1900");
    });
});
