import { describe, it, expect, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createI18n } from "vue-i18n";
import { reactive } from "vue";
import MetadataFields from "../src/components/MetadataFields.vue";

const i18n = createI18n({
    legacy: false,
    locale: "en",
    messages: {
        en: {
            collocation: { filterBy: "Filter by" },
            searchForm: {
                exactDate: "exact",
                rangeDate: "range",
                exactDateLabel: "Exact {field}",
                rangeDateLabel: "{field} range",
                dateFrom: "From",
                dateTo: "To",
            },
        },
    },
});

function mountMetadataFields(overrides = {}) {
    const defaults = {
        fields: [
            { value: "author", label: "Author", example: "Voltaire" },
            { value: "title", label: "Title", example: "Candide" },
        ],
        inputStyles: { author: "text", title: "text" },
        modelValue: reactive({ author: "", title: "" }),
        ...overrides,
    };

    return mount(MetadataFields, {
        props: defaults,
        global: {
            plugins: [i18n],
        },
    });
}

describe("MetadataFields", () => {
    // ----- Text inputs -----
    it("renders text inputs for text fields", () => {
        const wrapper = mountMetadataFields();
        const inputs = wrapper.findAll("input[type='text']");
        expect(inputs.length).toBe(2);
    });

    it("shows placeholder text", () => {
        const wrapper = mountMetadataFields();
        const input = wrapper.find("#author-input-filter");
        expect(input.attributes("placeholder")).toBe("Voltaire");
    });

    it("binds v-model to modelValue", async () => {
        const modelValue = reactive({ author: "", title: "" });
        const wrapper = mountMetadataFields({ modelValue });

        await wrapper.find("#author-input-filter").setValue("Rousseau");
        expect(modelValue.author).toBe("Rousseau");
    });

    // ----- ID prefixing -----
    it("prefixes IDs when idPrefix is set", () => {
        const wrapper = mountMetadataFields({ idPrefix: "primary-" });
        expect(wrapper.find("#primary-author-input-filter").exists()).toBe(true);
        expect(wrapper.find("#author-input-filter").exists()).toBe(false);
    });

    it("does not prefix IDs when idPrefix is empty", () => {
        const wrapper = mountMetadataFields({ idPrefix: "" });
        expect(wrapper.find("#author-input-filter").exists()).toBe(true);
    });

    // ----- Date fields -----
    describe("date/int fields", () => {
        function mountWithDateField(modelValue = {}, overrides = {}) {
            return mountMetadataFields({
                fields: [{ value: "year", label: "Year", example: "1800" }],
                inputStyles: { year: "int" },
                modelValue: reactive({ year: "", ...modelValue }),
                ...overrides,
            });
        }

        it("renders exact date input by default", () => {
            const wrapper = mountWithDateField();
            const input = wrapper.find("#year-input-filter");
            expect(input.exists()).toBe(true);
        });

        it("auto-detects range from modelValue for int fields", () => {
            const wrapper = mountWithDateField({ year: "1800-1900" });
            // Should show range inputs (From/To) instead of single input
            expect(wrapper.text()).toContain("From");
            expect(wrapper.text()).toContain("To");
        });

        it("auto-detects range from modelValue for date fields", () => {
            const wrapper = mountMetadataFields({
                fields: [{ value: "pub_date", label: "Date", example: "1800" }],
                inputStyles: { pub_date: "date" },
                modelValue: reactive({ pub_date: "1800<=>1900" }),
            });
            expect(wrapper.text()).toContain("From");
            expect(wrapper.text()).toContain("To");
        });

        it("uses prop dateType/dateRange when provided", () => {
            const wrapper = mountMetadataFields({
                fields: [{ value: "year", label: "Year", example: "1800" }],
                inputStyles: { year: "int" },
                modelValue: reactive({ year: "" }),
                dateType: reactive({ year: "range" }),
                dateRange: reactive({ year: { start: "1750", end: "1850" } }),
            });
            expect(wrapper.text()).toContain("From");
            expect(wrapper.text()).toContain("To");
        });
    });

    // ----- Dropdown fields -----
    describe("dropdown fields", () => {
        it("renders select with options", () => {
            const wrapper = mountMetadataFields({
                fields: [{ value: "genre", label: "Genre", example: "" }],
                inputStyles: { genre: "dropdown" },
                modelValue: reactive({ genre: "" }),
                choiceValues: {
                    genre: [
                        { text: "Novel", value: "novel" },
                        { text: "Poetry", value: "poetry" },
                    ],
                },
                selectedValues: reactive({ genre: "" }),
            });
            const options = wrapper.findAll("option");
            expect(options.length).toBe(2);
            expect(options[0].text()).toBe("Novel");
        });
    });

    // ----- Checkbox fields -----
    describe("checkbox fields", () => {
        it("renders checkboxes for checkbox fields", () => {
            const wrapper = mountMetadataFields({
                fields: [{ value: "language", label: "Language", example: "" }],
                inputStyles: { language: "checkbox" },
                modelValue: reactive({ language: "" }),
                choiceValues: {
                    language: [
                        { text: "French", value: "french" },
                        { text: "English", value: "english" },
                    ],
                },
                checkedValues: reactive({ french: false, english: false }),
            });
            const checkboxes = wrapper.findAll("input[type='checkbox']");
            expect(checkboxes.length).toBe(2);
        });
    });
});
