import { defineStore } from "pinia";

export const useMainStore = defineStore("main", {
    state: () => ({
        formData: {
            // Initialize with basic defaults to avoid undefined errors
            // This should match the structure from App.vue defaultFieldValues
            report: "home",
            q: "",
            method: "proxy",
            cooc_order: "yes",
            method_arg: "",
            arg_phrase: "",
            results_per_page: 25,
            start: "",
            end: "",
            colloc_filter_choice: "",
            colloc_within: "sent",
            filter_frequency: 100,
            approximate: "no",
            approximate_ratio: 100,
            start_date: "",
            end_date: "",
            year_interval: "",
            sort_by: "rowid",
            first_kwic_sorting_option: "",
            second_kwic_sorting_option: "",
            third_kwic_sorting_option: "",
            start_byte: "",
            end_byte: "",
            group_by: "",
        },
        reportValues: {},
        resultsLength: 0,
        textNavigationCitation: {},
        textObject: "",
        navBar: "",
        tocElements: {},
        byte: "",
        searching: false,
        currentReport: "concordance",
        description: {
            start: 0,
            end: 0,
            results_per_page: 25,
            termGroups: [],
        },
        aggregationCache: {
            results: [],
            query: {},
        },
        sortedKwicCache: {
            queryParams: {},
            results: [],
            totalResults: 0,
        },
        totalResultsDone: false,
        showFacets: true,
        urlUpdate: "",
        metadataUpdate: {},
        searchableMetadata: { display: [], inputStyle: [], choiceValues: [] },
    }),

    actions: {
        // Replace the old mutations with actions
        updateFormData(payload) {
            this.formData = payload;
        },

        setDefaultFields(payload) {
            for (let field in payload) {
                this.formData[field] = payload[field];
            }
        },

        updateFormDataField(payload) {
            this.formData[payload.key] = payload.value;
        },

        updateAllMetadata(payload) {
            this.formData = { ...this.formData, ...payload };
        },

        setReportValues(payload) {
            this.reportValues = payload;
        },

        updateCitation(payload) {
            this.textNavigationCitation = payload;
        },

        updateDescription(payload) {
            this.description = payload;
        },

        updateResultsLength(payload) {
            this.resultsLength = payload;
        },

        updateStartEndDate(payload) {
            this.formData = {
                ...this.formData,
                start_date: payload.startDate,
                end_date: payload.endDate,
            };
        },

        // Helper action to mimic the old updateField functionality
        updateField({ path, value }) {
            const pathArray = path.split(".");
            let target = this;

            // Navigate to the parent of the field to update
            for (let i = 0; i < pathArray.length - 1; i++) {
                target = target[pathArray[i]];
            }

            // Set the final value
            target[pathArray[pathArray.length - 1]] = value;
        },
    },
});
