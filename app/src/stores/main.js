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
        defaultFields: {},
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

        // Derive default form values + per-report field whitelists from
        // philoConfig. Called once at app startup; both pieces of state are
        // then read by App.vue (initial form state, route-driven updates),
        // SearchForm.vue (reset), and utils.paramsFilter (whitelist lookup).
        initFromConfig(philoConfig) {
            const defaults = {
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
                year_interval: philoConfig.time_series_interval,
                sort_by: "rowid",
                first_kwic_sorting_option: "",
                second_kwic_sorting_option: "",
                third_kwic_sorting_option: "",
                start_byte: "",
                end_byte: "",
                group_by: philoConfig.aggregation_config[0].field,
            };
            for (const field of philoConfig.metadata) {
                defaults[field] = "";
            }
            this.defaultFields = defaults;

            const commonFields = ["q", "approximate", "approximate_ratio", ...philoConfig.metadata];
            this.reportValues = {
                concordance: new Set([
                    ...commonFields,
                    "method", "cooc_order", "method_arg", "results_per_page",
                    "sort_by", "hit_num", "start", "end",
                    "frequency_field", "word_property",
                ]),
                kwic: new Set([
                    ...commonFields,
                    "method", "cooc_order", "method_arg", "results_per_page",
                    "first_kwic_sorting_option",
                    "second_kwic_sorting_option",
                    "third_kwic_sorting_option",
                    "start", "end",
                    "frequency_field", "word_property",
                ]),
                collocation: new Set([
                    ...commonFields,
                    "start", "colloc_filter_choice", "filter_frequency",
                    "colloc_within", "method_arg", "q_attribute", "q_attribute_value",
                ]),
                time_series: new Set([
                    ...commonFields,
                    "method", "cooc_order", "method_arg",
                    "start_date", "end_date", "year_interval", "max_time",
                ]),
                aggregation: new Set([
                    ...commonFields,
                    "method", "cooc_order", "method_arg", "group_by",
                ]),
            };
        },

        // Reset formData fields to the previously-computed defaults (used by
        // SearchForm's "reset" action).
        resetFormDataToDefaults() {
            for (const field in this.defaultFields) {
                this.formData[field] = this.defaultFields[field];
            }
        },

        updateFormDataField(payload) {
            this.formData[payload.key] = payload.value;
        },

        updateAllMetadata(payload) {
            this.formData = { ...this.formData, ...payload };
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
