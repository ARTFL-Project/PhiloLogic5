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
            colloc_lemmas: "no",
            filter_frequency: 100,
            filter_df_pct: "1",
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
            cutTerms: [],
        },
        totalResultsDone: false,
        searchError: "",
        showFacets: true,
        urlUpdate: "",
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
                // collocates counted as lemmas where the database has them
                colloc_lemmas: philoConfig.collocation_lemmas ? "yes" : "no",
                filter_frequency: 100,
                filter_df_pct: "1",
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
            // The values kept of each word property, for collocates
            const propertyFields = Object.keys(philoConfig.word_attributes || {}).map((name) => `colloc_attr_${name}`);
            for (const field of propertyFields) {
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
                    "start", "colloc_filter_choice", "filter_frequency", "filter_df_pct",
                    "colloc_within", "method_arg", "q_attribute", "q_attribute_value",
                    "colloc_lemmas", ...propertyFields,
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
                bibliography: new Set([
                    ...commonFields,
                    "results_per_page", "sort_by", "start", "end", "frequency_field",
                ]),
            };
        },

        updateFormDataField(payload) {
            this.formData[payload.key] = payload.value;
        },

        updateDescription(payload) {
            this.description = payload;
        },

        updateResultsLength(payload) {
            this.resultsLength = payload;
        },

        toggleFacets() {
            this.showFacets = !this.showFacets;
        },

        updateStartEndDate(payload) {
            this.formData = {
                ...this.formData,
                start_date: payload.startDate,
                end_date: payload.endDate,
            };
        },
    },
});
