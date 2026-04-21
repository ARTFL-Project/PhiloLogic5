/**
 * Shared test helpers for PhiloLogic5 Vue component tests.
 */
import { createI18n } from "vue-i18n";
import { createPinia, setActivePinia } from "pinia";
import { createRouter, createMemoryHistory } from "vue-router";
import { vi } from "vitest";
import en from "../src/locales/en.json";
import {
    paramsFilter,
    paramsToRoute,
    paramsToUrlString,
    copyObject,
    saveToLocalStorage,
    mergeResults,
    sortResults,
    deepEqual,
    dictionaryLookup,
    dateRangeHandler,
    buildBiblioCriteria,
    extractSurfaceFromCollocate,
    debug,
    isOnlyFacetChange,
    buildTocTree,
} from "../src/mixins.js";

/** Create a fresh i18n instance with English messages */
export function createTestI18n() {
    return createI18n({
        legacy: false,
        locale: "en",
        messages: { en },
    });
}

/** Create and activate a fresh Pinia store */
export function createTestPinia() {
    const pinia = createPinia();
    setActivePinia(pinia);
    return pinia;
}

/** Minimal $philoConfig for tests */
export function createTestConfig(overrides = {}) {
    return {
        valid_config: true,
        dbname: "testdb",
        db_url: "/testdb",
        access_control: false,
        metadata: ["author", "title", "year"],
        available_metadata: ["author", "title", "year"],
        metadata_aliases: { author: "Author", title: "Title", year: "Year" },
        metadata_input_style: { author: "text", title: "text", year: "int" },
        metadata_choice_values: {},
        facets: [],
        search_reports: ["concordance", "kwic", "collocation", "time_series", "aggregation"],
        time_series_year_field: "year",
        time_series_interval: 10,
        time_series_start_end_date: { start_date: 1450, end_date: 1800 },
        concordance_citation: [],
        bibliography_citation: [],
        kwic_metadata_sorting_fields: [],
        kwic_bibliography_fields: [],
        aggregation_config: [{ field: "author", object_level: "doc", field_citation: [], break_up_field: null, break_up_field_citation: null }],
        dictionary_bibliography: false,
        word_attributes: {},
        word_property_aliases: {},
        landing_page_browsing: "default",
        collocation_fields_to_compare: ["author", "title"],
        academic_citation: { link: "", collection: "", citation: [], custom_url: "" },
        report_error_link: "",
        header_in_footer: false,
        header_in_toc: false,
        concordance_biblio_sorting: [],
        concordance_formatting_regex: [],
        kwic_formatting_regex: [],
        navigation_formatting_regex: [],
        respect_text_line_breaks: false,
        skip_table_of_contents: false,
        external_page_images: false,
        page_images_url_root: "",
        page_image_extension: "",
        dictionary_lookup: { url_root: "", keywords: false },
        dictionary_lookup_keywords: { immutable_key_values: {}, variable_key_values: {}, selected_keyword: "" },
        query_parser_regex: [],
        stopwords: "",
        search_examples: {},
        results_summary: [{ field: "author", object_level: "doc" }],
        concordance_citation: [],
        navigation_citation: [],
        table_of_contents_citation: [],
        default_landing_page_browsing: [],
        default_landing_page_display: {},
        simple_landing_citation: [],
        dico_letter_range: [],
        ascii_conversion: true,
        logo: "",
        theme: "",
        ...overrides,
    };
}

/** Create a mock $http with configurable responses */
export function createMockHttp(responses = {}) {
    return {
        get: vi.fn((url) => {
            for (const [pattern, response] of Object.entries(responses)) {
                if (url.includes(pattern)) {
                    return Promise.resolve({ data: response });
                }
            }
            return Promise.resolve({ data: {} });
        }),
    };
}

/** Create a test router with memory history */
export function createTestRouter(route = {}) {
    const routeDefaults = { name: "concordance", path: "/concordance", query: {}, params: {}, ...route };
    const router = createRouter({
        history: createMemoryHistory(),
        routes: [
            { path: "/", name: "home", component: { template: "<div />" } },
            { path: "/concordance", name: "concordance", component: { template: "<div />" } },
            { path: "/kwic", name: "kwic", component: { template: "<div />" } },
            { path: "/collocation", name: "collocation", component: { template: "<div />" } },
            { path: "/time_series", name: "time_series", component: { template: "<div />" } },
            { path: "/aggregation", name: "aggregation", component: { template: "<div />" } },
            { path: "/bibliography", name: "bibliography", component: { template: "<div />" } },
            { path: "/landing", name: "landing", component: { template: "<div />" } },
            { path: "/navigate/:pathInfo([\\d/]+)", name: "textNavigation", component: { template: "<div />" } },
            { path: "/navigate/:pathInfo/table-of-contents", name: "tableOfContents", component: { template: "<div />" } },
        ],
    });
    // Push initial route with query
    const queryStr = Object.keys(routeDefaults.query).length > 0
        ? `?${new URLSearchParams(routeDefaults.query).toString()}`
        : "";
    router.push(`${routeDefaults.path}${queryStr}`);
    return router;
}

/** The mixin methods that App.vue registers globally */
const mixinMethods = {
    paramsFilter,
    paramsToRoute,
    paramsToUrlString,
    copyObject,
    saveToLocalStorage,
    mergeResults,
    sortResults,
    deepEqual,
    dictionaryLookup,
    dateRangeHandler,
    buildBiblioCriteria,
    extractSurfaceFromCollocate,
    debug,
    isOnlyFacetChange,
    buildTocTree,
};

/** Common global config for mounting components */
export function createGlobalConfig(overrides = {}) {
    const pinia = createTestPinia();
    const i18n = createTestI18n();
    const config = createTestConfig(overrides.philoConfig);
    const http = overrides.http || createMockHttp();
    const router = createTestRouter(overrides.route);

    return {
        plugins: [pinia, i18n, router],
        provide: {
            $http: http,
            $dbUrl: "/testdb",
            $philoConfig: config,
        },
        mixins: [{ methods: mixinMethods }],
        stubs: {
            "router-link": { template: '<a><slot /></a>', props: ["to"] },
            "router-view": { template: "<div />" },
            ...(overrides.stubs || {}),
        },
        mocks: {
            $philoConfig: config,
            $dbUrl: "/testdb",
            $scrollTo: vi.fn(),
            ...(overrides.mocks || {}),
        },
        directives: {
            scroll: { mounted() {}, unmounted() {} },
        },
    };
}
