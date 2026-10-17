<template>
    <div>
        <h1 class="visually-hidden">{{ $t("searchForm.searchInterface") }}</h1>
        <div class="card shadow" style="border: transparent">
            <form @submit.prevent="onSubmit()" @reset="onReset" role="search">
                <!-- for the fields with autocomplete: their description, and the number of suggestions shown -->
                <span :id="instructionsId" class="visually-hidden">{{ $t("searchForm.autocompleteInstructions") }}</span>
                <span class="visually-hidden" role="status">{{ openList
                    ? $t("searchForm.autocompleteCount", { term: openList.term }, openList.count) : "" }}</span>
                <div id="form-body">
                    <div id="initial-form">
                        <!-- Mobile: Dropdown selector for report types -->
                        <div class="d-block d-sm-none px-3 pt-3">
                            <label for="report-type-mobile-select" class="form-label fw-bold">
                                {{ $t('searchForm.selectReportType') }}
                            </label>
                            <select class="form-select" id="report-type-mobile-select" v-model="selectedReport"
                                @change="reportChange(selectedReport)"
                                :aria-label="$t('searchForm.selectReportType')">
                                <option v-for="searchReport in reports" :key="searchReport" :value="searchReport">
                                    {{ $t(`searchForm.${searchReport}`) }}
                                </option>
                            </select>
                        </div>

                        <!-- Desktop: Button group for report types -->
                        <div class="btn-group d-none d-sm-flex" role="group" id="report" style="width: 100%; top: -1px">
                            <button type="button" :id="searchReport" v-for="searchReport in reports"
                                @click="reportChange(searchReport)" :key="searchReport"
                                class="btn btn-secondary rounded-0" :class="{ active: selectedReport == searchReport }"
                                :aria-pressed="selectedReport == searchReport">
                                <span v-if="searchReport != 'kwic'">{{ $t(`searchForm.${searchReport}`) }}</span>
                                <span v-else><span class="d-md-inline d-sm-none">{{ $t(`searchForm.${searchReport}`)
                                        }}</span><span class="d-md-none">{{ $t("searchForm.shortKwic") }}</span></span>
                            </button>
                        </div>
                        <div id="search_terms_container" class="p-3">
                            <div class="row" id="search_terms">
                                <div class="cols-12 cols-md-8">
                                    <div class="input-group" id="q-group">
                                        <button class="btn btn-outline-secondary" type="button" id="search-terms-label"
                                            tabindex="-1">
                                            {{ $t("searchForm.searchTerms") }}
                                        </button>
                                        <button class="btn btn-outline-info" type="button" data-bs-toggle="modal"
                                            data-bs-target="#search-tips" @mouseover="showTips = true"
                                            @mouseleave="showTips = false"
                                            :aria-label="$t('searchForm.searchTipsButton')">
                                            <span v-if="!showTips">?</span>
                                            <span v-if="showTips">{{ $t("searchForm.tips") }}</span>
                                        </button>
                                        <input type="text" class="form-control" id="query-term-input"
                                            aria-labelledby="search-terms-label" v-model="queryTermTyped"
                                            v-bind="comboboxAttrs('q')" @input="onChange('q')"
                                            @keydown="onKeydown('q', $event)" @blur="closeAutocomplete('q')"
                                            autocomplete="off" />

                                        <AutocompleteList v-if="autoCompleteResults.q.length > 0" :id="listId('q')"
                                            :results="autoCompleteResults.q" :active-index="arrowCounters.q"
                                            :style="autoCompletePosition('q')" @toggle="toggle('q', $event)" />
                                        <button type="submit" class="btn btn-secondary" id="button-search">
                                            {{ $t("searchForm.search") }}
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div id="head-search-container" class="px-3 pt-1 pb-3" v-if="dictionary">
                            <div class="input-group" id="head-group">
                                <button type="button" class="btn btn-outline-secondary"
                                    :id="metadataDisplay[headIndex].value + '-label'">
                                    {{ metadataDisplay[headIndex].label }}
                                </button>
                                <input type="text" class="form-control"
                                    :id="metadataDisplay[headIndex].value + '-input'"
                                    :aria-labelledby="metadataDisplay[headIndex].value + '-label'"
                                    :name="metadataDisplay[headIndex].value"
                                    :placeholder="metadataDisplay[headIndex].example" v-model="metadataValues.head"
                                    v-bind="comboboxAttrs('head')" @input="onChange('head')"
                                    @keydown="onKeydown('head', $event)" @blur="closeAutocomplete('head')"
                                    autocomplete="off" />
                                <AutocompleteList v-if="autoCompleteResults.head?.length > 0" :id="listId('head')"
                                    :results="autoCompleteResults.head" :active-index="arrowCounters.head"
                                    :style="autoCompletePosition('head')" @toggle="toggle('head', $event)" />
                            </div>
                        </div>
                        <div id="search-buttons">
                            <div class="input-group">
                                <button type="reset" id="reset_form" class="btn btn-outline-secondary">
                                    {{ $t("searchForm.clear") }}
                                </button>
                                <button type="button" id="show-search-form" class="btn btn-secondary"
                                    @click="toggleForm()">
                                    <span v-if="!formOpen">{{ $t("searchForm.showSearchOptions") }}</span>
                                    <span v-else>{{ $t("searchForm.hideSearchOptions") }}</span>
                                </button>
                            </div>
                        </div>
                    </div>
                    <transition name="slide-fade">
                        <div id="search-elements" v-if="formOpen" class="ps-3 pe-3 pb-3 shadow" role="region"
                            :aria-label="$t('searchForm.advancedSearchOptions')">
                            <div class="mt-1" role="group" aria-labelledby="search-terms-params-heading">
                                <h2 id="search-terms-params-heading">{{
                                    $t("searchForm.searchTermsParameters") }}:</h2>
                                <div class="form-check form-switch form-check-inline" id="approximate"
                                    style="height: 31px">
                                    <input class="form-check-input" type="checkbox" id="approximate-input"
                                        :checked="approximateSelected" @change="toggleApproximate" />
                                    <label class="form-check-label" for="approximate-input">{{
                                        $t("searchForm.approximateMatch") }}
                                    </label>
                                </div>

                                <select class="form-select form-select-sm d-inline-block"
                                    style="max-width: fit-content; margin-left: 0.5rem"
                                    v-model="draft.approximate_ratio" :disabled="!approximateSelected"
                                    :aria-label="$t('searchForm.approximateMatchLabel')">
                                    <option v-for="value in approximateValues" :key="value.value" :value="value.value">
                                        {{ value.text }}
                                    </option>
                                </select>
                                <!-- Checkbox to determine if we follow co-occurrence word order -->
                                <div class="form-check form-switch" id="co-occurrence-order" style="height: 31px"
                                    v-if="selectedReport != 'collocation'">
                                    <input class="form-check-input" type="checkbox" id="co-occurrence-order-input"
                                        :checked="coocOrder" @change="toggleCoocOrder" />
                                    <label class=" form-check-label" for="co-occurrence-order-input">{{
                                        $t("searchForm.coOccurrenceWordOrder") }}</label>
                                </div>
                                <div class="input-group mb-4" v-if="selectedReport != 'collocation'">
                                    <button class="btn btn-outline-secondary" type="button" tabindex="-1">
                                        {{ $t("searchForm.searchCoOccurrences") }}</button><select class="form-select"
                                        style="width: fit-content; max-width: fit-content" v-model="draft.method"
                                        :aria-label="$t('searchForm.coOccurrenceMethodLabel')">
                                        <option v-for="value in methodOptions" :key="value.value" :value="value.value">
                                            {{ value.text }}
                                        </option>
                                    </select>
                                    <button class="btn btn-outline-secondary" type="button" id="method-arg-label"
                                        tabindex="-1"
                                        v-if="draft.method == 'proxy' || draft.method == 'exact_cooc'">
                                        {{ $t("searchForm.howMany") }}?
                                    </button>
                                    <input class="form-control" type="text" name="method_arg" id="method-arg"
                                        aria-labelledby="method-arg-label"
                                        v-if="draft.method == 'proxy' || draft.method == 'exact_cooc'"
                                        v-model="draft.method_arg" />
                                    <span class="input-group-text ms-0"
                                        v-if="draft.method == 'proxy' || draft.method == 'exact_cooc'">{{
                                            $t("searchForm.wordsSentence") }}</span>
                                </div>
                                <CollocationOptions v-if="selectedReport == 'collocation'" :form="draft" />
                                <div role="group" aria-labelledby="filter-by-field-heading">
                                    <h2 class="mt-2" id="filter-by-field-heading">{{ $t("searchForm.filterByField") }}:
                                    </h2>
                                    <MetadataFields
                                        :fields="metadataDisplayFiltered"
                                        :input-styles="metadataInputStyle"
                                        :choice-values="metadataChoiceValues"
                                        :model-value="metadataValues"
                                        :checked-values="metadataChoiceChecked"
                                        :selected-values="metadataChoiceSelected"
                                        :date-type="dateType"
                                        :date-range="dateRange">
                                        <template #text-input="{ field }">
                                            <button type="button" class="btn btn-outline-secondary" tabindex="-1"
                                                :id="field.value + '-label'">
                                                {{ field.label }}
                                            </button>
                                            <input type="text" class="form-control"
                                                :aria-labelledby="field.value + '-label'"
                                                :id="field.value + 'input-filter'" :name="field.value"
                                                :placeholder="field.example"
                                                v-model="metadataValues[field.value]"
                                                v-bind="comboboxAttrs(field.value)" @input="onChange(field.value)"
                                                @keydown="onKeydown(field.value, $event)"
                                                @blur="closeAutocomplete(field.value)" autocomplete="off" />
                                            <AutocompleteList v-if="autoCompleteResults[field.value].length > 0"
                                                :id="listId(field.value)" :results="autoCompleteResults[field.value]"
                                                :active-index="arrowCounters[field.value]"
                                                :style="autoCompletePosition(field.value)"
                                                @toggle="toggle(field.value, $event)" />
                                        </template>
                                    </MetadataFields>
                                </div>
                                <div class="mt-1" id="time-series-params" v-if="selectedReport === 'time_series'"
                                    role="group" aria-labelledby="time-series-params-heading">
                                    <h2 id="time-series-params-heading">{{ $t("searchForm.timeSeriesParams") }}:</h2>
                                    <div class="input-group mt-1 pb-2">
                                        <button type="button" class="btn btn-outline-secondary" tabindex="-1">{{
                                            $t("searchForm.dateRange")
                                            }}</button>
                                        <label for="start_date" class="d-inline-flex align-self-center mx-2">{{
                                            $t("searchForm.dateFrom") }}</label>
                                        <input type="text" class="form-control" name="start_date" id="start_date"
                                            style="max-width: 65px; text-align: center" v-model="draft.start_date" />
                                        <label for="end_date" class="d-inline-flex align-self-center mx-2">{{
                                            $t("searchForm.dateTo") }}</label>
                                        <input type="text" class="form-control" name="end_date" id="end_date"
                                            style="max-width: 65px; text-align: center" v-model="draft.end_date" />
                                    </div>
                                    <div class="input-group">
                                        <button type="button" class="btn btn-outline-secondary" id="year-interval-label"
                                            tabindex="-1">
                                            {{ $t("searchForm.yearInterval") }}
                                        </button>
                                        <span class="d-inline-flex align-self-center mx-2">{{ $t("searchForm.every")
                                            }}</span>
                                        <input type="text" class="form-control" name="year_interval" id="year_interval"
                                            aria-labelledby="year-interval-label"
                                            style="max-width: 50px; text-align: center"
                                            v-model="draft.year_interval" />
                                        <span class="d-inline-flex align-self-center mx-2">{{ $t("searchForm.years")
                                            }}</span>
                                    </div>
                                </div>
                                <div class="input-group mt-4" v-if="selectedReport === 'aggregation'">
                                    <button type="button" class="btn btn-outline-secondary">{{
                                        $t("searchForm.groupResultsBy")
                                        }}</button>
                                    <select class="form-select" :aria-label="$t('searchForm.groupResultsByLabel')"
                                        style="max-width: fit-content" v-model="draft.group_by">
                                        <option v-for="aggregationOption in aggregationOptions"
                                            :key="aggregationOption.text" :value="aggregationOption.value">
                                            {{ aggregationOption.text }}
                                        </option>
                                    </select>
                                </div>
                                <div role="group" aria-labelledby="display-options-heading"
                                    v-if="['concordance', 'bibliography'].includes(selectedReport)">
                                    <h2 class="mt-3" id="display-options-heading">
                                        {{ $t("searchForm.displayOptions") }}:
                                    </h2>
                                    <div class="input-group pb-2"
                                        v-if="['concordance', 'bibliography'].includes(selectedReport)">
                                        <label class="btn btn-outline-secondary" for="sort-results-select">
                                            {{ $t("searchForm.sortResultsBy") }}
                                        </label>
                                        <select class="form-select" style="max-width: fit-content"
                                            id="sort-results-select" :aria-label="$t('searchForm.sortResultsByLabel')"
                                            v-model="draft.sort_by">
                                            <option v-for="sortValue in sortValues" :key="sortValue.value"
                                                :value="sortValue.value">
                                                {{ sortValue.text }}
                                            </option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </transition>
                </div>
            </form>
        </div>
        <div class="d-flex justify-content-center position-relative" v-if="searching">
            <div style="position: absolute; z-index: 50; top: 30px">
                <ProgressSpinner :xl="true" />
            </div>
        </div>
        <div class="modal fade" id="search-tips" tabindex="-1" aria-labelledby="search-tips-title" aria-hidden="true"
            role="dialog">
            <SearchTips></SearchTips>
        </div>
    </div>
</template>

<script setup>
import { computed, inject, onMounted, reactive, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { storeToRefs } from "pinia";
import { useI18n } from "vue-i18n";
import { useMainStore } from "../stores/main";
import { copyObject, dateRangeHandler, defaultCommonWords, paramsToRoute } from "../utils.js";
import { useAutocomplete } from "../composables/useAutocomplete";
import AutocompleteList from "./AutocompleteList.vue";
import CollocationOptions from "./CollocationOptions.vue";
import MetadataFields from "./MetadataFields.vue";
import SearchTips from "./SearchTips";
import ProgressSpinner from "./ProgressSpinner";

// ── Injects, route, router, store, i18n ──────────────────────────────────────
const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const philoConfig = inject("$philoConfig");
const route = useRoute();
const router = useRouter();
const { t } = useI18n();
const store = useMainStore();
const {
    formData,
    searching,
    currentReport,
    searchableMetadata,
} = storeToRefs(store);

// ── Autocomplete (composable) ────────────────────────────────────────────────
const metadataValues = reactive({});
const queryTermTyped = ref(route.query.q || formData.value.q || "");
const {
    autoCompleteResults,
    arrowCounters,
    autoCompletePosition,
    onChange,
    onKeydown,
    toggle,
    closeAutocomplete,
    clearAutoCompletePopup,
    comboboxAttrs,
    listId,
    instructionsId,
    openList,
} = useAutocomplete({
    http: $http,
    dbUrl: $dbUrl,
    philoConfig,
    metadataValues,
    route,
    queryTerm: queryTermTyped,
});

// ── The form's search ────────────────────────────────────────────────────────
// The form edits a search of its own until it is run: the search shown (the store's formData, from the URL) stays as
// it is, for the results and their links. The form shows it again when the URL changes, as the values the page sets
// in it (a time series' dates).
const draft = reactive({ ...formData.value });
const selectedReport = ref(currentReport.value); // the report the form runs, the one shown until another is picked
let shown = { ...formData.value };

function showSearch() {
    for (const key of Object.keys(draft)) {
        if (!(key in formData.value)) delete draft[key];
    }
    Object.assign(draft, formData.value);
    shown = { ...formData.value };
}

watch(
    formData,
    (search) => {
        for (const [key, value] of Object.entries(search)) {
            if (shown[key] !== value) draft[key] = value;
        }
        shown = { ...search };
    },
    { deep: true }
);
watch(currentReport, (report) => {
    selectedReport.value = report;
});

// ── Local state ──────────────────────────────────────────────────────────────
const dictionary = philoConfig.dictionary;
const metadataInputStyle = philoConfig.metadata_input_style;
const reports = philoConfig.search_reports;
// Labels as computed, or as message keys, follow the language chosen
const approximateValues = computed(() => [
    { text: t("searchForm.similarity", { n: 90 }), value: "90" },
    { text: t("searchForm.similarity", { n: 80 }), value: "80" },
]);
const methodOptions = computed(() => [
    { text: t("searchForm.within"), value: "proxy" },
    { text: t("searchForm.withinExactly"), value: "exact_cooc" },
    { text: t("common.sameSentence"), value: "sentence" },
]);
const aggregationOptions = philoConfig.aggregation_config.map((f) => ({
    text: philoConfig.metadata_aliases[f.field] || f.field.charAt(0).toUpperCase() + f.field.slice(1),
    value: f.field,
}));

const headIndex = ref(0);
const formOpen = ref(false);
// The switches, as the search has them ("yes", "no")
const approximateSelected = computed({
    get: () => draft.approximate === "yes",
    set: (on) => {
        draft.approximate = on ? "yes" : "no";
    },
});
const coocOrder = computed({
    get: () => draft.cooc_order === "yes",
    set: (on) => {
        draft.cooc_order = on ? "yes" : "no";
    },
});
const metadataDisplay = ref([]);
const metadataChoiceValues = reactive({});
const metadataChoiceChecked = reactive({});
const metadataChoiceSelected = reactive({});
const showTips = ref(false);
const dateType = reactive({});
const dateRange = reactive({});

// ── Computed ─────────────────────────────────────────────────────────────────
const sortValues = computed(() => {
    const values = [{ value: "rowid", text: t("searchForm.selectSort") }];
    for (const fields of philoConfig.concordance_biblio_sorting) {
        const label = fields.map((f) => philoConfig.metadata_aliases[f] || f);
        values.push({ text: label.join(", "), value: fields });
    }
    return values;
});

const metadataDisplayFiltered = computed(() => {
    let metadataInForm;
    if (selectedReport.value === "time_series") {
        metadataInForm = metadataDisplay.value.filter(
            (f) => philoConfig.time_series_year_field !== f.value
        );
    } else {
        metadataInForm = copyObject(metadataDisplay.value);
    }
    if (!dictionary) return metadataInForm;
    const localMetadataDisplay = copyObject(metadataInForm);
    localMetadataDisplay.splice(headIndex.value, 1);
    return localMetadataDisplay;
});

// ── Form sync helpers ────────────────────────────────────────────────────────
function updateInputData() {
    showSearch();
    queryTermTyped.value = formData.value.q || "";
    for (const field of philoConfig.metadata) {
        const style = philoConfig.metadata_input_style[field];
        if (style === "text") {
            metadataValues[field] = formData.value[field];
        } else if (style === "dropdown") {
            metadataChoiceSelected[field] = formData.value[field];
        } else if (style === "checkbox") {
            metadataChoiceChecked[field] = formData.value[field].split(" | ");
        }
    }
}

// ── Toggle handlers ──────────────────────────────────────────────────────────
function toggleCoocOrder() {
    coocOrder.value = !coocOrder.value;
}

function toggleApproximate() {
    approximateSelected.value = !approximateSelected.value;
    draft.approximate_ratio = approximateSelected.value ? "90" : "";
}

function toggleForm() {
    formOpen.value = !formOpen.value;
}

// ── Submit / reset / report change ───────────────────────────────────────────
function onSubmit() {
    draft.report = selectedReport.value;
    formOpen.value = false;
    const metadataChoices = Object.fromEntries(
        Object.entries(metadataChoiceChecked).map(([key, val]) => [key, val.join(" | ")])
    );
    const metadataSelected = Object.fromEntries(
        Object.entries(metadataChoiceSelected).map(([key, val]) => [key, val])
    );
    // dateRangeHandler mutates metadataValues in place — no reassignment.
    dateRangeHandler(metadataInputStyle, dateRange, dateType, metadataValues);
    clearAutoCompletePopup();

    if (selectedReport.value === "collocation" && draft.colloc_within === "sent") {
        draft.method_arg = "";
    }
    if (selectedReport.value === "collocation" && !draft.colloc_filter_choice) {
        draft.colloc_filter_choice = defaultCommonWords(philoConfig);
    }

    router.push(
        paramsToRoute({
            ...draft,
            ...metadataValues,
            ...metadataChoices,
            ...metadataSelected,
            q: queryTermTyped.value.trim(),
            start: "",
            end: "",
            byte: "",
            start_date: draft.start_date,
            end_date: draft.end_date,
            // the filter of links made before collocate filters could be combined, which the form now has
            q_attribute: "",
            q_attribute_value: "",
            method_arg: draft.method_arg,
        })
    );
}

function onReset() {
    Object.assign(draft, store.defaultFields);
    for (const field of philoConfig.metadata) {
        metadataValues[field] = "";
    }
    queryTermTyped.value = "";
    for (const m in metadataInputStyle) {
        if (metadataInputStyle[m] === "date" || metadataInputStyle[m] === "int") {
            dateType[m] = "exact";
            dateRange[m] = { start: "", end: "" };
        }
    }
}

function reportChange(report) {
    draft.report = report === "landing_page" ? philoConfig.search_reports[0] : report;
    if (report === "collocation") {
        draft.colloc_filter_choice = defaultCommonWords(philoConfig);
    }
    selectedReport.value = report;
    if (!formOpen.value) toggleForm();
}

// ── Watchers ─────────────────────────────────────────────────────────────────
watch(() => route.fullPath, updateInputData);

// ── Lifecycle ────────────────────────────────────────────────────────────────
onMounted(() => {
    // Initialize queryTermTyped with formData.q if not set from route
    if (!queryTermTyped.value && formData.value?.q) {
        queryTermTyped.value = formData.value.q;
    }
});

// ── Initial dispatch (replaces created()) ────────────────────────────────────
for (const metadataField of philoConfig.metadata) {
    const metadataObj = {
        label: philoConfig.metadata_aliases[metadataField] ||
            metadataField[0].toUpperCase() + metadataField.slice(1),
        value: metadataField,
        example: philoConfig.search_examples[metadataField],
    };
    metadataDisplay.value.push(metadataObj);

    if (formData.value[metadataField] !== "") {
        const style = philoConfig.metadata_input_style[metadataField];
        if (["text", "date", "int"].includes(style)) {
            metadataValues[metadataField] = formData.value[metadataField];
        } else if (style === "checkbox") {
            metadataChoiceChecked[metadataField] = formData.value[metadataField].split(" | ");
        }
    }
    if (metadataField === "head") {
        headIndex.value = metadataDisplay.value.length - 1;
    }
    if (philoConfig.metadata_input_style[metadataField] === "dropdown") {
        metadataChoiceSelected[metadataField] = formData.value[metadataField] || "";
    }
}

for (const metadata in philoConfig.metadata_choice_values) {
    metadataChoiceValues[metadata] = philoConfig.metadata_choice_values[metadata].map((c) => ({
        text: c.label,
        value: c.value,
    }));
}

for (const m in metadataInputStyle) {
    dateType[m] = "exact";
    dateRange[m] = { start: "", end: "" };
}

searchableMetadata.value = {
    display: metadataDisplay.value,
    choiceValues: metadataChoiceValues,
    inputStyle: metadataInputStyle,
};
</script>

<style scoped>
input[type="text"] {
    opacity: 1;
}

.input-group,
#search-elements h2 {
    width: fit-content;
}

#report .btn {
    font-variant: small-caps;
    font-size: 1rem !important;
}

.dico-margin {
    margin-top: 210px !important;
}

#search-elements {
    position: absolute;
    z-index: 50;
    background-color: #fff;
    width: 100%;
    left: 0;
}

#search-elements.dico {
    margin-top: 168px;
}

#search-elements>h2 {
    margin-top: 15px;
    margin-bottom: 15px;
}

#search-elements .btn-outline-secondary,
#q-group .btn-outline-secondary,
#head-group .btn {
    pointer-events: none;
    /*disable hover effect*/
}

#search_terms .input-group,
#search-elements .input-group,
#head-group {
    max-width: 700px;
    width: 100%;
}

#report label {
    font-size: 1.1em;
    font-variant: small-caps;
    text-transform: capitalize;
}

.search_box {
    width: 250px;
    vertical-align: middle;
}

#search_field {
    font-weight: 400;
}

#more_options {
    width: 200px;
}

#more_options:hover {
    cursor: pointer;
}

#search_terms_container {
    padding-top: 15px;
    padding-bottom: 15px;
}

#search_terms,
#head-search-container {
    text-align: center;
}

#head-search-container {
    margin-top: -10px;
    padding-bottom: 15px;
}

.no-example {
    display: none;
}

#method,
.metadata_fields,
#collocation-options>div,
#time-series-options>div {
    margin-top: 10px;
}

#initial-form .btn-primary.active {
    box-shadow: 0px 0px;
    border-top: 0px;
}

#tip-text {
    display: none;
}

#tip-btn:hover #tip-text {
    display: inline;
}

#tip-btn:hover #tip {
    display: none;
}

#search_terms .row {
    text-align: left;
    padding-left: 20px;
    padding-right: 20px;
}

#search-buttons {
    position: absolute;
    top: 3rem;
    right: 1rem;
    z-index: 51;
}

.radio-group {
    border-radius: 0;
}

.radio-btn-group span:last-of-type label {
    border-top-right-radius: 0.25rem;
    border-bottom-right-radius: 0.25rem;
}

@media (max-width: 992px) {
    #search-buttons {
        position: initial;
        margin-bottom: 1rem;
        margin-left: 0;
        margin-right: 0;
        text-align: center;
    }

    #search-elements {
        margin-top: -1rem;
    }
}

@media (max-width: 768px) {
    #collocation-options .row {
        margin-left: -15px;
    }
}

/*Dico layout changes*/

#search_elements {
    border-top-width: 0px;
}

.select {
    font-size: inherit;
    position: relative;
    display: inline-block;
    width: 100%;
    text-align: center;
}

.select select {
    outline: none;
    -webkit-appearance: none;
    -moz-appearance: none;
    appearance: none;
    display: block;
    padding: 6px 12px;
    margin: 0;
    transition: border-color 0.2s;
    border: 1px solid #ccc;
    border-radius: 0px;
    background: #fff;
    color: #555;
    line-height: normal;
    font-family: inherit;
    font-size: inherit;
    line-height: inherit;
    width: 100%;
}

::placeholder {
    opacity: 0.4;
}

input:focus::placeholder {
    opacity: 0;
}

.code-block {
    font-family: monospace;
    font-size: 120%;
    background-color: #ededed;
    padding: 0px 4px;
}

.date-range {
    display: inline-block;
    width: auto;
}

.slide-fade-enter-active,
.slide-fade-leave-active {
    transition: all 0.3s ease-out;
}

.slide-fade-enter-from,
.slide-fade-leave-to {
    transform: translateY(-30px);
    opacity: 0;
}

h5 {
    font-variant: small-caps;
    font-weight: 700;
    font-size: 1.15rem;
}

#search-terms-params-heading,
#filter-by-field-heading,
#collocation-params-heading,
#display-options-heading,
#time-series-params-heading {
    font-variant: small-caps;
    font-weight: 700;
    font-size: 1.15rem;
}
</style>