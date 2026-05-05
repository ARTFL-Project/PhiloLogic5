<template>
    <div>
        <h1 class="visually-hidden">{{ $t("searchForm.searchInterface") }}</h1>
        <div class="card shadow" style="border: transparent">
            <form @submit.prevent @reset="onReset" @keyup.enter="onSubmit()" role="search">
                <div id="form-body">
                    <div id="initial-form">
                        <!-- Mobile: Dropdown selector for report types -->
                        <div class="d-block d-sm-none px-3 pt-3">
                            <label for="report-type-mobile-select" class="form-label fw-bold">
                                {{ $t('searchForm.selectReportType') }}
                            </label>
                            <select class="form-select" id="report-type-mobile-select" v-model="currentReport"
                                @change="reportChange(currentReport)"
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
                                class="btn btn-secondary rounded-0" :class="{ active: currentReport == searchReport }"
                                :aria-pressed="currentReport == searchReport">
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
                                            @input="onChange('q')" @keyup.down="onArrowDown('q')"
                                            @keyup.up="onArrowUp('q')" @keyup.enter="onEnter('q')"
                                            @keyup.escape="clearAutoCompletePopup" autocomplete="off" />

                                        <ul id="autocomplete-q" class="autocomplete-results shadow"
                                            :style="autoCompletePosition('q')" v-if="autoCompleteResults.q.length > 0"
                                            role="listbox" :aria-label="$t('searchForm.autocompleteResults')">
                                            <li tabindex="-1" v-for="(result, i) in autoCompleteResults.q" :key="result"
                                                @click="setResult(result, 'q')" class="autocomplete-result"
                                                :class="{ 'is-active': i === arrowCounters.q }" v-html="result"
                                                role="option" :aria-selected="i === arrowCounters.q"></li>
                                        </ul>
                                        <button class="btn btn-secondary" id="button-search" @click="onSubmit()">
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
                                    @input="onChange('head')"
                                    @keydown.down="onArrowDown(metadataDisplay[headIndex].value)"
                                    @keydown.up="onArrowUp(metadataDisplay[headIndex].value)"
                                    @keyup.enter="onEnter(metadataDisplay[headIndex].value)"
                                    @keyup.escape="clearAutoCompletePopup" autocomplete="off" />
                                <ul :id="'autocomplete-' + metadataDisplay[headIndex].value"
                                    class="autocomplete-results shadow"
                                    :style="autoCompletePosition(metadataDisplay[headIndex].value)"
                                    v-if="autoCompleteResults[metadataDisplay[headIndex].value].length > 0"
                                    role="listbox" :aria-label="$t('searchForm.autocompleteResults')">
                                    <li tabindex="-1"
                                        v-for="(result, i) in autoCompleteResults[metadataDisplay[headIndex].value]"
                                        :key="result" @click="setResult(result, metadataDisplay[headIndex].value)"
                                        class="autocomplete-result" :class="{
                                            'is-active': i === arrowCounters[metadataDisplay[headIndex].value],
                                        }" v-html="result" role="option"
                                        :aria-selected="i === arrowCounters[metadataDisplay[headIndex].value]"></li>
                                </ul>
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
                                    v-model="formData.approximate_ratio" :disabled="!approximateSelected"
                                    :aria-label="$t('searchForm.approximateMatchLabel')">
                                    <option v-for="value in approximateValues" :key="value.value" :value="value.value">
                                        {{ value.text }}
                                    </option>
                                </select>
                                <!-- Checkbox to determine if we follow co-occurrence word order -->
                                <div class="form-check form-switch" id="co-occurrence-order" style="height: 31px"
                                    v-if="currentReport != 'collocation'">
                                    <input class="form-check-input" type="checkbox" id="co-occurrence-order-input"
                                        :checked="coocOrder" @change="toggleCoocOrder" />
                                    <label class=" form-check-label" for="co-occurrence-order-input">{{
                                        $t("searchForm.coOccurrenceWordOrder") }}</label>
                                </div>
                                <div class="input-group mb-4" v-if="currentReport != 'collocation'">
                                    <button class="btn btn-outline-secondary" type="button" tabindex="-1">
                                        {{ $t("searchForm.searchCoOccurrences") }}</button><select class="form-select"
                                        style="width: fit-content; max-width: fit-content" v-model="formData.method"
                                        :aria-label="$t('searchForm.coOccurrenceMethodLabel')">
                                        <option v-for="value in methodOptions" :key="value.value" :value="value.value">
                                            {{ value.text }}
                                        </option>
                                    </select>
                                    <button class="btn btn-outline-secondary" type="button" id="method-arg-label"
                                        tabindex="-1"
                                        v-if="formData.method == 'proxy' || formData.method == 'exact_cooc'">
                                        {{ $t("searchForm.howMany") }}?
                                    </button>
                                    <input class="form-control" type="text" name="method_arg" id="method-arg"
                                        aria-labelledby="method-arg-label"
                                        v-if="formData.method == 'proxy' || formData.method == 'exact_cooc'"
                                        v-model="formData.method_arg" />
                                    <span class="input-group-text ms-0"
                                        v-if="formData.method == 'proxy' || formData.method == 'exact_cooc'">{{
                                            $t("searchForm.wordsSentence") }}</span>
                                </div>
                                <div class="mt-1" id="collocation-params" v-if="currentReport == 'collocation'"
                                    role="group" aria-labelledby="collocation-params-heading">
                                    <h2 id="collocation-params-heading">{{ $t("searchForm.collocationParams") }}:</h2>
                                    <div class="form-check">
                                        <input class="form-check-input" type="radio" name="colloc_within"
                                            id="collocSentence" value="sent" checked v-model="formData.colloc_within">
                                        <label class="form-check-label" for="collocSentence">
                                            {{ $t("searchForm.collocatesWithinSentence") }}
                                        </label>
                                    </div>
                                    <div class="form-check mt-1">
                                        <input class="form-check-input" type="radio" name="colloc_within"
                                            id="collocWithinN" value="n" v-model="formData.colloc_within">
                                        <label class="form-check-label" for="collocWithinN">
                                            {{ $t("searchForm.collocatesWithin") }} <input type="number"
                                                name="method_arg" class="form-control form-control-sm" id="collocNWords"
                                                :aria-label="$t('searchForm.collocNWordsLabel')" style="display: inline-block; width: 70px; text-align: center; height: 22px !important;
min-height: initial; min-height: fit-content;" v-model="formData.method_arg"> {{ $t("searchForm.words") }}
                                        </label>
                                    </div>
                                    <div class="mt-2">
                                        <button v-if="wordAttributes" type="button" class="btn btn-outline-secondary"
                                            tabindex="-1"
                                            style="border-top-right-radius: 0; border-bottom-right-radius: 0">
                                            {{ $t("searchForm.filterCollocate") }}
                                        </button>
                                        <div class="btn-group d-inline-block" role="group">
                                            <button class="btn btn-secondary dropdown-toggle"
                                                style="border-top-left-radius: 0; border-bottom-left-radius: 0"
                                                type="button" id="attribute-selector'" data-bs-toggle="dropdown"
                                                aria-expanded="false">
                                                {{ collocFilteringSelected.text || collocationOptions[0].text }}
                                            </button>
                                            <ul class="dropdown-menu" aria-labelledby="attribute-selector">
                                                <li v-for="option in collocationOptions" :key="option.value">
                                                    <a class="dropdown-item"
                                                        @click="collocFilteringSelected = option">{{
                                                            option.text }}</a>
                                                </li>
                                            </ul>
                                        </div>
                                        <div class="dropdown d-inline-block ms-2"
                                            v-if="collocFilteringSelected.value == 'attribute'">
                                            <button class="btn btn-secondary dropdown-toggle"
                                                style="border-top-left-radius: 0; border-bottom-left-radius: 0"
                                                type="button" id="attribute-selector'" data-bs-toggle="dropdown"
                                                aria-expanded="false">
                                                {{ $philoConfig.word_property_aliases[attributeSelected] ||
                                                    attributeSelected || $t("searchForm.selectAttributeType") }}
                                            </button>
                                            <ul class="dropdown-menu" aria-labelledby="attribute-selector">
                                                <li v-for="(_, attribute) in wordAttributes" :key="attribute">
                                                    <a class="dropdown-item" @click="attributeSelected = attribute">
                                                        {{ $philoConfig.word_property_aliases[attribute] || attribute }}
                                                    </a>
                                                </li>
                                            </ul>
                                        </div>
                                        <div class="dropdown d-inline-block ms-2"
                                            v-if="collocFilteringSelected.value == 'attribute' && attributeSelected.length > 0">
                                            <button class="btn btn-secondary dropdown-toggle" type="button"
                                                id="attributeValues" data-bs-toggle="dropdown" aria-expanded="false">
                                                {{ wordAttributeSelected.toUpperCase() ||
                                                    $t("searchForm.selectAttributeValue")
                                                }}
                                            </button>
                                            <ul class="dropdown-menu" aria-labelledby="attributeValues">
                                                <li v-for="attributeValue in wordAttributes[attributeSelected]"
                                                    :key="attributeValue">
                                                    <a class="dropdown-item"
                                                        @click="wordAttributeSelected = attributeValue">{{
                                                            attributeValue }}</a>
                                                </li>
                                            </ul>
                                        </div>
                                        <div class="input-group d-inline ms-2" style="width: fit-content"
                                            v-if="collocFilteringSelected.value == 'frequency'">
                                            <button class="btn btn-outline-secondary" style="height: fit-content"
                                                id="filter-frequency-label">
                                                {{ $t("searchForm.wordFiltering") }}
                                            </button>
                                            <input type="text" class="form-control d-inline-block" id="filter-frequency"
                                                name="filter_frequency" placeholder="100"
                                                v-model="formData.filter_frequency"
                                                aria-labelledby="filter-frequency-label"
                                                style="width: 60px; text-align: center" />
                                        </div>
                                    </div>
                                </div>
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
                                                @input="onChange(field.value)"
                                                @keydown.down="onArrowDown(field.value)"
                                                @keydown.up="onArrowUp(field.value)"
                                                @keyup.enter="onEnter(field.value)"
                                                @keyup.escape="clearAutoCompletePopup" autocomplete="off" />
                                            <ul :id="'autocomplete-' + field.value"
                                                class="autocomplete-results shadow"
                                                :style="autoCompletePosition(field.value)"
                                                v-if="autoCompleteResults[field.value].length > 0">
                                                <li tabindex="-1"
                                                    v-for="(result, i) in autoCompleteResults[field.value]"
                                                    :key="result" @click="setResult(result, field.value)"
                                                    class="autocomplete-result"
                                                    :class="{ 'is-active': i === arrowCounters[field.value] }"
                                                    v-html="result"></li>
                                            </ul>
                                        </template>
                                    </MetadataFields>
                                </div>
                                <div class="mt-1" id="time-series-params" v-if="currentReport === 'time_series'"
                                    role="group" aria-labelledby="time-series-params-heading">
                                    <h2 id="time-series-params-heading">{{ $t("searchForm.timeSeriesParams") }}:</h2>
                                    <div class="input-group mt-1 pb-2">
                                        <button class="btn btn-outline-secondary" tabindex="-1">{{
                                            $t("searchForm.dateRange")
                                            }}</button>
                                        <label for="start_date" class="d-inline-flex align-self-center mx-2">{{
                                            $t("searchForm.dateFrom") }}</label>
                                        <input type="text" class="form-control" name="start_date" id="start_date"
                                            style="max-width: 65px; text-align: center" v-model="formData.start_date" />
                                        <label for="end_date" class="d-inline-flex align-self-center mx-2">{{
                                            $t("searchForm.dateTo") }}</label>
                                        <input type="text" class="form-control" name="end_date" id="end_date"
                                            style="max-width: 65px; text-align: center" v-model="formData.end_date" />
                                    </div>
                                    <div class="input-group">
                                        <button class="btn btn-outline-secondary" id="year-interval-label"
                                            tabindex="-1">
                                            {{ $t("searchForm.yearInterval") }}
                                        </button>
                                        <span class="d-inline-flex align-self-center mx-2">{{ $t("searchForm.every")
                                            }}</span>
                                        <input type="text" class="form-control" name="year_interval" id="year_interval"
                                            aria-labelledby="year-interval-label"
                                            style="max-width: 50px; text-align: center"
                                            v-model="formData.year_interval" />
                                        <span class="d-inline-flex align-self-center mx-2">{{ $t("searchForm.years")
                                            }}</span>
                                    </div>
                                </div>
                                <div class="input-group mt-4" v-if="currentReport === 'aggregation'">
                                    <button class="btn btn-outline-secondary">{{ $t("searchForm.groupResultsBy")
                                        }}</button>
                                    <select class="form-select" :aria-label="$t('searchForm.groupResultsByLabel')"
                                        style="max-width: fit-content" v-model="formData.group_by">
                                        <option v-for="aggregationOption in aggregationOptions"
                                            :key="aggregationOption.text" :value="aggregationOption.value">
                                            {{ aggregationOption.text }}
                                        </option>
                                    </select>
                                </div>
                                <div role="group" aria-labelledby="display-options-heading"
                                    v-if="['concordance', 'bibliography'].includes(currentReport)">
                                    <h2 class="mt-3" id="display-options-heading">
                                        {{ $t("searchForm.displayOptions") }}:
                                    </h2>
                                    <div class="input-group pb-2"
                                        v-if="['concordance', 'bibliography'].includes(currentReport)">
                                        <label class="btn btn-outline-secondary" for="sort-results-select">
                                            {{ $t("searchForm.sortResultsBy") }}
                                        </label>
                                        <select class="form-select" style="max-width: fit-content"
                                            id="sort-results-select" :aria-label="$t('searchForm.sortResultsByLabel')"
                                            v-model="formData.sort_by">
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
import { computed, inject, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { storeToRefs } from "pinia";
import { useI18n } from "vue-i18n";
import { useMainStore } from "../stores/main";
import { copyObject, dateRangeHandler, paramsToRoute } from "../utils.js";
import { useAutocomplete } from "../composables/useAutocomplete";
import MetadataFields from "./MetadataFields.vue";  // eslint-disable-line no-unused-vars
import SearchTips from "./SearchTips";  // eslint-disable-line no-unused-vars
import ProgressSpinner from "./ProgressSpinner";  // eslint-disable-line no-unused-vars

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
    metadataUpdate,
    searchableMetadata,
} = storeToRefs(store);

// ── Autocomplete (composable) ────────────────────────────────────────────────
const metadataValues = reactive({});
const autocomplete = useAutocomplete({
    http: $http,
    dbUrl: $dbUrl,
    philoConfig,
    metadataValues,
    route,
    onSelect(field, value) {
        store.updateFormDataField({ key: field, value });
    },
});
// Add "q" to the shared reactive objects for query term autocomplete
autocomplete.autoCompleteResults.q = [];
autocomplete.arrowCounters.q = -1;
const {
    autoCompleteResults,
    arrowCounters,
    autoCompletePosition,
    onArrowDown,
    onArrowUp,
    clearAutoCompletePopup,
    onChange: metadataOnChange,
    setMetadataResult,
} = autocomplete;

// ── Local state ──────────────────────────────────────────────────────────────
const dictionary = philoConfig.dictionary;
const metadataInputStyle = philoConfig.metadata_input_style;
const reports = philoConfig.search_reports;
const wordAttributes = philoConfig.word_attributes;
const approximateValues = [
    { text: t("searchForm.similarity", { n: 90 }), value: "90" },
    { text: t("searchForm.similarity", { n: 80 }), value: "80" },
];
const methodOptions = [
    { text: t("searchForm.within"), value: "proxy" },
    { text: t("searchForm.withinExactly"), value: "exact_cooc" },
    { text: t("common.sameSentence"), value: "sentence" },
];
const collocationOptions = ref([
    { text: t("searchForm.mostFrequentTerms"), value: "frequency" },
    { text: t("searchForm.stopwords"), value: "stopwords" },
]);
const aggregationOptions = philoConfig.aggregation_config.map((f) => ({
    text: philoConfig.metadata_aliases[f.field] || f.field.charAt(0).toUpperCase() + f.field.slice(1),
    value: f.field,
}));

const headIndex = ref(0);
const formOpen = ref(false);
const approximateSelected = ref(false);
const coocOrder = ref(true);
const metadataDisplay = ref([]);
const metadataChoiceValues = reactive({});
const metadataChoiceChecked = reactive({});
const metadataChoiceSelected = reactive({});
const selectedSortValues = ref("rowid");  // eslint-disable-line no-unused-vars
const showTips = ref(false);
const queryTermTyped = ref(route.query.q || formData.value.q || "");
const dateType = reactive({});
const dateRange = reactive({});
const attributeSelected = ref("");
const wordAttributeSelected = ref("");
const collocFilteringSelected = ref({ text: "", value: "" });

// Internal handle (not reactive — just a setTimeout id)
let qTimeout = null;
let onDocumentClick = null;

// ── Computed ─────────────────────────────────────────────────────────────────
const statFieldSelected = computed(() => getLoadedStatField());  // eslint-disable-line no-unused-vars

const sortValues = computed(() => {
    const values = [{ value: "rowid", text: "select" }];
    for (const fields of philoConfig.concordance_biblio_sorting) {
        const label = fields.map((f) => philoConfig.metadata_aliases[f] || f);
        values.push({ text: label.join(", "), value: fields });
    }
    return values;
});

const metadataDisplayFiltered = computed(() => {
    let metadataInForm;
    if (currentReport.value === "time_series") {
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

function getLoadedStatField() {
    const queryParam = formData.value?.group_by;
    if (!queryParam) return "";
    return (
        philoConfig.metadata_aliases[queryParam] ||
        queryParam.charAt(0).toUpperCase() + queryParam.slice(1)
    );
}

// ── Toggle handlers ──────────────────────────────────────────────────────────
function toggleCoocOrder() {           // eslint-disable-line no-unused-vars
    coocOrder.value = !coocOrder.value;
}

function toggleApproximate() {         // eslint-disable-line no-unused-vars
    approximateSelected.value = !approximateSelected.value;
    store.updateFormDataField({
        key: "approximate_ratio",
        value: approximateSelected.value ? "90" : "",
    });
}

function toggleForm() {
    formOpen.value = !formOpen.value;
}

function selectApproximate(approximateValue) {     // eslint-disable-line no-unused-vars
    store.updateFormDataField({ key: "approximate_ratio", value: approximateValue });
}

// ── Submit / reset / report change ───────────────────────────────────────────
function onSubmit() {                  // eslint-disable-line no-unused-vars
    formData.value.report = currentReport.value;
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

    if (currentReport.value === "collocation" && formData.value.colloc_within === "sent") {
        store.updateFormDataField({ key: "method_arg", value: "" });
    }
    store.updateFormDataField({
        key: "colloc_filter_choice",
        value: collocFilteringSelected.value.value,
    });
    if (
        formData.value.colloc_filter_choice === "frequency" ||
        formData.value.colloc_filter_choice === "stopwords"
    ) {
        attributeSelected.value = "";
        wordAttributeSelected.value = "";
    }

    router.push(
        paramsToRoute({
            ...formData.value,
            ...metadataValues,
            ...metadataChoices,
            ...metadataSelected,
            q: queryTermTyped.value.trim(),
            start: "",
            end: "",
            byte: "",
            start_date: formData.value.start_date,
            end_date: formData.value.end_date,
            q_attribute: attributeSelected.value,
            q_attribute_value: wordAttributeSelected.value,
            method_arg: formData.value.method_arg,
        })
    );
}

function onReset() {                   // eslint-disable-line no-unused-vars
    store.resetFormDataToDefaults();
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

function reportChange(report) {        // eslint-disable-line no-unused-vars
    formData.value.report = report === "landing_page" ? philoConfig.search_reports[0] : report;
    if (report === "collocation") {
        store.updateFormDataField({
            key: "colloc_filter_choice",
            value: philoConfig.stopwords.length > 0 ? "stopwords" : "frequency",
        });
    }
    currentReport.value = report;
    if (!formOpen.value) toggleForm();
}

// ── Autocomplete handlers (q + metadata fan-out) ────────────────────────────
function onChange(field) {              // eslint-disable-line no-unused-vars
    if (field !== "q") {
        metadataOnChange(field);
        return;
    }
    if (!philoConfig.autocomplete.includes(field)) return;
    if (qTimeout) clearTimeout(qTimeout);
    qTimeout = setTimeout(() => {
        const currentQueryTerm = route.query.q;
        if (
            queryTermTyped.value.replace('"', "").length > 1 &&
            queryTermTyped.value !== currentQueryTerm
        ) {
            $http
                .get(`${$dbUrl}/scripts/autocomplete_term.py`, {
                    params: { term: queryTermTyped.value },
                })
                .then((response) => {
                    autoCompleteResults.q = response.data;
                });
        }
    }, 200);
}

function onEnter(field) {               // eslint-disable-line no-unused-vars
    const result = autoCompleteResults[field][arrowCounters[field]];
    setResult(result, field);
}

function setResult(inputString, field) {
    if (field !== "q") {
        setMetadataResult(inputString, field);
        return;
    }
    if (typeof inputString !== "undefined") {
        const inputGroup = inputString.replace(/<[^>]+>/g, "").split(/(\s*\|\s*|\s*OR\s*|\s+|\s*NOT\s*)/);
        let lastInput = inputGroup.pop();
        if (lastInput.match(/"/)) {
            if (lastInput.startsWith('"')) lastInput = lastInput.slice(1);
            if (lastInput.endsWith('"')) lastInput = lastInput.slice(0, -1);
        }
        // word property autocomplete (no quotes) vs regular term (quoted)
        queryTermTyped.value = lastInput.includes(":")
            ? `${inputGroup.join("")}${lastInput}`
            : `${inputGroup.join("")}"${lastInput.trim()}"`;
    }
    autoCompleteResults.q = [];
    arrowCounters.q = -1;
}

// ── Watchers ─────────────────────────────────────────────────────────────────
watch(() => route.fullPath, updateInputData);

watch(metadataUpdate, (metadata) => {
    for (const field of metadata) {
        const style = philoConfig.metadata_input_style[field];
        if (style === "text") {
            metadataValues[field] = metadata[field];
        } else if (style === "dropdown") {
            metadataChoiceSelected[field] = metadata[field];
        } else if (style === "checkbox") {
            metadataChoiceChecked[field] = metadata[field].split(" | ");
        }
    }
    for (const m of philoConfig.metadata) {
        autoCompleteResults[m] = [];
        arrowCounters[m] = -1;
    }
});

// Two-way sync for cooc_order / approximate between local UI state and store
watch(coocOrder, (newValue) => {
    store.updateFormDataField({ key: "cooc_order", value: newValue ? "yes" : "no" });
});
watch(
    () => formData.value.cooc_order,
    (newValue) => { coocOrder.value = newValue === "yes"; },
    { immediate: true }
);

watch(approximateSelected, (newValue) => {
    store.updateFormDataField({ key: "approximate", value: newValue ? "yes" : "no" });
});
watch(
    () => formData.value.approximate,
    (newValue) => { approximateSelected.value = newValue === "yes"; },
    { immediate: true }
);

// ── Lifecycle ────────────────────────────────────────────────────────────────
onMounted(() => {
    // Initialize queryTermTyped with formData.q if not set from route
    if (!queryTermTyped.value && formData.value?.q) {
        queryTermTyped.value = formData.value.q;
    }

    // Global click listener to clear autocomplete popup. Captured in a named
    // function (not an arrow-on-the-fly) so we can remove it on unmount.
    nextTick(() => {
        onDocumentClick = () => clearAutoCompletePopup();
        document.addEventListener("click", onDocumentClick);
    });
});

onBeforeUnmount(() => {
    if (onDocumentClick) {
        document.removeEventListener("click", onDocumentClick);
    }
    if (qTimeout) clearTimeout(qTimeout);
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

if (Object.keys(philoConfig.word_attributes).length > 0) {
    // Place word attribute option at the second position
    collocationOptions.value.splice(1, 0, {
        text: t("searchForm.selectAttribute"),
        value: "attribute",
    });
}

if (formData.value.colloc_filter_choice && formData.value.colloc_filter_choice.length > 0) {
    for (const collocFilter of collocationOptions.value) {
        if (collocFilter.value === formData.value.colloc_filter_choice) {
            collocFilteringSelected.value = collocFilter;
            if (collocFilteringSelected.value.value === "attribute") {
                attributeSelected.value =
                    philoConfig.word_property_aliases[route.query.q_attribute] ||
                    route.query.q_attribute;
                wordAttributeSelected.value = route.query.q_attribute_value;
            }
            break;
        }
    }
}

if (collocFilteringSelected.value.value === "") {
    const fallback = philoConfig.stopwords.length > 0 ? "stopwords" : "frequency";
    collocFilteringSelected.value = collocationOptions.value.find((o) => o.value === fallback);
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

.autocomplete {
    position: relative;
}

.autocomplete-results {
    padding: 0;
    margin: 3px 0 0 15px;
    border: 1px solid #eeeeee;
    border-top-width: 0px;
    max-height: 216px;
    overflow-y: scroll;
    width: 267px;
    position: absolute;
    left: 0;
    background-color: #fff;
    z-index: 100;
    top: 34px;
    font-size: 1.2rem;
}

.autocomplete-result {
    list-style: none;
    text-align: left;
    padding: 4px 12px;
    cursor: pointer;
    font-size: 1.2rem;
}

.autocomplete-result:hover,
.is-active {
    background-color: #ddd;
    color: black;
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