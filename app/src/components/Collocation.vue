<template>
    <div id="collocation-container" class="container-fluid mt-4">
        <div v-if="isInvalidCollocationQuery" class="alert alert-warning mx-2 mt-2" role="alert">
            <strong>{{ $t('collocation.invalidQuery') }}</strong>
            <p class="mb-0">{{ $t('collocation.invalidQueryExplanation') }}</p>
        </div>
        <!-- Mobile: Dropdown selector for collocation methods -->
        <div class="d-block d-sm-none mt-3 mx-2">
            <label for="colloc-method-mobile-select" class="form-label fw-bold">
                {{ $t('collocation.methodSelectionTabs') }}
            </label>
            <select class="form-select" id="colloc-method-mobile-select" v-model="mode"
                @change="handleMobileMethodChange" :disabled="isInvalidCollocationQuery"
                :aria-label="$t('collocation.methodSelectionTabs')">
                <option value="frequency">{{ $t("collocation.collocation") }}</option>
                <option value="compare">{{ $t("collocation.compareTo") }}</option>
                <option value="similar">{{ $t("collocation.similarUsage") }}</option>
                <option value="timeSeries">{{ $t("collocation.timeSeries") }}</option>
            </select>
        </div>

        <!-- Desktop: Tab navigation for collocation methods -->
        <div class="d-none d-sm-block mt-3" style="padding: 0 0.5rem">
            <ul class="nav nav-tabs" id="colloc-method-switch" role="tablist"
                :aria-label="$t('collocation.methodSelectionTabs')">
                <li class="nav-item" role="presentation">
                    <button class="nav-link shadow-sm" id="frequency-tab" data-bs-toggle="tab"
                        :class="{ active: mode === 'frequency' }" data-bs-target="#frequency-tab-pane"
                        type="button" role="tab" :aria-selected="mode === 'frequency'"
                        :disabled="isInvalidCollocationQuery" @click="!isInvalidCollocationQuery && setMode('frequency')">
                        {{ $t("collocation.collocation") }}
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link shadow-sm" id="compare-tab" data-bs-toggle="tab"
                        :class="{ active: mode === 'compare' }" data-bs-target="#compare-tab-pane" type="button"
                        role="tab" :aria-selected="mode === 'compare'" :disabled="isInvalidCollocationQuery"
                        @click="!isInvalidCollocationQuery && setMode('compare')">
                        {{ $t("collocation.compareTo") }}
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link shadow-sm" id="similar-tab" data-bs-toggle="tab"
                        :class="{ active: mode === 'similar' }" data-bs-target="#similar-tab-pane" type="button"
                        role="tab" :aria-selected="mode === 'similar'" :disabled="isInvalidCollocationQuery"
                        @click="!isInvalidCollocationQuery && setMode('similar')">
                        {{ $t("collocation.similarUsage") }}
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link shadow-sm" id="time-series-tab" data-bs-toggle="tab"
                        :class="{ active: mode === 'timeSeries' }" data-bs-target="#time-series-tab-pane"
                        type="button" role="tab" :aria-selected="mode === 'timeSeries'"
                        :disabled="isInvalidCollocationQuery" @click="!isInvalidCollocationQuery && setMode('timeSeries')">
                        {{ $t("collocation.timeSeries") }}
                    </button>
                </li>
            </ul>
        </div>
        <results-summary :description="results.description" :filter-list="filterList" :colloc-method="mode"
            v-if="mode === 'frequency'" style="margin-top:0 !important;"></results-summary>
        <div role="region" :aria-label="$t('collocation.collocationResults')">
            <div class="card shadow-sm mx-2 p-3" style="border-top-width: 0;" v-if="mode == 'compare'"
                role="region" :aria-label="$t('collocation.compareTo')">
                <div class="row">
                    <!-- Primary corpus criteria -->
                    <div class="col-12 col-md-6 mb-3 mb-md-0" role="region"
                        :aria-label="$t('collocation.primaryCorpusCriteria')">
                        <h6 class="fw-bold border-bottom pb-2">{{ $t('collocation.primaryCorpusCriteria') }}</h6>
                        <!-- Spacer to align with the comparison side's alert -->
                        <div class="alert alert-info p-1 mb-2 d-none d-md-block invisible" aria-hidden="true">&nbsp;
                        </div>
                        <MetadataFields :fields="metadataDisplay" :input-styles="metadataInputStyle"
                            :choice-values="metadataChoiceValues" :model-value="formData" :checked-values="{}"
                            :selected-values="{}" id-prefix="primary-" />
                    </div>
                    <!-- Comparison corpus criteria -->
                    <div class="col-12 col-md-6 compare-divider" role="region"
                        :aria-label="$t('collocation.comparisonCorpusCriteria')">
                        <h6 class="fw-bold border-bottom pb-2">{{ $t('collocation.comparisonCorpusCriteria') }}</h6>
                        <div class="alert alert-info p-1 mb-2" style="width: fit-content" role="note">
                            {{ $t('collocation.emptySearch') }}</div>
                        <MetadataFields :fields="metadataDisplay" :input-styles="metadataInputStyle"
                            :choice-values="metadataChoiceValues" :model-value="comparedMetadataValues"
                            :checked-values="metadataChoiceChecked" :selected-values="metadataChoiceSelected"
                            :date-type="dateType" :date-range="dateRange" id-prefix="compare-">
                            <template #text-input="{ field }">
                                <label class="btn btn-outline-secondary"
                                    :for="'compare-' + field.value + '-input-filter'">
                                    {{ field.label }}
                                </label>
                                <input type="text" class="form-control" :id="'compare-' + field.value + '-input-filter'"
                                    :name="field.value" :placeholder="field.example"
                                    v-model="comparedMetadataValues[field.value]"
                                    @input="autocompleteOnChange(field.value)" @keydown.down="onArrowDown(field.value)"
                                    @keydown.up="onArrowUp(field.value)" @keyup.enter="onEnter(field.value)"
                                    @keyup.escape="clearAutoCompletePopup" autocomplete="off"
                                    :aria-label="`${$t('collocation.filterBy')} ${field.label}`" />
                                <ul :id="'compare-autocomplete-' + field.value" class="autocomplete-results shadow"
                                    :style="autoCompletePosition(field.value)"
                                    v-if="autoCompleteResults[field.value].length > 0">
                                    <li tabindex="-1" v-for="(result, i) in autoCompleteResults[field.value]"
                                        :key="result" @click="setMetadataResult(result, field.value)"
                                        class="autocomplete-result"
                                        :class="{ 'is-active': i === arrowCounters[field.value] }" v-html="result"></li>
                                </ul>
                            </template>
                        </MetadataFields>
                    </div>
                </div>
                <div class="mt-1">
                    <button type="button" class="btn btn-secondary" style="width: fit-content"
                        @click="getOtherCollocates()">{{
                            $t('collocation.runComparison') }}
                    </button>
                </div>
            </div>
            <div class="card mx-2 p-3" style="border-top-width: 0;" v-if="mode === 'similar'">
                <div class="d-flex align-items-center flex-wrap mt-2">
                    <label for="similarity-field-select" class="me-2 fw-bold">
                        {{ $t("collocation.compareBy") }}
                    </label>
                    <div class="btn-group" style="width: fit-content;" role="group">
                        <button class="btn btn-secondary dropdown-toggle" type="button" id="similarity-field-select"
                            data-bs-toggle="dropdown" aria-expanded="false"
                            :aria-label="`${$t('collocation.compareBy')}: ${similarFieldSelected || $t('collocation.selectField')}`">
                            {{ similarFieldSelected || $t('collocation.selectField') }}
                        </button>
                        <ul class="dropdown-menu" aria-labelledby="similarity-field-select">
                            <li v-for="field in fieldsToCompare" :key="field.value">
                                <button type="button" class="dropdown-item" @click="similarCollocDistributions(field)">
                                    {{ field.label }}
                                </button>
                            </li>
                        </ul>
                    </div>
                    <span class="ms-2">{{ $t("collocation.mostSimilarUsage") }}</span>
                </div>

                <bibliography-criteria class="ms-2 mt-3" :biblio="biblio" :query-report="formData.report"
                    :results-length="resultsLength" :hide-criteria-string="true"></bibliography-criteria>

                <div class="mt-2" style="display: flex; align-items: center;" v-if="similarSearching">
                    <div class="alert alert-info p-1 mb-0 d-inline-block" style="width: fit-content" role="alert"
                        aria-live="polite" aria-atomic="true">
                        {{ similarSearchProgress }}...
                    </div>
                    <progress-spinner class="px-2" :progress="progressPercent" />
                </div>
            </div>
            <div class="card shadow-sm mx-2 p-3" style="border-top-width: 0;" v-if="mode === 'timeSeries'">
                <bibliography-criteria :biblio="biblio" :query-report="formData.report"
                    :results-length="resultsLength"></bibliography-criteria>
                <div class="input-group mt-2">
                    <button class="btn btn-outline-secondary">
                        <label for="year_interval">{{ $t("searchForm.yearInterval") }}</label>
                    </button>
                    <span class="d-inline-flex align-self-center mx-2">{{ $t("searchForm.every") }}</span>
                    <input type="text" class="form-control" name="year_interval" id="year_interval"
                        style="max-width: 50px; text-align: center" v-model="timeSeriesInterval" />
                    <span class="d-inline-flex align-self-center mx-2">{{ $t("searchForm.years") }}</span>
                </div>
                <button type="button" class="btn btn-secondary mt-2" style="width: fit-content"
                    @click="getCollocatesOverTime()">{{
                        $t('collocation.searchEvolution') }}</button>
            </div>

            <!-- Results below -->
            <div class="row my-3 pe-1" style="padding: 0 0.5rem" v-if="resultsLength && mode == 'frequency'">
                <div class="col-12 col-sm-4">
                    <div class="card shadow-sm">
                        <table class="table table-borderless caption-top"
                            aria-label="$t('collocation.collocatesTable')">
                            <caption class="visually-hidden">
                                {{ $t('collocation.collocatesTableCaption') }}
                            </caption>
                            <thead class="table-header">
                                <tr>
                                    <th scope="col" id="collocate-header">{{ $t("collocation.collocate") }}</th>
                                    <th scope="col" id="count-header">{{ $t("collocation.count") }}</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr style="line-height: 1.75rem" v-for="(word, index) in sortedList"
                                    :key="word.collocate" :tabindex="0" @click="collocateClick(word)"
                                    @keydown.enter="collocateClick(word)" @keydown.space.prevent="collocateClick(word)"
                                    :aria-label="`${word.collocate} ${word.count}`">
                                    <td class="text-view" :id="`collocate-${index}`">{{ word.collocate }}</td>
                                    <td :id="`count-${index}`">{{ word.count }}</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
                <div class="col-12 col-sm-8">
                    <div class="card shadow-sm">
                        <word-cloud v-if="mode == 'frequency' && sortedList.length > 0"
                            :word-weights="sortedList" label="frequency" :click-handler="collocateClick"></word-cloud>
                    </div>
                </div>
            </div>
            <div v-if="mode === 'compare'">
                <div class="card shadow-sm mx-2 my-3 p-2" v-if="comparativeSearchStarted">
                    <div class="row mt-2">
                        <div class="col-6" id="primary-biblio">
                            <bibliography-criteria :biblio="biblio" :query-report="formData.report"
                                :results-length="resultsLength"></bibliography-criteria>
                        </div>
                        <div id="other-biblio" class="col-6" style="border-left: solid 1px rgba(0, 0, 0, 0.176)">
                            <bibliography-criteria :biblio="otherBiblio" :query-report="formData.report"
                                :results-length="resultsLength"></bibliography-criteria>
                        </div>
                    </div>
                    <ul class="nav nav-tabs mt-2" style="margin-left: -.5rem" id="colloc-tab" role="tablist"
                        :aria-label="$t('collocation.comparisonResultsTabs')">
                        <li class="nav-item" role="presentation">
                            <button class="nav-link active" style="border-left-width: 0" id="frequent-tab"
                                data-bs-toggle="tab" data-bs-target="#freq-tab-pane" type="button" role="tab"
                                aria-controls="freq-tab-pane" aria-selected="true"
                                :aria-label="$t('collocation.frequentCollocates')">
                                {{ $t('collocation.frequentCollocates') }}
                            </button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="rep-tab" data-bs-toggle="tab" data-bs-target="#rep-tab-pane"
                                type="button" role="tab" aria-controls="rep-tab-pane" aria-selected="false"
                                :aria-label="$t('collocation.overRepresentedCollocates')">
                                {{ $t('collocation.overRepresentedCollocates') }}
                            </button>
                        </li>
                    </ul>
                    <div class="tab-content" id="colloc-tab-content">
                        <div class="tab-pane fade show active" id="freq-tab-pane" role="tabpanel"
                            aria-labelledby="frequent-tab" tabindex="0"
                            :aria-label="$t('collocation.frequentCollocatesPanel')">
                            <div class="row gx-5">
                                <div class="col-6" role="region" :aria-label="$t('collocation.primaryCorpusResults')">
                                    <word-cloud :word-weights="sortedList" label="primary"
                                        :click-handler="collocateClick"></word-cloud>
                                </div>
                                <div class="col-6" style="border-left: solid 1px rgba(0, 0, 0, 0.176)" role="region"
                                    :aria-label="$t('collocation.comparisonCorpusResults')">
                                    <div class="d-flex justify-content-center position-relative"
                                        v-if="compareSearching">
                                        <progress-spinner :progress="progressPercent" :lg="true" />
                                    </div>
                                    <word-cloud v-if="otherCollocates.length > 0" :word-weights="otherCollocates"
                                        label="secondary" :click-handler="otherCollocateClick"></word-cloud>
                                </div>
                            </div>
                        </div>
                        <div class="tab-pane fade" id="rep-tab-pane" role="tabpanel" aria-labelledby="rep-tab"
                            tabindex="0" :aria-label="$t('collocation.overRepresentedCollocatesPanel')">
                            <div class="row gx-5">
                                <div class="col-6" role="region" :aria-label="$t('collocation.overRepresentedResults')">
                                    <word-cloud v-if="overRepresented.length > 0" :word-weights="overRepresented"
                                        :click-handler="collocateClick" label="over"></word-cloud>
                                </div>
                                <div class="col-6" style="border-left: solid 1px rgba(0, 0, 0, 0.176)" role="region"
                                    :aria-label="$t('collocation.underRepresentedResults')">
                                    <div class="d-flex justify-content-center position-relative"
                                        v-if="compareSearching">
                                        <progress-spinner :progress="progressPercent" :lg="true" />
                                    </div>
                                    <word-cloud v-if="underRepresented.length > 0" :word-weights="underRepresented"
                                        :click-handler="otherCollocateClick" label="under"></word-cloud>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div v-if="mode == 'similar'" class="mx-2" style="margin-bottom: 6rem;">
                <div class="card" v-if="similarDistributions.length > 0" role="region"
                    :aria-label="$t('collocation.similarUsageResults')">
                    <h2 class="visually-hidden">{{ $t('collocation.similarUsageResults') }}</h2>
                    <h3 class="sim-dist">
                        {{ $t("collocation.similarUsagePattern") }}
                    </h3>
                    <ul class="list-group list-group-flush" :aria-label="$t('collocation.similarUsagePattern')">
                        <li v-for="(item, index) in similarDistributions" :key="item[0]">
                            <button type="button"
                                class="list-group-item position-relative w-100 text-start border-0 pb-1"
                                style="text-align: justify" @click="similarToComparative(item[0])"
                                @keydown.enter="similarToComparative(item[0])"
                                @keydown.space.prevent="similarToComparative(item[0])"
                                :aria-label="`${$t('collocation.compareTo')} ${item[0]}, ${$t('collocation.count')}: ${item[1]}${item[2] && item[2].length > 0 ? ', ' + $t('collocation.sharedCollocates') + ': ' + item[2].join(', ') : ''}`">
                                <span class="sim-metadata">{{ item[0] }}</span>
                                <span class="badge text-bg-secondary position-absolute" style="right: 1rem; top: 0.5rem"
                                    aria-hidden="true">
                                    {{ item[1] }}
                                </span>
                                <br v-if="item[2] && item[2].length > 0">
                                <small v-if="item[2] && item[2].length > 0"
                                    style="font-size: 0.9em; color: #495057; display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: calc(100% - 4rem)"
                                    aria-hidden="true">
                                    <strong>{{ $t("collocation.sharedCollocates") }}:</strong> {{ item[2].join(', ') }}
                                </small>
                            </button>
                            <hr v-if="index < similarDistributions.length - 1" class="my-0"
                                style="opacity: 1; border-color: rgba(0, 0, 0, 0.125);" aria-hidden="true">
                        </li>
                    </ul>
                </div>
            </div>
            <div v-if="mode == 'timeSeries'" class="mx-2 my-3">
                <div v-if="searching" role="status" aria-live="polite" aria-atomic="true"
                    :aria-label="$t('common.loading')">
                    {{ $t('collocation.similarCollocGatheringMessage') }}...
                </div>
                <div v-if="collocationTimePeriods.length > 0" class="row" role="region"
                    :aria-label="$t('collocation.timeSeriesResults')">
                    <h2 class="visually-hidden">{{ $t('collocation.timeSeriesResults') }}</h2>
                    <div class="col-12 col-md-6" v-for="(period, index) in collocationTimePeriods" :key="period.year"
                        :id="`period-${period.periodYear}`">
                        <article class="card mb-3" v-if="period.done" role="article"
                            :aria-labelledby="`period-${index}-title`">
                            <header class="card-header p-2 d-flex align-items-center">
                                <h3 class="mb-0" :id="`period-${index}-title`">
                                    {{ period.periodYear }}
                                </h3>
                            </header>
                            <div class="btn-group w-100 rounded-0" role="group"
                                :aria-label="`${$t('collocation.viewToggle')} ${period.periodYear}`">
                                <button class="btn btn-sm rounded-0"
                                    :class="period.showDistinctive ? 'btn-secondary active' : 'btn-outline-secondary'"
                                    @click="period.showDistinctive = true" :aria-pressed="period.showDistinctive"
                                    :aria-label="$t('collocation.overRepresentedCollocates')">
                                    {{ $t('collocation.overRepresentedCollocates') }}
                                </button>
                                <button class="btn btn-sm rounded-0"
                                    :class="!period.showDistinctive ? 'btn-secondary active' : 'btn-outline-secondary'"
                                    @click="period.showDistinctive = false" :aria-pressed="!period.showDistinctive"
                                    :aria-label="$t('collocation.frequentCollocates')">
                                    {{ $t('collocation.frequentCollocates') }}
                                </button>
                            </div>
                            <div class="card-body pt-2" role="region"
                                :aria-label="`${period.showDistinctive ? $t('collocation.overRepresentedCollocates') : $t('collocation.frequentCollocates')} ${period.periodYear}`">
                                <word-cloud
                                    :word-weights="period.showDistinctive ? period.distinctive : period.frequent"
                                    :label="period.periodYear"
                                    :click-handler="collocateTimeSeriesClick(period.periodYear)">
                                </word-cloud>
                            </div>
                        </article>
                        <div style="margin-top: 5em; width: 100%; text-align: center" v-else>
                            <p class="mb-1" aria-hidden="true">{{ $t('collocation.gatheringTimeSeriesPeriod') }}...</p>
                            <progress-spinner :message="$t('collocation.gatheringTimeSeriesPeriod')" />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import { computed, inject, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { storeToRefs } from "pinia";
import { useI18n } from "vue-i18n";
import { useMainStore } from "../stores/main";
import {
    buildBiblioCriteria,
    dateRangeHandler,
    debug,
    extractSurfaceFromCollocate,
    paramsFilter,
    paramsToRoute,
} from "../utils.js";
import { useAutocomplete } from "../composables/useAutocomplete";
import BibliographyCriteria from "./BibliographyCriteria";
import MetadataFields from "./MetadataFields.vue";
import ProgressSpinner from "./ProgressSpinner";
import ResultsSummary from "./ResultsSummary";
import WordCloud from "./WordCloud.vue";

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
    currentReport,
    resultsLength,
    searching,
    searchableMetadata,
} = storeToRefs(store);

// ── Autocomplete (composable) ────────────────────────────────────────────────
const comparedMetadataValues = reactive({});
const autocomplete = useAutocomplete({
    http: $http,
    dbUrl: $dbUrl,
    philoConfig,
    metadataValues: comparedMetadataValues,
    route,
});
const {
    autoCompleteResults,
    arrowCounters,
    autoCompletePosition,
    onArrowDown,
    onArrowUp,
    onEnter,
    onChange: autocompleteOnChange,
    setMetadataResult,
    clearAutoCompletePopup,
} = autocomplete;

// ── Shared state ─────────────────────────────────────────────────────────────
const mode = ref("frequency");
const results = ref({});
const filterList = ref([]);
const biblio = ref({});
const sortedList = ref([]);
const collocatesFilePath = ref("");
const relativeFrequencies = ref({});

// ── Metadata helpers ─────────────────────────────────────────────────────────
const metadataDisplay = ref([]);
const metadataInputStyle = ref([]);
const metadataChoiceValues = ref([]);
const metadataChoiceChecked = ref({});
const metadataChoiceSelected = ref({});
const dateType = reactive({});
const dateRange = reactive({});

function buildMetadata(metadata) {
    metadataDisplay.value = metadata.display;
    metadataInputStyle.value = metadata.inputStyle;
    metadataChoiceValues.value = metadata.choiceValues;
    for (const field in metadataInputStyle.value) {
        dateType[field] = "exact";
        dateRange[field] = { start: "", end: "" };
    }
}

// ── Computed ─────────────────────────────────────────────────────────────────
const fieldsToCompare = computed(() => {
    return philoConfig.collocation_fields_to_compare.map(field => ({
        label: philoConfig.metadata_aliases[field] || field,
        value: field,
    }));
});

const isInvalidCollocationQuery = computed(() => {
    if (!formData.value.q) return false;
    let query = formData.value.q.trim();
    // Remove quoted tokens, lemma/attribute queries
    query = query.replace(/"[^"]*"/g, "").replace(/\S+:\S+/g, "");
    // Split by OR operators and filter them out
    let parts = query.split(/\s*(\||OR)\s*/i).filter(p => p.trim() && !p.match(/^\|$|^OR$/i));
    // Remove NOT patterns
    parts = parts.map(p => p.replace(/\s+NOT\s+\S+/gi, ""));
    // Check for multiple consecutive words
    return parts.some(p => p.trim().split(/\s+/).filter(w => w.length > 0).length > 1);
});

// ── Shared collocate handlers ────────────────────────────────────────────────
function collocateCleanup(collocate) {
    if (collocate.surfaceForm.startsWith("lemma:") || collocate.surfaceForm.search(/\w+:.*/) !== -1) {
        return `${formData.value.q} ${collocate.surfaceForm}`;
    }
    return `${formData.value.q} "${collocate.surfaceForm}"`;
}

function concordanceMethod() {
    return formData.value.colloc_within === "n" ? "proxy" : "sentence";
}

function collocateClick(item) {
    router.push(
        paramsToRoute({
            ...formData.value,
            report: "concordance",
            q: collocateCleanup(item),
            method: concordanceMethod(),
            cooc_order: "no",
        })
    );
}

function otherCollocateClick(item) {
    router.push(
        paramsToRoute({
            ...comparedMetadataValues,
            report: "concordance",
            q: collocateCleanup(item),
            method: concordanceMethod(),
            cooc_order: "no",
        })
    );
}

// ── URL + mode coordination ──────────────────────────────────────────────────
function getNonEmptyComparedMetadata() {
    const out = {};
    for (const [field, value] of Object.entries(comparedMetadataValues)) {
        if (value && value !== "") {
            out[`compare_${field}`] = value;
        }
    }
    return out;
}

// Clear in place to preserve the reactive reference (other code holds it).
function clearComparedMetadata() {
    for (const key of Object.keys(comparedMetadataValues)) {
        delete comparedMetadataValues[key];
    }
}

function restoreComparedMetadataFromUrl() {
    clearComparedMetadata();
    for (const [key, value] of Object.entries(route.query)) {
        if (key.startsWith("compare_")) {
            comparedMetadataValues[key.substring(8)] = value;
        }
    }
}

function updateCollocationUrl() {
    const urlParams = {
        ...formData.value,
        collocation_method: mode.value,
    };
    if (mode.value === "similar" && similarFieldSelected.value) {
        urlParams.similarity_by = similarFieldSelected.value;
        delete urlParams.time_series_interval;
    }
    if (mode.value === "timeSeries") {
        urlParams.time_series_interval = timeSeriesInterval.value;
        delete urlParams.similarity_by;
    }
    if (mode.value === "compare") {
        Object.assign(urlParams, getNonEmptyComparedMetadata());
        delete urlParams.similarity_by;
        delete urlParams.time_series_interval;
    }
    router.push(paramsToRoute(urlParams));
}

function setMode(newMode, { updateUrl = true } = {}) {
    mode.value = newMode;
    if (updateUrl) updateCollocationUrl();
}

function handleMobileMethodChange() {
    setMode(mode.value, { updateUrl: false });
}

// ── Primary fetch (shared by frequency / compare / similar entry paths) ──────
function updateCollocation() {
    if (isInvalidCollocationQuery.value) {
        searching.value = false;
        return;
    }
    $http.get(`${$dbUrl}/reports/collocation.py`, { params: paramsFilter(formData.value) })
        .then((response) => {
            resultsLength.value = response.data.results_length;
            filterList.value = response.data.filter_list;
            collocatesFilePath.value = response.data.file_path;
            searching.value = false;
            if (resultsLength.value) {
                sortedList.value = extractSurfaceFromCollocate(response.data.collocates);
                runPostFetchModeAction();
            }
        })
        .catch((error) => {
            searching.value = false;
            debug({ $options: { name: "collocation-report" } }, error);
        });
}

// After the primary fetch completes, similar/compare modes need a follow-up fetch.
function runPostFetchModeAction() {
    mode.value = route.query.collocation_method || "frequency";
    if (mode.value === "similar") {
        setMode("similar", { updateUrl: false });
        similarCollocDistributions({ value: route.query.similarity_by });
    } else if (mode.value === "compare") {
        getOtherCollocates();
    }
}

function fetchResults() {
    if (isInvalidCollocationQuery.value) {
        searching.value = false;
        return;
    }
    relativeFrequencies.value = {};
    searching.value = true;
    overRepresented.value = [];
    underRepresented.value = [];
    comparativeSearchStarted.value = false;
    similarDistributions.value = [];
    collocationTimePeriods.value = [];
    similarFieldSelected.value = "";
    updateCollocation();
}

// ── Mode: compare ────────────────────────────────────────────────────────────
const otherCollocates = ref([]);
const otherBiblio = ref({});
const overRepresented = ref([]);
const underRepresented = ref([]);
const compareSearching = ref(false);
const comparativeSearchStarted = ref(false);
const wholeCorpus = ref(false);

function getOtherCollocates() {
    wholeCorpus.value = Object.keys(comparedMetadataValues).length === 0;
    setMode("compare");
    Object.assign(
        comparedMetadataValues,
        dateRangeHandler(metadataInputStyle.value, dateRange, dateType, comparedMetadataValues),
    );
    const params = {
        q: formData.value.q,
        colloc_filter_choice: formData.value.colloc_filter_choice,
        colloc_within: formData.value.colloc_within,
        filter_frequency: formData.value.filter_frequency,
        q_attribute: formData.value.q_attribute || "",
        q_attribute_value: formData.value.q_attribute_value || "",
        ...comparedMetadataValues,
    };
    comparativeSearchStarted.value = true;
    compareSearching.value = true;
    otherCollocates.value = [];
    $http.get(`${$dbUrl}/reports/collocation.py`, { params: paramsFilter(params) })
        .then((response) => {
            compareSearching.value = false;
            if (response.data.results_length) {
                otherCollocates.value = extractSurfaceFromCollocate(response.data.collocates);
                comparativeCollocations(response.data.file_path);
            }
        })
        .catch((error) => {
            compareSearching.value = false;
            debug({ $options: { name: "collocation-report" } }, error);
        });
}

function comparativeCollocations(otherFilePath) {
    comparativeSearchStarted.value = true;
    Object.assign(
        comparedMetadataValues,
        dateRangeHandler(metadataInputStyle.value, dateRange, dateType, comparedMetadataValues),
    );
    otherBiblio.value = buildBiblioCriteria(philoConfig, comparedMetadataValues, comparedMetadataValues);
    overRepresented.value = [];
    underRepresented.value = [];
    $http.get(`${$dbUrl}/scripts/comparative_collocations.py`, {
        params: {
            primary_file_path: collocatesFilePath.value,
            other_file_path: otherFilePath,
            whole_corpus: wholeCorpus.value,
        },
    }).then((response) => {
        overRepresented.value = extractSurfaceFromCollocate(response.data.top);
        underRepresented.value = extractSurfaceFromCollocate(response.data.bottom);
        relativeFrequencies.value = { top: overRepresented.value, bottom: underRepresented.value };
    }).catch((error) => {
        debug({ $options: { name: "collocation-report" } }, error);
    });
}

// ── Mode: similar ────────────────────────────────────────────────────────────
const similarDistributions = ref([]);
const cachedDistributions = ref("");
const similarFieldSelected = ref("");
const similarSearchProgress = ref("");
const similarSearching = ref(false);

function similarCollocDistributions(field) {
    similarFieldSelected.value = field.value;
    similarSearching.value = true;
    similarSearchProgress.value = t("collocation.similarCollocGatheringMessage");
    similarDistributions.value = [];
    setMode("similar");
    $http.get(`${$dbUrl}/reports/collocation.py`, {
        params: {
            q: formData.value.q,
            colloc_filter_choice: formData.value.colloc_filter_choice,
            colloc_within: formData.value.colloc_within,
            filter_frequency: formData.value.filter_frequency,
            map_field: field.value,
            q_attribute: formData.value.q_attribute || "",
            q_attribute_value: formData.value.q_attribute_value || "",
        },
    }).then((response) => {
        getMostSimilarCollocDistribution(response.data.file_path);
    }).catch((error) => {
        similarSearching.value = false;
        debug({ $options: { name: "collocation-report" } }, error);
    });
}

function getMostSimilarCollocDistribution(filePath) {
    similarSearchProgress.value = t("collocation.similarCollocCompareMessage");
    $http.get(`${$dbUrl}/scripts/get_similar_collocate_distributions.py`, {
        params: {
            primary_file_path: collocatesFilePath.value,
            file_path: filePath,
        },
    }).then((response) => {
        similarDistributions.value = response.data.similar || [];
        cachedDistributions.value = filePath;
        similarSearching.value = false;
    }).catch((error) => {
        similarSearching.value = false;
        debug({ $options: { name: "collocation-report" } }, error);
    });
}

function similarToComparative(field) {
    $http.get(`${$dbUrl}/scripts/get_collocate_distribution.py`, {
        params: { file_path: cachedDistributions.value, field },
    }).then((response) => {
        // Reset compare metadata before applying the chosen similar result —
        // any leftover values from a previous compare run would otherwise
        // silently scope the comparison.
        clearComparedMetadata();
        comparedMetadataValues[similarFieldSelected.value] = field;
        mode.value = "compare";
        otherCollocates.value = extractSurfaceFromCollocate(response.data.collocates);
        wholeCorpus.value = false;
        comparativeCollocations(response.data.file_path);
    }).catch((error) => {
        debug({ $options: { name: "collocation-report" } }, error);
    });
}

// ── Mode: timeSeries ─────────────────────────────────────────────────────────
const timeSeriesInterval = ref(10);
const collocationTimePeriods = ref([]);
const progressPercent = ref(0);

function getCollocatesOverTime() {
    collocationTimePeriods.value = [];
    searching.value = true;
    setMode("timeSeries");
    const interval = parseInt(timeSeriesInterval.value);
    const params = {
        ...paramsFilter(formData.value),
        time_series_interval: interval,
        map_field: "year",
    };

    // Expand year range by one interval on each side for context
    if (formData.value.year) {
        const yearParts = formData.value.year.split("-");
        if (yearParts.length === 2) {
            const [startYear, endYear] = yearParts;
            if (startYear && endYear) {
                params.year = `${parseInt(startYear) - interval}-${parseInt(endYear) + interval}`;
            } else if (startYear) {
                params.year = `${parseInt(startYear) - interval}-`;
            } else if (endYear) {
                params.year = `-${parseInt(endYear) + interval}`;
            }
        } else if (yearParts[0]) {
            const year = parseInt(yearParts[0]);
            params.year = `${year - interval}-${year + interval}`;
        }
    }

    $http.get(`${$dbUrl}/reports/collocation.py`, { params }).then((response) => {
        searching.value = false;
        collocationTimeSeries(response.data.file_path, 0);
    }).catch((error) => {
        searching.value = false;
        debug({ $options: { name: "collocation-report" } }, error);
    });
}

function collocationTimeSeries(filePath, periodNumber) {
    collocationTimePeriods.value[periodNumber] = { year: periodNumber, done: false };
    $http.get(`${$dbUrl}/scripts/collocation_time_series.py`, {
        params: {
            file_path: filePath,
            year_interval: timeSeriesInterval.value,
            period_number: periodNumber,
        },
    }).then((response) => {
        if (response.data.period) {
            const period = response.data.period;
            const year = period.year;
            const interval = parseInt(timeSeriesInterval.value);
            collocationTimePeriods.value[periodNumber] = {
                year,
                frequent: extractSurfaceFromCollocate(period.collocates.frequent || []),
                distinctive: extractSurfaceFromCollocate(period.collocates.distinctive || []),
                periodYear: `${year}-${year + interval}`,
                showDistinctive: true,
                done: true,
            };
        }
        if (!response.data.done) {
            collocationTimeSeries(filePath, periodNumber + 1);
        }
    }).catch((error) => {
        debug({ $options: { name: "collocation-report" } }, error);
    });
}

function collocateTimeSeriesClick(period) {
    return (item) => {
        const method = formData.value.colloc_within === "n" ? "proxy_unordered" : "sentence_unordered";
        router.push(
            paramsToRoute({
                ...formData.value,
                report: "concordance",
                q: collocateCleanup(item),
                method,
                year: period,
            })
        );
    };
}

// ── Watchers ─────────────────────────────────────────────────────────────────
// View-only params don't affect the primary collocation results — they only
// switch which panel is rendered or feed mode-specific secondary fetches.
// Changing only these (e.g. clicking a tab) must NOT trigger a re-search.
function isViewOnlyParam(key) {
    return (
        key === "collocation_method" ||
        key === "similarity_by" ||
        key === "time_series_interval" ||
        key.startsWith("compare_")
    );
}

function shouldRefetchOnQueryChange(newQuery, oldQuery) {
    const allKeys = new Set([...Object.keys(newQuery), ...Object.keys(oldQuery)]);
    for (const key of allKeys) {
        if (newQuery[key] === oldQuery[key]) continue;
        if (!isViewOnlyParam(key)) return true;
    }
    return false;
}

watch(
    () => route.query,
    (newQuery, oldQuery = {}) => {
        if (route.name !== "collocation") return;

        // Sync mode + biblio cheaply on every route change.
        const newMode = newQuery.collocation_method || "frequency";
        if (newMode !== mode.value) mode.value = newMode;
        if (newMode === "compare") restoreComparedMetadataFromUrl();
        biblio.value = buildBiblioCriteria(philoConfig, route.query, formData.value);

        // Bail out if only the view changed (tab click, similarity dropdown
        // selection, time-series interval, etc.). Primary results stay valid;
        // mode-specific secondary fetches are user-triggered via their buttons.
        if (!shouldRefetchOnQueryChange(newQuery, oldQuery)) return;

        // Real search params changed — refetch the right path for the active mode.
        if (newMode === "timeSeries") {
            getCollocatesOverTime();
        } else {
            fetchResults();  // .then() runs runPostFetchModeAction for similar/compare
        }
    }
);

watch(searchableMetadata, (newVal) => buildMetadata(newVal), { deep: true });

// ── Lifecycle: register and tear down the global click listener ──────────────
// (Fixes the leak in the previous Options-API implementation, which never
// removed this listener — every Collocation mount accumulated another one.)
function onDocumentClick() {
    clearAutoCompletePopup();
}
onMounted(() => {
    document.addEventListener("click", onDocumentClick);
});
onBeforeUnmount(() => {
    document.removeEventListener("click", onDocumentClick);
});

// ── Initial dispatch (replaces created()) ────────────────────────────────────
formData.value.report = "collocation";
currentReport.value = "collocation";
buildMetadata(searchableMetadata.value);
biblio.value = buildBiblioCriteria(philoConfig, route.query, formData.value);
mode.value = route.query.collocation_method || "frequency";

switch (mode.value) {
    case "frequency":
        fetchResults();
        break;
    case "similar":
        setMode("similar", { updateUrl: false });
        fetchResults();
        break;
    case "timeSeries":
        setMode("timeSeries", { updateUrl: false });
        timeSeriesInterval.value = route.query.time_series_interval || 10;
        getCollocatesOverTime();
        break;
    case "compare":
        setMode("compare", { updateUrl: false });
        restoreComparedMetadataFromUrl();
        fetchResults();
        break;
}
</script>

<style lang="scss" scoped>
@use "../assets/styles/theme.module.scss" as theme;

th {
    font-variant: small-caps;
    background-color: theme.$card-header-color !important;
    color: white !important;
    border-color: theme.$card-header-color !important;
}

.table tbody tr {
    cursor: pointer;
    transition: all 0.2s ease;
    border: 1px solid transparent;
}

.table tbody tr:hover {
    transform: scale(1.01);
    background-color: rgba(theme.$link-color, 0.15) !important;
    border-color: rgba(theme.$link-color, 0.3);
    box-shadow: inset 0 0 8px rgba(theme.$link-color, 0.1);
    cursor: pointer;
    z-index: 1;
}

.table tbody tr:active {
    transform: scale(0.98);
}

.table tbody tr:hover {
    transform: scale(1.01) !important;
    background-color: rgba(theme.$link-color, 0.15) !important;
    border: 1px solid rgba(theme.$link-color, 0.3) !important;
    box-shadow: inset 0 0 8px rgba(theme.$link-color, 0.1) !important;
    cursor: pointer !important;
    z-index: 1 !important;
}

.table tbody tr:hover td {
    background-color: rgba(theme.$link-color, 0.15) !important;
    color: inherit !important;
}

.table tbody tr:active {
    transform: scale(0.98);
}

.table tbody tr:focus {
    background-color: rgba(theme.$button-color, 0.15) !important;
    color: theme.$button-color !important;
    outline: 2px solid theme.$button-color;
    outline-offset: -2px;
}

/* Ensure cells inherit the row colors */
.table tbody tr:hover td {
    color: inherit;
}

.table tbody tr:focus td {
    color: inherit;
}

#description {
    position: relative;
}

#export-results {
    position: absolute;
    right: 0;
    padding: 0.125rem 0.25rem;
    font-size: 0.8rem !important;
}

.cloud-word {
    display: inline-block;
    padding: 5px;
    cursor: pointer;
    line-height: initial;
}

.table th,
.table td {
    padding: 0.45rem 0.75rem;
}

#filter-list {
    position: absolute;
    z-index: 100;
}

#filter-list .list-group-item {
    border-width: 0px;
    padding: 0.1rem;
}

#close-filter-list {
    width: fit-content;
    float: right;
    padding: 0 0.2rem;
    position: absolute;
    right: 0;
}

.input-group {
    max-width: 700px;
    width: 100%;
}

input[type="text"]:focus {
    opacity: 1;
}

::placeholder {
    opacity: 0.4;
}

input:focus::placeholder {
    opacity: 0;
}

#colloc-tab button {
    font-variant: small-caps;
    font-size: 1rem;
}

.nav-link {
    border-bottom: 1px solid #dee2e6;
    background-color: #fff;
    color: theme.$link-color;
}

.nav-link.active {
    color: theme.$link-color;
    font-weight: bold;
}

#colloc-tab .nav-link:hover {
    background-color: rgba(theme.$link-color, 0.1);
    border-color: theme.$link-color;
}

#colloc-method-switch .nav-link:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.sim-dist {
    text-align: center;
    font-variant: small-caps;
    font-size: 1rem;
    color: #fff;
    background-color: theme.$link-color;
    padding: 0.5rem;
    margin-bottom: 0;
}

.sim-metadata {
    display: inline-block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: calc(100% - 4rem);
    vertical-align: bottom;
    padding-bottom: 0.15rem;
    color: theme.$link-color;
    font-weight: 500;
}

.colloc-cloud-title {
    text-align: center;
    background: theme.$link-color;
    color: #fff;
}

.card-header {
    text-align: center;

    h3,
    h4 {
        width: 100%;
        font-variant: small-caps;
        font-size: 1rem;
    }
}

.btn-group {
    border: none;
    border-bottom: solid 1px rgba(0, 0, 0, 0.176);
}

.btn-sm {
    font-size: 0.8rem;
    padding: 0.4rem 0.5rem;
    flex: 1;
}

.badge {
    font-size: 0.75rem;
}

@media (min-width: 768px) {
    .compare-divider {
        border-left: solid 1px rgba(0, 0, 0, 0.176);
    }
}

.alert-warning {
    background-color: rgba(theme.$card-header-color, 0.05);
    border-color: theme.$card-header-color;
    color: #000;
}

// Similarity search list items - same hover effect as facets
.list-group-item {
    cursor: pointer;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    transform: scale(1);
}

.list-group-item:hover {
    transform: scale(1.01);
    background-color: rgba(theme.$link-color, 0.15) !important;
    border-color: rgba(theme.$link-color, 0.3) !important;
    box-shadow: inset 0 0 8px rgba(theme.$link-color, 0.1);
    z-index: 1;
}

.list-group-item:active {
    transform: scale(0.98);
    background-color: rgba(theme.$link-color, 0.2) !important;
}

.list-group-item:hover .badge {
    background-color: theme.$link-color !important;
    transform: scale(1.1);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

.list-group-item:focus {
    outline: 2px solid theme.$link-color;
    outline-offset: -2px;
    box-shadow: inset 0 0 0 0.2rem rgba(theme.$link-color, 0.25);
    z-index: 2;
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
</style>