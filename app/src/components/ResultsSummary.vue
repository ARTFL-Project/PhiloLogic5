<template>
    <div id="results-summary-container" class="mt-4 ms-2 me-2">
        <div class="card shadow-sm px-3 py-2" :class="{ 'colloc-no-top-border': formData.report == 'collocation' }">
            <section id="initial_report" role="region" :aria-label="$t('resultsSummary.summaryRegion')">
                <div id="description">
                    <h1 class="page-heading">
                        {{ $t('resultsSummary.heading') }}
                    </h1>

                    <button type="button" class="btn btn-secondary btn-sm"
                        style="margin-top: -0.5rem; margin-right: -1rem" id="export-results" data-bs-toggle="modal"
                        data-bs-target="#export-modal" :aria-label="$t('resultsSummary.exportResults')">
                        {{ $t("resultsSummary.exportResults") }}
                    </button>

                    <div class="modal fade" tabindex="-1" id="export-modal" role="dialog"
                        aria-labelledby="export-modal-header">
                        <div class="modal-dialog" role="document">
                            <div class="modal-content">
                                <div class="modal-header">
                                    <h2 class="modal-title" id="export-modal-header">{{
                                        $t('resultsSummary.exportResults')
                                    }}</h2>
                                    <button type="button" class="btn-close" data-bs-dismiss="modal"
                                        :aria-label="$t('common.close')"></button>
                                </div>
                                <div class="modal-body">
                                    <export-results></export-results>
                                </div>
                            </div>
                        </div>
                    </div>

                    <search-arguments class="pt-4" :result-start="descriptionStart"
                        :result-end="descriptionEnd"></search-arguments>

                    <div v-if="['concordance', 'kwic', 'bibliography'].includes(formData.report)">
                        <div id="result-stats" class="pb-2">
                            <span class="d-inline-flex align-items-center">
                                <template v-if="totalResultsDone">
                                    {{ $t("resultsSummary.totalOccurrences", { n: resultsLength }) }}
                                </template>
                                <progress-spinner progress="0" :sm="true" v-else />
                            </span>
                            <span class="d-inline-flex" style=" align-items: center;" v-if="fieldSummary.length > 0">
                                <span>&nbsp;{{ $t("resultsSummary.spreadAcross") }}&nbsp;</span>
                                <progress-spinner progress="0" :sm="true" v-if="!hitlistStatsDone" />
                                <span v-else>
                                    <span v-for="(stat, statIndex) in statsDescription" :key="stat.field">
                                        <router-link :to="`/aggregation?${stat.link}&group_by=${stat.field}`"
                                            class="stat-link" v-if="stat.link.length > 0">
                                            {{ stat.count }} {{ stat.label }}(s)<span class="visually-hidden"> - {{
                                                $t('resultsSummary.browseByField', { label: stat.label }) }}</span>
                                        </router-link>
                                        <span v-else>{{ stat.count }} {{ stat.label }}(s)</span>
                                        <span v-if="statIndex != statsDescription.length - 1">&nbsp;{{ $t("common.and")
                                            }}&nbsp;</span>
                                    </span>
                                </span>
                            </span>
                        </div>

                        <div id="search-hits">
                            <strong v-if="resultsLength > 0" class="d-inline-flex align-items-center">
                                <span v-if="totalResultsDone">{{
                                    $t("resultsSummary.displayingHits", {
                                        start: descriptionStart,
                                        end: descriptionEnd,
                                        total: resultsLength,
                                    })
                                }}</span>
                                <span v-else>
                                    {{ $t("resultsSummary.displayingHitsPartial", {
                                        start: descriptionStart,
                                        end: descriptionEnd,
                                    }) }}
                                    <progress-spinner progress="0" :sm="true" class="ms-1" />
                                </span>
                            </strong>
                            <strong v-else>{{ $t("resultsSummary.noResults") }}</strong>

                            <button type="button" class="btn rounded-pill btn-outline-secondary btn-sm ms-1"
                                style="margin-top: -0.05rem" data-bs-toggle="modal"
                                data-bs-target="#results-bibliography">
                                {{ $t("resultsSummary.fromTheseTitles") }}<span class="visually-hidden"> - {{
                                    $t("resultsSummary.showBibliography") }}</span>
                            </button>
                        </div>

                        <div class="modal fade" tabindex="-1" id="results-bibliography" aria-hidden="true"
                            aria-labelledby="biblio-modal-title">
                            <results-bibliography></results-bibliography>
                        </div>
                    </div>

                    <div v-if="formData.report == 'aggregation' && groupByLabel">
                        <div id="result-stats" class="pb-2" v-if="resultsLength > 0">
                            {{
                                $t("resultsSummary.groupByOccurrences", {
                                    total: resultsLength,
                                    n: groupLength,
                                    label: groupByLabel.toLowerCase(),
                                })
                            }}
                        </div>
                        <div id="result-stats" class="pb-2" v-else>
                            <strong>{{ $t("resultsSummary.noResults") }}</strong>
                        </div>
                    </div>

                    <!-- Collocation filter section -->
                    <div v-if="formData.report === 'collocation'">
                        <div>
                            <span>
                                <span>
                                    <button type="button" class="btn btn-link p-0" @click="toggleFilterList($event)"
                                        v-if="colloc_filter_choice === 'frequency'"
                                        :aria-label="$t('resultsSummary.commonWords', { n: filter_frequency }) + ' ' + $t('resultsSummary.filtered')"
                                        :aria-expanded="showFilteredWords" aria-controls="filter-list">
                                        {{ $t("resultsSummary.commonWords", { n: filter_frequency }) }}
                                    </button>
                                    <button type="button" class="btn btn-link p-0" @click="toggleFilterList($event)"
                                        v-if="colloc_filter_choice === 'stopwords'"
                                        :aria-label="$t('resultsSummary.commonStopwords') + ' ' + $t('resultsSummary.filtered')"
                                        :aria-expanded="showFilteredWords" aria-controls="filter-list">
                                        {{ $t("resultsSummary.commonStopwords") }}
                                    </button>
                                    {{ $t("resultsSummary.filtered") }}.
                                </span>
                            </span>

                            <div class="card ps-3 pe-3 pb-3 shadow-lg" id="filter-list" v-if="showFilteredWords"
                                role="region" :aria-label="$t('resultsSummary.filteredWordsList')">
                                <button type="button" class="btn btn-secondary" id="close-filter-list"
                                    @click="toggleFilterList($event)" :aria-label="$t('common.close')">
                                    <span class="icon-x"></span>
                                </button>
                                <div class="row mt-4">
                                    <div class="col" v-for="wordGroup in splitFilterList" :key="wordGroup[0]">
                                        <ul class="list-group list-group-flush">
                                            <li class="list-group-item" v-for="word in wordGroup" :key="word">
                                                {{ word }}
                                            </li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>
        </div>

        <!-- Report switch buttons and Results per page control -->
        <div class="mt-4 mb-3" v-if="formData.report == 'concordance' || formData.report == 'kwic'">
            <div class="row d-flex align-items-center">
                <div class="col-12" :class="showFacets && $philoConfig.facets.length > 0 ?
                    (formData.report === 'kwic' ? 'col-md-8 col-xl-9' : 'col-md-9 col-xl-9') : ''">
                    <div class="d-flex flex-wrap justify-content-between align-items-center gap-2">
                        <div class="flex-shrink-0" v-if="['concordance', 'kwic'].includes(formData.report)">
                            <div class="btn-group" role="group" id="report_switch"
                                :aria-label="$t('resultsSummary.reportViewOptions')">
                                <button type="button" class="btn btn-secondary"
                                    :class="{ active: formData.report === 'concordance' }"
                                    @click="switchReport('concordance')"
                                    :aria-label="$t('resultsSummary.concordanceBig')"
                                    :aria-pressed="formData.report === 'concordance'">
                                    <span class="d-none d-lg-inline">{{ $t("resultsSummary.concordanceBig")
                                        }}</span>
                                    <span class="d-inline d-lg-none">{{
                                        $t("resultsSummary.concordanceSmall") }}</span>
                                </button>
                                <button type="button" class="btn btn-secondary"
                                    :class="{ active: formData.report === 'kwic' }" @click="switchReport('kwic')"
                                    :aria-label="$t('resultsSummary.kwicBig')"
                                    :aria-pressed="formData.report === 'kwic'">
                                    <span class="d-none d-lg-inline">{{ $t("resultsSummary.kwicBig") }}</span>
                                    <span class="d-none d-sm-inline d-lg-none">{{ $t("resultsSummary.kwicSmall")
                                        }}</span>
                                    <span class="d-inline d-sm-none">KWIC</span>
                                </button>
                            </div>
                        </div>
                        <div class="flex-shrink-0">
                            <div class="btn-group" role="group" :aria-label="$t('kwic.resultsPerPageControl')">
                                <button type="button" class="btn btn-outline-secondary results-label-btn"
                                    style="border-right: solid" tabindex="-1">
                                    {{ $t("kwic.resultsDisplayed") }}
                                </button>
                                <div class="dropdown d-inline-block">
                                    <button class="btn btn-light dropdown-toggle" style="
                                            border-left: 0 !important;
                                            border-bottom-left-radius: 0;
                                            border-top-left-radius: 0;
                                        " type="button" id="results-per-page-content-toggle" data-bs-toggle="dropdown"
                                        aria-expanded="false">
                                        {{ formData.results_per_page }}
                                    </button>
                                    <ul class="dropdown-menu" aria-labelledby="results-per-page-content-toggle">
                                        <li v-for="number in resultsPerPageOptions" :key="number">
                                            <button type="button" class="dropdown-item"
                                                @click="switchResultsPerPage(number)"
                                                :aria-label="`${$t('kwic.showResults', { count: number })}`">
                                                {{ number }}
                                            </button>
                                        </li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <!-- Show facets button row -->
            <div class="row d-none d-sm-flex mt-2" v-if="!showFacets && $philoConfig.facets.length > 0">
                <div class="col-12" :class="showFacets && $philoConfig.facets.length > 0 ?
                    (formData.report === 'kwic' ? 'col-md-8 col-xl-9' : 'col-md-9 col-xl-9') : ''">
                    <div class="d-flex justify-content-end">
                        <button type="button" class="btn btn-secondary btn-sm" @click="store.toggleFacets()"
                            :aria-label="$t('common.showFacetsLabel')">
                            {{ $t("common.showFacets") }}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import { Modal } from "bootstrap";
import { computed, inject, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { storeToRefs } from "pinia";
import ExportResults from "./ExportResults";  // eslint-disable-line no-unused-vars
import ProgressSpinner from "./ProgressSpinner";  // eslint-disable-line no-unused-vars
import ResultsBibliography from "./ResultsBibliography";  // eslint-disable-line no-unused-vars
import SearchArguments from "./SearchArguments";  // eslint-disable-line no-unused-vars
import { useMainStore } from "../stores/main";
import {
    debug,
    deepEqual,
    isOnlyFacetChange,
    paramsFilter,
    paramsToRoute,
    paramsToUrlString,
} from "../utils.js";

const props = defineProps(["description", "filterList", "groupLength"]);

const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const philoConfig = inject("$philoConfig");
const route = useRoute();
const router = useRouter();
const store = useMainStore();
const {
    formData,
    resultsLength,
    totalResultsDone,
    showFacets,
    searching,
} = storeToRefs(store);

const fieldSummary = philoConfig.results_summary;
const hits = ref("");
const descriptionStart = ref(1);
const descriptionEnd = ref(formData.value.results_per_page);
const statsDescription = ref([]);
const hitlistStatsDone = ref(false);
const groupByLabel = ref(
    route.query.group_by in philoConfig.metadata_aliases
        ? philoConfig.metadata_aliases[route.query.group_by]
        : route.query.group_by
);
const showFilteredWords = ref(false);
const currentQuery = ref({});
const resultsPerPageOptions = [25, 100, 500, 1000];

const splitFilterList = computed(() => {
    if (!props.filterList || props.filterList.length === 0) return [];
    const arrayLength = props.filterList.length;
    const chunkSize = arrayLength / 5;
    const splittedList = [];
    for (let index = 0; index < arrayLength; index += chunkSize) {
        splittedList.push(props.filterList.slice(index, index + chunkSize));
    }
    return splittedList;
});

function buildDescription() {
    let start;
    let end;
    if (
        typeof props.description === "undefined" ||
        props.description.start === "" ||
        props.description.start == 0
    ) {
        start = 1;
        end = parseInt(formData.value.results_per_page);
    } else {
        start = props.description.start || route.query.start || 1;
        end = formData.value.end || parseInt(formData.value.results_per_page);
    }
    if (end > resultsLength.value) end = resultsLength.value;
    const resultsPerPage = parseInt(formData.value.results_per_page);
    if (resultsLength.value && end <= resultsPerPage && end <= resultsLength.value) {
        descriptionStart.value = start;
        descriptionEnd.value = end;
    } else if (resultsLength.value) {
        descriptionStart.value = start;
        descriptionEnd.value = resultsPerPage > resultsLength.value ? resultsLength.value : end;
    }
}

function updateTotalResults() {
    // If the report response already provided the final count, skip the extra request
    if (totalResultsDone.value) {
        hits.value = buildDescription();
        return;
    }
    const params = { ...formData.value };
    if (formData.value.report === "time_series") {
        const start = formData.value.start_date || philoConfig.time_series_start_end_date.start_date;
        const end = formData.value.end_date || philoConfig.time_series_start_end_date.end_date;
        params.year = `${start}-${end}`;
    }
    totalResultsDone.value = false;
    $http
        .get(`${$dbUrl}/scripts/get_total_results.py`, { params: paramsFilter(params) })
        .then((response) => {
            resultsLength.value = response.data;
            hits.value = buildDescription();
            totalResultsDone.value = true;
        })
        .catch((error) => {
            debug({ $options: { name: "ResultsSummary" } }, error);
        });
}

function getHitListStats() {
    hitlistStatsDone.value = false;
    $http
        .get(`${$dbUrl}/scripts/get_hitlist_stats.py`, {
            params: paramsFilter({ ...formData.value }),
        })
        .then((response) => {
            hitlistStatsDone.value = true;
            const out = [];
            for (const stat of response.data.stats) {
                const label = stat.field in philoConfig.metadata_aliases
                    ? philoConfig.metadata_aliases[stat.field].toLowerCase()
                    : stat.field;
                let link = "";
                if (stat.link_field) {
                    link = paramsToUrlString({
                        ...formData.value,
                        report: "aggregation",
                        start: "",
                        end: "",
                        group_by: "",
                    });
                    if (link.length === 0) link = "/aggregation?";
                }
                out.push({ label, field: stat.field, count: stat.count, link });
            }
            statsDescription.value = out;
        })
        .catch((error) => {
            debug({ $options: { name: "ResultsSummary" } }, error);
        });
}

function updateDescriptions() {
    const modalEl = document.getElementById("results-bibliography");
    if (modalEl) {
        Modal.getOrCreateInstance(modalEl).hide();
    }
    buildDescription();
    // For reports that set searching/totalResultsDone (concordance/kwic/bibliography),
    // defer updateTotalResults + getHitListStats until the report response arrives
    // (via the `searching` watcher below). This avoids blocking extra Gunicorn workers
    // with hits.finish() calls while the main request is still in flight.
    if (!["concordance", "kwic", "bibliography"].includes(formData.value.report)) {
        updateTotalResults();
    }
}

function switchReport(reportName) {
    formData.value.report = reportName;
    formData.value.results_per_page = 25;
    router.push(paramsToRoute({ ...formData.value }));
}

function switchResultsPerPage(number) {
    formData.value.results_per_page = parseInt(number);
    router.push(
        paramsToRoute({ ...formData.value, results_per_page: number, start: "1", end: number })
    );
}

function toggleFilterList(event) {
    event.preventDefault();
    showFilteredWords.value = !showFilteredWords.value;
}


watch(
    () => route.fullPath,
    (newPath, oldPath) => {
        const newQuery = router.resolve(newPath).query;
        const oldQuery = router.resolve(oldPath || "").query;
        if (["concordance", "kwic", "bibliography"].includes(formData.value.report)) {
            if (!isOnlyFacetChange(newQuery, oldQuery)) {
                updateDescriptions();
            }
        } else {
            updateDescriptions();
        }
    }
);

watch(searching, (newVal, oldVal) => {
    // When a concordance/kwic/bibliography report response arrives (searching
    // goes true→false), fire the deferred secondary requests. Other reports
    // (collocation/time_series/aggregation) call updateTotalResults directly
    // from updateDescriptions(), so skip them here.
    if (
        oldVal === true && newVal === false &&
        ["concordance", "kwic", "bibliography"].includes(formData.value.report)
    ) {
        if (!totalResultsDone.value) updateTotalResults();
        const newQuery = {
            ...formData.value,
            start: "",
            end: "",
            first_kwic_sorting_option: "",
            second_kwic_sorting_option: "",
            third_kwic_sorting_option: "",
        };
        if (!deepEqual(newQuery, currentQuery.value) || Object.keys(statsDescription.value).length === 0) {
            getHitListStats();
            currentQuery.value = newQuery;
        }
        buildDescription();
    }
});

currentQuery.value = {
    ...formData.value,
    start: "",
    end: "",
    first_kwic_sorting_option: "",
    second_kwic_sorting_option: "",
    third_kwic_sorting_option: "",
};
updateDescriptions();
</script>

<style lang="scss" scoped>
@use "../assets/styles/theme.module.scss" as theme;

#description {
    position: relative;
}

.page-heading {
    position: absolute;
    left: -1rem;
    top: -0.5rem;
    font-size: 1rem;
    font-weight: 500;
    margin: 0;
    padding: 0.125rem 0.25rem;
    background: transparent;
    border-bottom: 1px solid theme.$link-color;
    border-right: 1px solid theme.$link-color;
    border-radius: 0.25rem;
    font-variant: small-caps;
    font-weight: 600;
    letter-spacing: 0.02em;
}

#export-results {
    position: absolute;
    right: 0;
    padding: 0.125rem 0.25rem;
    font-size: 0.8rem !important;
}

#results-bibliography .modal-header {
    padding-bottom: 0.5rem;
}

#results-bibliography .modal-header button {
    padding: 0.5rem;
}

#results-bibliography .modal-header h5 {
    line-height: 1;
}

#close-filter-list {
    width: fit-content;
    float: right;
    padding: 0 0.2rem;
    position: absolute;
    right: 0;
}

#close-filter-list .icon-x {
    background-color: #fff !important;
}

#filter-list .list-group-item {
    border-width: 0px;
    padding: 0.1rem;
}

.stat-link,
.btn-link {
    text-decoration: none;
}

.results-label-btn {
    cursor: default !important;
    pointer-events: none;
}

.results-label-btn:hover,
.results-label-btn:focus {
    background-color: var(--bs-btn-bg) !important;
    border-color: var(--bs-btn-border-color) !important;
    color: var(--bs-btn-color) !important;
}

.colloc-no-top-border {
    border-top-width: 0 !important;
    border-top-left-radius: 0%;
}
</style>