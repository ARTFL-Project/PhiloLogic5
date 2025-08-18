<template>
    <div id="results-summary-container" class="mt-4 ms-2 me-2">
        <div class="card shadow-sm px-3 py-2" :class="{ 'colloc-no-top-border': report == 'collocation' }">
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

                    <div v-if="['concordance', 'kwic', 'bibliography'].includes(report)">
                        <div id="result-stats" class="pb-2">
                            {{ $t("resultsSummary.totalOccurrences", { n: resultsLength }) }}
                            <span class="d-inline-flex" style=" align-items: center;" v-if="fieldSummary.length > 0">
                                <span>{{ $t("resultsSummary.spreadAcross") }}&nbsp;</span>
                                <progress-spinner progress="0" :sm="true" v-if="!hitlistStatsDone" />
                                <span v-else>
                                    <span v-for="(stat, statIndex) in statsDescription" :key="stat.field">
                                        <router-link :to="`/aggregation?${stat.link}&group_by=${stat.field}`"
                                            class="stat-link" v-if="stat.link.length > 0"
                                            :aria-label="$t('resultsSummary.viewAggregation', { count: stat.count, label: stat.label })">
                                            {{ stat.count }} {{ stat.label }}(s)
                                        </router-link>
                                        <span v-else>{{ stat.count }} {{ stat.label }}(s)</span>
                                        <span v-if="statIndex != statsDescription.length - 1">&nbsp;{{ $t("common.and")
                                            }}&nbsp;</span>
                                    </span>
                                </span>
                            </span>
                        </div>

                        <div id="search-hits">
                            <strong v-if="resultsLength > 0">{{
                                $t("resultsSummary.displayingHits", {
                                    start: descriptionStart,
                                    end: descriptionEnd,
                                    total: resultsLength,
                                })
                            }}</strong>
                            <strong v-else>{{ $t("resultsSummary.noResults") }}</strong>

                            <button type="button" class="btn rounded-pill btn-outline-secondary btn-sm ms-1"
                                style="margin-top: -0.05rem" data-bs-toggle="modal"
                                data-bs-target="#results-bibliography"
                                :aria-label="$t('resultsSummary.showBibliography')">
                                {{ $t("resultsSummary.fromTheseTitles") }}
                            </button>
                        </div>

                        <div class="modal fade" tabindex="-1" id="results-bibliography" aria-hidden="true"
                            aria-labelledby="biblio-modal-title">
                            <results-bibliography></results-bibliography>
                        </div>
                    </div>

                    <div v-if="report == 'aggregation' && groupByLabel">
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

                    <!-- Progress bar section -->
                    <div v-if="['collocation', 'time_series'].includes(report)">
                        <div class="progress ms-3 me-3 mb-3" v-if="runningTotal != resultsLength" role="progressbar"
                            :aria-valuenow="Math.floor((runningTotal / resultsLength) * 100)" :aria-valuemax="100"
                            aria-valuemin="0" :aria-label="$t('resultsSummary.progressLabel', {
                                percent: Math.floor((runningTotal / resultsLength) * 100)
                            })" aria-live="polite">
                            <div class="progress-bar"
                                :style="`width: ${((runningTotal / resultsLength) * 100).toFixed(2)}%`">
                                {{ Math.floor((runningTotal / resultsLength) * 100) }}%
                            </div>
                            <span class="visually-hidden">
                                {{ $t('resultsSummary.progressDescription', {
                                    current: runningTotal,
                                    total: resultsLength,
                                    percent: Math.floor((runningTotal / resultsLength) * 100)
                                }) }}
                            </span>
                        </div>

                        <!-- Collocation filter section -->
                        <div v-if="report == 'collocation'">
                            <span>
                                <span>
                                    <button type="button" class="btn btn-link p-0" @click="toggleFilterList($event)"
                                        v-if="colloc_filter_choice === 'frequency'"
                                        :aria-label="$t('resultsSummary.showFilteredWords')"
                                        :aria-expanded="showFilteredWords" aria-controls="filter-list">
                                        {{ $t("resultsSummary.commonWords", { n: filter_frequency }) }}
                                    </button>
                                    <button type="button" class="btn btn-link p-0" @click="toggleFilterList($event)"
                                        v-if="colloc_filter_choice === 'stopwords'"
                                        :aria-label="$t('resultsSummary.showFilteredWords')"
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
                                    &times;
                                </button>
                                <div class="row mt-4">
                                    <div class="col" v-for="wordGroup in splitFilterList" :key="wordGroup[0]">
                                        <ul class="list-group list-group-flush" role="list">
                                            <li class="list-group-item" v-for="word in wordGroup" :key="word"
                                                role="listitem">
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
        <div class="mt-4 mb-3" v-if="report == 'concordance' || report == 'kwic'">
            <div class="row d-none d-sm-flex align-items-center"
                :style="report === 'concordance' ? 'padding-right: 0.5rem' : ''"
                :class="report === 'kwic' ? 'px-2' : ''">
                <div class="col-12" :class="showFacets && $philoConfig.facets.length > 0 ?
                    (report === 'kwic' ? 'col-md-8 col-xl-9' : 'col-md-9 col-xl-9') : ''">
                    <div class="d-flex justify-content-start align-items-center">
                        <div class="flex-shrink-0" v-if="['concordance', 'kwic'].includes(report)">
                            <div class="btn-group" role="group" id="report_switch"
                                :aria-label="$t('resultsSummary.reportViewOptions')">
                                <button type="button" class="btn btn-secondary"
                                    :class="{ active: report === 'concordance' }" @click="switchReport('concordance')"
                                    :aria-label="$t('resultsSummary.switchToConcordance')"
                                    :aria-pressed="report === 'concordance'">
                                    <span class="d-none d-lg-inline">{{ $t("resultsSummary.concordanceBig")
                                        }}</span>
                                    <span class="d-inline d-lg-none">{{
                                        $t("resultsSummary.concordanceSmall") }}</span>
                                </button>
                                <button type="button" class="btn btn-secondary" :class="{ active: report === 'kwic' }"
                                    @click="switchReport('kwic')" :aria-label="$t('resultsSummary.switchToKwic')"
                                    :aria-pressed="report === 'kwic'">
                                    <span class="d-none d-lg-inline">{{ $t("resultsSummary.kwicBig") }}</span>
                                    <span class="d-inline d-lg-none">{{ $t("resultsSummary.kwicSmall")
                                        }}</span>
                                </button>
                            </div>
                        </div>
                        <div class="flex-shrink-1 ms-auto">
                            <div class="btn-group" role="group" :aria-label="$t('kwic.resultsPerPageControl')">
                                <button type="button" class="btn btn-outline-secondary results-label-btn"
                                    style="border-right: solid">
                                    {{ $t("kwic.resultsDisplayed") }}
                                </button>
                                <div class="dropdown d-inline-block">
                                    <button class="btn btn-light dropdown-toggle" style="
                                            border-left: 0 !important;
                                            border-bottom-left-radius: 0;
                                            border-top-left-radius: 0;
                                        " type="button" id="results-per-page-content-toggle" data-bs-toggle="dropdown"
                                        aria-expanded="false">
                                        {{ results_per_page }}
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
            <div class="row d-none d-sm-flex mt-2" v-if="!showFacets && facets.length > 0"
                :style="report === 'concordance' ? 'padding-right: 0.5rem' : ''"
                :class="report === 'kwic' ? 'px-2' : ''">
                <div class="col-12" :class="showFacets && $philoConfig.facets.length > 0 ?
                    (report === 'kwic' ? 'col-md-8 col-xl-9' : 'col-md-9 col-xl-9') : ''">
                    <div class="d-flex justify-content-end">
                        <button type="button" class="btn btn-secondary btn-sm" @click="toggleFacets()"
                            :aria-label="$t('common.showFacetsLabel')">
                            {{ $t("common.showFacets") }}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
import { Modal } from "bootstrap";
import ExportResults from "./ExportResults";
import ProgressSpinner from "./ProgressSpinner";
import ResultsBibliography from "./ResultsBibliography";
import searchArguments from "./SearchArguments";

import { mapFields } from "vuex-map-fields";

export default {
    name: "ResultsSummary",
    components: {
        searchArguments,
        ResultsBibliography,
        ExportResults,
        ProgressSpinner
    },
    props: ["description", "runningTotal", "filterList", "groupLength"],
    computed: {
        ...mapFields([
            "formData.report",
            "formData.results_per_page",
            "formData.first_kwic_sorting_option",
            "formData.second_kwic_sorting_option",
            "formData.third_kwic_sorting_option",
            "formData.start",
            "formData.end",
            "formData.group_by",
            "formData.colloc_filter_choice",
            "formData.filter_frequency",
            "formData.start_date",
            "formData.end_date",
            "currentReport",
            "resultsLength",
            "aggregationCache",
            "totalResultsDone",
            "showFacets",
        ]),
        splitFilterList: function () {
            if (!this.filterList || this.filterList.length === 0) {
                return [];
            }
            let arrayLength = this.filterList.length;
            let chunkSize = arrayLength / 5;
            let splittedList = [];
            for (let index = 0; index < arrayLength; index += chunkSize) {
                let myChunk = this.filterList.slice(index, index + chunkSize);
                splittedList.push(myChunk);
            }
            return splittedList;
        },
    },
    inject: ["$http"],
    data() {
        return {
            facets: this.$philoConfig.facets,
            fieldSummary: this.$philoConfig.results_summary,
            hits: "",
            descriptionStart: 1,
            descriptionEnd: this.$store.state.formData.results_per_page,
            statsDescription: [],
            resultsPerPage: 0,
            hitlistStatsDone: false,
            showBiblio: false,
            groupByLabel:
                this.$route.query.group_by in this.$philoConfig.metadata_aliases
                    ? this.$philoConfig.metadata_aliases[this.$route.query.group_by]
                    : this.$route.query.group_by,
            showFilteredWords: false,
            currentQuery: {},
            resultsPerPageOptions: [25, 100, 500, 1000],
        };
    },
    created() {
        this.currentQuery = {
            ...this.$store.state.formData,
            start: "",
            end: "",
            first_kwic_sorting_option: "",
            second_kwic_sorting_option: "",
            third_kwic_sorting_option: "",
        };
        this.updateDescriptions();
    },
    watch: {
        $route: "updateDescriptions",
    },
    methods: {
        updateDescriptions() {
            let modalEl = document.getElementById("results-bibliography");
            if (modalEl) {
                // hide modal if open
                let modal = Modal.getOrCreateInstance(modalEl);
                modal.hide();
            }
            this.buildDescription();
            this.updateTotalResults();
            if (["concordance", "kwic", "bibliography"].includes(this.report)) {
                let newQuery = {
                    ...this.$store.state.formData,
                    start: "",
                    end: "",
                    first_kwic_sorting_option: "",
                    second_kwic_sorting_option: "",
                    third_kwic_sorting_option: "",
                };
                if (!this.deepEqual(newQuery, this.currentQuery) || Object.keys(this.statsDescription).length == 0) {
                    this.getHitListStats();
                    this.currentQuery = newQuery;
                }
            }
        },
        buildDescription() {
            let start;
            let end;
            if (
                typeof this.description == "undefined" ||
                this.description.start === "" ||
                this.description.start == 0
            ) {
                start = 1;
                end = parseInt(this.results_per_page);
            } else {
                start = this.description.start || this.$route.query.start || 1;
                end = this.end || parseInt(this.results_per_page);
            }
            if (end > this.resultsLength) {
                end = this.resultsLength;
            }
            let resultsPerPage = parseInt(this.results_per_page);
            let description;
            if (this.resultsLength && end <= resultsPerPage && end <= this.resultsLength) {
                this.descriptionStart = start;
                this.descriptionEnd = end;
            } else if (this.resultsLength) {
                if (resultsPerPage > this.resultsLength) {
                    this.descriptionStart = start;
                    this.descriptionEnd = this.resultsLength;
                } else {
                    this.descriptionStart = start;
                    this.descriptionEnd = end;
                }
            }
            return description;
        },
        updateTotalResults() {
            let params = { ...this.$store.state.formData };
            if (this.report == "time_series") {
                params.year = `${this.start_date || this.$philoConfig.time_series_start_end_date.start_date}-${this.end_date || this.$philoConfig.time_series_start_end_date.end_date
                    }`;
            }
            this.totalResultsDone = false;
            this.$http
                .get(`${this.$dbUrl}/scripts/get_total_results.py`, {
                    params: this.paramsFilter(params),
                })
                .then((response) => {
                    this.resultsLength = response.data;
                    this.hits = this.buildDescription();
                    this.totalResultsDone = true;
                })
                .catch((error) => {
                    this.debug(this, error);
                });
        },
        getHitListStats() {
            this.hitlistStatsDone = false;
            this.$http
                .get(`${this.$dbUrl}/scripts/get_hitlist_stats.py`, {
                    params: this.paramsFilter({ ...this.$store.state.formData }),
                })
                .then((response) => {
                    this.hitlistStatsDone = true;
                    let statsDescription = [];
                    for (let stat of response.data.stats) {
                        let label = "";
                        if (stat.field in this.$philoConfig.metadata_aliases) {
                            label = this.$philoConfig.metadata_aliases[stat.field].toLowerCase();
                        } else {
                            label = stat.field;
                        }
                        let link = "";
                        if (stat.link_field) {
                            link = this.paramsToUrlString({
                                ...this.$store.state.formData,
                                report: "aggregation",
                                start: "",
                                end: "",
                                group_by: "",
                            });
                            if (link.length == 0) {
                                link = "/aggregation?";
                            }
                        }
                        statsDescription.push({
                            label: label,
                            field: stat.field,
                            count: stat.count,
                            link: link,
                        });
                    }
                    this.statsDescription = statsDescription;
                })
                .catch((error) => {
                    this.debug(this, error);
                });
        },
        switchReport(reportName) {
            this.report = reportName;
            this.first_kwic_sorting_option = "";
            this.second_kwic_sorting_option = "";
            this.third_kwic_sorting_option = "";
            this.results_per_page = 25;
            this.$router.push(this.paramsToRoute({ ...this.$store.state.formData }));
        },
        switchResultsPerPage(number) {
            this.results_per_page = parseInt(number);
            this.$router.push(
                this.paramsToRoute({ ...this.$store.state.formData, results_per_page: number, start: "1", end: number })
            );
        },
        showFacets() { },
        showResultsBiblio() {
            if (!this.showBiblio) {
                this.showBiblio = true;
            } else {
                this.showBiblio = false;
            }
        },
        toggleFilterList(event) {
            event.preventDefault();
            if (this.showFilteredWords == true) {
                this.showFilteredWords = false;
            } else {
                this.showFilteredWords = true;
            }
        },
        toggleFacets() {
            this.showFacets = !this.showFacets;
        },
    },
};
</script>

<style lang="scss" scoped>
@import "../assets/styles/theme.module.scss";

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
    border-bottom: 1px solid $link-color;
    border-right: 1px solid $link-color;
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