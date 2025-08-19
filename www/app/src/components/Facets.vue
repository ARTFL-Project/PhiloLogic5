<template>
    <div id="facet-search" class="d-none d-sm-block mr-2">
        <div class="card shadow-sm" title="Title" header-tag="header" id="facet-panel-wrapper">
            <div class="card-header text-center">
                <h3 class="h6 mb-0">{{ $t("facets.browseByFacet") }}</h3>
            </div>
            <button type="button" class="btn btn-secondary btn-sm close-box" @click="toggleFacets()"
                :aria-label="$t('facets.closeFacets')">
                ×
            </button>

            <transition name="slide-fade">
                <div class="list-group" flush id="select-facets" v-if="showFacetSelection" role="group"
                    :aria-label="$t('facets.selectFacetType')">
                    <span class="dropdown-header text-center">{{ $t("facets.frequencyBy") }}</span>
                    <button type="button" class="list-group-item list-group-item-action facet-selection"
                        v-for="facet in facets" :key="facet.alias" @click="getFacet(facet)"
                        :aria-label="`${$t('facets.selectFacet')} ${facet.alias}`">
                        {{ facet.alias }}
                    </button>
                </div>
            </transition>

            <transition name="slide-fade">
                <div class="list-group mt-3" style="border-top: 0" flush id="select-word-properties"
                    v-if="showFacetSelection && formData.report != 'bibliography' && philoConfig.words_facets.length > 0"
                    role="group" :aria-label="$t('facets.selectWordProperty')">
                    <span class="dropdown-header text-center">{{ $t("facets.wordProperty") }}</span>
                    <button type="button" class="list-group-item list-group-item-action facet-selection"
                        v-for="facet in wordFacets" :key="facet.facet" @click="getFacet(facet)"
                        :aria-label="`${$t('facets.selectWordProperty')} ${facet.alias}`">
                        {{ facet.alias }}
                    </button>
                </div>
            </transition>

            <transition name="slide-fade">
                <div class="list-group mt-3" style="border-top: 0"
                    v-if="showFacetSelection && formData.report != 'bibliography'" role="group"
                    :aria-label="$t('facets.selectCollocation')">
                    <span class="dropdown-header text-center">{{ $t("facets.collocates") }}</span>
                    <button type="button" class="list-group-item list-group-item-action facet-selection"
                        @click="getFacet(collocationFacet)" v-if="formData.report !== 'bibliography'"
                        :aria-label="`${$t('facets.selectCollocation')} ${$t('common.sameSentence')}`">
                        {{ $t("common.sameSentence") }}
                    </button>
                </div>
            </transition>

            <transition name="options-slide">
                <button type="button" class="btn btn-link m-2 text-center"
                    style="width: 100%; font-size: 90%; opacity: 0.8" v-if="!showFacetSelection"
                    @click="showFacetOptions()" :aria-label="$t('facets.showFacetOptions')">
                    {{ $t("facets.showOptions") }}
                </button>
            </transition>
        </div>

        <!-- Loading indicator -->
        <div class="d-flex justify-content-center position-relative" v-if="loading" role="status"
            :aria-label="$t('common.loadingFacets')">
            <div class="spinner-border text-secondary"
                style="width: 4rem; height: 4rem; position: absolute; z-index: 50; top: 10px" aria-hidden="true">
            </div>
            <span class="visually-hidden">{{ $t("common.loadingFacets") }}...</span>
        </div>

        <!-- Facet results -->
        <div class="card mt-3 shadow-sm" id="facet-results" v-if="showFacetResults" role="region"
            :aria-label="$t('facets.facetResultsRegion')">
            <div class="card-header text-center">
                <h4 class="mb-0 h6">{{ $t("facets.frequencyByLabel", { label: selectedFacet.alias }) }}</h4>
                <button type="button" class="btn btn-secondary btn-sm close-box" @click="hideFacets()"
                    :aria-label="$t('facets.hideFacetResults')">
                    ×
                </button>
            </div>

            <!-- Frequency toggle buttons -->
            <div class="btn-group btn-group-sm shadow-sm" role="group" :aria-label="$t('facets.frequencyTypeToggle')"
                v-if="percent == 100 && formData.report !== 'bibliography' && facet.type === 'facet'">
                <button type="button" class="btn btn-light" :class="{ active: showingRelativeFrequencies === false }"
                    @click="displayAbsoluteFrequencies()" :aria-pressed="showingRelativeFrequencies === false"
                    :aria-label="$t('facets.showAbsoluteFrequency')">
                    {{ $t("common.absoluteFrequency") }}
                </button>
                <button type="button" class="btn btn-light" :class="{ active: showingRelativeFrequencies }"
                    @click="displayRelativeFrequencies()" :aria-pressed="showingRelativeFrequencies"
                    :aria-label="$t('facets.showRelativeFrequency')">
                    {{ $t("common.relativeFrequency") }}
                </button>
            </div>

            <div class="m-2 text-center" style="opacity: 0.7">
                {{ $t("facets.top500Results", { label: selectedFacet.alias }) }}
            </div>

            <!-- Progress bar -->
            <div class="progress my-3 mb-3" v-if="percent != 100" role="progressbar" :aria-valuenow="runningTotal"
                :aria-valuemin="0" :aria-valuemax="resultsLength" :aria-label="$t('facets.loadingProgress')">
                <div class="progress-bar" :style="`width: ${((runningTotal / resultsLength) * 100)}%`"
                    :aria-hidden="true">
                </div>
                <span class="visually-hidden">
                    {{ $t('facets.progressDescription', {
                        current: runningTotal,
                        total: resultsLength,
                        percent: ((runningTotal / resultsLength) * 100).toFixed(0)
                    }) }}
                </span>
            </div>

            <!-- Facet results list -->
            <div class="list-group facet-results-container" flush role="list"
                :aria-label="$t('facets.facetResultsList')">
                <!-- Facet link -->
                <button type="button" class="list-group-item list-group-item-action facet-result-item"
                    v-if="facet.type == 'facet'" v-for="result in facetResults" :key="result.label" role="listitem"
                    @click="facetClick(result.metadata)"
                    :aria-label="`${$t('facets.filterBy')} ${result.label}, ${result.count} ${$t('facets.occurrences')}`">
                    <div class="d-flex justify-content-between align-items-start">
                        <span class="sidebar-text text-content-area text-view">
                            {{ result.label }}
                        </span>
                        <span class="badge bg-secondary rounded-pill" aria-hidden="true">
                            {{ result.count }}
                        </span>
                    </div>

                    <!-- Relative frequency description -->
                    <div class="relative-frequency-info" v-if="showingRelativeFrequencies">
                        <small class="text-muted" :aria-label="$t('facets.relativeFrequencyLabel', {
                            total: fullResults.unsorted[result.label].count,
                            wordCount: fullRelativeFrequencies[result.label].total_count,
                        })">
                            {{
                                $t("facets.relativeFrequencyDescription", {
                                    total: fullResults.unsorted[result.label].count,
                                    wordCount: fullRelativeFrequencies[result.label].total_count,
                                })
                            }}
                        </small>
                    </div>
                </button>

                <!-- Property link -->
                <button type="button" class="list-group-item list-group-item-action facet-result-item"
                    v-else-if="facet.type == 'property' && facet.facet != 'lemma'" v-for="result in facetResults"
                    :key="result.label" role="listitem" @click="propertyToConcordance(result.q)"
                    :aria-label="`${$t('facets.searchFor')} ${result.label}, ${result.count} ${$t('facets.occurrences')}`">
                    <div class="d-flex justify-content-between align-items-start">
                        <span class="sidebar-text text-content-area">
                            {{ result.label }}
                        </span>
                        <span class="badge bg-secondary rounded-pill" aria-hidden="true">
                            {{ result.count }}
                        </span>
                    </div>
                </button>

                <!-- Lemma (non-clickable) -->
                <div class="list-group-item facet-result-item non-clickable"
                    v-else-if="facet.type == 'property' && facet.facet == 'lemma'" v-for="result in facetResults"
                    :key="result.label" role="listitem"
                    :aria-label="`${result.label}, ${result.count} ${$t('facets.occurrences')}`">
                    <div class="d-flex justify-content-between align-items-start">
                        <span class="text-content-area">
                            {{ result.label }}
                        </span>
                        <span class="badge bg-secondary rounded-pill" aria-hidden="true">
                            {{ result.count }}
                        </span>
                    </div>
                </div>

                <!-- Collocation link -->
                <button type="button" class="list-group-item list-group-item-action facet-result-item"
                    v-else-if="facet.type == 'collocationFacet'" v-for="result in facetResults" :key="result.label"
                    role="listitem" @click="collocationToConcordance(result.collocate)"
                    :aria-label="`${$t('facets.searchCollocation')} ${result.collocate}, ${result.count} ${$t('facets.occurrences')}`">
                    <div class="d-flex justify-content-between align-items-start">
                        <span class="sidebar-text text-content-area">
                            {{ result.collocate }}
                        </span>
                        <span class="badge bg-secondary rounded-pill" aria-hidden="true">
                            {{ result.count }}
                        </span>
                    </div>
                </button>
            </div>
        </div>
    </div>
</template>

<script>
import { mapStores, mapWritableState } from "pinia";
import { useMainStore } from "../stores/main";

export default {
    name: "facets-report",
    computed: {
        ...mapWritableState(useMainStore, [
            "formData",
            "resultsLength",
            "showFacets",
            "urlUpdate"
        ]),
        ...mapStores(useMainStore),
    },
    inject: ["$http"],
    data() {
        return {
            philoConfig: this.$philoConfig,
            facets: [],
            queryArgs: {},
            showFacetSelection: true,
            showFacetResults: false,
            collocationFacet: {
                facet: "all_collocates",
                alias: this.$t("facets.collocate"),
                type: "collocationFacet",
            },
            loading: false,
            moreResults: false,
            done: true,
            facet: {},
            selectedFacet: {},
            showingRelativeFrequencies: false,
            fullResults: {},
            relativeFrequencies: [],
            absoluteFrequencies: [],
            interrupt: false,
            selected: "",
            runningTotal: 0,
        };
    },
    created() {
        this.facets = this.populateFacets();
        this.wordFacets = this.populateWordFacets();
    },
    watch: {
        urlUpdate() {
            this.facetResults = [];
            this.fullResults = {};
            this.showFacetSelection = true;
            this.showFacetResults = false;
        },
    },
    methods: {
        populateFacets() {
            let facetConfig = this.philoConfig.facets;
            let facets = [];
            let alias;
            for (let i = 0; i < facetConfig.length; i++) {
                let facet = facetConfig[i];
                if (!this.$philoConfig.metadata.includes(facet)) {
                    continue;
                }
                if (facet in this.philoConfig.metadata_aliases) {
                    alias = this.philoConfig.metadata_aliases[facet];
                } else {
                    alias = facet;
                }
                facets.push({
                    facet: facet,
                    alias: alias,
                    type: "facet",
                });
            }
            return facets;
        },
        populateWordFacets() {
            let wordFacets = [];
            for (let wordProperty of this.philoConfig.words_facets) {
                let alias = wordProperty;
                if (wordProperty in this.$philoConfig.word_property_aliases) {
                    alias = this.$philoConfig.word_property_aliases[wordProperty];
                }
                wordFacets.push({
                    facet: wordProperty,
                    alias: alias,
                    type: "property",
                });
            }
            return wordFacets;
        },
        getFacet(facetObj) {
            this.relativeFrequencies = [];
            this.absoluteFrequencies = [];
            this.showingRelativeFrequencies = false;
            this.facet = facetObj;
            this.selectedFacet = facetObj;
            this.selected = facetObj.alias;
            let urlString
            if (facetObj.type === "facet") {
                urlString = this.paramsToUrlString({
                    ...this.formData,
                    frequency_field: facetObj.alias,
                });
            } else if (facetObj.type === "collocationFacet") {
                urlString = this.paramsToUrlString({
                    ...this.formData,
                    report: "collocation",
                });
            } else if (facetObj.type === "property") {
                urlString = this.paramsToUrlString({
                    ...this.formData,
                    word_property: facetObj.facet,
                });
            }
            if (typeof sessionStorage[urlString] !== "undefined" && this.philoConfig.production === true) {
                this.loading = true;
                this.fullResults = JSON.parse(sessionStorage[urlString]);
                this.facetResults = this.fullResults.sorted.slice(0, 500);
                this.loading = false;
                this.percent = 100;
                this.showFacetResults = true;
                this.showFacetSelection = false;
            } else {
                this.done = false;
                let fullResults = {};
                this.loading = true;
                this.moreResults = true;
                this.percent = 0;
                let queryParams = this.copyObject(this.formData);
                if (facetObj.type === "facet") {
                    queryParams.frequency_field = facetObj.facet;
                } else if (facetObj.type === "collocationFacet") {
                    queryParams.report = "collocation";
                } else if (facetObj.type === "property") {
                    queryParams.word_property = facetObj.facet;
                }
                this.populateSidebar(facetObj, fullResults, 0, queryParams);
            }
        },
        populateSidebar(facet, fullResults, start, queryParams) {
            if (this.moreResults) {
                let params = this.paramsFilter({
                    ...queryParams,
                    start: start.toString(),
                })
                this.showFacetSelection = false;
                let scriptName = "/scripts/get_frequency.py";
                if (facet.type === "property") {
                    scriptName = "/scripts/get_word_property_count.py";
                }
                if (facet.type != "collocationFacet") {
                    this.$http.get(`${this.$dbUrl}/${scriptName}`, {
                        params: params
                    })
                        .then((response) => {
                            let results = response.data.results;
                            if (facet.type == "facet") {
                                this.moreResults = response.data.more_results;
                                let merge;
                                if (!this.interrupt && this.selected == facet.alias) {
                                    if (facet.type === "collocationFacet") {
                                        merge = this.mergeResults(fullResults.unsorted, response.data.collocates);
                                    } else {
                                        merge = this.mergeResults(fullResults.unsorted, results);
                                    }
                                    this.facetResults = merge.sorted.slice(0, 500);
                                    this.loading = false;
                                    this.showFacetResults = true;
                                    fullResults = merge;
                                    this.runningTotal = response.data.hits_done;
                                    start = response.data.hits_done;
                                    this.populateSidebar(facet, fullResults, start, queryParams);
                                } else {
                                    this.interrupt = false;
                                }
                            } else {
                                this.loading = false;
                                this.showFacetResults = true;
                                this.moreResults = false;
                                this.facetResults = results;
                                this.populateSidebar(facet, results, start, queryParams);
                            }
                        })
                        .catch((error) => {
                            this.debug(this, error);
                            this.loading = false;
                        });

                } else {
                    this.$http
                        .post(`${this.$dbUrl}/reports/collocation.py`, {
                            current_collocates: fullResults,
                        },
                            {
                                params: this.paramsFilter(params),
                            })
                        .then((response) => {
                            this.moreResults = response.data.more_results;
                            this.runningTotal = response.data.hits_done;
                            start = response.data.hits_done;
                            this.loading = false;
                            if (response.data.results_length) {
                                this.showFacetResults = true;
                                if (this.moreResults) {
                                    this.facetResults = this.extractSurfaceFromCollocate(response.data.collocates.slice(0, 100));
                                    this.populateSidebar(facet, response.data.collocates, start, queryParams);
                                }
                                else {
                                    this.facetResults = this.extractSurfaceFromCollocate(response.data.collocates.slice(0, 100));
                                    this.populateSidebar(facet, response.data.collocates, start, queryParams);
                                }
                            }

                        })
                        .catch((error) => {
                            this.loading = false;
                            this.debug(this, error);
                        });
                }
            } else {
                this.loading = false;
                this.runningTotal = this.resultsLength;
                this.fullResults = fullResults;
                this.percent = 100;
                let urlString = this.paramsToUrlString({
                    ...queryParams,
                    frequency_field: this.selectedFacet.alias,
                });
                this.saveToLocalStorage(urlString, fullResults);
            }
        },
        roundToTwo(num) {
            return +(Math.round(num + "e+2") + "e-2");
        },
        getRelativeFrequencies() {
            let relativeResults = {};
            for (let label in this.fullResults.unsorted) {
                let resultObj = this.fullResults.unsorted[label];
                relativeResults[label] = {
                    count: this.roundToTwo((resultObj.count / resultObj.total_word_count) * 10000),
                    url: resultObj.url,
                    label: label,
                    total_count: resultObj.total_word_count,
                    metadata: resultObj.metadata,
                };
            }
            this.fullRelativeFrequencies = relativeResults;
            let sortedRelativeResults = this.sortResults(this.fullRelativeFrequencies);
            this.facetResults = this.copyObject(sortedRelativeResults.slice(0, 500));
            this.showingRelativeFrequencies = true;
            this.loading = false;
            this.percent = 100;
        },
        displayRelativeFrequencies() {
            this.loading = true;
            if (this.relativeFrequencies.length == 0) {
                this.absoluteFrequencies = this.copyObject(this.facetResults);
                this.percent = 0;
                this.fullRelativeFrequencies = {};
                this.getRelativeFrequencies();
            } else {
                this.absoluteFrequencies = this.copyObject(this.facetResults);
                this.facetResults = this.relativeFrequencies;
                this.showingRelativeFrequencies = true;
                this.loading = false;
            }
        },
        displayAbsoluteFrequencies() {
            this.loading = true;
            this.relativeFrequencies = this.copyObject(this.facetResults);
            this.facetResults = this.absoluteFrequencies;
            this.showingRelativeFrequencies = false;
            this.loading = false;
        },
        collocationToConcordance(word) {
            this.formData.q = `${this.formData.q} "${word}"`;
            this.mainStore.updateFormDataField({
                key: "method",
                value: "cooc",
            });
            this.formData.start = "";
            this.formData.end = "";
            this.formData.report = "concordance";
            this.$router.push(this.paramsToRoute({ ...this.formData }));
        },
        propertyToConcordance(query) {
            this.formData.q = query;
            this.formData.start = "";
            this.formData.end = "";
            this.formData.report = "concordance";
            this.$router.push(this.paramsToRoute({ ...this.formData }));
        },
        showFacetOptions() {
            this.showFacetSelection = true;
        },
        hideFacets() {
            this.showFacetResults = false;
            this.showFacetSelection = true;
        },
        occurrence(count) {
            if (count == 1) {
                return "occurrence";
            } else {
                return "occurrences";
            }
        },
        facetClick(metadata) {
            let metadataValue;
            metadataValue = `"${metadata[this.selectedFacet.facet]}"`;
            this.mainStore.updateFormDataField({
                key: this.selectedFacet.facet,
                value: metadataValue,
            });
            this.$router.push(
                this.paramsToRoute({
                    ...this.formData,
                    start: "0",
                    end: "0",
                })
            );
        },
        toggleFacets() {
            if (this.showFacets) {
                this.showFacets = false;
            } else {
                this.showFacets = true;
            }
        },
    },
};
</script>

<style scoped lang="scss">
@use "sass:color";
@use "../assets/styles/theme.module.scss" as theme;

.card-header {
    font-variant: small-caps;
}

.dropdown-header {
    padding: 0.5rem 0;
    font-variant: small-caps;
}

.list-group-item {
    border-left-width: 0;
    border-right-width: 0;
}

.close-box {
    position: absolute;
    top: 1px;
    right: 0;
}

.sidebar-text {
    cursor: pointer;
}

.sidebar-count {
    width: 100%;
    display: inline-block;
}

.facet-selection {
    width: 100%;
    cursor: pointer;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    background-color: transparent !important;
}

.facet-selection:hover {
    font-weight: 700;
    background-color: rgba(theme.$link-color, 0.15) !important;
    border-color: transparent !important;
    /* Hide borders on hover */
    box-shadow: 0 2px 8px rgba(theme.$link-color, 0.15);
    z-index: 1;
}

.btn-link {
    text-decoration: none !important;
    color: theme.$link-color;
    transition: all 0.2s ease-in-out;
    border-radius: 0.25rem;
    position: relative;
}

.btn-link:hover {
    /* Subtle movement effect - global styles handle accessibility */
    transform: translateX(2px);
}

.btn-link:active {
    transform: translateX(1px);
}

/* Facet results container - prevent overflow */
.facet-results-container {
    overflow: hidden;
    border-radius: 0.375rem;
}

/* Facet result items - make entire item clickable with zoom effect */
.facet-result-item {
    cursor: pointer;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    transform: scale(1);
    border: 1px solid transparent;
    background-color: transparent !important;
    text-align: left;
    width: 100%;
    padding: 0.75rem 1rem;
    position: relative;
    margin: 0 !important;
}

.facet-result-item:not(.non-clickable):hover {
    transform: scale(1.01);
    background-color: rgba(theme.$link-color, 0.15) !important;
    border-color: rgba(theme.$link-color, 0.3);
    box-shadow: inset 0 0 8px rgba(theme.$link-color, 0.1);
    z-index: 1;
}

.facet-result-item:not(.non-clickable):active {
    transform: scale(0.98);
    background-color: rgba(theme.$link-color, 0.2) !important;
}

/* Non-clickable lemma items */
.facet-result-item.non-clickable {
    cursor: default;
    opacity: 0.7;
}

.facet-result-item.non-clickable:hover {
    transform: none;
    background-color: transparent !important;
    border-color: transparent;
    box-shadow: none;
}

/* Badge animation on hover */
.facet-result-item:not(.non-clickable):hover .badge {
    background-color: theme.$link-color !important;
    transform: scale(1.1);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Text styling within facet items */
.facet-result-item .sidebar-text {
    color: theme.$link-color;
    font-weight: 500;
    transition: color 0.25s ease;
}

.facet-result-item:hover .sidebar-text {
    color: color.adjust(theme.$link-color, $lightness: -15%);
}

/* Relative frequency info styling */
.relative-frequency-info {
    margin-top: 0.25rem;
    line-height: 1.2;
}

.relative-frequency-info small {
    opacity: 0.8;
    transition: opacity 0.25s ease;
}

.facet-result-item:hover .relative-frequency-info small {
    opacity: 1;
}

/* Focus styles for accessibility */
.facet-selection:focus {
    outline: 2px solid theme.$link-color;
    outline-offset: -2px;
    box-shadow: 0 0 0 0.2rem rgba(theme.$link-color, 0.25);
}

.facet-result-item:focus {
    outline: 2px solid theme.$link-color;
    outline-offset: -2px;
    box-shadow: inset 0 0 0 0.2rem rgba(theme.$link-color, 0.25);
    z-index: 2;
}

.facet-result-item:focus:not(:focus-visible) {
    outline: none;
    box-shadow: inset 0 0 8px rgba(theme.$link-color, 0.15);
}

.facet-result-item:focus-visible {
    outline: 2px solid theme.$link-color;
    outline-offset: -2px;
    box-shadow: inset 0 0 0 0.2rem rgba(theme.$link-color, 0.25);
}

/* Frequency switcher enhanced styles */
.btn-group .btn-light {
    transition: all 0.2s ease-in-out;
    border-color: rgba(theme.$link-color, 0.25);
    background-color: rgba(theme.$link-color, 0.02);
    color: rgba(theme.$link-color, 0.8);
    font-weight: 500;
    min-width: 0;
    /* Prevent width growth */
    flex: 1;
    /* Equal width distribution */
}

.btn-group .btn-light:hover {
    border-color: rgba(theme.$link-color, 0.4);
    background-color: rgba(theme.$link-color, 0.08);
    color: theme.$link-color;
}

.btn-group .btn-light.active {
    background-color: rgba(theme.$link-color, 0.15) !important;
    color: color.adjust(theme.$link-color, $lightness: -10%) !important;
    border-color: rgba(theme.$link-color, 0.5) !important;
    /* Keep same font-weight to prevent text size changes */
}

.btn-group .btn-light.active:hover {
    background-color: rgba(theme.$link-color, 0.2) !important;
    color: color.adjust(theme.$link-color, $lightness: -15%) !important;
}

/* Smooth transitions for all interactive elements */
.badge {
    font-size: 0.75rem;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Keep existing transitions */
.slide-fade-enter-active {
    transition: all 0.3s ease-out;
}

.slide-fade-leave-active {
    transition: all 0.3s ease-out;
}

.slide-fade-enter,
.slide-fade-leave-to {
    transform: translateY(-10px);
    height: 0;
    opacity: 0;
}

.options-slide-fade-enter-active {
    transition: all 0.3s ease-in;
}

.options-slide-fade-leave-active {
    transition: all 0.3s ease-in;
}

.options-slide-fade-enter,
.options-slide-fade-leave-to {
    opacity: 0;
}
</style>
