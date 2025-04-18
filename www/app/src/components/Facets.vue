<template>
    <div id="facet-search" class="d-none d-sm-block mr-2">
        <div class="card shadow-sm" title="Title" header-tag="header" id="facet-panel-wrapper">
            <div class="card-header text-center">
                <h6 class="mb-0">{{ $t("facets.browseByFacet") }}</h6>
            </div>
            <button type="button" class="btn btn-secondary btn-sm close-box" @click="toggleFacets()">x</button>
            <transition name="slide-fade">
                <div class="list-group" flush id="select-facets" v-if="showFacetSelection">
                    <span class="dropdown-header text-center">{{ $t("facets.frequencyBy") }}</span>
                    <div class="list-group-item facet-selection" v-for="facet in facets" :key="facet.alias"
                        @click="getFacet(facet)">
                        {{ facet.alias }}
                    </div>
                </div>
            </transition>
            <transition name="slide-fade">
                <div class="list-group" flush id="select-word-properties"
                    v-if="showFacetSelection && report != 'bibliography' && philoConfig.words_facets.length > 0">
                    <span class="dropdown-header text-center">{{ $t("facets.wordProperty") }}</span>
                    <div class="list-group-item facet-selection" v-for="facet in wordFacets" :key="facet.facet"
                        @click="getFacet(facet)">
                        {{ facet.alias }}
                    </div>
                </div>
            </transition>
            <transition name="slide-fade">
                <div class="list-group mt-2" style="border-top: 0"
                    v-if="showFacetSelection && report != 'bibliography'">
                    <span class="dropdown-header text-center">{{ $t("facets.collocates") }}</span>
                    <div class="list-group-item facet-selection" @click="getFacet(collocationFacet)"
                        v-if="report !== 'bibliography'">
                        {{ $t("common.sameSentence") }}
                    </div>
                </div>
            </transition>
            <transition name="options-slide">
                <div class="m-2 text-center" style="width: 100%; font-size: 90%; opacity: 0.8; cursor: pointer"
                    v-if="!showFacetSelection" @click="showFacetOptions()">
                    {{ $t("facets.showOptions") }}
                </div>
            </transition>
        </div>
        <div class="d-flex justify-content-center position-relative" v-if="loading">
            <div class="spinner-border text-secondary" role="status"
                style="width: 4rem; height: 4rem; position: absolute; z-index: 50; top: 10px">
                <span class="visually-hidden">{{ $t("common.loading") }}...</span>
            </div>
        </div>
        <div class="card mt-3 shadow-sm" id="facet-results" v-if="showFacetResults">
            <div class="card-header text-center">
                <h6 class="mb-0">{{ $t("facets.frequencyByLabel", { label: selectedFacet.alias }) }}</h6>
                <button type="button" class="btn btn-secondary btn-sm close-box" @click="hideFacets()">x</button>
            </div>
            <div class="btn-group btn-group-sm shadow-sm" role="group"
                v-if="percent == 100 && report !== 'bibliography' && facet.type === 'facet'">
                <button type="button" class="btn btn-light" :class="{ active: showingRelativeFrequencies === false }"
                    @click="displayAbsoluteFrequencies()">
                    {{ $t("common.absoluteFrequency") }}
                </button>
                <button type="button" class="btn btn-light" :class="{ active: showingRelativeFrequencies }"
                    @click="displayRelativeFrequencies()">
                    {{ $t("common.relativeFrequency") }}
                </button>
            </div>
            <div class="m-2 text-center" style="opacity: 0.7">
                {{ $t("facets.top500Results", { label: selectedFacet.alias }) }}
            </div>
            <div class="progress my-3 mb-3" :max="resultsLength" show-progress variant="secondary"
                v-if="percent != 100">
                <div class="progress-bar" :value="runningTotal"
                    :label="`${((runningTotal / resultsLength) * 100).toFixed(2)}%`"></div>
            </div>
            <div class="list-group" flush>
                <div class="list-group-item" v-for="result in facetResults" :key="result.label">
                    <div>
                        <a href class="sidebar-text text-content-area text-view" v-if="facet.type == 'facet'"
                            @click.prevent="facetClick(result.metadata)">{{ result.label }}</a>
                        <a href class="sidebar-text text-content-area" text-view
                            v-else-if="facet.type == 'property' && facet.facet != 'lemma'"
                            @click.prevent="propertyToConcordance(result.q)">{{ result.label }}</a>
                        <span class="text-content-area" text-view
                            v-else-if="facet.type == 'property' && facet.facet == 'lemma'">{{ result.label }}</span>
                        <a href class="sidebar-text text-content-area" v-else-if="facet.type == 'collocationFacet'"
                            @click.prevent="collocationToConcordance(result.collocate)">{{ result.collocate }}</a>
                        <div class="badge bg-secondary rounded-pill float-end">{{ result.count }}</div>
                    </div>
                    <div style="line-height: 70%; padding-bottom: 15px; font-size: 85%"
                        v-if="showingRelativeFrequencies">
                        <div style="display: inline-block; opacity: 0.8">
                            {{
                                $t("facets.relativeFrequencyDescription", {
                                    total: fullResults.unsorted[result.label].count,
                                    wordCount: fullRelativeFrequencies[result.label].total_count,
                                })
                            }}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>
<script>
import { mapFields } from "vuex-map-fields";

export default {
    name: "facets-report",
    computed: {
        ...mapFields([
            "formData.report",
            "formData.q",
            "fornData.method",
            "formData.start",
            "formData.end",
            "formData.metadataFields",
            "resultsLength",
            "showFacets",
            "urlUpdate",
        ]),
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
                    ...this.$store.state.formData,
                    frequency_field: facetObj.alias,
                });
            } else if (facetObj.type === "collocationFacet") {
                urlString = this.paramsToUrlString({
                    ...this.$store.state.formData,
                    report: "collocation",
                });
            } else if (facetObj.type === "property") {
                urlString = this.paramsToUrlString({
                    ...this.$store.state.formData,
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
                let queryParams = this.copyObject(this.$store.state.formData);
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
            this.q = `${this.q} "${word}"`;
            this.$store.commit("updateFormDataField", {
                key: "method",
                value: "cooc",
            });
            this.start = "";
            this.end = "";
            this.report = "concordance";
            this.$router.push(this.paramsToRoute({ ...this.$store.state.formData }));
        },
        propertyToConcordance(query) {
            this.q = query;
            this.start = "";
            this.end = "";
            this.report = "concordance";
            this.$router.push(this.paramsToRoute({ ...this.$store.state.formData }));
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
            this.$store.commit("updateFormDataField", {
                key: this.selectedFacet.facet,
                value: metadataValue,
            });
            this.$router.push(
                this.paramsToRoute({
                    ...this.$store.state.formData,
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
<style scoped>
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

.list-group-item {
    position: relative;
    padding: 0.5rem 1.25rem;
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
}

.facet-selection:hover {
    font-weight: 700;
}

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
