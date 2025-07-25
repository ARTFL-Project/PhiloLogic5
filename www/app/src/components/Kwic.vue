<template>
    <div class="container-fluid">
        <results-summary :description="results.description"></results-summary>
        <div style="position: relative" v-if="!showFacets && philoConfig.facets.length > 0">
            <button type="button" class="btn btn-secondary" style="position: absolute; bottom: 1rem; right: 0.5rem"
                @click="toggleFacets()" :aria-label="$t('common.showFacetsLabel')">
                {{ $t("common.showFacets") }}
            </button>
        </div>
        <div class="row px-2">
            <main class="col-12" :class="{ 'col-md-8': showFacets, 'col-xl-9': showFacets }" role="main"
                :aria-label="$t('kwic.resultsRegion')">
                <div class="card p-2 ml-2 shadow-sm">
                    <div class="p-2 mb-1">
                        <!-- Sorting controls -->
                        <div class="btn-group" role="group" :aria-label="$t('kwic.sortingControls')">
                            <button type="button" class="btn btn-sm btn-outline-secondary" style="border-right: solid">
                                {{ $t("kwic.sortResultsBy") }}
                            </button>
                            <div class="btn-group" v-for="(fields, index) in sortingFields" :key="index">
                                <div class="dropdown">
                                    <button class="btn btn-light btn-sm dropdown-toggle sort-toggle"
                                        :style="index == 0 ? 'border-left: 0 !important' : ''" :id="`kwicDrop${index}`"
                                        data-bs-toggle="dropdown" aria-expanded="false">
                                        {{ sortingSelection[index] }}
                                    </button>
                                    <ul class="dropdown-menu" :aria-labelledby="`kwicDrop${index}`">
                                        <li v-for="(selection, fieldIndex) in fields" :key="fieldIndex">
                                            <button type="button" class="dropdown-item"
                                                @click="updateSortingSelection(index, selection)"
                                                :aria-label="`${$t('kwic.selectSortCriteria', { criteria: selection.label })}`">
                                                {{ selection.label }}
                                            </button>
                                        </li>
                                    </ul>
                                </div>
                            </div>
                            <button type="button" class="btn btn-secondary btn-sm" id="sort-button"
                                @click="sortResults()" :aria-label="$t('kwic.applySorting')">
                                {{ $t("kwic.sort") }}
                            </button>
                        </div>

                        <!-- Results per page control -->
                        <div class="float-lg-end mt-lg-0 mt-md-2">
                            <div class="btn-group" role="group" :aria-label="$t('kwic.resultsPerPageControl')">
                                <button type="button" class="btn btn-sm btn-outline-secondary"
                                    style="border-right: solid">
                                    {{ $t("kwic.resultsDisplayed") }}
                                </button>
                                <div class="dropdown d-inline-block">
                                    <button class="btn btn-sm btn-light dropdown-toggle" style="
                                            border-left: 0 !important;
                                            border-bottom-left-radius: 0;
                                            border-top-left-radius: 0;
                                        " type="button" id="kwic-results-per-page" data-bs-toggle="dropdown"
                                        aria-expanded="false">
                                        {{ results_per_page }}
                                    </button>
                                    <ul class="dropdown-menu" aria-labelledby="kwic-results-per-page">
                                        <li v-for="number in ['25', '100', '500', '1000']" :key="number">
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
                        <div class="progress mt-3" v-if="runningTotal != resultsLength" role="progressbar"
                            :aria-valuenow="runningTotal" :aria-valuemin="0" :aria-valuemax="resultsLength"
                            :aria-label="$t('kwic.loadingProgress')">
                            <div class="progress-bar"
                                :style="`width: ${((runningTotal / resultsLength) * 100).toFixed(2)}%`"
                                :aria-hidden="true">
                                {{ Math.floor((runningTotal / resultsLength) * 100) }}%
                            </div>
                            <span class="visually-hidden">
                                {{ $t('kwic.progressDescription', {
                                    current: runningTotal,
                                    total: resultsLength,
                                    percent: Math.floor((runningTotal / resultsLength) * 100)
                                }) }}
                            </span>
                        </div>
                    </div>

                    <!-- KWIC results -->
                    <div id="kwic-concordance" role="region" :aria-label="$t('kwic.concordanceRegion')"
                        aria-live="polite">
                        <transition-group tag="div" :css="false" v-on:before-enter="onBeforeEnter" v-on:enter="onEnter">
                            <div class="kwic-line" v-for="(result, kwicIndex) in filteredKwic(results.results)"
                                :key="result.philo_id.join('-')" :data-index="kwicIndex"
                                :aria-label="`${$t('kwic.resultNumber')} ${results.description.start + kwicIndex}`">
                                <span v-html="initializePos(kwicIndex)"></span>
                                <div class="kwic-biblio-container" style="display: inline-block; position: relative;"
                                    @mouseover="showFullBiblio($event)" @mouseleave="hideFullBiblio($event)"
                                    @focus="showFullBiblio($event)" @blur="hideFullBiblio($event)"
                                    :aria-describedby="`full-biblio-${kwicIndex}`"
                                    :aria-label="`${$t('kwic.viewFullText')} ${result.fullBiblio}`">
                                    <router-link :to="result.citation_links.div1" class="kwic-biblio">
                                        <span class="short-biblio" v-html="result.shortBiblio"></span>
                                        <div :id="`full-biblio-${kwicIndex}`" class="full-biblio" role="tooltip"
                                            :aria-hidden="true">
                                            {{ result.fullBiblio }}
                                        </div>
                                    </router-link>
                                </div>
                                <span v-html="result.context"></span>
                            </div>
                        </transition-group>
                    </div>
                </div>
                <pages></pages>
            </main>

            <aside class="col col-md-4 col-xl-3" v-if="showFacets" role="complementary"
                :aria-label="$t('common.facetsRegion')">
                <facets></facets>
            </aside>
        </div>
    </div>
</template>

<script>
import gsap from "gsap";
import { computed } from "vue";
import { mapFields } from "vuex-map-fields";
import facets from "./Facets";
import pages from "./Pages";
import ResultsSummary from "./ResultsSummary";

export default {
    name: "kwic-report",
    components: {
        ResultsSummary,
        facets,
        pages,
    },
    computed: {
        ...mapFields([
            "formData.report",
            "formData.q",
            "formData.start",
            "formData.end",
            "formData.results_per_page",
            "formData.first_kwic_sorting_option",
            "formData.second_kwic_sorting_option",
            "formData.third_kwic_sorting_option",
            "resultsLength",
            "searching",
            "currentReport",
            "description",
            "sortedKwicCache",
            "urlUpdate",
            "showFacets",
        ]),
        sortingFields() {
            let sortingFields = [
                {
                    label: this.$t("common.none"),
                    field: "",
                },
                {
                    label: this.$t("kwic.searchedTerms"),
                    field: "q",
                },
                {
                    label: this.$t("kwic.wordsLeft"),
                    field: "left",
                },
                {
                    label: this.$t("kwic.wordsRight"),
                    field: "right",
                },
            ];
            for (let field of this.philoConfig.kwic_metadata_sorting_fields) {
                if (field in this.philoConfig.metadata_aliases) {
                    let label = this.philoConfig.metadata_aliases[field];
                    sortingFields.push({
                        label: label,
                        field: field,
                    });
                } else {
                    sortingFields.push({
                        label: field[0].toUpperCase() + field.slice(1),
                        field: field,
                    });
                }
            }
            return [sortingFields, sortingFields, sortingFields];
        },
        sortKeys() {
            let sortKeys = {
                q: this.$t("kwic.searchedTerms"),
                left: this.$t("kwic.wordsLeft"),
                right: this.$t("kwic.wordsRight"),
            };
            for (let field of this.philoConfig.kwic_metadata_sorting_fields) {
                if (field in this.philoConfig.metadata_aliases) {
                    let label = this.philoConfig.metadata_aliases[field];
                    sortKeys[field] = label;
                } else {
                    sortKeys[field] = field[0].toUpperCase() + field.slice(1);
                }
            }
            return sortKeys;
        },
        sortingSelection() {
            let sortingSelection = [];
            if (this.first_kwic_sorting_option !== "") {
                sortingSelection.push(this.sortKeys[this.first_kwic_sorting_option]);
            }
            if (this.second_kwic_sorting_option !== "") {
                sortingSelection.push(this.sortKeys[this.second_kwic_sorting_option]);
            }
            if (this.third_kwic_sorting_option !== "") {
                sortingSelection.push(this.sortKeys[this.third_kwic_sorting_option]);
            }
            if (sortingSelection.length === 0) {
                sortingSelection = [this.$t("common.none"), this.$t("common.none"), this.$t("common.none")];
            } else if (sortingSelection.length === 1) {
                sortingSelection.push(this.$t("common.none"));
                sortingSelection.push(this.$t("common.none"));
            } else if (sortingSelection.length === 2) {
                sortingSelection.push(this.$t("common.none"));
            }
            return sortingSelection;
        },
    },
    inject: ["$http"],
    provide() {
        return {
            results: computed(() => this.results.results),
        };
    },
    data() {
        return {
            philoConfig: this.$philoConfig,
            results: { description: { end: 0 } },
            searchParams: {},
            sortedResults: [],
            loading: false,
            runningTotal: 0,
            cachePath: "",
        };
    },
    created() {
        this.report = "kwic";
        this.currentReport = "kwic";
        this.fetchResults();
    },
    watch: {
        urlUpdate() {
            if (this.report == "kwic") {
                this.fetchResults();
            }
        },
    },
    methods: {
        buildFullCitation(metadataField) {
            let citationList = [];
            let biblioFields = this.philoConfig.kwic_bibliography_fields;
            if (typeof biblioFields === "undefined" || biblioFields.length === 0) {
                biblioFields = this.philoConfig.metadata.slice(0, 2);
                biblioFields.push("head");
            }
            for (var i = 0; i < biblioFields.length; i++) {
                if (biblioFields[i] in metadataField) {
                    var biblioField = metadataField[biblioFields[i]] || "";
                    if (biblioField.length > 0) {
                        citationList.push(biblioField);
                    }
                }
            }
            if (citationList.length > 0) {
                return citationList.join(", ");
            } else {
                return "NA";
            }
        },
        filteredKwic(results) {
            let filteredResults = [];
            if (typeof results != "undefined" && Object.keys(results).length) {
                for (let resultObject of results) {
                    resultObject.fullBiblio = this.buildFullCitation(resultObject.metadata_fields);
                    resultObject.shortBiblio = resultObject.fullBiblio.slice(0, 30);
                    filteredResults.push(resultObject);
                }
            }
            return filteredResults;
        },
        showFullBiblio(event) {
            let target = event.currentTarget.querySelector(".full-biblio");
            target.classList.add("show");
        },
        hideFullBiblio(event) {
            let target = event.currentTarget.querySelector(".full-biblio");
            target.classList.remove("show");
        },
        updateSortingSelection(index, selection) {
            if (index === 0) {
                if (selection.label == this.$t("common.none")) {
                    this.first_kwic_sorting_option = "";
                } else {
                    this.first_kwic_sorting_option = selection.field;
                }
            } else if (index == 1) {
                if (selection.label == this.$t("common.none")) {
                    this.second_kwic_sorting_option = "";
                } else {
                    this.second_kwic_sorting_option = selection.field;
                }
            } else {
                if (selection.label == this.$t("common.none")) {
                    this.third_kwic_sorting_option = "";
                } else {
                    this.third_kwic_sorting_option = selection.field;
                }
            }
        },
        fetchResults() {
            this.results = { description: { end: 0 }, results: [] };
            this.searchParams = { ...this.$store.state.formData };
            if (this.first_kwic_sorting_option === "") {
                this.searching = true;
                this.$http
                    .get(`${this.$dbUrl}/reports/kwic.py`, {
                        params: this.paramsFilter(this.searchParams),
                    })
                    .then((response) => {
                        this.results = response.data;
                        this.resultsLength = response.data.results_length;
                        this.runningTotal = response.data.results_length;
                        this.results.description = response.data.description;
                        this.searching = false;
                    })
                    .catch((error) => {
                        this.searching = false;
                        this.error = error.toString();
                        this.debug(this, error);
                    });
            } else {
                if (this.start == "") {
                    this.start = "0";
                    this.end = this.results_per_page;
                }
                this.searching = true;
                this.runningTotal = 0;
                this.recursiveLookup(0);
            }
        },
        recursiveLookup(hitsDone) {
            this.$http
                .get(`${this.$dbUrl}/scripts/get_neighboring_words.py`, {
                    params: {
                        ...this.paramsFilter({ ...this.$store.state.formData }),
                        hits_done: hitsDone,
                        max_time: 5,
                    },
                })
                .then((response) => {
                    this.searching = false;
                    hitsDone = response.data.hits_done;
                    this.runningTotal = hitsDone;
                    this.cachePath = response.data.cache_path;
                    if (hitsDone < this.resultsLength) {
                        this.recursiveLookup(hitsDone);
                    } else {
                        this.getKwicResults(hitsDone);
                    }
                });
        },
        getKwicResults(hitsDone) {
            let start = parseInt(this.start);
            let end = 0;
            if (this.results_per_page === "") {
                end = start + 25;
            } else {
                end = start + parseInt(this.results_per_page);
            }
            this.$http
                .get(`${this.$dbUrl}/scripts/get_sorted_kwic.py`, {
                    params: {
                        hits_done: hitsDone,
                        ...this.paramsFilter({ ...this.$store.state.formData }),
                        start: start,
                        end: end,
                        cache_path: this.cachePath,
                    },
                })
                .then((response) => {
                    this.results = response.data;
                    this.searching = false;
                });
        },
        initializePos(index) {
            let start = this.results.description.start;
            let currentPos = start + index;
            let currentPosLength = currentPos.toString().length;
            let endPos = start + parseInt(this.results_per_page) || 25;
            let endPosLength = endPos.toString().length;
            let spaces = endPosLength - currentPosLength + 1;
            return currentPos + "." + Array(spaces).join("&nbsp");
        },
        sortResults() {
            this.results.results = [];
            this.$router.push(this.paramsToRoute({ ...this.$store.state.formData }));
        },
        dicoLookup() { },
        onBeforeEnter(el) {
            el.style.opacity = 0;
        },
        onEnter(el, done) {
            gsap.to(el, {
                opacity: 1,
                delay: el.dataset.index * 0.0075,
                onComplete: done,
            });
        },
        toggleFacets() {
            if (this.showFacets) {
                this.showFacets = false;
            } else {
                this.showFacets = true;
            }
        },
        switchResultsPerPage(number) {
            this.$router.push(
                this.paramsToRoute({ ...this.$store.state.formData, results_per_page: number, start: "1", end: number })
            );
        },
    },
};
</script>

<style scoped lang="scss">
@import "../assets/styles/theme.module.scss";

.sort-toggle {
    border-bottom-left-radius: 0;
    border-top-left-radius: 0;
    border-bottom-right-radius: 0;
    border-top-right-radius: 0;
}

#kwic-concordance {
    font-family: monospace;
}

.kwic-line {
    line-height: 180%;
    white-space: nowrap;
    overflow: hidden;
}

.kwic-biblio {
    font-weight: 400 !important;
    z-index: 10;
    padding-right: 0.5rem;
}

.kwic-biblio:focus {
    outline: 2px solid $button-color;
    outline-offset: 2px;
}

.short-biblio {
    width: 200px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    display: inline-block;
    vertical-align: bottom;
    margin-left: -5px;
    padding-left: 5px;
}

.kwic-biblio-container {
    display: inline-block;
    position: relative;
    overflow: visible;
    /* Ensure tooltip isn't clipped */
}

.full-biblio {
    z-index: 10;
    display: none;
    opacity: 0;
    cursor: pointer;
}

.full-biblio.show {
    position: absolute;
    background-color: $button-color;
    color: #fff;
    display: inline !important;
    padding: 0px 5px;
    margin-left: -5px;
    opacity: 1;
    left: 0;
    transition: all 0.2s ease;
}

.full-biblio[aria-hidden="false"] {
    display: block !important;
}

:deep(.highlight) {
    border: none !important;
}

#sort-button {
    color: #fff !important;
    background-color: $button-color !important;
    outline: 2px solid rgba(255, 255, 255, 0.5);
    outline-offset: 2px;
    box-shadow: 0 0 0 0.2rem rgba(255, 255, 255, 0.25);
}

.progress {
    border-radius: 0.375rem;
    overflow: hidden;
}

.progress-bar {
    background-color: $button-color !important;
}

.dropdown-item:focus {
    outline: 2px solid $button-color;
    outline-offset: -2px;
    background-color: rgba($button-color, 0.1);
}

:deep(.kwic-before) {
    text-align: right;
    overflow: hidden;
    display: inline-block;
    position: absolute;
}

:deep(.inner-before) {
    float: right;
}

:deep(.kwic-after) {
    text-align: left;
    display: inline-block;
}

:deep(.kwic-text) {
    display: inline-block;
    overflow: hidden;
    vertical-align: bottom;
}

#kwic-switch {
    margin-left: -3px;
}

@media (min-width: 1300px) {
    :deep(.kwic-highlight) {
        margin-left: 330px;
    }

    :deep(.kwic-before) {
        width: 330px;
    }
}

@media (min-width: 992px) and (max-width: 1299px) {
    :deep(.kwic-highlight) {
        margin-left: 230px;
    }

    :deep(.kwic-before) {
        width: 230px;
    }
}

@media (min-width: 768px) and (max-width: 991px) {
    :deep(.kwic-highlight) {
        margin-left: 120px;
    }

    :deep(.kwic-before) {
        width: 120px;
    }

    :deep(.kwic-line) {
        font-size: 12px;
    }
}

@media (max-width: 767px) {
    :deep(.kwic-highlight) {
        margin-left: 200px;
    }

    :deep(.kwic-before) {
        width: 200px;
    }

    :deep(.kwic-line) {
        font-size: 12px;
    }
}
</style>
