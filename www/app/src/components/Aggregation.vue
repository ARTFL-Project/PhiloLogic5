<template>
    <div class="container-fluid mt-4">
        <results-summary :groupLength="aggregationResults.length"></results-summary>
        <div class="card shadow mt-4 ms-2 me-2" v-if="resultsLength" v-scroll="handleFullResultsScroll">
            <div id="aggregation-results" class="list-group" role="main" :aria-label="$t('aggregation.resultsRegion')">
                <article class="list-group-item pt-3 pb-3" role="article"
                    v-for="(result, resultIndex) in aggregationResults.slice(0, lastResult)" :key="resultIndex"
                    :aria-labelledby="`result-heading-${resultIndex}`">

                    <!-- Toggle button with proper accessibility -->
                    <button type="button" class="btn btn-outline-secondary btn-sm d-inline-block"
                        style="padding: 0 0.25rem; margin-right: 0.5rem" :id="`button-${resultIndex}`"
                        @click="toggleBreakUp(resultIndex)" v-if="result.break_up_field.length > 0"
                        :aria-expanded="breakUpFields[resultIndex].show" :aria-controls="`breakdown-${resultIndex}`"
                        :aria-label="breakUpFields[resultIndex].show ?
                            $t('aggregation.collapseBreakdown') :
                            $t('aggregation.expandBreakdown')">
                        <span aria-hidden="true">{{ breakUpFields[resultIndex].show ? '−' : '+' }}</span>
                    </button>

                    <!-- Count badge with screen reader context -->
                    <span class="badge rounded-pill bg-secondary" style="font-size: 100%"
                        :aria-label="`${$t('aggregation.countLabel')}: ${result.count}`">
                        {{ result.count }}
                    </span>

                    <!-- Main citation with proper heading -->
                    <div :id="`result-heading-${resultIndex}`" class="d-inline-block">
                        <citations :citation="result.citation"></citations>
                    </div>

                    <!-- Breakdown summary -->
                    <span class="d-inline-block ps-1" v-if="breakUpFields[resultIndex].results.length"
                        :aria-label="`${$t('aggregation.distributedAcross')} ${breakUpFields[resultIndex].results.length} ${breakUpFieldName}(s)`">
                        {{ $t("common.across") }} {{ breakUpFields[resultIndex].results.length }}
                        {{ breakUpFieldName }}(s)
                    </span>

                    <!-- Performance warning -->
                    <h6 class="ms-4 mt-2" role="alert"
                        v-if="breakUpFields[resultIndex].show && breakUpFields[resultIndex].results.length > 1000">
                        {{ $t("aggregation.performance") }}
                    </h6>

                    <!-- Expandable breakdown section -->
                    <div class="list-group ms-4 mt-2" v-if="breakUpFields[resultIndex].show"
                        :id="`breakdown-${resultIndex}`" role="group"
                        :aria-label="`${$t('aggregation.breakdownResults')} ${breakUpFieldName}`">

                        <article class="list-group-item"
                            v-for="(value, key) in breakUpFields[resultIndex].results.slice(0, breakUpFields[resultIndex].limit)"
                            :key="key" role="article" :aria-labelledby="`breakdown-item-${resultIndex}-${key}`">

                            <span class="badge rounded-pill bg-secondary"
                                :aria-label="`${$t('aggregation.countLabel')}: ${value.count}`">
                                {{ value.count }}
                            </span>

                            <div :id="`breakdown-item-${resultIndex}-${key}`" class="d-inline-block">
                                <citations :citation="buildCitationObject(
                                    statsConfig.break_up_field,
                                    statsConfig.break_up_field_citation,
                                    value.metadata_fields
                                )"></citations>
                            </div>
                        </article>
                    </div>
                </article>
            </div>
        </div>
    </div>
</template>
<script>
import { mapFields } from "vuex-map-fields";
import citations from "./Citations";
import ResultsSummary from "./ResultsSummary";

export default {
    name: "aggregation-report",
    components: { citations, ResultsSummary },
    inject: ["$http"],
    provide() {
        return {
            results: this.aggregationResults,
        };
    },
    computed: {
        ...mapFields([
            "formData.report",
            "resultsLength",
            "aggregationCache",
            "searching",
            "currentReport",
            "urlUpdate",
        ]),
        statsConfig() {
            for (let fieldObject of this.$philoConfig.aggregation_config) {
                if (fieldObject.field == this.$route.query.group_by) {
                    return fieldObject;
                }
            }
            return null;
        },
    },
    data() {
        return {
            loading: false,
            aggregationResults: [],
            lastResult: 50,
            infiniteId: 0,
            groupedByField: this.$route.query.group_by,
            breakUpFields: [],
            breakUpFieldName: "",
        };
    },
    created() {
        this.report = "aggregation";
        this.currentReport = "aggregation";
        this.fetchResults();
    },
    watch: {
        urlUpdate() {
            if (this.report == "aggregation") {
                this.groupedByField = this.$route.query.group_by;
                this.fetchResults();
            }
        },
    },
    methods: {
        fetchResults() {
            if (
                this.deepEqual(
                    { ...this.aggregationCache.query, start: "", end: "" },
                    { ...this.$route.query, start: "", end: "" }
                )
            ) {
                this.aggregationResults = this.aggregationCache.results;
                this.breakUpFields = this.aggregationResults.map((results) => ({
                    show: false,
                    results: results.break_up_field,
                }));
                this.resultsLength = this.aggregationCache.totalResults;
            } else {
                this.searching = true;
                this.$http
                    .get(`${this.$dbUrl}/reports/aggregation.py`, {
                        params: this.paramsFilter({
                            ...this.$store.state.formData,
                        }),
                    })
                    .then((response) => {
                        this.infiniteId += 1;
                        this.aggregationResults = Object.freeze(this.buildStatResults(response.data.results));
                        this.lastResult = 50;
                        this.breakUpFields = this.aggregationResults.map((results) => ({
                            show: false,
                            results: results.break_up_field,
                            limit: 1000,
                        }));
                        this.breakUpFieldName =
                            this.$philoConfig.metadata_aliases[response.data.break_up_field] ||
                            response.data.break_up_field;
                        if (typeof this.breakUpFieldName != "undefined" || this.breakUpFieldName != null) {
                            this.breakUpFieldName = this.breakUpFieldName.toLowerCase();
                        }
                        this.resultsLength = response.data.total_results;
                        this.searching = false;
                    })
                    .catch((error) => {
                        this.searching = false;
                        this.debug(this, error);
                    });
            }
        },
        handleFullResultsScroll() {
            let scrollPosition = document.getElementById("aggregation-results").getBoundingClientRect().bottom - 200;
            if (scrollPosition < window.innerHeight) {
                this.lastResult += 50;
            }
        },
        buildStatResults(results) {
            let resultsWithCiteObject = [];
            for (let result of results) {
                result.citation = this.buildCitationObject(
                    this.groupedByField,
                    this.statsConfig.field_citation,
                    result.metadata_fields
                );
                resultsWithCiteObject.push(result);
            }
            return resultsWithCiteObject;
        },
        buildCitationObject(fieldToLink, citationObject, metadataFields) {
            let citations = [];
            for (let citation of citationObject) {
                let label = metadataFields[citation.field];
                if (label == null || label.length == 0) {
                    if (citation["field"] != fieldToLink) {
                        continue;
                    }
                }
                if (citation["field"] == fieldToLink) {
                    let queryParams = {
                        ...this.$store.state.formData,
                        start: "0",
                        end: "25",
                    };
                    if (label == null || label.length == 0) {
                        queryParams[fieldToLink] = ""; // Should be NULL, but that's broken in the philo lib
                        label = this.$t("common.na");
                    } else {
                        queryParams[fieldToLink] = `"${label}"`;
                    }
                    if (fieldToLink != this.groupedByField) {
                        queryParams[this.groupedByField] = `"${metadataFields[this.groupedByField]}"`;
                    }
                    let link = "";
                    // workaround for broken NULL searches
                    if (queryParams[fieldToLink].length) {
                        link = this.paramsToRoute({
                            ...queryParams,
                            report: "concordance",
                        });
                        citations.push({ ...citation, href: link, label: label });
                    } else {
                        citations.push({ ...citation, href: "", label: label });
                    }
                } else {
                    citations.push({ ...citation, href: "", label: label });
                }
            }
            return citations;
        },
        toggleBreakUp(resultIndex) {
            this.breakUpFields[resultIndex].show = !this.breakUpFields[resultIndex].show;
        },
    },
};
</script>
<style scoped>
#description {
    position: relative;
}

#export-results {
    position: absolute;
    right: 0;
    padding: 0.125rem 0.25rem;
    font-size: 0.8rem !important;
}
</style>