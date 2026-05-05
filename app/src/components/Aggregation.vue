<template>
    <div class="container-fluid mt-4">
        <results-summary :groupLength="aggregationResults.length"></results-summary>
        <div class="card shadow mt-4 ms-2 me-2" v-if="resultsLength" v-scroll="handleFullResultsScroll">
            <div id="aggregation-results" class="list-group" role="region"
                :aria-label="$t('aggregation.resultsRegion')">
                <article class="list-group-item pt-3 pb-3" role="article"
                    v-for="(result, resultIndex) in aggregationResults.slice(0, lastResult)" :key="resultIndex">

                    <button type="button" class="btn btn-outline-secondary btn-sm d-inline-block"
                        style="padding: 0 0.25rem; margin-right: 0.5rem" :id="`button-${resultIndex}`"
                        @click="toggleBreakUp(resultIndex)" v-if="result.break_up_field.length > 0"
                        :aria-expanded="breakUpFields[resultIndex].show" :aria-controls="`breakdown-${resultIndex}`"
                        :aria-describedby="`result-heading-${resultIndex}`"
                        :aria-label="breakUpFields[resultIndex].show ?
                            $t('aggregation.collapseBreakdown', { name: result.metadata_fields[groupedByField] || $t('common.na') }) :
                            $t('aggregation.expandBreakdown', { name: result.metadata_fields[groupedByField] || $t('common.na') })">
                        <span aria-hidden="true">{{ breakUpFields[resultIndex].show ? '−' : '+' }}</span>
                    </button>

                    <span class="badge rounded-pill bg-secondary">
                        {{ result.count }}
                    </span>

                    <span :id="`result-heading-${resultIndex}`">
                        <citations :citation="result.citation" :result-number="resultIndex + 1"></citations>
                    </span>

                    <!-- Breakdown summary -->
                    <span class="d-inline-block ps-1" v-if="breakUpFields[resultIndex].results.length">
                        {{ $t("common.across") }} {{ breakUpFields[resultIndex].results.length }}
                        {{ breakUpFieldName }}(s)
                    </span>

                    <!-- Performance warning -->
                    <h6 class="ms-4 mt-2" role="alert"
                        v-if="breakUpFields[resultIndex].show && breakUpFields[resultIndex].results.length > 1000">
                        {{ $t("aggregation.performance") }}
                    </h6>

                    <!-- Expandable breakdown section -->
                    <div class="breakdown-container ms-4" v-if="breakUpFields[resultIndex].show"
                        :id="`breakdown-${resultIndex}`" role="group">

                        <article class="breakdown-item"
                            v-for="(value, key) in breakUpFields[resultIndex].results.slice(0, breakUpFields[resultIndex].limit)"
                            :key="key" role="article" :aria-labelledby="`breakdown-item-${resultIndex}-${key}`">

                            <div class="breakdown-content">
                                <span class="badge rounded-pill bg-info breakdown-badge">
                                    {{ value.count }}
                                </span>

                                <div :id="`breakdown-item-${resultIndex}-${key}`" class="breakdown-citation">
                                    <citations :citation="buildCitationObject(
                                        statsConfig.break_up_field,
                                        statsConfig.break_up_field_citation,
                                        value.metadata_fields
                                    )"></citations>
                                </div>
                            </div>
                        </article>
                    </div>
                </article>
            </div>
        </div>
    </div>
</template>
<script setup>
import { computed, inject, provide, ref, watch } from "vue";
import { useRoute } from "vue-router";
import { storeToRefs } from "pinia";
import { useI18n } from "vue-i18n";
import { useMainStore } from "../stores/main";
import { debug, deepEqual, paramsFilter, paramsToRoute } from "../utils.js";
import citations from "./Citations";
import ResultsSummary from "./ResultsSummary";

const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const philoConfig = inject("$philoConfig");
const route = useRoute();
const { t } = useI18n();
const store = useMainStore();
const {
    formData,
    resultsLength,
    aggregationCache,
    searching,
    currentReport,
    urlUpdate,
} = storeToRefs(store);

const aggregationResults = ref([]);
const lastResult = ref(50);
const infiniteId = ref(0);
const groupedByField = ref(route.query.group_by);
const breakUpFields = ref([]);
const breakUpFieldName = ref("");

const statsConfig = computed(() => {
    for (const fieldObject of philoConfig.aggregation_config) {
        if (fieldObject.field === route.query.group_by) return fieldObject;
    }
    return null;
});

provide("results", aggregationResults);

function fetchResults() {
    if (
        deepEqual(
            { ...aggregationCache.value.query, start: "", end: "" },
            { ...route.query, start: "", end: "" }
        )
    ) {
        aggregationResults.value = aggregationCache.value.results;
        breakUpFields.value = aggregationResults.value.map((r) => ({
            show: false,
            results: r.break_up_field,
        }));
        resultsLength.value = aggregationCache.value.totalResults;
        return;
    }

    searching.value = true;
    $http
        .get(`${$dbUrl}/reports/aggregation.py`, {
            params: paramsFilter({ ...formData.value }),
        })
        .then((response) => {
            infiniteId.value += 1;
            aggregationResults.value = Object.freeze(buildStatResults(response.data.results));
            lastResult.value = 50;
            breakUpFields.value = aggregationResults.value.map((r) => ({
                show: false,
                results: r.break_up_field,
                limit: 1000,
            }));
            const aliasName =
                philoConfig.metadata_aliases[response.data.break_up_field] ||
                response.data.break_up_field;
            breakUpFieldName.value = aliasName ? aliasName.toLowerCase() : "";
            resultsLength.value = response.data.total_results;
            searching.value = false;
        })
        .catch((error) => {
            searching.value = false;
            debug({ $options: { name: "aggregation-report" } }, error);
        });
}

function handleFullResultsScroll() {
    const scrollPosition =
        document.getElementById("aggregation-results").getBoundingClientRect().bottom - 200;
    if (scrollPosition < window.innerHeight) {
        lastResult.value += 50;
    }
}

function buildStatResults(results) {
    return results.map((result) => {
        result.citation = buildCitationObject(
            groupedByField.value,
            statsConfig.value.field_citation,
            result.metadata_fields
        );
        return result;
    });
}

function buildCitationObject(fieldToLink, citationObject, metadataFields) {
    const out = [];
    for (const citation of citationObject) {
        let label = metadataFields[citation.field];
        if ((label == null || label.length === 0) && citation.field !== fieldToLink) {
            continue;
        }
        if (citation.field === fieldToLink) {
            const queryParams = { ...formData.value, start: "0", end: "25" };
            if (label == null || label.length === 0) {
                // Should be NULL, but that's broken in the philo lib
                queryParams[fieldToLink] = "";
                label = t("common.na");
            } else {
                queryParams[fieldToLink] = `"${label}"`;
            }
            if (fieldToLink !== groupedByField.value) {
                queryParams[groupedByField.value] = `"${metadataFields[groupedByField.value]}"`;
            }
            // workaround for broken NULL searches
            const href = queryParams[fieldToLink].length
                ? paramsToRoute({ ...queryParams, report: "concordance" })
                : "";
            out.push({ ...citation, href, label });
        } else {
            out.push({ ...citation, href: "", label });
        }
    }
    return out;
}

function toggleBreakUp(resultIndex) {
    breakUpFields.value[resultIndex].show = !breakUpFields.value[resultIndex].show;
}

watch(urlUpdate, () => {
    if (formData.value.report === "aggregation") {
        groupedByField.value = route.query.group_by;
        fetchResults();
    }
});

formData.value.report = "aggregation";
currentReport.value = "aggregation";
fetchResults();
</script>
<style scoped lang="scss">
@use "../assets/styles/theme.module.scss" as theme;

#description {
    position: relative;
}

#export-results {
    position: absolute;
    right: 0;
    padding: 0.125rem 0.25rem;
    font-size: 0.8rem !important;
}

.badge {
    font-size: 100% !important;
}

.breakdown-container {
    padding: 0.75rem;
    margin-top: 0.5rem;
    position: relative;
}

/* Continuous vertical line for tree structure */
.breakdown-container::after {
    content: '';
    position: absolute;
    left: 0.5rem;
    top: 0;
    bottom: 0;
    width: 1px;
    background-color: theme.$link-color;
}

.breakdown-item {
    padding: 0.5rem 0;
    border-bottom: 1px solid #e9ecef;
    position: relative;
    padding-left: 1rem;
}

.breakdown-item:last-child {
    border-bottom: none;
}

/* Horizontal connector line */
.breakdown-item::before {
    content: '';
    position: absolute;
    left: -0.15rem;
    top: 50%;
    width: 0.5rem;
    height: 1px;
    background-color: theme.$link-color;
}

/* Hide the vertical line after the last item */
.breakdown-item:last-child::after {
    content: '';
    position: absolute;
    left: 0.5rem;
    top: 50%;
    bottom: 0;
    width: 1px;
    background-color: white;
}

.breakdown-content {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.breakdown-badge {
    font-size: 0.85em;
    min-width: 2.5rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    background-color: white !important;
    color: theme.$button-color !important;
    border: 1px solid theme.$button-color !important;
}

.breakdown-citation {
    flex: 1;
    font-size: 0.95em;
    color: #495057;
}

.list-group-item {
    border-left: 4px solid transparent;
    transition: border-left-color 0.2s ease;
}

.list-group-item:hover {
    border-left-color: theme.$link-color;
}

/* Expand/collapse button */
.btn-outline-secondary {
    border-color: theme.$link-color;
    color: theme.$link-color;
    transition: all 0.2s ease;
}

.btn-outline-secondary:hover {
    background-color: theme.$link-color;
    border-color: theme.$link-color;
    color: white;
}

.btn-outline-secondary[aria-expanded="true"] {
    background-color: theme.$button-color;
    border-color: theme.$button-color;
    color: white;
}

.bg-secondary {
    background-color: theme.$button-color !important;
    color: white !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}
</style>