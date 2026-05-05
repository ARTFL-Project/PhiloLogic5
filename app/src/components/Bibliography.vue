<template>
    <div class="container-fluid">
        <results-summary :description="results.description"></results-summary>

        <div class="row bibliography-layout mt-3 pe-1">
            <!-- Facets sidebar - appears first in DOM for mobile accessibility -->
            <div role="region" class="col-12 col-md-3 col-xl-3 facets-column" :aria-label="$t('common.facetsRegion')"
                v-if="showFacets">
                <facets></facets>
            </div>

            <!-- Results column for regular bibliography -->
            <div role="region" class="col-12" :class="{ 'col-md-9': showFacets, 'col-xl-9': showFacets }"
                :aria-label="$t('bibliography.bibliographyResultsRegion')"
                v-if="!philoConfig.dictionary_bibliography || results.result_type == 'doc'">
                <transition-group tag="div" :css="false" v-on:before-enter="beforeEnter" v-on:enter="enter">
                    <article class="card philologic-occurrence mx-2 mb-4 shadow-sm"
                        v-for="(result, index) in results.results" :key="result.philo_id.join('-')" role="article">
                        <div class="row citation-container">
                            <div class="col-12">
                                <div class="cite d-flex align-items-center" :data-id="result.philo_id.join(' ')">
                                    <span class="number flex-shrink-0" aria-hidden="true">{{
                                        results.description.start + index
                                    }}</span>
                                    <div :id="`citation-${results.description.start + index}`" class="flex-grow-1">
                                        <citations :citation="result.citation"
                                            :result-number="results.description.start + index"></citations>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </article>
                </transition-group>
            </div>

            <!-- Results column for dictionary bibliography -->
            <div role="region" class="col-12" :class="{ 'col-md-9': showFacets, 'col-xl-9': showFacets }"
                :aria-label="$t('bibliography.dictionaryResultsRegion')"
                v-if="philoConfig.dictionary_bibliography && results.result_type != 'doc'">
                <div class="list-group" v-for="(group, groupKey) in results.results" :key="groupKey" role="group">
                    <article class="list-group-item p-0" v-for="(result, index) in group" :key="index"
                        style="border-width: 0">
                        <div class="card philologic-occurrence mx-2 mb-4 shadow-sm">
                            <header class="citation-dico-container">
                                <div class="cite" :data-id="result.philo_id.join(' ')"
                                    :id="`dict-citation-${groupKey}-${index}`">
                                    <span class="number" aria-hidden="true">{{ results.description.start + index
                                        }}</span>
                                    <citations :citation="result.citation"
                                        :result-number="results.description.start + index"></citations>
                                </div>
                            </header>
                            <div class="pt-3 px-3 text-content" select-word :position="result.position">
                                <div v-html="result.context"></div>
                            </div>
                        </div>
                    </article>
                </div>
            </div>

            <div class="pages-wrapper">
                <pages></pages>
            </div>
        </div>
    </div>
</template>
<script setup>
import { computed, inject, provide, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import { useMainStore } from "../stores/main";
import { useFadeTransition } from "../composables/useFadeTransition";
import { debug, isOnlyFacetChange, paramsFilter } from "../utils.js";
import citations from "./Citations";
import facets from "./Facets";
import pages from "./Pages";
import ResultsSummary from "./ResultsSummary";

const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const philoConfig = inject("$philoConfig");
const store = useMainStore();
const {
    formData,
    currentReport,
    urlUpdate,
    showFacets,
    searching,
    totalResultsDone,
} = storeToRefs(store);

const results = ref({});
const resultType = ref("doc");

provide("results", computed(() => results.value.results));

function fetchResults() {
    totalResultsDone.value = false;
    results.value = {};
    searching.value = true;
    $http
        .get(`${$dbUrl}/reports/bibliography.py`, {
            params: paramsFilter({ ...formData.value }),
        })
        .then((response) => {
            if (!philoConfig.dictionary_bibliography || response.data.doc_level) {
                results.value = response.data;
                resultType.value = results.value.result_type;
            } else {
                results.value = dictionaryBibliography(response.data);
            }
            if (response.data.results_length !== undefined) {
                store.updateResultsLength(parseInt(response.data.results_length));
            }
            if (response.data.query_done) {
                totalResultsDone.value = true;
            }
            searching.value = false;
        })
        .catch((error) => {
            searching.value = false;
            debug({ $options: { name: "bibliography-report" } }, error);
        });
}

function dictionaryBibliography(data) {
    const groupedResults = [];
    let currentTitle = data.results[0].metadata_fields.title;
    let titleGroup = [];
    for (let i = 0; i < data.results.length; i += 1) {
        if (data.results[i].metadata_fields.title !== currentTitle) {
            groupedResults.push(titleGroup);
            titleGroup = [];
            currentTitle = data.results[i].metadata_fields.title;
        }
        data.results[i].position = i + 1;
        titleGroup.push(data.results[i]);
        if (i + 1 === data.results.length) {
            groupedResults.push(titleGroup);
        }
    }
    data.results = groupedResults;
    return data;
}

const { beforeEnter, enter } = useFadeTransition();

watch(urlUpdate, (newUrl, oldUrl) => {
    if (!isOnlyFacetChange(newUrl, oldUrl)) {
        fetchResults();
    }
});

formData.value.report = "bibliography";
currentReport.value = "bibliography";
fetchResults();
</script>
<style scoped>
.citation-container {
    border-width: 0 !important;
}

.citation-dico-container {
    border-bottom: solid 1px #eee !important;
}

.number {
    background-color: rgb(78, 93, 108);
    color: #fff;
    font-size: 1rem;
    line-height: 1.5;
    padding: 0.375rem 0.75rem;
    display: inline-block;
    margin-right: 5px;
    border-radius: 0.25rem;
}

.text-content {
    text-align: justify;
}

/* Mobile layout: facets above results */
@media (max-width: 767px) {
    .bibliography-layout {
        display: flex;
        flex-direction: column;
    }

    .facets-column {
        order: 1;
        margin-bottom: 1rem;
    }

    .bibliography-layout>div[role="region"]:not(.facets-column) {
        order: 2;
    }

    .pages-wrapper {
        order: 3;
        width: 100%;
    }
}

/* Desktop layout: facets on right side */
@media (min-width: 768px) {
    .bibliography-layout {
        display: flex;
        flex-direction: row;
        flex-wrap: wrap;
    }

    .facets-column {
        order: 2;
        padding-left: 0 !important;
    }

    .bibliography-layout>div[role="region"]:not(.facets-column) {
        order: 1;
    }

    .pages-wrapper {
        order: 3;
        width: 100%;
    }
}
</style>
