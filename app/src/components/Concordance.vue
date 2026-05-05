<template>
    <div class="container-fluid">
        <results-summary :description="results.description"></results-summary>

        <div class="row concordance-layout px-2">
            <!-- Facets sidebar - appears first in DOM for mobile accessibility -->
            <div role="region" class="col-12 col-md-3 col-xl-3 facets-column" :aria-label="$t('common.facetsRegion')"
                v-if="showFacets">
                <facets></facets>
            </div>

            <!-- Results column -->
            <div role="region" class="col-12" :class="{ 'col-md-9': showFacets, 'col-xl-9': showFacets }"
                :aria-label="$t('concordance.resultsRegion')">

                <transition-group tag="div" :css="false" v-on:before-enter="onBeforeEnter" v-on:enter="onEnter">
                    <article class="card philologic-occurrence text-view mb-3 shadow-sm"
                        v-for="(result, index) in results.results" :key="result.philo_id.join('-')" :data-index="index">

                        <!-- Citation header -->
                        <div class="row citation-container g-0">
                            <div class="col-10 col-md-11">
                                <span class="cite" :id="`result-${index}-heading`">
                                    <span class="number" :aria-describedby="`result-number-desc-${index}`">
                                        {{ results.description.start + index }}
                                        <span :id="`result-number-desc-${index}`" class="visually-hidden">
                                            {{ $t('concordance.resultNumber') }} {{ results.description.start + index }}
                                        </span>
                                    </span>
                                    <citations :citation="result.citation"
                                        :result-number="results.description.start + index"></citations>
                                </span>
                            </div>
                            <div class="col-2 col-md-1">
                                <button type="button" class="btn btn-secondary more-context"
                                    @click="moreContext(index, $event)"
                                    :aria-label="`${$t('concordance.showMoreContext')} ${$t('common.forResult')} ${results.description.start + index}`">
                                    <span class="more-text">{{ $t("concordance.more") }}</span>
                                    <i class="bi bi-plus-square more-icon" aria-hidden="true"></i>
                                </button>
                            </div>
                        </div>

                        <!-- Concordance content -->
                        <div class="row">
                            <div class="col m-2 concordance-text" :position="results.description.start + index"
                                @keyup="dicoLookup($event, result.metadata_fields.year)">
                                <div class="default-length" v-html="result.context"></div>
                                <div class="more-length"></div>
                            </div>
                        </div>
                    </article>
                </transition-group>
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
import { useI18n } from "vue-i18n";
import { useMainStore } from "../stores/main";
import { useFadeTransition } from "../composables/useFadeTransition";
import { debug, isOnlyFacetChange, paramsFilter } from "../utils.js";
import Citations from "./Citations";  // eslint-disable-line no-unused-vars
import Facets from "./Facets";  // eslint-disable-line no-unused-vars
import Pages from "./Pages";  // eslint-disable-line no-unused-vars
import ResultsSummary from "./ResultsSummary";  // eslint-disable-line no-unused-vars

const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const { t } = useI18n();
const store = useMainStore();
const {
    formData,
    searching,
    currentReport,
    urlUpdate,
    showFacets,
    totalResultsDone,
} = storeToRefs(store);

const results = ref({ description: { end: 0 }, results: [] });
const searchParams = ref({});

const { beforeEnter: onBeforeEnter, enter: onEnter } = useFadeTransition();

provide("results", computed(() => results.value.results));

function fetchResults() {
    totalResultsDone.value = false;
    results.value = { description: { end: 0 } };
    searchParams.value = { ...formData.value };
    searching.value = true;
    $http
        .get(`${$dbUrl}/reports/concordance.py`, {
            params: paramsFilter(searchParams.value),
        })
        .then((response) => {
            results.value = response.data;
            store.updateResultsLength(parseInt(response.data.results_length));
            if (response.data.query_done) totalResultsDone.value = true;
            searching.value = false;
        })
        .catch((error) => {
            searching.value = false;
            debug({ $options: { name: "concordance-report" } }, error);
        });
}

function moreContext(index, event) {
    const button = event.target.closest("button");
    const textSpan = button.querySelector(".more-text");
    const icon = button.querySelector(".more-icon");
    const defaultNode = document.getElementsByClassName("default-length")[index];
    const moreNode = document.getElementsByClassName("more-length")[index];
    const resultNumber = results.value.description.start + index - 1;
    const localParams = { hit_num: resultNumber, ...searchParams.value };

    const isExpanded =
        textSpan.innerHTML === t("concordance.less") ||
        icon.classList.contains("bi-dash-square");

    if (isExpanded) {
        defaultNode.style.display = "block";
        moreNode.style.display = "none";
        textSpan.innerHTML = t("concordance.more");
        icon.classList.remove("bi-dash-square");
        icon.classList.add("bi-plus-square");
        return;
    }

    const showMore = () => {
        defaultNode.style.display = "none";
        moreNode.style.display = "block";
        textSpan.innerHTML = t("concordance.less");
        icon.classList.remove("bi-plus-square");
        icon.classList.add("bi-dash-square");
    };

    if (moreNode.innerHTML.length === 0) {
        $http
            .get(`${$dbUrl}/scripts/get_more_context.py`, {
                params: paramsFilter(localParams),
            })
            .then((response) => {
                moreNode.innerHTML = response.data;
                showMore();
            })
            .catch((error) => {
                debug({ $options: { name: "concordance-report" } }, error);
            });
    } else {
        showMore();
    }
}

// Template binds @keyup="dicoLookup(...)" but the original implementation was
// empty. Kept as a stub until a real implementation exists.
function dicoLookup() {}

watch(urlUpdate, (newUrl, oldUrl) => {
    if (!isOnlyFacetChange(newUrl, oldUrl)) fetchResults();
});

formData.value.report = "concordance";
currentReport.value = "concordance";
fetchResults();
</script>

<style>
.concordance-text {
    text-align: justify;
}

.philologic-occurrence {
    left: 0;
    position: relative;
}

.separator {
    padding: 5px;
    font-size: 60%;
    display: inline-block;
    vertical-align: middle;
}

.more-context {
    line-height: 1.8;
    position: absolute;
    right: 0;
}

/* Show text by default on larger screens (> 991px / Bootstrap lg) */
.more-text {
    display: inline-block;
}

.more-icon {
    display: none;
    font-size: 1.125rem;
    vertical-align: middle;
}

/* At smaller viewports or zoom, switch to icon-only */
@media (max-width: 991px) {
    .more-text {
        display: none;
    }

    .more-icon {
        display: inline-block;
    }

    .more-context {
        padding: 0.25rem 0.5rem;
        min-width: auto;
        line-height: 1.2;
    }

    .number {
        padding: 0.25rem 0.5rem !important;
        line-height: 1.2 !important;
    }
}

.more_context,
.citation-container {
    border-bottom: solid 1px #eee !important;
}

.number {
    color: #fff;
    font-size: 1rem;
    line-height: 1.5;
    padding: 0.375rem 0.75rem;
    display: inline-block;
    margin-right: 5px;
    border-radius: 0.25rem;
}

.hit_n {
    vertical-align: 5px;
}

.cite {
    height: 38px;
    display: inline-block;
}

.philologic-doc {
    font-variant: small-caps;
    font-weight: 700;
}

.citation-separator {
    margin-left: 8px;
    padding-left: 8px;
    border-left: double 3px darkgray;
}

.page-display {
    margin-left: 8px;
    padding-left: 8px;
    border-left: double 3px darkgray;
}

.citation-small-caps {
    font-variant: small-caps;
}

/* Concordance styling for theater */
.xml-speaker {
    font-weight: 700;
}

.xml-sp+.xml-l,
.xml-sp+.xml-ab {
    display: inline;
}

.xml-stage {
    font-style: italic;
}

/* Mobile layout: facets above results */
@media (max-width: 767px) {
    .concordance-layout {
        display: flex;
        flex-direction: column;
    }

    .facets-column {
        order: 1;
        margin-bottom: 1rem;
    }

    .concordance-layout>div[role="region"]:not(.facets-column) {
        order: 2;
    }

    .pages-wrapper {
        order: 3;
        width: 100%;
    }
}

/* Desktop layout: facets on right side */
@media (min-width: 768px) {
    .concordance-layout {
        display: flex;
        flex-direction: row;
        flex-wrap: wrap;
    }

    .facets-column {
        order: 2;
        padding-left: 0 !important;
    }

    .concordance-layout>div[role="region"]:not(.facets-column) {
        order: 1;
    }

    .pages-wrapper {
        order: 3;
        width: 100%;
    }
}
</style>
