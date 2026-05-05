<template>
    <div class="container-fluid">
        <results-summary :description="results.description"></results-summary>
        <div class="row px-2 kwic-layout">
            <!-- Facets sidebar - appears first in DOM for mobile accessibility -->
            <div role="region" class="col-12 col-md-4 col-xl-3 facets-column" :aria-label="$t('common.facetsRegion')"
                v-if="showFacets">
                <facets></facets>
            </div>

            <!-- Results column -->
            <div role="region" class="col-12" :class="{ 'col-md-8': showFacets, 'col-xl-9': showFacets }"
                :aria-label="$t('kwic.resultsRegion')">
                <div class="card p-2 ml-2 shadow-sm">
                    <div class="p-2 mb-1">
                        <!-- Sorting controls -->
                        <div class="btn-group">
                            <button type="button" class="btn btn-sm btn-outline-secondary" style="border-right: solid"
                                tabindex="-1" id="sort-group-label">
                                {{ $t("kwic.sortResultsBy") }}
                            </button>
                            <div role="group" aria-labelledby="sort-group-label" style="display: contents;">
                                <div class="btn-group" v-for="(fields, index) in sortingFields" :key="index">
                                    <div class="dropdown">
                                        <button class="btn btn-light btn-sm dropdown-toggle sort-toggle"
                                            :style="index == 0 ? 'border-left: 0 !important' : ''"
                                            :id="`kwicDrop${index}`" data-bs-toggle="dropdown" aria-expanded="false"
                                            :aria-label="getSortingAriaLabel(index)">
                                            {{ sortingSelection[index] }}
                                        </button>
                                        <ul class="dropdown-menu" :aria-labelledby="`kwicDrop${index}`">
                                            <li>
                                                <span class="dropdown-header">{{ sortingLabels[index] }}</span>
                                            </li>
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
                            </div>
                            <button type="button" class="btn btn-secondary btn-sm" id="sort-button"
                                @click="sortResults()" :aria-label="$t('kwic.applySorting')">
                                {{ $t("kwic.sort") }}
                            </button>
                        </div>
                    </div>

                    <div class="progress mt-3" v-if="runningTotal != resultsLength" role="progressbar"
                        :aria-valuenow="runningTotal" :aria-valuemin="0" :aria-valuemax="resultsLength">
                        <div class="progress-bar"
                            :style="`width: ${((runningTotal / resultsLength) * 100).toFixed(2)}%`" :aria-hidden="true">
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

                    <!-- KWIC results -->
                    <div id="kwic-concordance">
                        <transition-group tag="div" :css="false" v-on:before-enter="onBeforeEnter" v-on:enter="onEnter">
                            <div v-for="(result, kwicIndex) in filteredKwic(results.results)"
                                :key="result.philo_id.join('-')" :data-index="kwicIndex">

                                <!-- Default KWIC view -->
                                <div class="kwic-line visual-kwic" aria-hidden="true">
                                    <span v-html="initializePos(kwicIndex)"></span>
                                    <div class="kwic-biblio-container"
                                        style="display: inline-block; position: relative;"
                                        @mouseover="showFullBiblio($event)" @mouseleave="hideFullBiblio($event)">
                                        <router-link :to="result.citation_links.div1" class="kwic-biblio"
                                            @focus="showFullBiblio($event)" @blur="hideFullBiblio($event)">
                                            <span class="short-biblio" v-html="result.shortBiblio"></span>
                                            <div :id="`full-biblio-${kwicIndex}`" class="full-biblio" role="tooltip"
                                                :aria-hidden="true">
                                                {{ result.fullBiblio }}
                                            </div>
                                        </router-link>
                                    </div>
                                    <div class="kwic-context">
                                        <span v-html="result.context"></span>
                                    </div>
                                </div>

                                <!-- Accessible version for screen readers -->
                                <div class="kwic-line accessible-kwic visually-hidden">
                                    <span>{{ results.description.start + kwicIndex }}.</span>
                                    <router-link :to="result.citation_links.div1" class="kwic-biblio" tabindex="-1"
                                        :aria-describedby="`kwic-context-${kwicIndex}`">
                                        {{ result.fullBiblio }}
                                    </router-link>
                                    <span :id="`kwic-context-${kwicIndex}`">
                                        {{ $t('kwic.contextDescription', {
                                            context: result.context.replace(/<[^>]*>/g, '').trim().substring(0, 60)
                                        }) }}
                                    </span>
                                    <div class="accessible-context">
                                        <span v-html="result.context"></span>
                                    </div>
                                </div>
                            </div>
                        </transition-group>
                    </div>
                </div>
            </div>

            <div class="pages-wrapper">
                <pages></pages>
            </div>
        </div>
    </div>
</template>

<script setup>
import { computed, inject, onBeforeUnmount, provide, ref, watch } from "vue";
import { useRouter } from "vue-router";
import { storeToRefs } from "pinia";
import { useI18n } from "vue-i18n";
import { useMainStore } from "../stores/main";
import { useFadeTransition } from "../composables/useFadeTransition";
import { debug, isOnlyFacetChange, paramsFilter, paramsToRoute } from "../utils.js";
import Facets from "./Facets";  // eslint-disable-line no-unused-vars
import Pages from "./Pages";  // eslint-disable-line no-unused-vars
import ResultsSummary from "./ResultsSummary";  // eslint-disable-line no-unused-vars

const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const philoConfig = inject("$philoConfig");
const router = useRouter();
const { t } = useI18n();
const store = useMainStore();
const {
    formData,
    resultsLength,
    searching,
    currentReport,
    urlUpdate,
    showFacets,
    totalResultsDone,
} = storeToRefs(store);

const results = ref({ description: { end: 0 } });
const searchParams = ref({});
const runningTotal = ref(0);

const { beforeEnter: onBeforeEnter, enter: onEnter } = useFadeTransition(0.0075);

provide("results", computed(() => results.value.results));

// AbortController for the streaming sorted-KWIC fetch. Held as a module-scope
// `let` (not a ref) since its identity isn't needed reactively — we just need
// to keep a handle to abort on unmount or query change.
let abortController = null;

// ── Sort options (writable computeds proxy formData) ─────────────────────────
const first_kwic_sorting_option = computed({
    get: () => formData.value.first_kwic_sorting_option,
    set: (value) => { formData.value.first_kwic_sorting_option = value; },
});
const second_kwic_sorting_option = computed({
    get: () => formData.value.second_kwic_sorting_option,
    set: (value) => { formData.value.second_kwic_sorting_option = value; },
});
const third_kwic_sorting_option = computed({
    get: () => formData.value.third_kwic_sorting_option,
    set: (value) => { formData.value.third_kwic_sorting_option = value; },
});

const sortingFields = computed(() => {
    const fields = [
        { label: t("common.none"), field: "" },
        { label: t("kwic.searchedTerms"), field: "q" },
        { label: t("kwic.wordsLeft"), field: "left" },
        { label: t("kwic.wordsRight"), field: "right" },
    ];
    for (const field of philoConfig.kwic_metadata_sorting_fields) {
        const label = field in philoConfig.metadata_aliases
            ? philoConfig.metadata_aliases[field]
            : field[0].toUpperCase() + field.slice(1);
        fields.push({ label, field });
    }
    return [fields, fields, fields];
});

const sortKeys = computed(() => {
    const keys = {
        q: t("kwic.searchedTerms"),
        left: t("kwic.wordsLeft"),
        right: t("kwic.wordsRight"),
    };
    for (const field of philoConfig.kwic_metadata_sorting_fields) {
        keys[field] = field in philoConfig.metadata_aliases
            ? philoConfig.metadata_aliases[field]
            : field[0].toUpperCase() + field.slice(1);
    }
    return keys;
});

const sortingSelection = computed(() => [
    first_kwic_sorting_option.value !== ""
        ? sortKeys.value[first_kwic_sorting_option.value]
        : t("kwic.firstLabel"),
    second_kwic_sorting_option.value !== ""
        ? sortKeys.value[second_kwic_sorting_option.value]
        : t("kwic.secondLabel"),
    third_kwic_sorting_option.value !== ""
        ? sortKeys.value[third_kwic_sorting_option.value]
        : t("kwic.thirdLabel"),
]);

const sortingLabels = computed(() => [
    t("kwic.firstLabel"),
    t("kwic.secondLabel"),
    t("kwic.thirdLabel"),
]);

// ── KWIC view helpers ────────────────────────────────────────────────────────
function buildFullCitation(metadataField) {
    let biblioFields = philoConfig.kwic_bibliography_fields;
    if (typeof biblioFields === "undefined" || biblioFields.length === 0) {
        biblioFields = philoConfig.metadata.slice(0, 2);
        biblioFields.push("head");
    }
    const out = [];
    for (const f of biblioFields) {
        if (f in metadataField) {
            const value = metadataField[f] || "";
            if (value.length > 0) out.push(value);
        }
    }
    return out.length > 0 ? out.join(", ") : "NA";
}

function filteredKwic(rawResults) {
    if (typeof rawResults === "undefined" || !Object.keys(rawResults).length) return [];
    return rawResults.map((resultObject) => {
        resultObject.fullBiblio = buildFullCitation(resultObject.metadata_fields);
        resultObject.shortBiblio = resultObject.fullBiblio.slice(0, 30);
        return resultObject;
    });
}

function showFullBiblio(event) {
    event.currentTarget.querySelector(".full-biblio").classList.add("show");
}

function hideFullBiblio(event) {
    event.currentTarget.querySelector(".full-biblio").classList.remove("show");
}

function initializePos(index) {
    const start = results.value.description.start;
    const currentPos = start + index;
    const currentPosLength = currentPos.toString().length;
    const endPos = start + parseInt(formData.value.results_per_page) || 25;
    const endPosLength = endPos.toString().length;
    const spaces = endPosLength - currentPosLength + 1;
    return currentPos + "." + Array(spaces).join("&nbsp");
}

// ── Sort UI handlers ─────────────────────────────────────────────────────────
function getSortingAriaLabel(index) {
    const current = sortingSelection.value[index];
    if (index === 0) return t("kwic.firstSortingCriteria", { current });
    if (index === 1) return t("kwic.secondSortingCriteria", { current });
    return t("kwic.thirdSortingCriteria", { current });
}

function updateSortingSelection(index, selection) {
    const value = selection.label === t("common.none") ? "" : selection.field;
    if (index === 0) first_kwic_sorting_option.value = value;
    else if (index === 1) second_kwic_sorting_option.value = value;
    else third_kwic_sorting_option.value = value;
}

function sortResults() {
    results.value.results = [];
    router.push(paramsToRoute({ ...formData.value }));
}

// ── Fetch path ───────────────────────────────────────────────────────────────
async function fetchSortedResults() {
    searching.value = true;
    runningTotal.value = 0;

    const params = new URLSearchParams(paramsFilter({ ...formData.value }));
    const url = `${$dbUrl}/scripts/get_sorted_kwic.py?${params}`;

    if (abortController) abortController.abort();
    abortController = new AbortController();

    const applyData = (data) => {
        if (data.progress) {
            runningTotal.value = data.progress.hits_done;
            resultsLength.value = data.progress.total;
        } else {
            results.value = data;
            resultsLength.value = data.results_length;
            runningTotal.value = data.results_length;
        }
    };

    try {
        const response = await fetch(url, { signal: abortController.signal });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        // eslint-disable-next-line no-constant-condition
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.trim()) continue;
                applyData(JSON.parse(line));
            }
        }

        if (buffer.trim()) applyData(JSON.parse(buffer));
    } catch (e) {
        if (e.name === "AbortError") return;
        debug({ $options: { name: "kwic-report" } }, e);
    } finally {
        searching.value = false;
        abortController = null;
    }
}

function fetchResults() {
    totalResultsDone.value = false;
    results.value = { description: { end: 0 }, results: [] };
    searchParams.value = { ...formData.value };
    const hasSorting =
        first_kwic_sorting_option.value !== "" ||
        second_kwic_sorting_option.value !== "" ||
        third_kwic_sorting_option.value !== "";

    if (hasSorting) {
        if (formData.value.start === "") {
            formData.value.start = "0";
            formData.value.end = formData.value.results_per_page;
        }
        fetchSortedResults();
        return;
    }

    searching.value = true;
    $http
        .get(`${$dbUrl}/reports/kwic.py`, { params: paramsFilter(searchParams.value) })
        .then((response) => {
            results.value = response.data;
            resultsLength.value = response.data.results_length;
            runningTotal.value = response.data.results_length;
            results.value.description = response.data.description;
            if (response.data.query_done) totalResultsDone.value = true;
            searching.value = false;
        })
        .catch((error) => {
            searching.value = false;
            debug({ $options: { name: "kwic-report" } }, error);
        });
}

watch(urlUpdate, (newUrl, oldUrl) => {
    if (!isOnlyFacetChange(newUrl, oldUrl)) {
        if (abortController) abortController.abort();
        fetchResults();
    }
});

onBeforeUnmount(() => {
    if (abortController) abortController.abort();
});

formData.value.report = "kwic";
currentReport.value = "kwic";
fetchResults();
</script>

<style scoped lang="scss">
@use "../assets/styles/theme.module.scss" as theme;

.sort-toggle {
    border-bottom-left-radius: 0;
    border-top-left-radius: 0;
    border-bottom-right-radius: 0;
    border-top-right-radius: 0;
}

.dropdown-menu .dropdown-header {
    font-weight: 700;
    text-transform: uppercase;
    font-size: 0.7rem;
    letter-spacing: 0.05em;
    color: theme.$link-color;
    padding: 0.25rem 1rem 0.4rem;
    margin-top: -0.25rem;
    margin-bottom: 0.25rem;
    border-bottom: 1px solid rgba(theme.$button-color, 0.2);
}

#kwic-concordance {
    font-family: monospace;
}

/* Visual KWIC layout for sighted users */
.visual-kwic {
    line-height: 1.8rem;
    white-space: nowrap;
    display: flex;
    overflow: hidden;
}

.visual-kwic .kwic-biblio-container {
    flex-shrink: 0;
}

.visual-kwic .kwic-context {
    overflow: hidden;
    white-space: nowrap;
    flex: 1;
    position: relative;
}

.visual-kwic :deep(.kwic-before) {
    text-align: right;
    display: inline-block;
    position: absolute;
}

/* Accessible version for screen readers */
.accessible-kwic {
    white-space: normal;
    line-height: 1.6;
    padding: 0.5rem 0;
}

.accessible-kwic .accessible-context {
    margin-top: 0.25rem;
}

.accessible-kwic .kwic-biblio {
    font-weight: bold;
    margin: 0 0.5rem;
}

.kwic-biblio {
    font-weight: 400 !important;
    z-index: 10;
    padding-right: 0.5rem;
}

.kwic-biblio:focus {
    outline: 2px solid theme.$button-color;
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
    background-color: theme.$button-color;
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
    background-color: theme.$button-color !important;
    outline: 2px solid rgba(255, 255, 255, 0.5);
    outline-offset: 2px;
    box-shadow: 0 0 0 0.2rem rgba(255, 255, 255, 0.25);
}

.progress {
    border-radius: 0.375rem;
    overflow: hidden;
}

.progress-bar {
    background-color: theme.$button-color !important;
}

.dropdown-item:focus {
    outline: 2px solid theme.$button-color;
    outline-offset: -2px;
    background-color: rgba(theme.$button-color, 0.1);
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

@media (min-width: 2000px) {
    .visual-kwic :deep(.kwic-highlight) {
        margin-left: 600px;
    }

    .visual-kwic :deep(.kwic-before) {
        width: 600px;
    }
}

@media (min-width: 1600px) and (max-width: 1999px) {
    .visual-kwic :deep(.kwic-highlight) {
        margin-left: 450px;
    }

    .visual-kwic :deep(.kwic-before) {
        width: 450px;
    }
}

@media (min-width: 1300px) and (max-width: 1599px) {
    .visual-kwic :deep(.kwic-highlight) {
        margin-left: 330px;
    }

    .visual-kwic :deep(.kwic-before) {
        width: 330px;
    }
}

@media (min-width: 992px) and (max-width: 1299px) {
    .visual-kwic :deep(.kwic-highlight) {
        margin-left: 230px;
    }

    .visual-kwic :deep(.kwic-before) {
        width: 230px;
    }
}

@media (min-width: 768px) and (max-width: 991px) {
    .visual-kwic :deep(.kwic-highlight) {
        margin-left: 120px;
    }

    .visual-kwic :deep(.kwic-before) {
        width: 120px;
    }

    .visual-kwic :deep(.kwic-line) {
        font-size: 12px;
    }
}

@media (max-width: 767px) {
    .visual-kwic :deep(.kwic-highlight) {
        margin-left: 200px;
    }

    .visual-kwic :deep(.kwic-before) {
        width: 200px;
    }

    .visual-kwic :deep(.kwic-line) {
        font-size: 12px;
    }

    /* Mobile layout: facets above results */
    .kwic-layout {
        display: flex;
        flex-direction: column;
    }

    .facets-column {
        order: 1;
        margin-bottom: 1rem;
    }

    .kwic-layout>div[role="region"]:not(.facets-column) {
        order: 2;
    }

    .pages-wrapper {
        order: 3;
        width: 100%;
    }
}

/* Desktop layout: facets on right side */
@media (min-width: 768px) {
    .kwic-layout {
        display: flex;
        flex-direction: row;
        flex-wrap: wrap;
    }

    .facets-column {
        order: 2;
    }

    .kwic-layout>div[role="region"]:not(.facets-column) {
        order: 1;
    }

    .pages-wrapper {
        order: 3;
        width: 100%;
    }
}
</style>
