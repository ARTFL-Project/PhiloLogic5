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
                        <div class="btn-group sorting-controls">
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
                                @click="sortResults()">
                                {{ $t("kwic.sort") }}
                            </button>
                        </div>
                    </div>

                    <!-- KWIC results -->
                    <div id="kwic-concordance">
                        <div>
                            <div v-for="(result, kwicIndex) in kwicResults"
                                :key="result.philo_id.join('-')" class="result-fade-in" :style="{ '--i': kwicIndex }">

                                <!-- the line, for everyone: its link named by the short citation, described by the full one -->
                                <div class="kwic-line visual-kwic">
                                    <span v-html="initializePos(kwicIndex)"></span>
                                    <div class="kwic-biblio-container"
                                        style="display: inline-block; position: relative;">
                                        <router-link :to="result.citation_links.div1" class="kwic-biblio"
                                            :aria-describedby="`full-biblio-${kwicIndex}`">
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
                            </div>
                        </div>
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
import { computed, inject, provide, ref, watch } from "vue";
import { useRouter } from "vue-router";
import { storeToRefs } from "pinia";
import { useI18n } from "vue-i18n";
import { useReport } from "../composables/useReport";
import { useMainStore } from "../stores/main";
import { paramsFilter, paramsToRoute } from "../utils.js";
import Facets from "./Facets";
import Pages from "./Pages";
import ResultsSummary from "./ResultsSummary";

const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const philoConfig = inject("$philoConfig");
const router = useRouter();
const { t } = useI18n();
const store = useMainStore();
const {
    formData,
    resultsLength,
    showFacets,
    totalResultsDone,
} = storeToRefs(store);

const results = ref({ description: { end: 0 } });
const searchParams = ref({});

provide("results", computed(() => results.value.results));

// ── Sort options ─────────────────────────────────────────────────────────────
// The dropdowns' choices, which Sort runs: the search shown (its links, the next page) keeps its own until then, and
// the dropdowns show them again when it changes
const SORT_OPTIONS = ["first_kwic_sorting_option", "second_kwic_sorting_option", "third_kwic_sorting_option"];
const sortChoices = ref(SORT_OPTIONS.map((option) => formData.value[option]));
watch(
    () => SORT_OPTIONS.map((option) => formData.value[option]),
    (shown) => {
        sortChoices.value = shown;
    }
);

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

const sortingSelection = computed(() =>
    sortChoices.value.map((choice, index) => (choice !== "" ? sortKeys.value[choice] : sortingLabels.value[index]))
);

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

// The lines with their citations, whole and cut short: made once per response, not at each render
const kwicResults = computed(() => {
    const rawResults = results.value.results;
    if (typeof rawResults === "undefined" || !Object.keys(rawResults).length) return [];
    return rawResults.map((resultObject) => {
        const fullBiblio = buildFullCitation(resultObject.metadata_fields);
        return { ...resultObject, fullBiblio, shortBiblio: fullBiblio.slice(0, 30) };
    });
});

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
    sortChoices.value[index] = selection.label === t("common.none") ? "" : selection.field;
}

function sortResults() {
    // a new sort order changes the URL, and the results are fetched again; the same leaves them as they are
    const choices = Object.fromEntries(SORT_OPTIONS.map((option, index) => [option, sortChoices.value[index]]));
    router.push(paramsToRoute({ ...formData.value, ...choices }));
}

// ── Fetch path ───────────────────────────────────────────────────────────────
// Sorted, the lines come from get_sorted_kwic.py, as one JSON response that can take a while
const { run: fetchResults } = useReport("kwic", async (signal) => {
    totalResultsDone.value = false;
    results.value = { description: { end: 0 }, results: [] };
    searchParams.value = { ...formData.value };
    const hasSorting = SORT_OPTIONS.some((option) => formData.value[option] !== "");

    if (hasSorting) {
        if (formData.value.start === "") {
            formData.value.start = "0";
            formData.value.end = formData.value.results_per_page;
        }
        const { data } = await $http.get(`${$dbUrl}/scripts/get_sorted_kwic.py`, {
            params: paramsFilter({ ...formData.value }),
            signal,
        });
        results.value = data;
        resultsLength.value = data.results_length;
        return;
    }

    const { data } = await $http.get(`${$dbUrl}/reports/kwic.py`, { params: paramsFilter(searchParams.value), signal });
    results.value = data;
    resultsLength.value = data.results_length;
    if (data.query_done) totalResultsDone.value = true;
});

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
    --fade-step: 7.5ms; /* the lines follow each other twice as fast as concordance cards */
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

.kwic-biblio-container:hover .full-biblio,
.kwic-biblio-container:focus-within .full-biblio {
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
    /* The context on a line of its own, under the citation: next to it, it was too narrow to show the keyword */
    .visual-kwic {
        flex-wrap: wrap;
    }

    .visual-kwic .kwic-context {
        flex-basis: 100%;
    }

    .visual-kwic :deep(.kwic-highlight) {
        margin-left: 40vw;
    }

    .visual-kwic :deep(.kwic-before) {
        width: 40vw;
    }

    .sorting-controls {
        flex-wrap: wrap;
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
