<template>
    <div id="facet-search">
        <div class="card shadow-sm" title="Title" header-tag="header" id="facet-panel-wrapper">
            <div class="card-header text-center">
                <h2 class="h6 mb-0">{{ $t("facets.browseByFacet") }}</h2>
            </div>
            <button type="button" class="btn btn-secondary btn-sm close-box" @click="store.toggleFacets()"
                :aria-label="$t('facets.closeFacets')">
                <span class="icon-x"></span>
            </button>

            <transition name="slide-fade">
                <div class="list-group" flush id="select-facets" v-if="showFacetSelection" role="group"
                    :aria-label="$t('facets.selectFacetType')">
                    <span class="dropdown-header text-center">{{ $t("facets.frequencyBy") }}</span>
                    <button type="button" class="list-group-item list-group-item-action facet-selection"
                        v-for="facet in facets" :key="facet.alias" @click="facetSearch(facet)"
                        :aria-describedby="`facet-desc-${facet.facet}`">
                        {{ facet.alias }}
                        <span :id="`facet-desc-${facet.facet}`" class="visually-hidden">
                            {{ $t('facets.selectFacet') }} {{ facet.alias }}
                        </span>
                    </button>
                </div>
            </transition>

            <transition name="slide-fade">
                <div class="list-group mt-3" style="border-top: 0" flush id="select-word-properties"
                    v-if="showFacetSelection && formData.report != 'bibliography' && philoConfig.words_facets.length > 0"
                    role="group" :aria-label="$t('facets.selectWordProperty')">
                    <span class="dropdown-header text-center">{{ $t("facets.wordProperty") }}</span>
                    <button type="button" class="list-group-item list-group-item-action facet-selection"
                        v-for="facet in wordFacets" :key="facet.facet" @click="facetSearch(facet)"
                        :aria-describedby="`word-facet-desc-${facet.facet}`">
                        {{ facet.alias }}
                        <span :id="`word-facet-desc-${facet.facet}`" class="visually-hidden">
                            {{ $t('facets.selectWordProperty') }} {{ facet.alias }}
                        </span>
                    </button>
                </div>
            </transition>

            <transition name="slide-fade">
                <div class="list-group mt-3" style="border-top: 0"
                    v-if="showFacetSelection && formData.report != 'bibliography'" role="group"
                    :aria-label="$t('facets.selectCollocation')">
                    <span class="dropdown-header text-center">{{ $t("facets.collocates") }}</span>
                    <button type="button" class="list-group-item list-group-item-action facet-selection"
                        @click="facetSearch(collocationFacet)" v-if="formData.report !== 'bibliography'"
                        :aria-describedby="'collocation-desc'">
                        {{ $t("common.sameSentence") }}
                        <span id="collocation-desc" class="visually-hidden">
                            {{ $t('facets.selectCollocation') }} {{ $t('common.sameSentence') }}
                        </span>
                    </button>
                </div>
            </transition>

            <transition name="options-slide">
                <button type="button" class="btn btn-link m-2 text-center"
                    style="width: 100%; font-size: 90%; opacity: 0.8" v-if="!showFacetSelection"
                    @click="showFacetOptions()" :aria-describedby="'show-options-desc'">
                    {{ $t("facets.showOptions") }}
                    <span id="show-options-desc" class="visually-hidden">
                        {{ $t('facets.showFacetOptions') }}
                    </span>
                </button>
            </transition>
        </div>

        <!-- Loading indicator -->
        <div class="d-flex justify-content-center position-relative" v-if="loading">
            <div class="position-absolute" style="z-index: 50; top: 10px">
                <progress-spinner :lg="true" :message="$t('common.loadingFacets')" />
            </div>
        </div>

        <!-- Facet results -->
        <div class="card mt-3 shadow-sm" id="facet-results" v-if="showFacetResults" role="region"
            :aria-label="$t('facets.facetResultsRegion')">
            <div class="card-header text-center">
                <h3 class="mb-0 h6">{{ $t("facets.frequencyByLabel", { label: selectedFacet.alias }) }}</h3>
                <button type="button" class="btn btn-secondary btn-sm close-box" @click="hideFacets()"
                    :aria-label="$t('facets.hideFacetResults')">
                    <span class="icon-x"></span>
                </button>
            </div>

            <!-- Frequency toggle buttons -->
            <div class="btn-group btn-group-sm shadow-sm" role="group" :aria-label="$t('facets.frequencyTypeToggle')"
                v-if="showFacetResults && formData.report !== 'bibliography' && selectedFacet.type === 'facet'">
                <button type="button" class="btn btn-light" :class="{ active: showingRelativeFrequencies === false }"
                    @click="toggleFrequencies()" :aria-pressed="showingRelativeFrequencies === false"
                    :aria-describedby="'absolute-freq-desc'">
                    {{ $t("common.absoluteFrequency") }}
                    <span id="absolute-freq-desc" class="visually-hidden">
                        {{ $t('facets.showAbsoluteFrequency') }}
                    </span>
                </button>
                <button type="button" class="btn btn-light" :class="{ active: showingRelativeFrequencies }"
                    @click="toggleFrequencies()" :aria-pressed="showingRelativeFrequencies"
                    :aria-describedby="'relative-freq-desc'">
                    {{ $t("common.relativeFrequency") }}
                    <span id="relative-freq-desc" class="visually-hidden">
                        {{ $t('facets.showRelativeFrequency') }}
                    </span>
                </button>
            </div>

            <div class="m-2 text-center" style="opacity: 0.7">
                {{ $t("facets.top100Results", { label: selectedFacet.alias }) }}
            </div>

            <!-- Facet results list -->
            <ul class="list-group facet-results-container" flush :aria-label="$t('facets.facetResultsList')">
                <!-- Facet link -->
                <li v-if="selectedFacet.type == 'facet'" v-for="result in facetResults" :key="result.label">
                    <button type="button" class="list-group-item list-group-item-action facet-result-item"
                        @click="facetClick(result.metadata)"
                        :aria-describedby="`facet-result-desc-${result.label.replace(/[^a-zA-Z0-9]/g, '-')}`">
                        <div class="d-flex justify-content-between align-items-start">
                            <span class="sidebar-text text-content-area text-view">
                                {{ result.label }}
                            </span>
                            <span class="badge bg-secondary rounded-pill" aria-hidden="true">
                                {{ result.count }}
                            </span>
                        </div>
                        <span :id="`facet-result-desc-${result.label.replace(/[^a-zA-Z0-9]/g, '-')}`"
                            class="visually-hidden">
                            {{ $t('facets.filterBy') }} {{ result.label }}, {{ result.count }} {{
                                $t('facets.occurrences') }}
                        </span>

                        <!-- Relative frequency description -->
                        <div class="relative-frequency-info"
                            v-if="showingRelativeFrequencies && result.absolute_count != null">
                            <small class="text-muted">
                                {{
                                    $t("facets.relativeFrequencyDescription", {
                                        total: result.absolute_count,
                                        wordCount: result.total_word_count,
                                    })
                                }}
                            </small>
                        </div>
                    </button>
                </li>

                <!-- Property link -->
                <li v-if="selectedFacet.type == 'property' && selectedFacet.facet != 'lemma'" v-for="result in facetResults"
                    :key="`property-${result.label}`">
                    <button type="button" class="list-group-item list-group-item-action facet-result-item"
                        @click="propertyToConcordance(result.q)"
                        :aria-describedby="`property-result-desc-${result.label.replace(/[^a-zA-Z0-9]/g, '-')}`">
                        <div class="d-flex justify-content-between align-items-start">
                            <span class="sidebar-text text-content-area">
                                {{ result.label }}
                            </span>
                            <span class="badge bg-secondary rounded-pill" aria-hidden="true">
                                {{ result.count }}
                            </span>
                        </div>
                        <span :id="`property-result-desc-${result.label.replace(/[^a-zA-Z0-9]/g, '-')}`"
                            class="visually-hidden">
                            {{ $t('facets.searchFor') }} {{ result.label }}, {{ result.count }} {{
                                $t('facets.occurrences') }}
                        </span>
                    </button>
                </li>

                <!-- Lemma (non-clickable) -->
                <li v-if="selectedFacet.type == 'property' && selectedFacet.facet == 'lemma'" v-for="result in facetResults"
                    :key="`lemma-${result.label}`">
                    <div class="list-group-item facet-result-item non-clickable"
                        :aria-describedby="`lemma-result-desc-${result.label.replace(/[^a-zA-Z0-9]/g, '-')}`">
                        <div class="d-flex justify-content-between align-items-start">
                            <span class="text-content-area">
                                {{ result.label }}
                            </span>
                            <span class="badge bg-secondary rounded-pill" aria-hidden="true">
                                {{ result.count }}
                            </span>
                        </div>
                        <span :id="`lemma-result-desc-${result.label.replace(/[^a-zA-Z0-9]/g, '-')}`"
                            class="visually-hidden">
                            {{ result.label }}, {{ result.count }} {{ $t('facets.occurrences') }}
                        </span>
                    </div>
                </li>

                <!-- Collocation link -->
                <li v-if="selectedFacet.type == 'collocationFacet'" v-for="result in facetResults"
                    :key="`colloc-${result.label}`">
                    <button type="button" class="list-group-item list-group-item-action facet-result-item"
                        @click="collocationToConcordance(result.collocate)"
                        :aria-describedby="`colloc-result-desc-${result.collocate.replace(/[^a-zA-Z0-9]/g, '-')}`">
                        <div class="d-flex justify-content-between align-items-start">
                            <span class="sidebar-text text-content-area">
                                {{ result.collocate }}
                            </span>
                            <span class="badge bg-secondary rounded-pill" aria-hidden="true">
                                {{ result.count }}
                            </span>
                        </div>
                        <span :id="`colloc-result-desc-${result.collocate.replace(/[^a-zA-Z0-9]/g, '-')}`"
                            class="visually-hidden">
                            {{ $t('facets.searchCollocation') }} {{ result.collocate }}, {{ result.count }} {{
                                $t('facets.occurrences') }}
                        </span>
                    </button>
                </li>
            </ul>
        </div>
    </div>
</template>

<script setup>
import { inject, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { storeToRefs } from "pinia";
import { useI18n } from "vue-i18n";
import { useMainStore } from "../stores/main";
import {
    copyObject,
    debug,
    extractSurfaceFromCollocate,
    isOnlyFacetChange,
    paramsFilter,
    paramsToRoute,
    paramsToUrlString,
    saveToLocalStorage,
} from "../utils.js";
import ProgressSpinner from "./ProgressSpinner";  // eslint-disable-line no-unused-vars

const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const philoConfig = inject("$philoConfig");
const route = useRoute();
const router = useRouter();
const { t } = useI18n();
const store = useMainStore();
const { formData } = storeToRefs(store);

const showFacetSelection = ref(true);
const showFacetResults = ref(false);
const collocationFacet = {
    facet: "collocation",
    alias: t("facets.collocate"),
    type: "collocationFacet",
};
const loading = ref(false);
const selectedFacet = ref({});
const showingRelativeFrequencies = ref(false);
const shouldShowRelativeFrequency = ref(false);
const fullResults = ref({});
const facetResults = ref([]);
const selected = ref("");

function populateFacets() {
    const facets = [];
    for (const facet of philoConfig.facets) {
        if (!philoConfig.metadata.includes(facet)) continue;
        const alias = philoConfig.metadata_aliases[facet] || facet;
        facets.push({ facet, alias, type: "facet" });
    }
    return facets;
}

function populateWordFacets() {
    const wordFacets = [];
    for (const wordProperty of philoConfig.words_facets || []) {
        const alias = philoConfig.word_property_aliases[wordProperty] || wordProperty;
        wordFacets.push({ facet: wordProperty, alias, type: "property" });
    }
    return wordFacets;
}

const facets = ref(populateFacets());
const wordFacets = ref(populateWordFacets());

function fetchFrequencyFacet(queryParams) {
    $http.get(`${$dbUrl}/scripts/get_frequency.py`, {
        params: paramsFilter(queryParams),
    }).then((response) => {
        fullResults.value = {
            absolute: response.data.results,
            relative: response.data.relative_results || [],
        };
        if (shouldShowRelativeFrequency.value && fullResults.value.relative.length) {
            facetResults.value = fullResults.value.relative;
        } else {
            facetResults.value = fullResults.value.absolute;
            showingRelativeFrequencies.value = false;
        }
        loading.value = false;
        showFacetResults.value = true;
        const urlString = paramsToUrlString({
            ...queryParams,
            frequency_field: selectedFacet.value.alias,
        });
        saveToLocalStorage(urlString, fullResults.value);
    }).catch((error) => {
        debug({ $options: { name: "facets-report" } }, error);
        loading.value = false;
    });
}

function fetchCollocationFacet(queryParams) {
    $http.get(`${$dbUrl}/reports/collocation.py`, {
        params: paramsFilter(queryParams),
    }).then((response) => {
        if (response.data.results_length) {
            facetResults.value = extractSurfaceFromCollocate(
                response.data.collocates.slice(0, 100)
            );
            fullResults.value = response.data.collocates;
            showFacetResults.value = true;
        }
        loading.value = false;
        const urlString = paramsToUrlString({
            ...queryParams,
            report: "collocation",
        });
        saveToLocalStorage(urlString, fullResults.value);
    }).catch((error) => {
        loading.value = false;
        debug({ $options: { name: "facets-report" } }, error);
    });
}

function fetchPropertyFacet(facet, queryParams) {
    $http.get(`${$dbUrl}/scripts/get_word_property_count.py`, {
        params: paramsFilter(queryParams),
    }).then((response) => {
        facetResults.value = response.data.results.slice(0, 100);
        fullResults.value = response.data.results;
        loading.value = false;
        showFacetResults.value = true;
        const urlString = paramsToUrlString({
            ...queryParams,
            word_property: facet.facet,
        });
        saveToLocalStorage(urlString, fullResults.value);
    }).catch((error) => {
        debug({ $options: { name: "facets-report" } }, error);
        loading.value = false;
    });
}

function getFacet(facetObj, updateUrl = true) {
    selectedFacet.value = facetObj;
    showFacetSelection.value = false;
    facetResults.value = [];

    showingRelativeFrequencies.value = !updateUrl && shouldShowRelativeFrequency.value;
    selected.value = facetObj.alias;

    let urlString;
    if (facetObj.type === "facet") {
        urlString = paramsToUrlString({ ...formData.value, frequency_field: facetObj.alias });
    } else if (facetObj.type === "collocationFacet") {
        urlString = paramsToUrlString({ ...formData.value, report: "collocation" });
    } else if (facetObj.type === "property") {
        urlString = paramsToUrlString({ ...formData.value, word_property: facetObj.facet });
    }

    if (typeof sessionStorage[urlString] !== "undefined" && philoConfig.production === true) {
        fullResults.value = JSON.parse(sessionStorage[urlString]);
        if (facetObj.type === "facet") {
            facetResults.value = (shouldShowRelativeFrequency.value && fullResults.value.relative)
                ? fullResults.value.relative
                : fullResults.value.absolute;
        } else if (facetObj.type === "collocationFacet") {
            facetResults.value = extractSurfaceFromCollocate(fullResults.value.slice(0, 100));
        } else {
            facetResults.value = fullResults.value.slice(0, 100);
        }
        showFacetResults.value = true;
        showFacetSelection.value = false;
        return;
    }

    loading.value = true;
    const queryParams = copyObject(formData.value);
    if (facetObj.type === "facet") {
        queryParams.frequency_field = facetObj.facet;
        fetchFrequencyFacet(queryParams);
    } else if (facetObj.type === "collocationFacet") {
        queryParams.report = "collocation";
        fetchCollocationFacet(queryParams);
    } else if (facetObj.type === "property") {
        queryParams.word_property = facetObj.facet;
        fetchPropertyFacet(facetObj, queryParams);
    }
}

function facetSearch(facetObj) {
    switch (facetObj.type) {
        case "facet":
            router.push(paramsToRoute({
                ...formData.value,
                facet: facetObj.facet,
                relative_frequency: false,
            }));
            break;
        case "collocationFacet":
            router.push(paramsToRoute({ ...formData.value, facet: "collocation" }));
            break;
        case "property":
            router.push(paramsToRoute({
                ...formData.value,
                facet: "property",
                word_property: facetObj.facet,
            }));
            break;
    }
}

function updateFacetUrl() {
    if (selectedFacet.value && selectedFacet.value.facet) {
        router.push(paramsToRoute({
            ...formData.value,
            facet: selectedFacet.value.facet,
            relative_frequency: showingRelativeFrequencies.value ? "true" : "false",
        }));
    }
}

function displayRelativeFrequencies() {
    facetResults.value = fullResults.value.relative;
    showingRelativeFrequencies.value = true;
    if (
        selectedFacet.value.facet !== route.query.facet ||
        showingRelativeFrequencies.value.toString() !== route.query.relative_frequency
    ) {
        updateFacetUrl();
    }
}

function displayAbsoluteFrequencies() {
    facetResults.value = fullResults.value.absolute;
    showingRelativeFrequencies.value = false;
    if (
        selectedFacet.value.facet !== route.query.facet ||
        showingRelativeFrequencies.value.toString() !== route.query.relative_frequency
    ) {
        updateFacetUrl();
    }
}

function checkFacetStateFromUrl() {
    if (!route.query.facet) {
        showFacetSelection.value = true;
        showFacetResults.value = false;
        return;
    }

    const facet = route.query.facet;
    shouldShowRelativeFrequency.value = route.query.relative_frequency === "true";

    // Clean up irrelevant facet params from store
    if (facet === "collocation") {
        store.updateFormDataField({ key: "word_property", value: "" });
        store.updateFormDataField({ key: "relative_frequency", value: "" });
    } else if (facet === "property") {
        store.updateFormDataField({ key: "relative_frequency", value: "" });
    }

    let facetObj;
    if (facet === "property") {
        facetObj = { type: "property", facet: route.query.word_property };
    } else if (facet === "collocation") {
        facetObj = { type: "collocationFacet", alias: t("facets.collocate"), facet: "collocation" };
    } else {
        facetObj = facets.value.find((f) => f.facet === facet);
    }
    if (facetObj) getFacet(facetObj, false);
}

function collocationToConcordance(word) {
    const routeParams = paramsToRoute({
        ...formData.value,
        q: `${formData.value.q} "${word}"`,
        method: "sentence",
        cooc_order: "no",
        start: "",
        end: "",
        report: "concordance",
    });
    delete routeParams.query.facet;
    delete routeParams.query.relative_frequency;
    delete routeParams.query.word_property;
    router.push(routeParams);
}

function propertyToConcordance(query) {
    formData.value.q = query;
    formData.value.start = "";
    formData.value.end = "";
    formData.value.report = "concordance";
    router.push(paramsToRoute({ ...formData.value }));
}

function showFacetOptions() {
    showFacetSelection.value = true;
}

function toggleFrequencies() {
    showingRelativeFrequencies.value = !showingRelativeFrequencies.value;
    router.push(paramsToRoute({
        ...formData.value,
        facet: selectedFacet.value.facet,
        relative_frequency: showingRelativeFrequencies.value ? "true" : "false",
    }));
}

function hideFacets(skipRouterPush = false) {
    showFacetResults.value = false;
    showFacetSelection.value = true;
    showingRelativeFrequencies.value = false;
    facetResults.value = [];
    fullResults.value = {};
    if (!skipRouterPush) {
        const routeParams = paramsToRoute({ ...formData.value });
        delete routeParams.query.facet;
        delete routeParams.query.relative_frequency;
        delete routeParams.query.word_property;
        router.push(routeParams);
    }
}

function facetClick(metadata) {
    store.updateFormDataField({
        key: selectedFacet.value.facet,
        value: `"${metadata[selectedFacet.value.facet]}"`,
    });
    const routeParams = paramsToRoute({ ...formData.value, start: "0", end: "0" });
    delete routeParams.query.facet;
    delete routeParams.query.relative_frequency;
    delete routeParams.query.word_property;
    router.push(routeParams);
}


watch(
    () => route.fullPath,
    (newPath, oldPath) => {
        const newQuery = router.resolve(newPath).query;
        const oldQuery = router.resolve(oldPath || "").query;

        // If facet param was removed from URL, hide facets immediately
        if (oldQuery.facet && !newQuery.facet) {
            showFacetResults.value = false;
            showFacetSelection.value = true;
            return;
        }

        if (!isOnlyFacetChange(newQuery, oldQuery)) {
            // Clear facet data when query changes (not just facet selection)
            facetResults.value = [];
            fullResults.value = {};
            checkFacetStateFromUrl();
        } else if (
            newQuery.facet !== oldQuery.facet ||
            newQuery.word_property !== oldQuery.word_property
        ) {
            checkFacetStateFromUrl();
        }

        // relative_frequency only affects regular metadata facets
        if (
            newQuery.relative_frequency !== oldQuery.relative_frequency &&
            newQuery.facet !== "collocation" &&
            newQuery.facet !== "property"
        ) {
            if (newQuery.relative_frequency === "false") {
                displayAbsoluteFrequencies();
            } else {
                displayRelativeFrequencies();
            }
        }
    }
);

checkFacetStateFromUrl();
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
    border-radius: 0.5rem;
}

.facet-result-item:not(.non-clickable):hover {
    transform: scale(1.01);
    background-color: rgba(theme.$link-color, 0.15) !important;
    border-color: rgba(theme.$link-color, 0.3);
    box-shadow: inset 0 0 8px rgba(theme.$link-color, 0.1);
    z-index: 1;
}

.facet-result-item:not(.non-clickable):hover:focus,
.facet-result-item:not(.non-clickable):hover:focus-visible {
    box-shadow: 0 0 0 3px theme.$button-color !important;
    z-index: 3;
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
    opacity: 1;
    color: #6c757d !important;
    transition: color 0.25s ease, opacity 0.25s ease;
}

.facet-result-item:hover .relative-frequency-info small {
    opacity: 1;
    color: #212529 !important;
}

/* Focus styles for accessibility */
.facet-selection:focus {
    outline: 2px solid theme.$button-color !important;
    outline-offset: -2px;
    box-shadow: 0 0 0 0.2rem rgba(theme.$button-color, 0.25) !important;
}

/* Override global focus-visible-only behavior for facet items */
.facet-result-item:focus {
    outline: 2px solid theme.$button-color !important;
    outline-offset: -4px !important;
    z-index: 3;
}

button.facet-result-item:focus-visible,
.list-group-item.facet-result-item:focus-visible {
    outline: 2px solid theme.$button-color !important;
    outline-offset: -4px !important;
    z-index: 3;
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

/* Reset list styling for semantic ul/li elements */
.facet-results-container {
    list-style: none;
    padding: 0;
    margin: 0;
}

.facet-results-container li {
    list-style: none;
    padding: 0;
    margin: 0;
}

</style>