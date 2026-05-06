<template>
    <div id="collocation-container" class="container-fluid mt-4">
        <div v-if="isInvalidCollocationQuery" class="alert alert-warning mx-2 mt-2" role="alert">
            <strong>{{ $t('collocation.invalidQuery') }}</strong>
            <p class="mb-0">{{ $t('collocation.invalidQueryExplanation') }}</p>
        </div>

        <!-- Mobile: Dropdown selector for collocation methods -->
        <div class="d-block d-sm-none mt-3 mx-2">
            <label for="colloc-method-mobile-select" class="form-label fw-bold">
                {{ $t('collocation.methodSelectionTabs') }}
            </label>
            <select class="form-select" id="colloc-method-mobile-select" v-model="mode"
                @change="handleMobileMethodChange" :disabled="isInvalidCollocationQuery"
                :aria-label="$t('collocation.methodSelectionTabs')">
                <option value="frequency">{{ $t("collocation.collocation") }}</option>
                <option value="compare">{{ $t("collocation.compareTo") }}</option>
                <option value="similar">{{ $t("collocation.similarUsage") }}</option>
                <option value="timeSeries">{{ $t("collocation.timeSeries") }}</option>
            </select>
        </div>

        <!-- Desktop: Tab navigation for collocation methods -->
        <div class="d-none d-sm-block mt-3" style="padding: 0 0.5rem">
            <ul class="nav nav-tabs" id="colloc-method-switch" role="tablist"
                :aria-label="$t('collocation.methodSelectionTabs')">
                <li class="nav-item" role="presentation">
                    <button class="nav-link shadow-sm" id="frequency-tab" data-bs-toggle="tab"
                        :class="{ active: mode === 'frequency' }" data-bs-target="#frequency-tab-pane" type="button"
                        role="tab" :aria-selected="mode === 'frequency'" :disabled="isInvalidCollocationQuery"
                        @click="!isInvalidCollocationQuery && setMode('frequency')">
                        {{ $t("collocation.collocation") }}
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link shadow-sm" id="compare-tab" data-bs-toggle="tab"
                        :class="{ active: mode === 'compare' }" data-bs-target="#compare-tab-pane" type="button"
                        role="tab" :aria-selected="mode === 'compare'" :disabled="isInvalidCollocationQuery"
                        @click="!isInvalidCollocationQuery && setMode('compare')">
                        {{ $t("collocation.compareTo") }}
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link shadow-sm" id="similar-tab" data-bs-toggle="tab"
                        :class="{ active: mode === 'similar' }" data-bs-target="#similar-tab-pane" type="button"
                        role="tab" :aria-selected="mode === 'similar'" :disabled="isInvalidCollocationQuery"
                        @click="!isInvalidCollocationQuery && setMode('similar')">
                        {{ $t("collocation.similarUsage") }}
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link shadow-sm" id="time-series-tab" data-bs-toggle="tab"
                        :class="{ active: mode === 'timeSeries' }" data-bs-target="#time-series-tab-pane" type="button"
                        role="tab" :aria-selected="mode === 'timeSeries'" :disabled="isInvalidCollocationQuery"
                        @click="!isInvalidCollocationQuery && setMode('timeSeries')">
                        {{ $t("collocation.timeSeries") }}
                    </button>
                </li>
            </ul>
        </div>

        <results-summary :description="results.description" :filter-list="filterList" :colloc-method="mode"
            v-if="mode === 'frequency'" style="margin-top:0 !important;"></results-summary>

        <div role="region" :aria-label="$t('collocation.collocationResults')">
            <Frequency v-if="mode === 'frequency'" ref="frequencyRef"
                :sorted-list="sortedList" :results-length="resultsLength"
                @pivot-to-compare="pivotToCompare" />

            <Compare v-if="mode === 'compare'" ref="compareRef"
                :sorted-list="sortedList" :biblio="biblio" :results-length="resultsLength"
                :collocates-file-path="collocatesFilePath"
                :compared-metadata-values="comparedMetadataValues"
                :date-type="dateType" :date-range="dateRange"
                :metadata-display="metadataDisplay" :metadata-input-style="metadataInputStyle"
                :metadata-choice-values="metadataChoiceValues"
                :metadata-choice-checked="metadataChoiceChecked"
                :metadata-choice-selected="metadataChoiceSelected" />

            <Similar v-if="mode === 'similar'" ref="similarRef"
                :biblio="biblio" :results-length="resultsLength"
                :collocates-file-path="collocatesFilePath"
                :fields-to-compare="fieldsToCompare"
                @field-selected="onSimilarFieldSelected"
                @pivot-to-compare="pivotToCompare" />

            <Evolution v-if="mode === 'timeSeries'" ref="evolutionRef"
                :biblio="biblio" :results-length="resultsLength" />
        </div>
    </div>
</template>

<script setup>
import { storeToRefs } from "pinia";
import { computed, inject, nextTick, onMounted, reactive, ref, useTemplateRef, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { useMainStore } from "../stores/main";
import {
    buildBiblioCriteria,
    debug,
    extractSurfaceFromCollocate,
    paramsFilter,
    paramsToRoute,
} from "../utils.js";
import ResultsSummary from "./ResultsSummary";
import Compare from "./collocation/Compare.vue";
import Evolution from "./collocation/Evolution.vue";
import Frequency from "./collocation/Frequency.vue";
import Similar from "./collocation/Similar.vue";

//  Injects, route, router, store
const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const philoConfig = inject("$philoConfig");
const route = useRoute();
const router = useRouter();
const store = useMainStore();
const {
    formData,
    currentReport,
    resultsLength,
    searching,
    searchableMetadata,
} = storeToRefs(store);

//  Child refs (for parent-driven actions)
const frequencyRef = useTemplateRef("frequencyRef");
const compareRef = useTemplateRef("compareRef");
const similarRef = useTemplateRef("similarRef");
const evolutionRef = useTemplateRef("evolutionRef");

//  Shared state
const mode = ref("frequency");
const results = ref({});
const filterList = ref([]);
const biblio = ref({});
const sortedList = ref([]);
const collocatesFilePath = ref("");

//  Compare metadata (parent-owned because it's URL-persisted and shared by
//  the cross-mode pivots from frequency and similar)
const comparedMetadataValues = reactive({});
const dateType = reactive({});
const dateRange = reactive({});

//  Metadata helpers (built from searchableMetadata, used by Compare's MetadataFields)
const metadataDisplay = ref([]);
const metadataInputStyle = ref([]);
const metadataChoiceValues = ref([]);
const metadataChoiceChecked = ref({});
const metadataChoiceSelected = ref({});

function buildMetadata(metadata) {
    metadataDisplay.value = metadata.display;
    metadataInputStyle.value = metadata.inputStyle;
    metadataChoiceValues.value = metadata.choiceValues;
    for (const field in metadataInputStyle.value) {
        dateType[field] = "exact";
        dateRange[field] = { start: "", end: "" };
    }
}

//  Similar mode's currently-selected field (kept here so updateCollocationUrl can serialize)
const similarFieldForUrl = ref("");
function onSimilarFieldSelected(value) {
    similarFieldForUrl.value = value;
    setMode("similar");  // updates URL with similarity_by
}

//  Computed
const fieldsToCompare = computed(() => {
    return philoConfig.collocation_fields_to_compare.map(field => ({
        label: philoConfig.metadata_aliases[field] || field,
        value: field,
    }));
});

const isInvalidCollocationQuery = computed(() => {
    if (!formData.value.q) return false;
    let query = formData.value.q.trim();
    // Remove quoted tokens, lemma/attribute queries
    query = query.replace(/"[^"]*"/g, "").replace(/\S+:\S+/g, "");
    // Split by OR operators and filter them out
    let parts = query.split(/\s*(\||OR)\s*/i).filter(p => p.trim() && !p.match(/^\|$|^OR$/i));
    // Remove NOT patterns
    parts = parts.map(p => p.replace(/\s+NOT\s+\S+/gi, ""));
    // Check for multiple consecutive words
    return parts.some(p => p.trim().split(/\s+/).filter(w => w.length > 0).length > 1);
});

//  URL + mode coordination
function getNonEmptyComparedMetadata() {
    const out = {};
    for (const [field, value] of Object.entries(comparedMetadataValues)) {
        if (value && value !== "") {
            out[`compare_${field}`] = value;
        }
    }
    return out;
}

// Clear in place to preserve the reactive reference (children hold refs to it).
function clearComparedMetadata() {
    for (const key of Object.keys(comparedMetadataValues)) {
        delete comparedMetadataValues[key];
    }
}

function restoreComparedMetadataFromUrl() {
    clearComparedMetadata();
    for (const [key, value] of Object.entries(route.query)) {
        if (key.startsWith("compare_")) {
            comparedMetadataValues[key.substring(8)] = value;
        }
    }
}

function updateCollocationUrl() {
    const urlParams = {
        ...formData.value,
        collocation_method: mode.value,
    };
    if (mode.value === "similar" && similarFieldForUrl.value) {
        urlParams.similarity_by = similarFieldForUrl.value;
        delete urlParams.time_series_interval;
    }
    if (mode.value === "timeSeries") {
        urlParams.time_series_interval = evolutionRef.value?.getInterval() || 10;
        delete urlParams.similarity_by;
    }
    if (mode.value === "compare") {
        Object.assign(urlParams, getNonEmptyComparedMetadata());
        delete urlParams.similarity_by;
        delete urlParams.time_series_interval;
    }
    router.push(paramsToRoute(urlParams));
}

function setMode(newMode, { updateUrl = true } = {}) {
    mode.value = newMode;
    if (updateUrl) updateCollocationUrl();
}

function handleMobileMethodChange() {
    setMode(mode.value, { updateUrl: false });
}

//  Primary fetch (shared by frequency / compare / similar entry paths)
function updateCollocation() {
    if (isInvalidCollocationQuery.value) {
        searching.value = false;
        return;
    }
    $http.get(`${$dbUrl}/reports/collocation.py`, { params: paramsFilter(formData.value) })
        .then((response) => {
            resultsLength.value = response.data.results_length;
            filterList.value = response.data.filter_list;
            collocatesFilePath.value = response.data.file_path;
            searching.value = false;
            if (resultsLength.value) {
                sortedList.value = extractSurfaceFromCollocate(response.data.collocates);
                runPostFetchModeAction();
            }
        })
        .catch((error) => {
            searching.value = false;
            debug({ $options: { name: "collocation-report" } }, error);
        });
}

// After the primary fetch completes, mode-specific follow-up runs in the active child.
async function runPostFetchModeAction() {
    mode.value = route.query.collocation_method || "frequency";
    await nextTick();  // ensure the active mode's child is mounted
    if (mode.value === "similar") {
        similarRef.value?.runSimilar(route.query.similarity_by);
    } else if (mode.value === "compare") {
        compareRef.value?.runFromMetadata();
    } else if (mode.value === "frequency") {
        frequencyRef.value?.fetchOutliers();
    }
}

function fetchResults() {
    if (isInvalidCollocationQuery.value) {
        searching.value = false;
        return;
    }
    searching.value = true;
    similarFieldForUrl.value = "";
    updateCollocation();
}

//  Cross-mode pivot: a child (frequency or similar) has selected a group
//  and wants to compare against it.
async function pivotToCompare(payload) {
    const { field, name, otherFilePath = null, otherCollocates = null } = payload;
    // IMPORTANT: populate comparedMetadataValues BEFORE setMode -- updateCollocationUrl
    // serializes the current values into compare_* URL params, and the route
    // watcher restores comparedMetadataValues from those params. If we set after,
    // the route watcher's restore would wipe what we just set.
    clearComparedMetadata();
    // Quote-wrap so MARC-formatted values (e.g. authors with commas/dashes/years)
    // hit the metadata parser as a single exact-match QUOTE token.
    const escaped = String(name).replace(/"/g, '\\"');
    comparedMetadataValues[field] = `"${escaped}"`;
    setMode("compare");
    await nextTick();  // wait for Compare to mount
    if (!compareRef.value) return;
    if (otherFilePath) {
        compareRef.value.runFromFilePath(otherFilePath, otherCollocates);
    } else {
        compareRef.value.runFromMetadata();
    }
}

//  Watchers
// View-only params don't affect the primary collocation results — they only
// switch which panel is rendered or feed mode-specific secondary fetches.
// Changing only these (e.g. clicking a tab) must NOT trigger a re-search.
function isViewOnlyParam(key) {
    return (
        key === "collocation_method" ||
        key === "similarity_by" ||
        key === "time_series_interval" ||
        key.startsWith("compare_")
    );
}

function shouldRefetchOnQueryChange(newQuery, oldQuery) {
    const allKeys = new Set([...Object.keys(newQuery), ...Object.keys(oldQuery)]);
    for (const key of allKeys) {
        if (newQuery[key] === oldQuery[key]) continue;
        if (!isViewOnlyParam(key)) return true;
    }
    return false;
}

watch(
    () => route.query,
    async (newQuery, oldQuery = {}) => {
        if (route.name !== "collocation") return;

        // Sync mode + biblio cheaply on every route change.
        const newMode = newQuery.collocation_method || "frequency";
        if (newMode !== mode.value) mode.value = newMode;
        if (newMode === "compare") restoreComparedMetadataFromUrl();
        biblio.value = buildBiblioCriteria(philoConfig, route.query, formData.value);

        // Bail out if only the view changed (tab click, similarity dropdown
        // selection, time-series interval, etc.). Primary results stay valid;
        // mode-specific secondary fetches are user-triggered via their buttons.
        if (!shouldRefetchOnQueryChange(newQuery, oldQuery)) return;

        // Real search params changed — refetch the right path for the active mode.
        if (newMode === "timeSeries") {
            await nextTick();
            evolutionRef.value?.runEvolution(parseInt(route.query.time_series_interval) || 10);
        } else {
            fetchResults();  // .then() runs runPostFetchModeAction for similar/compare/frequency
        }
    }
);

watch(searchableMetadata, (newVal) => buildMetadata(newVal), { deep: true });

//  Initial dispatch (before first render so the right child mounts)
formData.value.report = "collocation";
currentReport.value = "collocation";
buildMetadata(searchableMetadata.value);
biblio.value = buildBiblioCriteria(philoConfig, route.query, formData.value);
mode.value = route.query.collocation_method || "frequency";
if (mode.value === "compare") restoreComparedMetadataFromUrl();
if (mode.value === "similar" && route.query.similarity_by) {
    similarFieldForUrl.value = route.query.similarity_by;
}

onMounted(async () => {
    switch (mode.value) {
        case "frequency":
            fetchResults();
            break;
        case "similar":
            fetchResults();  // primary fetch then runPostFetchModeAction → similarRef.runSimilar
            break;
        case "timeSeries":
            await nextTick();
            evolutionRef.value?.runEvolution(parseInt(route.query.time_series_interval) || 10);
            break;
        case "compare":
            fetchResults();  // primary fetch then runPostFetchModeAction → compareRef.runFromMetadata
            break;
    }
});
</script>

<style lang="scss" scoped>
@use "../assets/styles/theme.module.scss" as theme;

.nav-link {
    border-bottom: 1px solid #dee2e6;
    background-color: #fff;
    color: theme.$link-color;
}

.nav-link.active {
    color: theme.$link-color;
    font-weight: bold;
}

#colloc-method-switch .nav-link:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.alert-warning {
    background-color: rgba(theme.$card-header-color, 0.05);
    border-color: theme.$card-header-color;
    color: #000;
}
</style>
