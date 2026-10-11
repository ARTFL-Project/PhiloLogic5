<template>
    <div id="collocation-container" class="container-fluid mt-4">
        <!-- Mobile: Dropdown selector for collocation methods -->
        <div class="d-block d-sm-none mt-3 mx-2">
            <label for="colloc-method-mobile-select" class="form-label fw-bold">
                {{ $t('collocation.methodSelectionTabs') }}
            </label>
            <select class="form-select" id="colloc-method-mobile-select" v-model="mode"
                @change="handleMobileMethodChange"
                :aria-label="$t('collocation.methodSelectionTabs')">
                <option value="frequency">{{ $t("collocation.collocation") }}</option>
                <option value="compare">{{ $t("collocation.compareTo") }}</option>
                <option value="similar">{{ $t("collocation.similarUsage") }}</option>
                <option value="senses">{{ $t("collocation.wordSenses") }}</option>
                <option value="senseTree">{{ $t("collocation.senseTree") }}</option>
                <option value="senseOnsets">{{ $t("collocation.senseOnsets") }}</option>
                <option value="risingWords">{{ $t("collocation.risingWords") }}</option>
                <option value="aheadOfTime">{{ $t("collocation.aheadOfTime") }}</option>
            </select>
        </div>

        <!-- Desktop: Tab navigation for collocation methods -->
        <div class="d-none d-sm-block mt-3" style="padding: 0 0.5rem">
            <ul class="nav nav-tabs" id="colloc-method-switch" role="tablist"
                :aria-label="$t('collocation.methodSelectionTabs')">
                <li class="nav-item" role="presentation">
                    <button class="nav-link shadow-sm" id="frequency-tab" data-bs-toggle="tab"
                        :class="{ active: mode === 'frequency' }" data-bs-target="#frequency-tab-pane" type="button"
                        role="tab" :aria-selected="mode === 'frequency'"
                        @click="setMode('frequency')">
                        {{ $t("collocation.collocation") }}
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link shadow-sm" id="compare-tab" data-bs-toggle="tab"
                        :class="{ active: mode === 'compare' }" data-bs-target="#compare-tab-pane" type="button"
                        role="tab" :aria-selected="mode === 'compare'"
                        @click="setMode('compare')">
                        {{ $t("collocation.compareTo") }}
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link shadow-sm" id="similar-tab" data-bs-toggle="tab"
                        :class="{ active: mode === 'similar' }" data-bs-target="#similar-tab-pane" type="button"
                        role="tab" :aria-selected="mode === 'similar'"
                        @click="setMode('similar')">
                        {{ $t("collocation.similarUsage") }}
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link shadow-sm" id="word-senses-tab" data-bs-toggle="tab"
                        :class="{ active: mode === 'senses' }" data-bs-target="#word-senses-tab-pane" type="button"
                        role="tab" :aria-selected="mode === 'senses'"
                        @click="setMode('senses')">
                        {{ $t("collocation.wordSenses") }}
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link shadow-sm" id="sense-tree-tab" data-bs-toggle="tab"
                        :class="{ active: mode === 'senseTree' }" data-bs-target="#sense-tree-tab-pane" type="button"
                        role="tab" :aria-selected="mode === 'senseTree'"
                        @click="setMode('senseTree')">
                        {{ $t("collocation.senseTree") }}
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link shadow-sm" id="sense-onsets-tab" data-bs-toggle="tab"
                        :class="{ active: mode === 'senseOnsets' }" data-bs-target="#sense-onsets-tab-pane" type="button"
                        role="tab" :aria-selected="mode === 'senseOnsets'"
                        @click="setMode('senseOnsets')">
                        {{ $t("collocation.senseOnsets") }}
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link shadow-sm" id="rising-words-tab" data-bs-toggle="tab"
                        :class="{ active: mode === 'risingWords' }" data-bs-target="#rising-words-tab-pane" type="button"
                        role="tab" :aria-selected="mode === 'risingWords'"
                        @click="setMode('risingWords')">
                        {{ $t("collocation.risingWords") }}
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link shadow-sm" id="ahead-of-time-tab" data-bs-toggle="tab"
                        :class="{ active: mode === 'aheadOfTime' }" data-bs-target="#ahead-of-time-tab-pane" type="button"
                        role="tab" :aria-selected="mode === 'aheadOfTime'"
                        @click="setMode('aheadOfTime')">
                        {{ $t("collocation.aheadOfTime") }}
                    </button>
                </li>
            </ul>
        </div>

        <results-summary :description="results.description" :filter-list="filterList" :colloc-method="mode"
            :stopwords-missing="stopwordsMissing"
            v-if="mode === 'frequency' || mode in detectors"
            style="margin-top:0 !important;"></results-summary>

        <div role="region" :aria-label="$t('collocation.collocationResults')">
            <Frequency v-if="mode === 'frequency'" ref="frequencyRef"
                :sorted-list="sortedList" :results-length="resultsLength"
                @pivot-to-compare="pivotToCompare" />
            <div class="mx-2 my-3" role="status"
                v-if="mode == 'frequency' && !searching && (!resultsLength || !sortedList.length)">
                {{ resultsLength ? $t("collocation.noCollocates") : $t("resultsSummary.noResults") }}
            </div>

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

            <WordSenses v-if="mode === 'senses'" ref="wordSensesRef"
                @filter-list="filterList = $event" @stopwords-missing="stopwordsMissing = $event" />

            <SenseTree v-if="mode === 'senseTree'" ref="senseTreeRef"
                @filter-list="filterList = $event" @stopwords-missing="stopwordsMissing = $event" />

            <SenseOnsets v-if="mode === 'senseOnsets'" ref="senseOnsetsRef"
                @filter-list="filterList = $event" @stopwords-missing="stopwordsMissing = $event" />

            <RisingWords v-if="mode === 'risingWords'" ref="risingWordsRef"
                @filter-list="filterList = $event" @stopwords-missing="stopwordsMissing = $event" />

            <AheadOfTime v-if="mode === 'aheadOfTime'" ref="aheadOfTimeRef"
                @filter-list="filterList = $event" @stopwords-missing="stopwordsMissing = $event" />
        </div>
    </div>
</template>

<script setup>
import { storeToRefs } from "pinia";
import { computed, defineAsyncComponent, inject, nextTick, onMounted, reactive, ref, useTemplateRef, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { useMainStore } from "../stores/main";
import {
    buildBiblioCriteria,
    debug,
    extractSurfaceFromCollocate,
    paramsFilter,
    paramsToRoute,
    quoteMetadataValue,
} from "../utils.js";
import ResultsSummary from "./ResultsSummary";
import Compare from "./collocation/Compare.vue";
import Frequency from "./collocation/Frequency.vue";
import Similar from "./collocation/Similar.vue";
import WordSenses from "./collocation/WordSenses.vue";
// Lazy-loaded: the sense tree only ships when the user opens its tab, not on entry to Collocation.
const SenseTree = defineAsyncComponent(() => import("./collocation/SenseTree.vue"));
const SenseOnsets = defineAsyncComponent(() => import("./collocation/SenseOnsets.vue"));
const RisingWords = defineAsyncComponent(() => import("./collocation/RisingWords.vue"));
const AheadOfTime = defineAsyncComponent(() => import("./collocation/AheadOfTime.vue"));

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
const wordSensesRef = useTemplateRef("wordSensesRef");
const senseTreeRef = useTemplateRef("senseTreeRef");
const senseOnsetsRef = useTemplateRef("senseOnsetsRef");
const risingWordsRef = useTemplateRef("risingWordsRef");
const aheadOfTimeRef = useTemplateRef("aheadOfTimeRef");

// The tabs that run their own detection on the query (no Detect button), by method
const detectors = {
    senses: wordSensesRef,
    senseTree: senseTreeRef,
    senseOnsets: senseOnsetsRef,
    risingWords: risingWordsRef,
    aheadOfTime: aheadOfTimeRef,
};

async function detect(method) {
    await nextTick();
    detectors[method].value?.runDetection();
}

//  Shared state
const mode = ref("frequency");

// The collocation method a URL asks for; links to the usage patterns over time and the word map, which the senses
// replaced, open the senses
function methodOf(query) {
    const method = query.collocation_method || "frequency";
    return method === "timeSeries" || method === "wordMap" ? "senses" : method;
}
const results = ref({});
const filterList = ref([]);
const stopwordsMissing = ref(false);
// Whether the primary collocation fetch has run for the current query. False
// when the page was loaded directly on a senses tab, which runs its own detection
// — so switching to frequency/similar/compare must trigger the primary fetch.
const primaryFetched = ref(false);
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
    if (mode.value === "compare") {
        Object.assign(urlParams, getNonEmptyComparedMetadata());
        delete urlParams.similarity_by;
        delete urlParams.time_series_interval;
    }
    if (mode.value in detectors) {
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
    setMode(mode.value);
}

//  Primary fetch (shared by frequency / compare / similar entry paths)
function updateCollocation() {
    $http.get(`${$dbUrl}/reports/collocation.py`, { params: paramsFilter(formData.value) })
        .then((response) => {
            resultsLength.value = response.data.results_length;
            filterList.value = response.data.filter_list;
            stopwordsMissing.value = Boolean(response.data.stopwords_missing);
            collocatesFilePath.value = response.data.file_path;
            searching.value = false;
            primaryFetched.value = true;
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
    mode.value = methodOf(route.query);
    await nextTick();  // ensure the active mode's child is mounted
    if (mode.value === "similar") {
        // A URL without the field to compare by (the server needs one): the first one configured
        const similarityBy = route.query.similarity_by || fieldsToCompare.value[0]?.value;
        if (!similarityBy) {
            setMode("frequency", { updateUrl: false });
            await nextTick();
            frequencyRef.value?.fetchOutliers();
            return;
        }
        similarRef.value?.runSimilar(similarityBy);
    } else if (mode.value === "compare") {
        compareRef.value?.runFromMetadata();
    } else if (mode.value === "frequency") {
        frequencyRef.value?.fetchOutliers();
    }
}

function fetchResults() {
    searching.value = true;
    similarFieldForUrl.value = "";
    updateCollocation();
}

//  Cross-mode pivot: a child (frequency or similar) has selected a group
//  and wants to compare against it.
async function pivotToCompare(payload) {
    const { field, name, otherFilePath = null, otherCollocates = null, focusDistinctive = false } = payload;
    // IMPORTANT: populate comparedMetadataValues BEFORE setMode -- updateCollocationUrl
    // serializes the current values into compare_* URL params, and the route
    // watcher restores comparedMetadataValues from those params. If we set after,
    // the route watcher's restore would wipe what we just set.
    clearComparedMetadata();
    // Quote-wrap so MARC-formatted values (e.g. authors with commas/dashes/years)
    // hit the metadata parser as a single exact-match QUOTE token.
    comparedMetadataValues[field] = quoteMetadataValue(name);
    setMode("compare");
    await nextTick();  // wait for Compare to mount
    if (!compareRef.value) return;
    if (otherFilePath) {
        compareRef.value.runFromFilePath(otherFilePath, otherCollocates);
    } else {
        compareRef.value.runFromMetadata({ focusDistinctive });
    }
}

//  Watchers
// View-only params don't affect the primary collocation results — they only
// switch which panel is rendered or feed mode-specific secondary fetches.
// Changing only these (e.g. clicking a tab) must NOT trigger a re-search.
function isViewOnlyParam(key) {
    return (
        key === "report" || // the route says which: links may have it, the URLs pushed here don't
        key === "collocation_method" ||
        key === "similarity_by" ||
        key === "time_series_interval" ||
        key.startsWith("compare_")
    );
}

function shouldRefetchOnQueryChange(newQuery, oldQuery) {
    const allKeys = new Set([...Object.keys(newQuery), ...Object.keys(oldQuery)]);
    for (const key of allKeys) {
        if ((newQuery[key] ?? "") === (oldQuery[key] ?? "")) continue; // missing is empty (method_arg)
        if (!isViewOnlyParam(key)) return true;
    }
    return false;
}

watch(
    () => route.query,
    async (newQuery, oldQuery = {}) => {
        if (route.name !== "collocation") return;

        // Sync mode + biblio cheaply on every route change.
        const oldMode = methodOf(oldQuery);
        const newMode = methodOf(newQuery);
        const modeChanged = newMode !== oldMode;
        if (newMode !== mode.value) mode.value = newMode;
        if (newMode === "compare") restoreComparedMetadataFromUrl();
        biblio.value = buildBiblioCriteria(philoConfig, route.query, formData.value);

        // The senses tabs run on tab switch (no Detect button).
        // The child is freshly v-if-mounted, so its result is null and needs a fetch.
        if (modeChanged && !shouldRefetchOnQueryChange(newQuery, oldQuery)) {
            if (newMode in detectors) {
                await detect(newMode);
                return;
            }
            // Switching to frequency/similar/compare, but the primary fetch
            // never ran (page was loaded directly on a detection tab) — run it
            // now so the view isn't empty.
            if (!primaryFetched.value) {
                fetchResults();
                return;
            }
        }

        // Bail out if only the view changed (tab click, similarity dropdown
        // selection, time-series interval, etc.). Primary results stay valid;
        // mode-specific secondary fetches are user-triggered via their buttons.
        if (!shouldRefetchOnQueryChange(newQuery, oldQuery)) return;

        // Real search params changed — the primary results are now stale.
        primaryFetched.value = false;

        // Refetch the right path for the active mode.
        if (newMode in detectors) {
            await detect(newMode);
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
mode.value = methodOf(route.query);
if (mode.value === "compare") restoreComparedMetadataFromUrl();
if (mode.value === "similar" && route.query.similarity_by) {
    similarFieldForUrl.value = route.query.similarity_by;
}

onMounted(async () => {
    if (mode.value in detectors) {
        await detect(mode.value);
        return;
    }
    switch (mode.value) {
        case "frequency":
            fetchResults();
            break;
        case "similar":
            fetchResults();  // primary fetch then runPostFetchModeAction → similarRef.runSimilar
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
