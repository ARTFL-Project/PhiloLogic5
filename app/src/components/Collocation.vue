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
                <option value="sensePeriods">{{ $t("collocation.sensePeriods") }}</option>
                <option value="senseOverview">{{ $t("collocation.senseOverview") }}</option>
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
                    <button class="nav-link shadow-sm" id="sense-periods-tab" data-bs-toggle="tab"
                        :class="{ active: mode === 'sensePeriods' }" data-bs-target="#sense-periods-tab-pane" type="button"
                        role="tab" :aria-selected="mode === 'sensePeriods'"
                        @click="setMode('sensePeriods')">
                        {{ $t("collocation.sensePeriods") }}
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link shadow-sm" id="sense-overview-tab" data-bs-toggle="tab"
                        :class="{ active: mode === 'senseOverview' }" data-bs-target="#sense-overview-tab-pane" type="button"
                        role="tab" :aria-selected="mode === 'senseOverview'"
                        @click="setMode('senseOverview')">
                        {{ $t("collocation.senseOverview") }}
                    </button>
                </li>
            </ul>
        </div>

        <results-summary :description="results.description" :filter-list="filterList" :colloc-method="mode"
            :stopwords-missing="stopwordsMissing"
            v-if="['frequency', 'senses', 'senseTree', 'sensePeriods', 'senseOverview'].includes(mode)"
            style="margin-top:0 !important;"></results-summary>

        <div role="region" :aria-label="$t('collocation.collocationResults')">
            <!-- Each tab is kept, with its results, while another is shown: shown again after the search changed, it
                 is a new one -->
            <KeepAlive :key="tabKeys.frequency">
                <Frequency v-if="mode === 'frequency'" ref="frequencyRef"
                    :sorted-list="sortedList" :results-length="resultsLength"
                    @pivot-to-compare="pivotToCompare" />
            </KeepAlive>
            <div class="mx-2 my-3" role="status"
                v-if="mode == 'frequency' && !searching && (!resultsLength || !sortedList.length)">
                {{ resultsLength ? $t("collocation.noCollocates") : $t("resultsSummary.noResults") }}
            </div>

            <KeepAlive :key="tabKeys.compare">
                <Compare v-if="mode === 'compare'" ref="compareRef"
                    :sorted-list="sortedList" :biblio="biblio" :results-length="resultsLength"
                    :collocates-file-path="collocatesFilePath"
                    :compared-metadata-values="comparedMetadataValues"
                    :date-type="dateType" :date-range="dateRange"
                    :primary-metadata-values="primaryMetadataValues"
                    :primary-date-type="primaryDateType" :primary-date-range="primaryDateRange"
                    :primary-choice-selected="primaryChoiceSelected"
                    :metadata-display="metadataDisplay" :metadata-input-style="metadataInputStyle"
                    :metadata-choice-values="metadataChoiceValues"
                    :metadata-choice-checked="metadataChoiceChecked"
                    :metadata-choice-selected="metadataChoiceSelected" @run="runComparison" />
            </KeepAlive>

            <KeepAlive :key="tabKeys.similar">
                <Similar v-if="mode === 'similar'" ref="similarRef"
                    :biblio="biblio" :results-length="resultsLength"
                    :collocates-file-path="collocatesFilePath"
                    :fields-to-compare="fieldsToCompare"
                    @field-selected="onSimilarFieldSelected"
                    @pivot-to-compare="pivotToCompare" />
            </KeepAlive>

            <KeepAlive :key="tabKeys.senses">
                <WordSenses v-if="mode === 'senses'" ref="wordSensesRef"
                    @filter-list="filterList = $event" @stopwords-missing="stopwordsMissing = $event" />
            </KeepAlive>

            <KeepAlive :key="tabKeys.senseTree">
                <SenseTree v-if="mode === 'senseTree'" ref="senseTreeRef"
                    @filter-list="filterList = $event" @stopwords-missing="stopwordsMissing = $event" />
            </KeepAlive>

            <KeepAlive :key="tabKeys.sensePeriods">
                <SensePeriods v-if="mode === 'sensePeriods'" ref="sensePeriodsRef"
                    @filter-list="filterList = $event" @stopwords-missing="stopwordsMissing = $event" />
            </KeepAlive>

            <KeepAlive :key="tabKeys.senseOverview">
                <SenseOverview v-if="mode === 'senseOverview'" ref="senseOverviewRef"
                    @filter-list="filterList = $event" @stopwords-missing="stopwordsMissing = $event" />
            </KeepAlive>
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
    dateRangeHandler,
    debug,
    extractSurfaceFromCollocate,
    isViewOnlyParam,
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
const SensePeriods = defineAsyncComponent(() => import("./collocation/SensePeriods.vue"));
const SenseOverview = defineAsyncComponent(() => import("./collocation/SenseOverview.vue"));

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
const sensePeriodsRef = useTemplateRef("sensePeriodsRef");
const senseOverviewRef = useTemplateRef("senseOverviewRef");

//  Shared state
const mode = ref("frequency");
// Each tab's instance (its KeepAlive's key) and the search it is of, by mode: a tab shown again after the search
// changed is a new one, its old one gone, while the one shown at a new search stays, to show its results as before
const tabKeys = reactive({});
const tabSearches = {};
let searchCount = 0; // the searches made on the page

function showTab(tab) {
    if (tabSearches[tab] !== searchCount) {
        tabSearches[tab] = searchCount;
        tabKeys[tab] = (tabKeys[tab] || 0) + 1;
    }
    mode.value = tab;
}

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
// The primary subcorpus as the compare form edits it: the search's until a comparison is run with it
const primaryMetadataValues = reactive({});
const primaryDateType = reactive({});
const primaryDateRange = reactive({});
const primaryChoiceSelected = reactive({});

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
        primaryDateType[field] = "exact";
        primaryDateRange[field] = { start: "", end: "" };
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
    restoreInputs(comparedMetadataValues, dateType, dateRange, metadataChoiceSelected.value);
}

function restorePrimaryMetadataFromUrl() {
    for (const field of philoConfig.metadata) {
        primaryMetadataValues[field] = route.query[field] || "";
    }
    restoreInputs(primaryMetadataValues, primaryDateType, primaryDateRange, primaryChoiceSelected);
}

// The inputs of metadata values as the URL gave them: a dropdown's choice, and a date in the exact input unless it is
// a range, which MetadataFields shows in the range inputs
function restoreInputs(values, types, ranges, selected) {
    for (const [field, style] of Object.entries(metadataInputStyle.value)) {
        if (style === "dropdown") {
            selected[field] = values[field] || "";
        } else if (["date", "int"].includes(style)
            && !String(values[field] || "").includes(style === "date" ? "<=>" : "-")) {
            types[field] = "exact";
            ranges[field] = { start: "", end: "" };
        }
    }
}

// The search's parameters without the comparison's: those of the URL it came from may have been erased since
function searchParams() {
    return Object.fromEntries(Object.entries(formData.value).filter(([key]) => !key.startsWith("compare_")));
}

// Run the comparison the form gives: both subcorpora go in the URL, which a reload keeps. A new primary subcorpus is
// searched first, by the route watcher, which then runs the comparison.
function runComparison() {
    for (const [values, types, ranges, selected] of [
        [primaryMetadataValues, primaryDateType, primaryDateRange, primaryChoiceSelected],
        [comparedMetadataValues, dateType, dateRange, metadataChoiceSelected.value],
    ]) {
        dateRangeHandler(metadataInputStyle.value, ranges, types, values);
        for (const [field, style] of Object.entries(metadataInputStyle.value)) {
            if (style === "dropdown") values[field] = selected[field] || "";
        }
    }
    const location = paramsToRoute({
        ...searchParams(),
        ...primaryMetadataValues,
        collocation_method: "compare",
        ...getNonEmptyComparedMetadata(),
    });
    const searchChanged = shouldRefetchOnQueryChange(router.resolve(location).query, route.query);
    router.push(location);
    if (!searchChanged) compareRef.value?.runFromMetadata();
}

function updateCollocationUrl() {
    const urlParams = {
        ...searchParams(),
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
    if (["senses", "senseTree", "sensePeriods", "senseOverview"].includes(mode.value)) {
        delete urlParams.similarity_by;
        delete urlParams.time_series_interval;
    }
    router.push(paramsToRoute(urlParams));
}

function setMode(newMode, { updateUrl = true } = {}) {
    showTab(newMode);
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
    showTab(methodOf(route.query));
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
        // A new search: the tab shown stays (a tab click may have shown it already), the others kept are of the last
        if (shouldRefetchOnQueryChange(newQuery, oldQuery)) {
            searchCount += 1;
            if (newMode === mode.value) tabSearches[mode.value] = searchCount;
        }
        if (newMode !== mode.value) showTab(newMode);
        if (newMode === "compare") restoreComparedMetadataFromUrl();
        restorePrimaryMetadataFromUrl();
        biblio.value = buildBiblioCriteria(philoConfig, route.query, formData.value);

        // The senses tabs run on tab switch (no Detect button): a tab kept with its result of this search keeps it.
        if (modeChanged && !shouldRefetchOnQueryChange(newQuery, oldQuery)) {
            if (newMode === "senses") {
                await nextTick();
                wordSensesRef.value?.runDetection();
                return;
            }
            if (newMode === "senseTree") {
                await nextTick();
                senseTreeRef.value?.runDetection();
                return;
            }
            if (newMode === "sensePeriods") {
                await nextTick();
                sensePeriodsRef.value?.runDetection();
                return;
            }
            if (newMode === "senseOverview") {
                await nextTick();
                senseOverviewRef.value?.runDetection();
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
        if (newMode === "senses") {
            await nextTick();
            wordSensesRef.value?.runDetection();
        } else if (newMode === "senseTree") {
            await nextTick();
            senseTreeRef.value?.runDetection();
        } else if (newMode === "sensePeriods") {
            await nextTick();
            sensePeriodsRef.value?.runDetection();
        } else if (newMode === "senseOverview") {
            await nextTick();
            senseOverviewRef.value?.runDetection();
        } else {
            fetchResults();  // .then() runs runPostFetchModeAction for similar/compare/frequency
        }
    }
);

watch(searchableMetadata, (newVal) => {
    buildMetadata(newVal);
    // On a page load, the input styles come after the values were restored from the URL
    restoreInputs(comparedMetadataValues, dateType, dateRange, metadataChoiceSelected.value);
    restoreInputs(primaryMetadataValues, primaryDateType, primaryDateRange, primaryChoiceSelected);
}, { deep: true });

//  Initial dispatch (before first render so the right child mounts)
formData.value.report = "collocation";
currentReport.value = "collocation";
buildMetadata(searchableMetadata.value);
biblio.value = buildBiblioCriteria(philoConfig, route.query, formData.value);
showTab(methodOf(route.query));
if (mode.value === "compare") restoreComparedMetadataFromUrl();
restorePrimaryMetadataFromUrl();
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
        case "senses":
            await nextTick();
            wordSensesRef.value?.runDetection();
            break;
        case "senseTree":
            await nextTick();
            senseTreeRef.value?.runDetection();
            break;
        case "sensePeriods":
            await nextTick();
            sensePeriodsRef.value?.runDetection();
            break;
        case "senseOverview":
            await nextTick();
            senseOverviewRef.value?.runDetection();
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
