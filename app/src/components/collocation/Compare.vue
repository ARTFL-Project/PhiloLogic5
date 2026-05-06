<template>
    <div>
        <!-- Comparison criteria card -->
        <div class="card shadow-sm mx-2 p-3" style="border-top-width: 0;" role="region"
            :aria-label="$t('collocation.compareTo')">
            <div class="row">
                <!-- Primary corpus criteria -->
                <div class="col-12 col-md-6 mb-3 mb-md-0" role="region"
                    :aria-label="$t('collocation.primaryCorpusCriteria')">
                    <h6 class="fw-bold border-bottom pb-2">{{ $t('collocation.primaryCorpusCriteria') }}</h6>
                    <!-- Spacer to align with the comparison side's alert -->
                    <div class="alert alert-info p-1 mb-2 d-none d-md-block invisible" aria-hidden="true">&nbsp;
                    </div>
                    <MetadataFields :fields="metadataDisplay" :input-styles="metadataInputStyle"
                        :choice-values="metadataChoiceValues" :model-value="formData" :checked-values="{}"
                        :selected-values="{}" id-prefix="primary-" />
                </div>
                <!-- Comparison corpus criteria -->
                <div class="col-12 col-md-6 compare-divider" role="region"
                    :aria-label="$t('collocation.comparisonCorpusCriteria')">
                    <h6 class="fw-bold border-bottom pb-2">{{ $t('collocation.comparisonCorpusCriteria') }}</h6>
                    <div class="alert alert-info p-1 mb-2" style="width: fit-content" role="note">
                        {{ $t('collocation.emptySearch') }}</div>
                    <MetadataFields :fields="metadataDisplay" :input-styles="metadataInputStyle"
                        :choice-values="metadataChoiceValues" :model-value="comparedMetadataValues"
                        :checked-values="metadataChoiceChecked" :selected-values="metadataChoiceSelected"
                        :date-type="dateType" :date-range="dateRange" id-prefix="compare-">
                        <template #text-input="{ field }">
                            <label class="btn btn-outline-secondary"
                                :for="'compare-' + field.value + '-input-filter'">
                                {{ field.label }}
                            </label>
                            <input type="text" class="form-control" :id="'compare-' + field.value + '-input-filter'"
                                :name="field.value" :placeholder="field.example"
                                v-model="comparedMetadataValues[field.value]"
                                @input="autocompleteOnChange(field.value)" @keydown.down="onArrowDown(field.value)"
                                @keydown.up="onArrowUp(field.value)" @keyup.enter="onEnter(field.value)"
                                @keyup.escape="clearAutoCompletePopup" autocomplete="off"
                                :aria-label="`${$t('collocation.filterBy')} ${field.label}`" />
                            <ul :id="'compare-autocomplete-' + field.value" class="autocomplete-results shadow"
                                :style="autoCompletePosition(field.value)"
                                v-if="autoCompleteResults[field.value].length > 0">
                                <li tabindex="-1" v-for="(result, i) in autoCompleteResults[field.value]"
                                    :key="result" @click="setMetadataResult(result, field.value)"
                                    class="autocomplete-result"
                                    :class="{ 'is-active': i === arrowCounters[field.value] }" v-html="result"></li>
                            </ul>
                        </template>
                    </MetadataFields>
                </div>
            </div>
            <div class="mt-1">
                <button type="button" class="btn btn-secondary" style="width: fit-content"
                    @click="runFromMetadata()">{{
                        $t('collocation.runComparison') }}
                </button>
            </div>
        </div>

        <!-- Comparison results -->
        <div class="card shadow-sm mx-2 my-3 p-2" v-if="comparativeSearchStarted">
            <div class="row mt-2">
                <div class="col-6" id="primary-biblio">
                    <bibliography-criteria :biblio="biblio" :query-report="formData.report"
                        :results-length="resultsLength"></bibliography-criteria>
                </div>
                <div id="other-biblio" class="col-6" style="border-left: solid 1px rgba(0, 0, 0, 0.176)">
                    <bibliography-criteria :biblio="otherBiblio" :query-report="formData.report"
                        :results-length="resultsLength"></bibliography-criteria>
                </div>
            </div>
            <ul class="nav nav-tabs mt-2" style="margin-left: -.5rem" id="colloc-tab" role="tablist"
                :aria-label="$t('collocation.comparisonResultsTabs')">
                <li class="nav-item" role="presentation">
                    <button class="nav-link active" style="border-left-width: 0" id="frequent-tab"
                        data-bs-toggle="tab" data-bs-target="#freq-tab-pane" type="button" role="tab"
                        aria-controls="freq-tab-pane" aria-selected="true"
                        :aria-label="$t('collocation.frequentCollocates')">
                        {{ $t('collocation.frequentCollocates') }}
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="rep-tab" data-bs-toggle="tab" data-bs-target="#rep-tab-pane"
                        type="button" role="tab" aria-controls="rep-tab-pane" aria-selected="false"
                        :aria-label="$t('collocation.overRepresentedCollocates')">
                        {{ $t('collocation.overRepresentedCollocates') }}
                    </button>
                </li>
            </ul>
            <div class="tab-content" id="colloc-tab-content">
                <div class="tab-pane fade show active" id="freq-tab-pane" role="tabpanel"
                    aria-labelledby="frequent-tab" tabindex="0"
                    :aria-label="$t('collocation.frequentCollocatesPanel')">
                    <div class="row gx-5">
                        <div class="col-6" role="region" :aria-label="$t('collocation.primaryCorpusResults')">
                            <word-cloud :word-weights="sortedList" label="primary"
                                :click-handler="onCollocateClick"></word-cloud>
                        </div>
                        <div class="col-6" style="border-left: solid 1px rgba(0, 0, 0, 0.176)" role="region"
                            :aria-label="$t('collocation.comparisonCorpusResults')">
                            <div class="d-flex justify-content-center position-relative" v-if="compareSearching">
                                <progress-spinner :lg="true" />
                            </div>
                            <word-cloud v-if="otherCollocates.length > 0" :word-weights="otherCollocates"
                                label="secondary" :click-handler="onOtherCollocateClick"></word-cloud>
                        </div>
                    </div>
                </div>
                <div class="tab-pane fade" id="rep-tab-pane" role="tabpanel" aria-labelledby="rep-tab"
                    tabindex="0" :aria-label="$t('collocation.overRepresentedCollocatesPanel')">
                    <div class="row gx-5">
                        <div class="col-6" role="region" :aria-label="$t('collocation.overRepresentedResults')">
                            <word-cloud v-if="overRepresented.length > 0" :word-weights="overRepresented"
                                :click-handler="onCollocateClick" label="over"></word-cloud>
                        </div>
                        <div class="col-6" style="border-left: solid 1px rgba(0, 0, 0, 0.176)" role="region"
                            :aria-label="$t('collocation.underRepresentedResults')">
                            <div class="d-flex justify-content-center position-relative" v-if="compareSearching">
                                <progress-spinner :lg="true" />
                            </div>
                            <word-cloud v-if="underRepresented.length > 0" :word-weights="underRepresented"
                                :click-handler="onOtherCollocateClick" label="under"></word-cloud>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import { storeToRefs } from "pinia";
import { inject, onBeforeUnmount, onMounted, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import { useAutocomplete } from "../../composables/useAutocomplete";
import { useMainStore } from "../../stores/main";
import {
    buildBiblioCriteria,
    collocateCleanup,
    concordanceMethod,
    dateRangeHandler,
    debug,
    extractSurfaceFromCollocate,
    paramsFilter,
    paramsToRoute,
} from "../../utils.js";
import BibliographyCriteria from "../BibliographyCriteria";
import MetadataFields from "../MetadataFields.vue";
import ProgressSpinner from "../ProgressSpinner";
import WordCloud from "../WordCloud.vue";

const props = defineProps({
    sortedList: { type: Array, required: true },
    biblio: { type: Object, required: true },
    resultsLength: { type: Number, default: 0 },
    collocatesFilePath: { type: String, default: "" },
    comparedMetadataValues: { type: Object, required: true },
    dateType: { type: Object, required: true },
    dateRange: { type: Object, required: true },
    metadataDisplay: { type: Array, required: true },
    metadataInputStyle: { type: [Array, Object], required: true },
    metadataChoiceValues: { type: Array, required: true },
    metadataChoiceChecked: { type: Object, default: () => ({}) },
    metadataChoiceSelected: { type: Object, default: () => ({}) },
});

const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const philoConfig = inject("$philoConfig");
const route = useRoute();
const router = useRouter();
const store = useMainStore();
const { formData } = storeToRefs(store);

// Autocomplete composable bound to the parent's comparedMetadataValues
const autocomplete = useAutocomplete({
    http: $http,
    dbUrl: $dbUrl,
    philoConfig,
    metadataValues: props.comparedMetadataValues,
    route,
});
const {
    autoCompleteResults,
    arrowCounters,
    autoCompletePosition,
    onArrowDown,
    onArrowUp,
    onEnter,
    onChange: autocompleteOnChange,
    setMetadataResult,
    clearAutoCompletePopup,
} = autocomplete;

const otherCollocates = ref([]);
const otherBiblio = ref({});
const overRepresented = ref([]);
const underRepresented = ref([]);
const compareSearching = ref(false);
const comparativeSearchStarted = ref(false);
const wholeCorpus = ref(false);

function onCollocateClick(item) {
    router.push(
        paramsToRoute({
            ...formData.value,
            report: "concordance",
            q: collocateCleanup(item, formData.value.q),
            method: concordanceMethod(formData.value.colloc_within),
            cooc_order: "no",
        })
    );
}

function onOtherCollocateClick(item) {
    router.push(
        paramsToRoute({
            ...props.comparedMetadataValues,
            report: "concordance",
            q: collocateCleanup(item, formData.value.q),
            method: concordanceMethod(formData.value.colloc_within),
            cooc_order: "no",
        })
    );
}

// Run the full compare flow: build params from comparedMetadataValues, fetch
// "other" collocates, then call comparativeCollocations.
function runFromMetadata() {
    wholeCorpus.value = Object.keys(props.comparedMetadataValues).length === 0;
    // dateRangeHandler mutates comparedMetadataValues in place
    dateRangeHandler(props.metadataInputStyle, props.dateRange, props.dateType, props.comparedMetadataValues);
    // Refresh the right-hand biblio criteria up front -- comparativeCollocations
    // would otherwise only update it after the round-trip, and not at all when
    // the new filter has zero hits.
    otherBiblio.value = buildBiblioCriteria(philoConfig, props.comparedMetadataValues, props.comparedMetadataValues);
    const params = {
        q: formData.value.q,
        colloc_filter_choice: formData.value.colloc_filter_choice,
        colloc_within: formData.value.colloc_within,
        filter_frequency: formData.value.filter_frequency,
        q_attribute: formData.value.q_attribute || "",
        q_attribute_value: formData.value.q_attribute_value || "",
        ...props.comparedMetadataValues,
    };
    comparativeSearchStarted.value = true;
    compareSearching.value = true;
    otherCollocates.value = [];
    $http.get(`${$dbUrl}/reports/collocation.py`, { params: paramsFilter(params) })
        .then((response) => {
            compareSearching.value = false;
            if (response.data.results_length) {
                otherCollocates.value = extractSurfaceFromCollocate(response.data.collocates);
                runComparativeCollocations(response.data.file_path);
            }
        })
        .catch((error) => {
            compareSearching.value = false;
            debug({ $options: { name: "collocation-compare" } }, error);
        });
}

// Skip the "other" collocation fetch and go straight to comparative_collocations
// using a pre-existing file_path. Used when pivoting from similar-usage where
// the "other" Counter has already been built.
function runFromFilePath(otherFilePath, otherCollocs = null) {
    if (otherCollocs) otherCollocates.value = otherCollocs;
    wholeCorpus.value = false;
    runComparativeCollocations(otherFilePath);
}

function runComparativeCollocations(otherFilePath) {
    comparativeSearchStarted.value = true;
    dateRangeHandler(props.metadataInputStyle, props.dateRange, props.dateType, props.comparedMetadataValues);
    otherBiblio.value = buildBiblioCriteria(philoConfig, props.comparedMetadataValues, props.comparedMetadataValues);
    overRepresented.value = [];
    underRepresented.value = [];
    $http.get(`${$dbUrl}/scripts/comparative_collocations.py`, {
        params: {
            primary_file_path: props.collocatesFilePath,
            other_file_path: otherFilePath,
            whole_corpus: wholeCorpus.value,
        },
    }).then((response) => {
        overRepresented.value = extractSurfaceFromCollocate(response.data.top);
        underRepresented.value = extractSurfaceFromCollocate(response.data.bottom);
    }).catch((error) => {
        debug({ $options: { name: "collocation-compare" } }, error);
    });
}

function reset() {
    otherCollocates.value = [];
    otherBiblio.value = {};
    overRepresented.value = [];
    underRepresented.value = [];
    comparativeSearchStarted.value = false;
}

// Document-click listener for the autocomplete popup
function onDocumentClick() {
    clearAutoCompletePopup();
}
onMounted(() => {
    document.addEventListener("click", onDocumentClick);
});
onBeforeUnmount(() => {
    document.removeEventListener("click", onDocumentClick);
});

defineExpose({ runFromMetadata, runFromFilePath, reset });
</script>

<style lang="scss" scoped>
@use "../../assets/styles/theme.module.scss" as theme;

#colloc-tab button {
    font-variant: small-caps;
    font-size: 1rem;
}

.nav-link {
    border-bottom: 1px solid #dee2e6;
    background-color: #fff;
    color: theme.$link-color;
}

.nav-link.active {
    color: theme.$link-color;
    font-weight: bold;
}

#colloc-tab .nav-link:hover {
    background-color: rgba(theme.$link-color, 0.1);
    border-color: theme.$link-color;
}

@media (min-width: 768px) {
    .compare-divider {
        border-left: solid 1px rgba(0, 0, 0, 0.176);
    }
}

.autocomplete-results {
    padding: 0;
    margin: 3px 0 0 15px;
    border: 1px solid #eeeeee;
    border-top-width: 0px;
    max-height: 216px;
    overflow-y: scroll;
    width: 267px;
    position: absolute;
    left: 0;
    background-color: #fff;
    z-index: 100;
    top: 34px;
    font-size: 1.2rem;
}

.autocomplete-result {
    list-style: none;
    text-align: left;
    padding: 4px 12px;
    cursor: pointer;
    font-size: 1.2rem;
}

.autocomplete-result:hover,
.is-active {
    background-color: #ddd;
    color: black;
}

input[type="text"]:focus {
    opacity: 1;
}

::placeholder {
    opacity: 0.4;
}

input:focus::placeholder {
    opacity: 0;
}
</style>
