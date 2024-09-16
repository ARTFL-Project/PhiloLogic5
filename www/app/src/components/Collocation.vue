<template>
    <div id="collocation-container" class="container-fluid mt-4">
        <div class="d-none d-sm-block mt-3" style="padding: 0 0.5rem">
            <ul class="nav nav-tabs" id="colloc-method-switch" role="tablist">
                <li class="nav-item" role="presentation">
                    <button class="nav-link shadow-sm" id="frequency-tab" data-bs-toggle="tab"
                        :class="{ active: collocMethod === 'frequency' }" data-bs-target="#frequency-tab-pane"
                        type="button" role="tab" aria-controls="frequency-tab-pane" aria-selected="true"
                        @click="getFrequency()">{{ $t("collocation.collocation") }}</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link shadow-sm" id="compare-tab" data-bs-toggle="tab"
                        :class="{ active: collocMethod === 'compare' }" data-bs-target="#compare-tab-pane" type="button"
                        role="tab" aria-controls="compare-tab-pane" aria-selected="false" @click="toggleCompare()">{{
                            $t("collocation.compareTo") }}</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link shadow-sm" id="similar-tab" data-bs-toggle="tab"
                        :class="{ active: collocMethod === 'similar' }" data-bs-target="#similar-tab-pane" type="button"
                        role="tab" aria-controls="similar-tab-pane" aria-selected="false" @click="toggleSimilar()">{{
                            $t("collocation.similarUsage") }}</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link shadow-sm" id="time-series-tab" data-bs-toggle="tab"
                        :class="{ active: collocMethod === 'timeSeries' }" data-bs-target="#time-series-tab-pane"
                        type="button" role="tab" aria-controls="time-series-tab-pane" aria-selected="false"
                        @click="toggleTimeSeries()">{{ $t("collocation.timeSeries") }}</button>
                </li>
            </ul>
        </div>
        <results-summary :description="results.description" :running-total="runningTotal" :filter-list="filterList"
            :colloc-method="collocMethod" v-if="collocMethod === 'frequency'"
            style="margin-top:0 !important;"></results-summary>
        <div class="card shadow-sm mx-2 p-2" style="border-top-width: 0;" v-if="collocMethod == 'compare'">
            <button id="toggle-metadata" class="btn btn-link" style="text-align: start;" type="button"
                data-bs-toggle="collapse" data-bs-target="#other-corpus-metadata" aria-expanded="false"
                aria-controls="other-corpus-metadata" @click="filterMetadataOpen = !filterMetadataOpen">
                <span v-if="!filterMetadataOpen">&#9654;</span>
                <span v-else>&#9660;</span>
                {{ $t('collocation.metadataFilter') }}
            </button>
            <div class="collapse my-2" id="other-corpus-metadata">
                <div class="alert alert-info p-1" style="width: fit-content" role="alert">
                    {{ $t('collocation.emptySearch') }}</div>
                <div class="row">
                    <div class="input-group pb-2" v-for="localField in metadataDisplay" :key="localField.value">
                        <div class="input-group pb-2" :id="localField.value + '-group'"
                            v-if="metadataInputStyle[localField.value] == 'text'">
                            <button type="button" class="btn btn-outline-secondary">
                                <label :for="localField.value + 'input-filter'">{{
                            localField.label
                        }}</label></button><input type="text" class="form-control"
                                :id="localField.value + 'input-filter'" :name="localField.value"
                                :placeholder="localField.example" v-model="comparedMetadataValues[localField.value]"
                                v-if="metadataInputStyle[localField.value] == 'text' &&
                            metadataInputStyle[localField.value] != 'date'
                            " />
                        </div>
                        <div class="input-group pb-2" :id="localField.value + '-group'"
                            v-if="metadataInputStyle[localField.value] == 'checkbox'">
                            <button style="border-top-right-radius: 0; border-bottom-right-radius: 0"
                                class="btn btn-outline-secondary me-2">
                                {{ localField.label }}
                            </button>
                            <div class="d-inline-block">
                                <div class="form-check d-inline-block ms-3" style="padding-top: 0.35rem"
                                    :id="localField.value" :options="metadataChoiceValues[localField.value]"
                                    v-for="metadataChoice in metadataChoiceValues[localField.value]"
                                    :key="metadataChoice.value" v-once>
                                    <input class="form-check-input" type="checkbox" :id="metadataChoice.value"
                                        v-model="metadataChoiceChecked[metadataChoice.value]" />
                                    <label class="form-check-label" :for="metadataChoice.value">{{
                            metadataChoice.text
                        }}</label>
                                </div>
                            </div>
                        </div>
                        <div class="input-group pb-2" :id="localField.value + '-group'"
                            v-if="metadataInputStyle[localField.value] == 'dropdown'">
                            <button type="button" class="btn btn-outline-secondary">
                                <label :for="localField.value + '-input-filter'">{{
                            localField.label
                        }}</label>
                            </button>
                            <select class="form-select" :id="localField.value + '-select'"
                                v-model="metadataChoiceSelected[localField.value]">
                                <option v-for="innerValue in metadataChoiceValues[localField.value]"
                                    :key="innerValue.value" :value="innerValue.value"
                                    :id="innerValue.value + '-select-option'">
                                    {{ innerValue.text }}
                                </option>
                            </select>
                        </div>
                        <div class="input-group pb-2" :id="localField.value + '-group'"
                            v-if="['date', 'int'].includes(metadataInputStyle[localField.value])">
                            <button type="button" class="btn btn-outline-secondary"
                                style="border-top-right-radius: 0; border-bottom-right-radius: 0">
                                <label :for="localField.value + '-date'">{{ localField.label
                                    }}</label>
                            </button>
                            <div class="btn-group" role="group">
                                <button class="btn btn-secondary dropdown-toggle"
                                    style="border-top-left-radius: 0; border-bottom-left-radius: 0" type="button"
                                    :id="localField.value + '-selector'" data-bs-toggle="dropdown"
                                    aria-expanded="false">
                                    {{ $t(`searchForm.${dateType[localField.value]}Date`) }}
                                </button>
                                <ul class="dropdown-menu" :aria-labelledby="localField.value + '-selector'">
                                    <li @click="dateTypeToggle(localField.value, 'exact')">
                                        <a class="dropdown-item">{{
                            $t("searchForm.exactDate") }}</a>
                                    </li>
                                    <li @click="dateTypeToggle(localField.value, 'range')">
                                        <a class="dropdown-item">{{
                            $t("searchForm.rangeDate") }}</a>
                                    </li>
                                </ul>
                            </div>
                            <input type="text" class="form-control" :id="localField.value + 'input-filter'"
                                :name="localField.value" :placeholder="localField.example"
                                v-model="comparedMetadataValues[localField.value]"
                                v-if="dateType[localField.value] == 'exact'" />
                            <span class="d-inline-block" v-if="dateType[localField.value] == 'range'">
                                <div class="input-group ms-3">
                                    <button class="btn btn-outline-secondary" type="button">
                                        <label for="query-term-input">{{
                            $t("searchForm.dateFrom")
                        }}</label>
                                    </button>
                                    <input type="text" class="form-control date-range"
                                        :id="localField.value + '-start-input-filter'"
                                        :name="localField.value + '-start'" :placeholder="localField.example"
                                        v-model="dateRange[localField.value].start" />
                                    <button class="btn btn-outline-secondary ms-3" type="button">
                                        <label for="query-term-input">{{
                            $t("searchForm.dateTo")
                        }}</label></button><input type="text" class="form-control date-range"
                                        :id="localField.value + 'end-input-filter'" :name="localField.value + '-end'"
                                        :placeholder="localField.example" v-model="dateRange[localField.value].end" />
                                </div>
                            </span>
                        </div>
                    </div>
                </div>
                <button type="button" class="btn btn-secondary" style="width: fit-content"
                    @click="getOtherCollocates({}, 0)">{{
                            $t('collocation.runComparison') }}
                </button>
            </div>
        </div>
        <div class="card mx-2 p-3" style="border-top-width: 0;" v-if="collocMethod === 'similar'">
            <div class="btn-group mt-2" style="width: fit-content;" role="group">
                <button class="btn btn-secondary dropdown-toggle" type="button" data-bs-toggle="dropdown"
                    aria-expanded="false">
                    {{ this.similarFieldSelected || "Select a field" }}
                </button>
                <ul class="dropdown-menu">
                    <li v-for="field in fieldsToCompare" :key="field.value"
                        @click="similarCollocDistributions(field, 0)"><a class="dropdown-item">{{ field.label }}</a>
                    </li>
                </ul>
                <span style="display: inline-block; margin-left: .5rem; padding-top: .5rem;">{{
                            $t("collocation.mostSimilarUsage") }}</span>
                <bibliography-criteria class="ms-2 pt-1" :biblio="biblio" :query-report="report"
                    :results-length="resultsLength" :hide-criteria-string="true"></bibliography-criteria>
            </div>
            <div class="mt-2" style="display: flex; align-items: center;" v-if="similarSearching">
                <div class="alert alert-info p-1 mb-0 d-inline-block" style="width: fit-content" role="alert">
                    {{ similarSearchProgress }}...
                </div>
                <progress-spinner class="px-2" :progress="progressPercent" />
            </div>
        </div>
        <div class="card shadow-sm mx-2 p-3" style="border-top-width: 0;" v-if="collocMethod === 'timeSeries'">
            <bibliography-criteria :biblio="biblio" :query-report="report"
                :results-length="resultsLength"></bibliography-criteria>
            <div class="input-group mt-2">
                <button class="btn btn-outline-secondary">
                    <label for="year_interval">{{ $t("searchForm.yearInterval") }}</label>
                </button>
                <span class="d-inline-flex align-self-center mx-2">{{ $t("searchForm.every") }}</span>
                <input type="text" class="form-control" name="year_interval" id="year_interval"
                    style="max-width: 50px; text-align: center" v-model="timeSeriesInterval" />
                <span class="d-inline-flex align-self-center mx-2">{{ $t("searchForm.years") }}</span>
            </div>
            <button type="button" class="btn btn-secondary mt-2" style="width: fit-content"
                @click="getCollocatesOverTime(0, true)">{{ $t('collocation.searchEvolution') }}</button>
        </div>

        <!-- Results below -->
        <div class="row my-3 pe-1" style="padding: 0 0.5rem" v-if="resultsLength && collocMethod == 'frequency'">
            <div class="col-12 col-sm-4">
                <div class="card shadow-sm">
                    <table class="table table-hover table-striped table-light table-borderless caption-top">
                        <thead class="table-header">
                            <tr>
                                <th scope="col">{{ $t("collocation.collocate") }}</th>
                                <th scope="col">{{ $t("collocation.count") }}</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="line-height: 1.75rem" v-for="word in sortedList" :key="word.collocate"
                                @click="collocateClick(word)">
                                <td class="text-view">{{ word.collocate }}</td>
                                <td>{{ word.count }}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            <div class="col-12 col-sm-8">
                <div class="card shadow-sm">
                    <word-cloud v-if="collocMethod == 'frequency' && sortedList.length > 0" :word-weights="sortedList"
                        label="" :click-handler="collocateClick"></word-cloud>
                </div>
            </div>
        </div>
        <div v-if="collocMethod === 'compare'">
            <div class="card shadow-sm mx-2 my-3 p-2" v-if="comparativeSearchStarted">
                <div class="row mt-2">
                    <div class="col-6">
                        <bibliography-criteria :biblio="biblio" :query-report="report"
                            :results-length="resultsLength"></bibliography-criteria>
                    </div>
                    <div class="col-6" style="border-left: solid 1px rgba(0, 0, 0, 0.176)">
                        <bibliography-criteria :biblio="otherBiblio" :query-report="report"
                            :results-length="resultsLength"></bibliography-criteria>
                    </div>
                </div>
                <ul class="nav nav-tabs mt-2" style="margin-left: -.5rem" id="colloc-tab" role="tablist">
                    <li class="nav-item" role="presentation">
                        <button class="nav-link active" style="border-left-width: 0" id="frequent-tab"
                            data-bs-toggle="tab" data-bs-target="#freq-tab-pane" type="button" role="tab"
                            aria-controls="freq-tab-pane" aria-selected="true">{{
                            $t('collocation.frequentCollocates')
                        }}</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="rep-tab" data-bs-toggle="tab" data-bs-target="#rep-tab-pane"
                            type="button" role="tab" aria-controls="rep-tab-pane" aria-selected="false">{{
                                $t('collocation.overRepresentedCollocates') }}</button>
                    </li>
                </ul>
                <div class="tab-content" id="colloc-tab-content">
                    <div class="tab-pane fade show active" id="freq-tab-pane" role="tabpanel" aria-labelledby="freq-tab"
                        tabindex="0">
                        <div class="row gx-5">
                            <div class="col-6">
                                <word-cloud :word-weights="sortedList" label=""
                                    :click-handler="collocateClick"></word-cloud>
                            </div>
                            <div class="col-6" style="border-left: solid 1px rgba(0, 0, 0, 0.176)">
                                <div class="d-flex justify-content-center position-relative" v-if="compareSearching">
                                    <progress-spinner :progress="progressPercent" :lg="true" />
                                </div>
                                <word-cloud v-if="otherCollocates.length > 0" :word-weights="otherCollocates" label=""
                                    :click-handler="otherCollocateClick"></word-cloud>
                            </div>
                        </div>
                    </div>
                    <div class="tab-pane fade" id="rep-tab-pane" role="tabpanel" aria-labelledby="rep-tab" tabindex="0">
                        <div class="row gx-5">
                            <div class="col-6">
                                <word-cloud v-if="overRepresented.length > 0" :word-weights="overRepresented"
                                    :click-handler="collocateClick"></word-cloud>
                            </div>
                            <div class="col-6" style="border-left: solid 1px rgba(0, 0, 0, 0.176)">
                                <div class="d-flex justify-content-center position-relative" v-if="compareSearching">
                                    <progress-spinner :progress="progressPercent" :lg="true" />
                                </div>
                                <word-cloud v-if="underRepresented.length > 0" :word-weights="underRepresented"
                                    :click-handler="otherCollocateClick"></word-cloud>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div v-if="collocMethod == 'similar'" class="mx-2" style="margin-bottom: 6rem;">
            <div class="card" v-if="mostSimilarDistributions.length > 0">
                <div class="row">
                    <div class="col-6 pe-0">
                        <h6 class="sim-dist">{{ $t("collocation.topSimilarUses") }}</h6>
                        <ul class="list-group list-group-flush mt-3">
                            <button type="button" class="list-group-item position-relative" style="text-align: justify"
                                v-for="metadataValue in mostSimilarDistributions" :key="metadataValue"
                                @click="similarToComparative(metadataValue[0])">{{
                            metadataValue[0] }} <span class="badge text-bg-secondary position-absolute"
                                    style="right: 1rem">{{
                            metadataValue[1]
                        }}</span></button>
                        </ul>
                    </div>
                    <div class="col-6 ps-0" style="border-left: solid 1px rgba(0, 0, 0, 0.176)">
                        <h6 class="sim-dist">{{ $t("collocation.topDissimilarUses") }}</h6>
                        <ul class="list-group list-group-flush mt-3">
                            <button type="button" class="list-group-item" style="text-align: justify"
                                v-for="metadataValue in mostDissimilarDistributions" :key="metadataValue"
                                @click="similarToComparative(metadataValue[0])">{{
                            metadataValue[0] }} <span class="badge text-bg-secondary position-absolute"
                                    style="right: 1rem">{{ metadataValue[1] }}</span></button>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        <div v-if="collocMethod == 'timeSeries' && collocationTimePeriods.length > 0" class="mx-2 my-3">

            <div v-for="timePeriods in collocationTimePeriods" :key="timePeriods.periodsCompared">
                <h6 class="time-series-colloc-title mb-0 mt-4">{{
                            $t("collocation.collocateBetweenPeriods",
                                {
                                    start:
                                        timePeriods.firstPeriodYear, end:
                                        timePeriods.secondPeriodYear
                                }) }}</h6>
                <div class="card mb-3 shadow-sm">
                    <div class="row">
                        <div class="col-6">
                            <h6 class="py-2 colloc-cloud-title" style="margin-right: -.75rem;">{{
                            timePeriods.firstPeriodYear }}</h6>
                            <word-cloud class="px-2" :word-weights="timePeriods.firstPeriod" label=""
                                :click-handler="collocateTimeSeriesClick(timePeriods.firstPeriodYear)"></word-cloud>
                        </div>
                        <div class="col-6" style="border-left: solid 1px rgba(0, 0, 0, 0.176)">
                            <h6 class="py-2 colloc-cloud-title" style="margin-left: -.75rem;">
                                {{ timePeriods.secondPeriodYear }}</h6>
                            <word-cloud class="px-2" :word-weights="timePeriods.secondPeriod" label=""
                                :click-handler="collocateTimeSeriesClick(timePeriods.secondPeriodYear)"></word-cloud>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
import { Collapse } from "bootstrap";
import { mapFields } from "vuex-map-fields";
import BibliographyCriteria from "./BibliographyCriteria";
import ProgressSpinner from "./ProgressSpinner";
import ResultsSummary from "./ResultsSummary";
import WordCloud from "./WordCloud.vue";

export default {
    name: "collocation-report",
    components: {
        ResultsSummary, WordCloud, BibliographyCriteria, ProgressSpinner
    },
    computed: {
        ...mapFields([
            "formData.report",
            "formData.colloc_filter_choice",
            "formData.q",
            "formData.filter_frequency",
            "formData.method_arg",
            "formData.colloc_within",
            "formData.q_attribute",
            "formData.q_attribute_value",
            "currentReport",
            "resultsLength",
            "searching",
            "urlUpdate",
            "accessAuthorized",
            "searchableMetadata"
        ]),
        formData() {
            return this.$store.state.formData;
        },
        fieldsToCompare() {
            let fields = []
            for (let field of this.philoConfig.collocation_fields_to_compare) {
                fields.push({ label: this.philoConfig.metadata_aliases[field] || field, value: field })
            }
            return fields
        }

    },
    inject: ["$http"],
    provide() {
        return {
            results: this.results.results,
        };
    },
    data() {
        return {
            philoConfig: this.$philoConfig,
            results: {},
            filterList: [],
            searchParams: {},
            biblio: {},
            moreResults: false,
            sortedList: [],
            collocateCounts: [],
            showFilteredWords: false,
            runningTotal: 0,
            collocCloudWords: [],
            collocMethod: "frequency",
            relativeFrequencies: {},
            overRepresented: [],
            underRepresented: [],
            loading: false,
            metadataDisplay: [],
            metadataInputStyle: [],
            metadataChoiceValues: [],
            comparedMetadataValues: {},
            dateRange: {},
            dateType: {},
            otherCollocates: [],
            otherBiblio: {},
            comparedTo: "wholeCorpus",
            filterMetadataOpen: false,
            compareSearching: false,
            comparativeSearchStarted: false,
            otherDone: false,
            fieldValuesToCompare: [],
            mostSimilarDistributions: [],
            mostDissimilarDistributions: [],
            cachedDistributions: "",
            similarFieldSelected: "",
            similarSearchProgress: "",
            similarSearching: false,
            timeSeriesInterval: 10,
            collocationTimePeriods: [],
            progressPercent: 0
        };
    },
    created() {
        this.report = "collocation";
        this.currentReport = "collocation";
        this.fetchResults();
        this.buildMetadata(this.searchableMetadata);
        this.biblio = this.buildBiblioCriteria(this.$philoConfig, this.$route.query, this.formData)
    },
    watch: {
        urlUpdate() {
            if (this.$route.name == "collocation") {
                this.fetchResults();
                this.biblio = this.buildBiblioCriteria(this.$philoConfig, this.$route.query, this.formData)
            }
        },
        searchableMetadata: {
            handler: function (newVal, oldVal) {
                this.buildMetadata(newVal)
            },
            deep: true,
        },
    },
    methods: {
        fetchResults() {
            this.localFormData = this.copyObject(this.$store.state.formData);
            this.searching = true;
            this.relativeFrequencies = {};
            this.collocMethod = "frequency"
            this.overRepresented = [];
            this.underRepresented = [];
            this.other_corpus_metadata = {};
            this.comparativeSearchStarted = false
            this.mostSimilarDistributions = []
            this.mostDissimilarDistributions = []
            this.collocationTimePeriods = []
            this.similarFieldSelected = ""
            this.updateCollocation({}, 0);
        },
        buildMetadata(metadata) {
            this.metadataDisplay = metadata.display;
            this.metadataInputStyle = metadata.inputStyle;
            this.metadataChoiceValues = metadata.choiceValues;
            for (let metadata in this.metadataInputStyle) {
                this.dateType[metadata] = "exact";
                this.dateRange[metadata] = { start: "", end: "" };
            }
        },
        updateCollocation(fullResults, start) {
            let params = {
                ...this.$store.state.formData,
                start: start.toString(),
                max_time: 2
            };
            this.$http
                .post(`${this.$dbUrl}/reports/collocation.py`, {
                    current_collocates: fullResults,
                },
                    {
                        params: this.paramsFilter(params),
                    })
                .then((response) => {
                    this.resultsLength = response.data.results_length;
                    this.moreResults = response.data.more_results;
                    this.runningTotal = response.data.hits_done;
                    this.filterList = response.data.filter_list
                    start = response.data.hits_done;
                    this.searching = false;
                    if (this.resultsLength) {
                        if (this.moreResults) {
                            this.sortedList = this.extractSurfaceFromCollocate(response.data.collocates.slice(0, 100));
                            this.updateCollocation(response.data.collocates, start);
                        }
                        else {
                            this.collocateCounts = response.data.collocates;
                            this.sortedList = this.extractSurfaceFromCollocate(response.data.collocates.slice(0, 100));
                            this.done = true
                        }
                    }

                })
                .catch((error) => {
                    this.searching = false;
                    this.debug(this, error);
                });
        },
        extractSurfaceFromCollocate(words) {
            let newWords = []
            for (let wordObj of words) {
                let collocate = `${wordObj[0]}`.replace(/lemma:/, "");
                if (collocate.search(/\w+:.*/) != -1) {
                    collocate = collocate.replace(/(\p{L}+):.*/u, "$1");
                }
                let surfaceForm = wordObj[0];
                newWords.push({ collocate: collocate, surfaceForm: surfaceForm, count: wordObj[1] });
            }
            return newWords
        },
        collocateCleanup(collocate) {
            let q
            if (collocate.surfaceForm.startsWith("lemma:")) {
                q = `${this.q} ${collocate.surfaceForm}`;
            } else if (collocate.surfaceForm.search(/\w+:.*/) != -1) {
                q = `${this.q} ${collocate.surfaceForm}`;
            }
            else {
                q = `${this.q} "${collocate.surfaceForm}"`;
            }
            return q
        },
        collocateClick(item) {
            let q = this.collocateCleanup(item)
            let method = "sentence"
            if (this.colloc_within == "n") {
                method = "proxy"
            }
            this.$router.push(
                this.paramsToRoute({
                    ...this.$store.state.formData,
                    report: "concordance",
                    q: q,
                    method: method,
                    cooc_order: "no"
                })
            );
        },
        otherCollocateClick(item) {
            let q = this.collocateCleanup(item)
            let method = "sentence"
            if (this.colloc_within == "n") {
                method = "proxy"
            }
            this.$router.push(
                this.paramsToRoute({
                    ...this.comparedMetadataValues,
                    report: "concordance",
                    q: q,
                    method: method,
                    cooc_order: "no"
                })
            );
        },
        dateTypeToggle(metadata, dateType) {
            this.dateRange[metadata] = { start: "", end: "" };
            this.comparedMetadataValues[metadata] = "";
            this.dateType[metadata] = dateType;
        },
        getFrequency() {
            this.collocMethod = "frequency";
        },
        toggleCompare() {
            this.collocMethod = "compare";
            this.filterMetadataOpen = true
            this.$nextTick(() => {
                let collapseElement = document.getElementById('other-corpus-metadata')
                new Collapse(collapseElement, {
                    toggle: true
                })
            })
        },
        toggleSimilar() {
            this.collocMethod = "similar";
        },
        toggleTimeSeries() {
            this.collocMethod = "timeSeries";
        },
        getOtherCollocates(fullResults, start) {
            if (Object.keys(this.comparedMetadataValues).length === 0) {
                this.wholeCorpus = true
            } else {
                this.wholeCorpus = false
            }
            if (Object.keys(fullResults).length === 0) {
                this.progressPercent = 0
            }
            this.collocMethod = 'compare';
            this.comparedMetadataValues = this.dateRangeHandler(this.metadataInputStyle, this.dateRange, this.dateType, this.comparedMetadataValues)
            let params = {
                q: this.q,
                colloc_filter_choice: this.colloc_filter_choice,
                colloc_within: this.colloc_within,
                filter_frequency: this.filter_frequency,
                q_attribute: this.q_attribute || "",
                q_attribute_value: this.q_attribute_value || "",
                ...this.comparedMetadataValues,
                start: start.toString(),
            };
            this.comparativeSearchStarted = true;
            this.compareSearching = true
            this.otherCollocates = [];
            this.$http
                .post(`${this.$dbUrl}/reports/collocation.py`, {
                    current_collocates: fullResults,
                },
                    {
                        params: this.paramsFilter(params),
                    })
                .then((response) => {
                    let resultsLength = response.data.results_length;
                    let moreResults = response.data.more_results;
                    let start = response.data.hits_done;
                    if (resultsLength) {
                        if (moreResults) {
                            this.progressPercent = Math.trunc((start / resultsLength) * 100)
                            this.getOtherCollocates(response.data.collocates, start);
                            console.log(this.progressPercent)
                        }
                        else {
                            this.compareSearching = false;
                            this.otherCollocates = this.extractSurfaceFromCollocate(response.data.collocates.slice(0, 100));
                            this.comparativeCollocations(response.data.collocates)
                        }
                    }

                })
                .catch((error) => {
                    this.searching = false;
                    this.debug(this, error);
                });
        },
        comparativeCollocations(otherCollocates) {
            let collapseElement = document.getElementById('other-corpus-metadata')
            if (collapseElement != null) {
                Collapse.getInstance(collapseElement).hide()
                this.filterMetadataOpen = false
            }
            this.comparativeSearchStarted = true;
            this.comparedMetadataValues = this.dateRangeHandler(this.metadataInputStyle, this.dateRange, this.dateType, this.comparedMetadataValues)
            this.otherBiblio = this.buildBiblioCriteria(this.$philoConfig, this.comparedMetadataValues, this.comparedMetadataValues)
            this.overRepresented = [];
            this.underRepresented = [];
            this.$http.post(`${this.$dbUrl}/scripts/comparative_collocations.py`, {
                all_collocates: this.collocateCounts,
                other_collocates: otherCollocates,
                whole_corpus: this.wholeCorpus,
            }, {
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded'
                }
            }).then((response) => {
                this.overRepresented = this.extractSurfaceFromCollocate(response.data.top);
                this.underRepresented = this.extractSurfaceFromCollocate(response.data.bottom);
                this.relativeFrequencies = { top: this.overRepresented, bottom: this.underRepresented };

            }).catch((error) => {
                this.debug(this, error);
            });
        },
        similarCollocDistributions(field, start, first) {
            this.similarFieldSelected = field.label
            this.similarSearching = true
            this.similarSearchProgress = this.$t("collocation.similarCollocGatheringMessage")
            this.mostSimilarDistributions = []
            if (typeof first === 'undefined') {
                first = true
                this.progressPercent = 0
            }
            else {
                first = false
            }
            this.$http
                .post(`${this.$dbUrl}/reports/collocation.py`, {
                    current_collocates: [],
                }, {
                    params: {
                        q: this.q, start: start.toString(),
                        colloc_filter_choice: this.colloc_filter_choice,
                        colloc_within: this.colloc_within,
                        filter_frequency: this.filter_frequency,
                        map_field: field.value,
                        q_attribute: this.q_attribute || "",
                        q_attribute_value: this.q_attribute_value || "",
                        first: first,
                        max_time: 2
                    }
                }).then((response) => {
                    if (response.data.more_results) {
                        this.progressPercent = Math.trunc((response.data.hits_done / response.data.results_length) * 100)
                        this.similarCollocDistributions(field, response.data.hits_done, first);
                    } else {
                        this.getMostSimilarCollocDistribution(response.data.file_path);
                    }
                }).catch((error) => {
                    this.debug(this, error);
                });

        },
        getMostSimilarCollocDistribution(filePath) {
            this.progressPercent = 0
            this.similarSearchProgress = this.$t("collocation.similarCollocCompareMessage")
            this.$http.post(`${this.$dbUrl}/scripts/get_similar_collocate_distributions.py`, {
                collocates: this.collocateCounts,
            },
                {
                    params: {
                        file_path: filePath,
                    }
                }).then((response) => {
                    this.mostSimilarDistributions = response.data.most_similar_distributions
                    this.mostDissimilarDistributions = response.data.most_dissimilar_distributions
                    this.cachedDistributions = filePath
                    this.similarSearching = false
                }).catch((error) => {
                    this.debug(this, error);
                });
        },
        similarToComparative(field) {
            this.$http.get(`${this.$dbUrl}/scripts/get_collocate_distribution.py`, {
                params: {
                    file_path: this.cachedDistributions,
                    field: field
                }
            }).then((response) => {
                this.comparedMetadataValues[this.similarFieldSelected] = field
                this.collocMethod = "compare";
                this.otherCollocates = this.extractSurfaceFromCollocate(response.data.collocates.slice(0, 100));
                this.wholeCorpus = false
                this.comparativeCollocations(response.data.collocates)
            }).catch((error) => {
                this.debug(this, error);
            });
        },
        getCollocatesOverTime(start, first) {
            this.collocationTimePeriods = []
            this.$http.post(`${this.$dbUrl}/reports/collocation.py`, {
                current_collocates: []
            }, {
                params: {
                    ...this.$store.state.formData,
                    max_time: 2,
                    time_series_interval: this.timeSeriesInterval,
                    map_field: "year",
                    start: start.toString(),
                    first: first
                }
            }).then((response) => {
                if (response.data.more_results) {
                    this.getCollocatesOverTime(response.data.hits_done, false);
                } else {
                    this.collocationTimeSeries(response.data.file_path, 0)
                }

            }).catch((error) => {
                this.debug(this, error);
            });
        },
        collocationTimeSeries(filePath, periodNumber) {
            this.$http.get(`${this.$dbUrl}/scripts/collocation_time_series.py`, {
                params: {
                    file_path: filePath,
                    year_interval: this.timeSeriesInterval,
                    period_number: periodNumber
                }
            }).then((response) => {
                this.collocationTimePeriods.push({
                    firstPeriod: this.extractSurfaceFromCollocate(response.data.first_period.collocates),
                    secondPeriod: this.extractSurfaceFromCollocate(response.data.second_period.collocates),
                    firstPeriodYear: `${response.data.first_period.year}-${response.data.first_period.year + parseInt(this.timeSeriesInterval)}`,
                    secondPeriodYear: `${response.data.second_period.year}-${response.data.second_period.year + parseInt(this.timeSeriesInterval)}`,
                })
                if (!response.data.done) {
                    periodNumber += 1
                    this.collocationTimeSeries(filePath, periodNumber)
                }
            }).catch((error) => {
                this.debug(this, error);
            });
        },
        collocateTimeSeriesClick(period) {
            let localClick = (item) => {
                let q = this.collocateCleanup(item)
                let method = "sentence_unordered"
                if (this.colloc_within == "n") {
                    method = "proxy_unordered"
                }
                this.$router.push(
                    this.paramsToRoute({
                        ...this.$store.state.formData,
                        report: "concordance",
                        q: q,
                        method: method,
                        year: period
                    })
                );
            }
            return localClick
        },
    },
};
</script>

<style lang="scss" scoped>
@import "../assets/styles/theme.module.scss";

th {
    background-color: $link-color;
    color: #fff;
    font-weight: 400;
    font-variant: small-caps;
}

tbody tr {
    cursor: pointer;
}

#description {
    position: relative;
}

#export-results {
    position: absolute;
    right: 0;
    padding: 0.125rem 0.25rem;
    font-size: 0.8rem !important;
}

.cloud-word {
    display: inline-block;
    padding: 5px;
    cursor: pointer;
    line-height: initial;
}

.table th,
.table td {
    padding: 0.45rem 0.75rem;
}

#filter-list {
    position: absolute;
    z-index: 100;
}

#filter-list .list-group-item {
    border-width: 0px;
    padding: 0.1rem;
}

#close-filter-list {
    width: fit-content;
    float: right;
    padding: 0 0.2rem;
    position: absolute;
    right: 0;
}

.input-group {
    max-width: 700px;
    width: 100%;
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

.btn-link {
    text-decoration: none;
    margin-left: -.75rem;
}

#colloc-tab button {
    font-variant: small-caps;
    font-size: 1rem;
}

.sim-dist {
    text-align: center;
    font-variant: small-caps;
    color: #fff;
    background-color: $link-color;
    padding: 0.5rem;
}

.colloc-cloud-title {
    text-align: center;
    background: $link-color;
    color: #fff;
}

.time-series-colloc-title {
    border: solid 1px rgba(0, 0, 0, 0.176);
    border-radius: 0.25rem 0.25rem 0 0;
    border-bottom-width: 0;
    padding: 0.5rem;
    width: fit-content;
    font-variant: small-caps;
}
</style>
<!-- Not scoped to apply to child -->
<style>
#results-summary-container .card {
    border-top-width: 0 !important;
    border-top-left-radius: 0%;
}
</style>