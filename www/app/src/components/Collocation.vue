<template>
    <div class="container-fluid mt-4">
        <results-summary :description="results.description" :running-total="runningTotal"
            :filter-list="filterList"></results-summary>
        <div class="row d-none d-sm-block mt-4" style="padding: 0 0.5rem">
            <div class="col col-sm-7 col-lg-8">
                <div class="btn-group" role="group" id="report_switch">
                    <button type="button" class="btn btn-secondary" :class="{ active: collocMethod === 'frequency' }"
                        @click="getFrequency()">
                        <span class="d-none d-sm-none d-md-inline">{{ $t("collocation.frequency") }}</span>
                    </button>
                    <button type="button" class="btn btn-secondary" :class="{ active: collocMethod === 'idf' }"
                        @click="getRelativeFrequency()">
                        <span class="d-none d-sm-none d-md-inline">{{ $t("collocation.idf") }}</span>
                    </button>
                    <button type="button" class="btn btn-secondary" :class="{ active: collocMethod === 'comparative' }"
                        @click="collocMethod = 'comparative'">
                        <span class="d-none d-sm-none d-md-inline">{{ $t("collocation.comparitive") }}</span>
                    </button>
                </div>
            </div>
        </div>
        <div class="card shadow-sm mt-2 ms-2 p-2" style="max-width: 720px;" v-if="collocMethod == 'comparative'">
            <div class="row">
                <div class="col-8">
                    <div class="input-group pb-2" v-for="localField in metadataDisplay" :key="localField.value">
                        <div class="input-group pb-2" :id="localField.value + '-group'"
                            v-if="metadataInputStyle[localField.value] == 'text'">
                            <button type="button" class="btn btn-outline-secondary">
                                <label :for="localField.value + 'input-filter'">{{
                                    localField.label
                                }}</label></button><input type="text" class="form-control"
                                :id="localField.value + 'input-filter'" :name="localField.value"
                                :placeholder="localField.example" v-model="metadataValues[localField.value]" v-if="metadataInputStyle[localField.value] == 'text' &&
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
                                <option v-for="innerValue in metadataChoiceValues[localField.value]" :key="innerValue.value"
                                    :value="innerValue.value" :id="innerValue.value + '-select-option'">
                                    {{ innerValue.text }}
                                </option>
                            </select>
                        </div>
                        <div class="input-group pb-2" :id="localField.value + '-group'"
                            v-if="['date', 'int'].includes(metadataInputStyle[localField.value])">
                            <button type="button" class="btn btn-outline-secondary"
                                style="border-top-right-radius: 0; border-bottom-right-radius: 0">
                                <label :for="localField.value + '-date'">{{ localField.label }}</label>
                            </button>
                            <div class="btn-group" role="group">
                                <button class="btn btn-secondary dropdown-toggle"
                                    style="border-top-left-radius: 0; border-bottom-left-radius: 0" type="button"
                                    :id="localField.value + '-selector'" data-bs-toggle="dropdown" aria-expanded="false">
                                    {{ $t(`searchForm.${dateType[localField.value]}Date`) }}
                                </button>
                                <ul class="dropdown-menu" :aria-labelledby="localField.value + '-selector'">
                                    <li @click="dateTypeToggle(localField.value, 'exact')">
                                        <a class="dropdown-item">{{ $t("searchForm.exactDate") }}</a>
                                    </li>
                                    <li @click="dateTypeToggle(localField.value, 'range')">
                                        <a class="dropdown-item">{{ $t("searchForm.rangeDate") }}</a>
                                    </li>
                                </ul>
                            </div>
                            <input type="text" class="form-control" :id="localField.value + 'input-filter'"
                                :name="localField.value" :placeholder="localField.example"
                                v-model="metadataValues[localField.value]" v-if="dateType[localField.value] == 'exact'" />
                            <span class="d-inline-block" v-if="dateType[localField.value] == 'range'">
                                <div class="input-group ms-3">
                                    <button class="btn btn-outline-secondary" type="button">
                                        <label for="query-term-input">{{
                                            $t("searchForm.dateFrom")
                                        }}</label>
                                    </button>
                                    <input type="text" class="form-control date-range"
                                        :id="localField.value + '-start-input-filter'" :name="localField.value + '-start'"
                                        :placeholder="localField.example" v-model="dateRange[localField.value].start" />
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
                <div class="col-4 position-relative">
                    <button type="button" class="btn btn-secondary position-absolute" style="bottom: 10px; right: 15px">Run
                        comparison
                    </button>
                </div>
            </div>

        </div>
        <div class="row mt-2 pe-1" style="padding: 0 0.5rem" v-if="resultsLength">
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
                            <tr style="line-height: 2rem" v-for="word in sortedList" :key="word.collocate"
                                @click="collocTableClick(word)">
                                <td class="text-view">{{ word.collocate }}</td>
                                <td>{{ word.count }}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            <div class="col-12 col-sm-8">
                <div class="card p-3 shadow-sm">
                    <div class="card-text">
                        <span class="cloud-word text-view" v-for="word in collocCloudWords" :key="word.word"
                            :style="getWordCloudStyle(word)" @click="collocTableClick(word)">{{ word.collocate }}</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>
<script>
import { mapFields } from "vuex-map-fields";
import ResultsSummary from "./ResultsSummary";
import variables from "../assets/styles/theme.module.scss";

export default {
    name: "collocation-report",
    components: {
        ResultsSummary,
    },
    computed: {
        ...mapFields([
            "formData.report",
            "formData.colloc_filter_choice",
            "formData.q",
            "formData.filter_frequency",
            "formData.arg_proxy",
            "formData.colloc_within",
            "currentReport",
            "resultsLength",
            "searching",
            "urlUpdate",
            "accessAuthorized",
            "searchableMetadata"
        ]),
        colorCodes: function () {
            let r = 0,
                g = 0,
                b = 0;
            // 3 digits
            if (this.cloudColor.length == 4) {
                r = "0x" + this.cloudColor[1] + this.cloudColor[1];
                g = "0x" + this.cloudColor[2] + this.cloudColor[2];
                b = "0x" + this.cloudColor[3] + this.cloudColor[3];

                // 6 digits
            } else if (this.cloudColor.length == 7) {
                r = "0x" + this.cloudColor[1] + this.cloudColor[2];
                g = "0x" + this.cloudColor[3] + this.cloudColor[4];
                b = "0x" + this.cloudColor[5] + this.cloudColor[6];
            }
            let colorCodes = {};
            var step = 0.03;
            for (let i = 0; i < 21; i += 1) {
                let rLocal = r - r * step * i;
                let gLocal = g - g * step * i;
                let bLocal = b - b * step * i;
                let opacityStep = i * 0.03;
                colorCodes[i] = `rgba(${rLocal}, ${gLocal}, ${bLocal}, ${0.4 + opacityStep})`;
            }
            return colorCodes;
        },
        splittedFilterList: function () {
            let arrayLength = this.filterList.length;
            let chunkSize = arrayLength / 5;
            let splittedList = [];
            for (let index = 0; index < arrayLength; index += chunkSize) {
                let myChunk = this.filterList.slice(index, index + chunkSize);
                splittedList.push(myChunk);
            }
            return splittedList;
        },
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
            moreResults: false,
            sortedList: [],
            showFilteredWords: false,
            runningTotal: 0,
            collocCloudWords: [],
            unboundListener: null,
            cloudColor: variables.color,
            collocMethod: "frequency",
            collocatesUnsorted: {},
            collocatesSorted: [],
            relativeFrequencies: {},
            loading: false,
            metadataDisplay: [],
            metadataInputStyle: [],
            metadataChoiceValues: [],
            metadataValues: {},
            dateRange: {},
            dateType: {},
        };
    },
    created() {
        this.report = "collocation";
        this.currentReport = "collocation";
        this.fetchResults();
    },
    watch: {
        urlUpdate() {
            if (this.$route.name == "collocation") {
                this.fetchResults();
            }
        },
        searchableMetadata: {
            handler: function (newVal, oldVal) {
                this.metadataDisplay = newVal.display;
                this.metadataInputStyle = newVal.inputStyle;
                this.metadataChoiceValues = newVal.choiceValues;
            },
            deep: true,
        },
    },
    methods: {
        fetchResults() {
            this.localFormData = this.copyObject(this.$store.state.formData);
            var collocObject = {};
            this.searching = true;
            this.relativeFrequencies = {};
            this.updateCollocation(collocObject, 0);
            // } else {
            //     this.retrieveFromStorage();
            // }
        },
        updateCollocation(fullResults, start) {
            let params = {
                ...this.$store.state.formData,
                start: start.toString(),
            };
            this.$http
                .get(`${this.$dbUrl}/reports/collocation.py`, {
                    params: this.paramsFilter(params),
                })
                .then((response) => {
                    let data = response.data;
                    this.resultsLength = data.results_length;
                    this.moreResults = data.more_results;
                    this.runningTotal = data.hits_done;
                    start = data.hits_done;
                    this.searching = false;
                    if (this.resultsLength) {
                        this.sortAndRenderCollocation(fullResults, data, start);
                    }
                })
                .catch((error) => {
                    this.searching = false;
                    this.debug(this, error);
                });
        },
        sortAndRenderCollocation(fullResults, data, start) {
            if (typeof fullResults === "undefined" || Object.keys(fullResults).length === 0) {
                fullResults = {};
                this.filterList = data.filter_list;
            }
            var collocates = this.mergeResults(fullResults, data.collocates);
            this.collocatesUnsorted = collocates.unsorted
            let sortedList = [];
            for (let word of collocates.sorted.slice(0, 100)) {
                let collocate = `${word.label}`.replace(/lemma:/, "");
                if (collocate.search(/\w+:.*/) != -1) {
                    collocate = collocate.replace(/(\p{L}+):.*/u, "$1");
                }
                let surfaceForm = word.label;
                sortedList.push({ collocate: collocate, surfaceForm: surfaceForm, count: word.count });
            }
            this.sortedList = sortedList;
            this.buildWordCloud();
            if (this.moreResults) {
                var tempFullResults = collocates.unsorted;
                var runningQuery = this.$store.state.formData;
                if (this.report === "collocation" && this.deepEqual(runningQuery, this.localFormData)) {
                    // make sure we're still running the same query
                    this.updateCollocation(tempFullResults, start);
                }
            } else {
                this.done = true;
                // Collocation cloud not showing when loading cached searches one after the other
                //saveToLocalStorage({results: this.sortedList, resultsLength: this.resultsLength, filterList: this.filterList});
            }
        },
        retrieveFromStorage() {
            let urlString = this.paramsToUrlString(this.$store.state.formData);
            var savedObject = JSON.parse(sessionStorage[urlString]);
            this.sortedList = savedObject.results;
            this.resultsLength = savedObject.resultsLength;
            this.filterList = savedObject.filterList;
            this.done = true;
            this.searching = false;
        },
        collocTableClick(item) {
            let q
            if (item.surfaceForm.startsWith("lemma:")) {
                q = `${this.q} ${item.surfaceForm}`;
            } else if (item.surfaceForm.search(/\w+:.*/) != -1) {
                q = `${this.q} ${item.surfaceForm}`;
            }
            else {
                q = `${this.q} "${item.surfaceForm}"`;
            }
            let method = "cooc"
            if (this.arg_proxy.length > 0) {
                method = 'proxy'
            }
            this.$router.push(
                this.paramsToRoute({
                    ...this.$store.state.formData,
                    report: "concordance",
                    q: q,
                    method: method,
                })
            );
        },
        buildWordCloud() {
            let lowestValue = this.sortedList[this.sortedList.length - 1].count;
            let higestValue = this.sortedList[0].count;
            let diff = higestValue - lowestValue;
            let coeff = diff / 20;

            var adjustWeight = function (count) {
                let adjustedCount = count - lowestValue;
                let adjustedWeight = Math.round(adjustedCount / coeff);
                adjustedWeight = parseInt(adjustedWeight);
                return adjustedWeight;
            };

            let weightedWordList = [];
            for (let wordObject of this.sortedList) {
                let adjustedWeight = adjustWeight(wordObject.count);
                weightedWordList.push({
                    collocate: wordObject.collocate,
                    surfaceForm: wordObject.surfaceForm,
                    weight: 1 + adjustedWeight / 10,
                    color: this.colorCodes[adjustedWeight],
                });
            }
            weightedWordList.sort(function (a, b) {
                return a.collocate.localeCompare(b.collocate);
            });
            this.collocCloudWords = weightedWordList;
        },
        getWordCloudStyle(word) {
            return `font-size: ${word.weight}rem; color: ${word.color}`;
        },
        dateTypeToggle(metadata, dateType) {
            this.dateRange[metadata] = { start: "", end: "" };
            this.metadataValues[metadata] = "";
            this.dateType[metadata] = dateType;
        },
        getFrequency() {
            this.collocMethod = "frequency";
            this.sortedList = this.collocatesSorted
            this.buildWordCloud()
        },
        getRelativeFrequency() {
            if (this.collocMethod == "frequency") {
                this.collocatesSorted = this.copyObject(this.sortedList)
            }
            this.collocMethod = "idf";
            if (Object.keys(this.relativeFrequencies).length === 0) {
                this.searching = true;
                this.$http.post(`${this.$dbUrl}/scripts/get_collocation_relative_proportions.py`, {
                    all_collocates: this.collocatesUnsorted,
                }, {
                    params: {
                        ...this.$store.state.formData,
                    },

                }, {
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded'
                    }
                }).then((response) => {
                    this.relativeFrequencies = response.data;
                    this.sortedList = response.data;
                    this.searching = false;
                    this.buildWordCloud();

                }).catch((error) => {
                    this.debug(this, error);
                });
            } else {
                this.sortedList = this.relativeFrequencies
                this.buildWordCloud();
            }
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
</style>

