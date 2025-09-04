<template>
    <div id="collocation-container" class="container-fluid mt-4">
        <div class="d-none d-sm-block mt-3" style="padding: 0 0.5rem">
            <ul class="nav nav-tabs" id="colloc-method-switch" role="tablist"
                :aria-label="$t('collocation.methodSelectionTabs')">
                <li class="nav-item" role="presentation">
                    <button class="nav-link shadow-sm" id="frequency-tab" data-bs-toggle="tab"
                        :class="{ active: collocMethod === 'frequency' }" data-bs-target="#frequency-tab-pane"
                        type="button" role="tab" aria-controls="frequency-tab-pane"
                        :aria-selected="collocMethod === 'frequency'" @click="getFrequency()"
                        :aria-label="$t('collocation.collocation')">
                        {{ $t("collocation.collocation") }}
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link shadow-sm" id="compare-tab" data-bs-toggle="tab"
                        :class="{ active: collocMethod === 'compare' }" data-bs-target="#compare-tab-pane" type="button"
                        role="tab" aria-controls="compare-tab-pane" :aria-selected="collocMethod === 'compare'"
                        @click="toggleCompare()" :aria-label="$t('collocation.compareTo')">
                        {{ $t("collocation.compareTo") }}
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link shadow-sm" id="similar-tab" data-bs-toggle="tab"
                        :class="{ active: collocMethod === 'similar' }" data-bs-target="#similar-tab-pane" type="button"
                        role="tab" aria-controls="similar-tab-pane" :aria-selected="collocMethod === 'similar'"
                        @click="toggleSimilar()" :aria-label="$t('collocation.similarUsage')">
                        {{ $t("collocation.similarUsage") }}
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link shadow-sm" id="time-series-tab" data-bs-toggle="tab"
                        :class="{ active: collocMethod === 'timeSeries' }" data-bs-target="#time-series-tab-pane"
                        type="button" role="tab" aria-controls="time-series-tab-pane"
                        :aria-selected="collocMethod === 'timeSeries'" @click="toggleTimeSeries()"
                        :aria-label="$t('collocation.timeSeries')">
                        {{ $t("collocation.timeSeries") }}
                    </button>
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
                            <label class="btn btn-outline-secondary" :for="localField.value + '-input-filter'">
                                {{ localField.label }}
                            </label>
                            <input type="text" class="form-control" :id="localField.value + '-input-filter'"
                                :name="localField.value" :placeholder="localField.example"
                                v-model="comparedMetadataValues[localField.value]"
                                :aria-label="`${$t('collocation.filterBy')} ${localField.label}`" v-if="metadataInputStyle[localField.value] == 'text' &&
                                    metadataInputStyle[localField.value] != 'date'" />
                        </div>

                        <div class="input-group pb-2" :id="localField.value + '-group'"
                            v-if="metadataInputStyle[localField.value] == 'checkbox'">
                            <span class="btn btn-outline-secondary me-2"
                                style="border-top-right-radius: 0; border-bottom-right-radius: 0">
                                {{ localField.label }}
                            </span>
                            <div class="d-inline-block">
                                <div class="form-check d-inline-block ms-3" style="padding-top: 0.35rem"
                                    :id="localField.value" :options="metadataChoiceValues[localField.value]"
                                    v-for="metadataChoice in metadataChoiceValues[localField.value]"
                                    :key="metadataChoice.value" v-once>
                                    <input class="form-check-input" type="checkbox" :id="metadataChoice.value"
                                        v-model="metadataChoiceChecked[metadataChoice.value]"
                                        :aria-label="`${$t('collocation.filterBy')} ${localField.label}: ${metadataChoice.text}`" />
                                    <label class="form-check-label" :for="metadataChoice.value">
                                        {{ metadataChoice.text }}
                                    </label>
                                </div>
                            </div>
                        </div>

                        <!-- Dropdown fields -->
                        <div class="input-group pb-2" :id="localField.value + '-group'"
                            v-if="metadataInputStyle[localField.value] == 'dropdown'">
                            <label class="btn btn-outline-secondary" :for="localField.value + '-select'">
                                {{ localField.label }}
                            </label>
                            <select class="form-select" :id="localField.value + '-select'"
                                v-model="metadataChoiceSelected[localField.value]"
                                :aria-label="`${$t('collocation.filterBy')} ${localField.label}`">
                                <option v-for="innerValue in metadataChoiceValues[localField.value]"
                                    :key="innerValue.value" :value="innerValue.value">
                                    {{ innerValue.text }}
                                </option>
                            </select>
                        </div>

                        <div class="input-group pb-2" :id="localField.value + '-group'"
                            v-if="['date', 'int'].includes(metadataInputStyle[localField.value])">
                            <span class="btn btn-outline-secondary"
                                style="border-top-right-radius: 0; border-bottom-right-radius: 0">
                                {{ localField.label }}
                            </span>
                            <div class="btn-group" role="group">
                                <button class="btn btn-secondary dropdown-toggle"
                                    style="border-top-left-radius: 0; border-bottom-left-radius: 0" type="button"
                                    :id="localField.value + '-selector'" data-bs-toggle="dropdown" aria-expanded="false"
                                    :aria-label="`${$t('collocation.selectDateType')} ${localField.label}`">
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
                            <input type="text" class="form-control" :id="localField.value + '-input-filter'"
                                :name="localField.value" :placeholder="localField.example"
                                v-model="comparedMetadataValues[localField.value]"
                                :aria-label="`${$t('collocation.filterBy')} ${localField.label}`"
                                v-if="dateType[localField.value] == 'exact'" />
                            <span class="d-inline-block" v-if="dateType[localField.value] == 'range'">
                                <div class="input-group ms-3">
                                    <label class="btn btn-outline-secondary"
                                        :for="localField.value + '-start-input-filter'">
                                        {{ $t("searchForm.dateFrom") }}
                                    </label>
                                    <input type="text" class="form-control date-range"
                                        :id="localField.value + '-start-input-filter'"
                                        :name="localField.value + '-start'" :placeholder="localField.example"
                                        v-model="dateRange[localField.value].start"
                                        :aria-label="`${$t('searchForm.dateFrom')} ${localField.label}`" />
                                    <label class="btn btn-outline-secondary ms-3"
                                        :for="localField.value + '-end-input-filter'">
                                        {{ $t("searchForm.dateTo") }}
                                    </label>
                                    <input type="text" class="form-control date-range"
                                        :id="localField.value + '-end-input-filter'" :name="localField.value + '-end'"
                                        :placeholder="localField.example" v-model="dateRange[localField.value].end"
                                        :aria-label="`${$t('searchForm.dateTo')} ${localField.label}`" />
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
            <div class="d-flex align-items-center flex-wrap mt-2">
                <div class="btn-group" style="width: fit-content;" role="group">
                    <button class="btn btn-secondary dropdown-toggle" type="button" data-bs-toggle="dropdown"
                        aria-expanded="false">
                        {{ this.similarFieldSelected || "Select a field" }}
                    </button>
                    <ul class="dropdown-menu">
                        <li v-for="field in fieldsToCompare" :key="field.value"
                            @click="similarCollocDistributions(field, 0)"><a class="dropdown-item">{{ field.label }}</a>
                        </li>
                    </ul>
                </div>
                <span class="ms-2">{{ $t("collocation.mostSimilarUsage") }}</span>
            </div>

            <bibliography-criteria class="ms-2 mt-3" :biblio="biblio" :query-report="formData.report"
                :results-length="resultsLength" :hide-criteria-string="true"></bibliography-criteria>

            <div class="mt-2" style="display: flex; align-items: center;" v-if="similarSearching">
                <div class="alert alert-info p-1 mb-0 d-inline-block" style="width: fit-content" role="alert">
                    {{ similarSearchProgress }}...
                </div>
                <progress-spinner class="px-2" :progress="progressPercent" />
            </div>
        </div>
        <div class="card shadow-sm mx-2 p-3" style="border-top-width: 0;" v-if="collocMethod === 'timeSeries'">
            <bibliography-criteria :biblio="biblio" :query-report="formData.report"
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
                    <table class="table table-borderless caption-top" aria-label="$t('collocation.collocatesTable')">
                        <caption class="visually-hidden">
                            {{ $t('collocation.collocatesTableCaption') }}
                        </caption>
                        <thead class="table-header">
                            <tr>
                                <th scope="col" id="collocate-header">{{ $t("collocation.collocate") }}</th>
                                <th scope="col" id="count-header">{{ $t("collocation.count") }}</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="line-height: 1.75rem" v-for="(word, index) in sortedList" :key="word.collocate"
                                :tabindex="0" @click="collocateClick(word)" @keydown.enter="collocateClick(word)"
                                @keydown.space.prevent="collocateClick(word)"
                                :aria-label="`${word.collocate} ${word.count}`">
                                <td class="text-view" :id="`collocate-${index}`">{{ word.collocate }}</td>
                                <td :id="`count-${index}`">{{ word.count }}</td>
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
                        <bibliography-criteria :biblio="biblio" :query-report="formData.report"
                            :results-length="resultsLength"></bibliography-criteria>
                    </div>
                    <div class="col-6" style="border-left: solid 1px rgba(0, 0, 0, 0.176)">
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
                                <word-cloud :word-weights="sortedList" label=""
                                    :click-handler="collocateClick"></word-cloud>
                            </div>
                            <div class="col-6" style="border-left: solid 1px rgba(0, 0, 0, 0.176)" role="region"
                                :aria-label="$t('collocation.comparisonCorpusResults')">
                                <div class="d-flex justify-content-center position-relative" v-if="compareSearching"
                                    role="status" :aria-label="$t('common.loading')">
                                    <progress-spinner :progress="progressPercent" :lg="true" />
                                </div>
                                <word-cloud v-if="otherCollocates.length > 0" :word-weights="otherCollocates" label=""
                                    :click-handler="otherCollocateClick"></word-cloud>
                            </div>
                        </div>
                    </div>
                    <div class="tab-pane fade" id="rep-tab-pane" role="tabpanel" aria-labelledby="rep-tab" tabindex="0"
                        :aria-label="$t('collocation.overRepresentedCollocatesPanel')">
                        <div class="row gx-5">
                            <div class="col-6" role="region" :aria-label="$t('collocation.overRepresentedResults')">
                                <word-cloud v-if="overRepresented.length > 0" :word-weights="overRepresented"
                                    :click-handler="collocateClick"></word-cloud>
                            </div>
                            <div class="col-6" style="border-left: solid 1px rgba(0, 0, 0, 0.176)" role="region"
                                :aria-label="$t('collocation.underRepresentedResults')">
                                <div class="d-flex justify-content-center position-relative" v-if="compareSearching"
                                    role="status" :aria-label="$t('common.loading')">
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
            <div class="card" v-if="mostSimilarDistributions.length > 0" role="region"
                :aria-label="$t('collocation.similarUsageResults')">
                <div class="row">
                    <div class="col-6 pe-0">
                        <h6 class="sim-dist" role="heading" aria-level="3">
                            {{ $t("collocation.topSimilarUses") }}
                        </h6>
                        <ul class="list-group list-group-flush mt-3" :aria-label="$t('collocation.topSimilarUses')">
                            <li v-for="(metadataValue, index) in mostSimilarDistributions" :key="metadataValue">
                                <button type="button"
                                    class="list-group-item position-relative w-100 text-start border-0"
                                    style="text-align: justify" @click="similarToComparative(metadataValue[0])"
                                    @keydown.enter="similarToComparative(metadataValue[0])"
                                    @keydown.space.prevent="similarToComparative(metadataValue[0])"
                                    :aria-label="`${$t('collocation.compareTo')} ${metadataValue[0]}, ${$t('collocation.count')}: ${metadataValue[1]}`">
                                    {{ metadataValue[0] }}
                                    <span class="badge text-bg-secondary position-absolute" style="right: 1rem"
                                        aria-hidden="true">
                                        {{ metadataValue[1] }}
                                    </span>
                                </button>
                                <hr v-if="index < mostSimilarDistributions.length - 1" class="my-0"
                                    style="opacity: 1; border-color: rgba(0, 0, 0, 0.125);" aria-hidden="true">
                            </li>
                        </ul>
                    </div>
                    <div class="col-6 ps-0" style="border-left: solid 1px rgba(0, 0, 0, 0.176)">
                        <h6 class="sim-dist" role="heading" aria-level="3">
                            {{ $t("collocation.topDissimilarUses") }}
                        </h6>
                        <ul class="list-group list-group-flush mt-3" :aria-label="$t('collocation.topDissimilarUses')">
                            <li v-for="(metadataValue, index) in mostDissimilarDistributions" :key="metadataValue">
                                <button type="button"
                                    class="list-group-item position-relative w-100 text-start border-0"
                                    style="text-align: justify" @click="similarToComparative(metadataValue[0])"
                                    @keydown.enter="similarToComparative(metadataValue[0])"
                                    @keydown.space.prevent="similarToComparative(metadataValue[0])"
                                    :aria-label="`${$t('collocation.compareTo')} ${metadataValue[0]}, ${$t('collocation.count')}: ${metadataValue[1]}`">
                                    {{ metadataValue[0] }}
                                    <span class="badge text-bg-secondary position-absolute" style="right: 1rem"
                                        aria-hidden="true">
                                        {{ metadataValue[1] }}
                                    </span>
                                </button>
                                <hr v-if="index < mostSimilarDistributions.length - 1" class="my-0"
                                    style="opacity: 1; border-color: rgba(0, 0, 0, 0.125);" aria-hidden="true">
                            </li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        <div v-if="collocMethod == 'timeSeries'" class="mx-2 my-3">
            <div v-if="searching" role="status" :aria-label="$t('common.loading')">
                {{ $t('collocation.similarCollocGatheringMessage') }}...
            </div>
            <div v-if="collocationTimePeriods.length > 0" class="row" role="region"
                :aria-label="$t('collocation.timeSeriesResults')">
                <div class="col-12 col-md-6" v-for="(period, index) in collocationTimePeriods" :key="period.year">
                    <article class="card mb-3" v-if="period.done" role="article"
                        :aria-labelledby="`period-${index}-title`">
                        <header class="card-header p-2 d-flex align-items-center">
                            <h6 class="mb-0" :id="`period-${index}-title`" role="heading" aria-level="4">
                                {{ period.periodYear }}
                            </h6>
                        </header>
                        <div class="btn-group w-100 rounded-0" role="group"
                            :aria-label="`${$t('collocation.viewToggle')} ${period.periodYear}`">
                            <button class="btn btn-sm rounded-0"
                                :class="period.showDistinctive ? 'btn-secondary active' : 'btn-outline-secondary'"
                                @click="period.showDistinctive = true" :aria-pressed="period.showDistinctive"
                                :aria-label="$t('collocation.overRepresentedCollocates')">
                                {{ $t('collocation.overRepresentedCollocates') }}
                            </button>
                            <button class="btn btn-sm rounded-0"
                                :class="!period.showDistinctive ? 'btn-secondary active' : 'btn-outline-secondary'"
                                @click="period.showDistinctive = false" :aria-pressed="!period.showDistinctive"
                                :aria-label="$t('collocation.frequentCollocates')">
                                {{ $t('collocation.frequentCollocates') }}
                            </button>
                        </div>
                        <div class="card-body pt-2" role="region"
                            :aria-label="`${period.showDistinctive ? $t('collocation.overRepresentedCollocates') : $t('collocation.frequentCollocates')} ${period.periodYear}`">
                            <word-cloud :word-weights="period.showDistinctive ? period.distinctive : period.frequent"
                                label="" :click-handler="collocateTimeSeriesClick(period.periodYear)">
                            </word-cloud>
                        </div>
                    </article>
                    <div style="margin-top: 5em; width: 100%; text-align: center" v-else role="status"
                        :aria-label="`${$t('common.loading')} ${$t('collocation.gatheringTimeSeriesPeriod')}`">
                        <p class="mb-1">{{ $t('collocation.gatheringTimeSeriesPeriod') }}...</p>
                        <progress-spinner />
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
import { Collapse } from "bootstrap";
import { mapStores, mapWritableState } from "pinia";
import { useMainStore } from "../stores/main";
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
        ...mapWritableState(useMainStore, [
            "formData",
            "currentReport",
            "resultsLength",
            "searching",
            "urlUpdate",
            "accessAuthorized",
            "searchableMetadata"
        ]),
        ...mapStores(useMainStore),
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
            metadataChoiceChecked: {},
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
            progressPercent: 0,
            distinctiveView: 'prev', // Default to showing comparison with previous period
        };
    },
    created() {
        this.formData.report = "collocation";
        this.currentReport = "collocation";
        this.buildMetadata(this.searchableMetadata);
        this.biblio = this.buildBiblioCriteria(this.$philoConfig, this.$route.query, this.formData)
        this.collocMethod = this.$route.query.collocation_method || 'frequency';

        switch (this.collocMethod) {
            case "frequency":
                this.fetchResults();
                break;
            case "similar":
                this.toggleSimilar(false);
                this.fetchResults();
                break;
            case "timeSeries":
                this.toggleTimeSeries(false);
                this.timeSeriesInterval = this.$route.query.time_series_interval || 10;
                this.getCollocatesOverTime(0, true);
                break;
            case "compare":
                this.toggleCompare(false);
                this.restoreComparedMetadataFromUrl();
                this.fetchResults();
                break;
        }
    },
    watch: {
        $route(newUrl, oldUrl) {
            if (this.$route.name == "collocation") {
                this.fetchResults();
                if (newUrl.query.collocation_method != oldUrl.query.collocation_method) {
                    this.setCollocationMethod(newUrl.query.collocation_method || 'frequency', false);
                }
                // Restore compared metadata if in compare mode
                if (newUrl.query.collocation_method === 'compare') {
                    this.restoreComparedMetadataFromUrl();
                }
                this.biblio = this.buildBiblioCriteria(this.$philoConfig, this.$route.query, this.formData)
            }
        },
        searchableMetadata: {
            handler: function (newVal, oldVal) {
                this.buildMetadata(newVal)
            },
            deep: true,
        }
    },
    methods: {
        fetchResults() {
            this.localFormData = this.copyObject(this.formData);
            this.relativeFrequencies = {};
            this.searching = true;
            this.fetchComplete = false; // Reset fetch completion status
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
                ...this.formData,
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
                            this.done = true;

                            // After base collocation results are loaded
                            this.checkCollocationMethod();
                        }
                    }
                    else {
                        this.fetchComplete = true; // Mark fetch as complete even with no results
                    }

                })
                .catch((error) => {
                    this.searching = false;
                    this.debug(this, error);
                });
        },
        setCollocationMethod(method, updateUrl = true) {
            switch (method) {
                case 'compare':
                    this.toggleCompare(updateUrl);
                    break;
                case 'similar':
                    this.toggleSimilar(updateUrl);
                    break;
                case 'timeSeries':
                    this.toggleTimeSeries(updateUrl);
                    this.getCollocatesOverTime(0, true)
                    break;
                default:
                    this.getFrequency(updateUrl);
                    break;
            }
        },
        checkCollocationMethod() {
            this.collocMethod = this.$route.query.collocation_method || 'frequency';
            switch (this.collocMethod) {
                case "frequency":
                    break;
                case "similar":
                    this.toggleSimilar(false);
                    this.similarCollocDistributions({ value: this.$route.query.similarity_by }, 0);
                    break;
                case "compare":
                    this.getOtherCollocates({}, 0);;
                    break;
            }
        },
        collocateCleanup(collocate) {
            let q
            if (collocate.surfaceForm.startsWith("lemma:")) {
                q = `${this.formData.q} ${collocate.surfaceForm}`;
            } else if (collocate.surfaceForm.search(/\w+:.*/) != -1) {
                q = `${this.formData.q} ${collocate.surfaceForm}`;
            }
            else {
                q = `${this.formData.q} "${collocate.surfaceForm}"`;
            }
            return q
        },
        collocateClick(item) {
            let q = this.collocateCleanup(item)
            let method = "sentence"
            if (this.formData.colloc_within == "n") {
                method = "proxy"
            }
            this.$router.push(
                this.paramsToRoute({
                    ...this.formData,
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
            if (this.formData.colloc_within == "n") {
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
        getFrequency(updateUrl = false) {
            this.collocMethod = "frequency";
            if (updateUrl) {
                this.updateCollocationUrl();
            }
        },
        toggleCompare(updateUrl = false) {
            this.collocMethod = "compare";
            this.filterMetadataOpen = true
            this.$nextTick(() => {
                let collapseElement = document.getElementById('other-corpus-metadata')
                if (collapseElement) {
                    try {
                        new Collapse(collapseElement, {
                            toggle: true
                        })
                    } catch (error) {
                        console.warn('Could not initialize Bootstrap Collapse:', error);
                    }
                }
            })
            if (updateUrl) {
                this.updateCollocationUrl();
            }
        },
        toggleSimilar(updateUrl = false) {
            this.collocMethod = "similar";
            if (updateUrl) {
                this.updateCollocationUrl();
            }
        },
        toggleTimeSeries(updateUrl = false) {
            this.collocMethod = "timeSeries";
            if (updateUrl) {
                this.updateCollocationUrl();
            }
        },
        checkCollocationStateFromUrl() {
            this.collocMethod = this.$route.query.collocation_method || 'frequency';
            if (this.collocMethod === "similar") {
                this.similarFieldSelected = this.$route.query.similarity_by
            }
        },
        updateCollocationUrl() {
            let urlParams = {
                ...this.formData,
                collocation_method: this.collocMethod,
            };

            // Add method-specific parameters
            if (this.collocMethod === 'similar' && this.similarFieldSelected) {
                urlParams.similarity_by = this.similarFieldSelected;
                delete urlParams.time_series_interval;
            }
            if (this.collocMethod === 'timeSeries') {
                urlParams.time_series_interval = this.timeSeriesInterval;
                delete urlParams.similarity_by;
            }
            if (this.collocMethod === 'compare') {
                // Add comparative metadata parameters with compare_ prefix
                const compareParams = this.getNonEmptyComparedMetadata();
                Object.assign(urlParams, compareParams);
                delete urlParams.similarity_by;
                delete urlParams.time_series_interval;
            }

            const routeParams = this.paramsToRoute(urlParams);
            this.$router.push(routeParams);
        },
        getNonEmptyComparedMetadata() {
            const nonEmptyMetadata = {};

            // Add all non-empty values from comparedMetadataValues with compare_ prefix
            for (const [field, value] of Object.entries(this.comparedMetadataValues)) {
                if (value && value !== '') {
                    nonEmptyMetadata[`compare_${field}`] = value;
                }
            }

            return nonEmptyMetadata;
        },
        restoreComparedMetadataFromUrl() {
            const urlParams = this.$route.query;

            // Clear existing values
            this.comparedMetadataValues = {};

            // Process compare_ prefixed parameters
            for (const [key, value] of Object.entries(urlParams)) {
                if (key.startsWith('compare_')) {
                    const field = key.substring(8); // Remove 'compare_' prefix
                    this.comparedMetadataValues[field] = value;
                }
            }
        },
        getOtherCollocates(fullResults, start) {
            // Hide the metadata form immediately when comparison starts
            if (Object.keys(fullResults).length === 0) {
                let collapseElement = document.getElementById('other-corpus-metadata')
                if (collapseElement != null) {
                    Collapse.getInstance(collapseElement).hide()
                    this.filterMetadataOpen = false
                }
            }

            if (Object.keys(this.comparedMetadataValues).length === 0) {
                this.wholeCorpus = true
            } else {
                this.wholeCorpus = false
            }
            if (Object.keys(fullResults).length === 0) {
                this.progressPercent = 0
            }
            this.collocMethod = 'compare';
            this.updateCollocationUrl(); // Update URL when running compare search
            this.comparedMetadataValues = this.dateRangeHandler(this.metadataInputStyle, this.dateRange, this.dateType, this.comparedMetadataValues)
            let params = {
                q: this.formData.q,
                colloc_filter_choice: this.formData.colloc_filter_choice,
                colloc_within: this.formData.colloc_within,
                filter_frequency: this.formData.filter_frequency,
                q_attribute: this.formData.q_attribute || "",
                q_attribute_value: this.formData.q_attribute_value || "",
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
            this.comparativeSearchStarted = true;
            this.comparedMetadataValues = this.dateRangeHandler(this.metadataInputStyle, this.dateRange, this.dateType, this.comparedMetadataValues)
            this.otherBiblio = this.buildBiblioCriteria(this.$philoConfig, this.comparedMetadataValues, this.comparedMetadataValues)
            this.overRepresented = [];
            this.underRepresented = [];
            this.$http.post(`${this.$dbUrl}/scripts/comparative_collocations.py`, {
                all_collocates: this.collocateCounts,
                other_collocates: otherCollocates,
                whole_corpus: this.wholeCorpus,
            }).then((response) => {
                this.overRepresented = this.extractSurfaceFromCollocate(response.data.top);
                this.underRepresented = this.extractSurfaceFromCollocate(response.data.bottom);
                this.relativeFrequencies = { top: this.overRepresented, bottom: this.underRepresented };

            }).catch((error) => {
                this.debug(this, error);
            });
        },
        similarCollocDistributions(field, start, first) {
            this.similarFieldSelected = field.value // Store the field value for URL
            this.similarSearching = true
            this.similarSearchProgress = this.$t("collocation.similarCollocGatheringMessage")
            this.mostSimilarDistributions = []
            if (typeof first === 'undefined') {
                first = true
                this.progressPercent = 0
                this.collocMethod = 'similar';
                this.updateCollocationUrl(); // Update URL when running similar usage search
            }
            else {
                first = false
            }
            this.$http
                .post(`${this.$dbUrl}/reports/collocation.py`, {
                    current_collocates: [],
                }, {
                    params: {
                        q: this.formData.q, start: start.toString(),
                        colloc_filter_choice: this.formData.colloc_filter_choice,
                        colloc_within: this.formData.colloc_within,
                        filter_frequency: this.formData.filter_frequency,
                        map_field: field.value,
                        q_attribute: this.formData.q_attribute || "",
                        q_attribute_value: this.formData.q_attribute_value || "",
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
            this.searching = true
            if (first) {
                this.collocMethod = 'timeSeries';
                this.updateCollocationUrl(); // Update URL when running time series search
            }
            const interval = parseInt(this.timeSeriesInterval)
            let params = {
                ...this.formData,
                max_time: 2,
                time_series_interval: interval,
                map_field: "year",
                start: start.toString(),
                first: first
            }

            // Handle year range modifications
            if (this.formData.year) {
                const yearParts = this.formData.year.split('-')
                if (yearParts.length === 2) {
                    const [startYear, endYear] = yearParts
                    if (startYear && endYear) {
                        // Both start and end years provided
                        params.year = `${parseInt(startYear) - interval}-${parseInt(endYear) + interval}`
                    } else if (startYear) {
                        // Only start year provided
                        params.year = `${parseInt(startYear) - interval}-`
                    } else if (endYear) {
                        // Only end year provided
                        params.year = `-${parseInt(endYear) + interval}`
                    }
                } else if (yearParts[0]) {
                    // Single year
                    const year = parseInt(yearParts[0])
                    params.year = `${year - interval}-${year + interval}`
                }
            }

            this.$http.post(`${this.$dbUrl}/reports/collocation.py`, {
                current_collocates: []
            }, {
                params: params
            }).then((response) => {
                if (response.data.more_results) {
                    this.getCollocatesOverTime(response.data.hits_done, false);
                } else {
                    this.searching = false
                    this.collocationTimeSeries(response.data.file_path, 0)
                }
            }).catch((error) => {
                this.debug(this, error);
            });
        },
        collocationTimeSeries(filePath, periodNumber) {
            this.collocationTimePeriods[periodNumber] = { year: periodNumber, done: false }
            this.$http.get(`${this.$dbUrl}/scripts/collocation_time_series.py`, {
                params: {
                    file_path: filePath,
                    year_interval: this.timeSeriesInterval,
                    period_number: periodNumber
                }
            }).then((response) => {
                if (response.data.period) {
                    const period = response.data.period;
                    const year = period.year;
                    const interval = parseInt(this.timeSeriesInterval);

                    const frequent = this.extractSurfaceFromCollocate(period.collocates.frequent || []);
                    const distinctive = this.extractSurfaceFromCollocate(period.collocates.distinctive || []);

                    this.collocationTimePeriods[periodNumber] = {
                        year: year,
                        frequent: frequent,
                        distinctive: distinctive,
                        periodYear: `${year}-${year + interval}`,
                        showDistinctive: true,  // Default to showing distinctive collocates
                        done: true
                    };
                }

                if (!response.data.done) {
                    periodNumber += 1;
                    this.collocationTimeSeries(filePath, periodNumber);
                }
            }).catch((error) => {
                this.debug(this, error);
            });
        },

        getDistinctiveCollocates(period) {
            if (!period) {
                console.log('Period is null/undefined');  // Debug missing period
                return [];
            }
            const result = this.distinctiveView === 'prev' ?
                period.distinctive_prev :
                period.distinctive_next;
            console.log('Distinctive collocates for period:', {
                view: this.distinctiveView,
                result: result
            });  // Debug result
            return result;
        },
        collocateTimeSeriesClick(period) {
            let localClick = (item) => {
                let q = this.collocateCleanup(item)
                let method = "sentence_unordered"
                if (this.formData.colloc_within == "n") {
                    method = "proxy_unordered"
                }
                this.$router.push(
                    this.paramsToRoute({
                        ...this.formData,
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
@use "../assets/styles/theme.module.scss" as theme;

th {
    font-variant: small-caps;
    background-color: theme.$card-header-color !important;
    color: white !important;
    border-color: theme.$card-header-color !important;
}

.table tbody tr {
    cursor: pointer;
    transition: all 0.2s ease;
    border: 1px solid transparent;
}

.table tbody tr:hover {
    transform: scale(1.01);
    background-color: rgba(theme.$link-color, 0.15) !important;
    border-color: rgba(theme.$link-color, 0.3);
    box-shadow: inset 0 0 8px rgba(theme.$link-color, 0.1);
    cursor: pointer;
    z-index: 1;
}

.table tbody tr:active {
    transform: scale(0.98);
}

.table tbody tr:hover {
    transform: scale(1.01) !important;
    background-color: rgba(theme.$link-color, 0.15) !important;
    border: 1px solid rgba(theme.$link-color, 0.3) !important;
    box-shadow: inset 0 0 8px rgba(theme.$link-color, 0.1) !important;
    cursor: pointer !important;
    z-index: 1 !important;
}

.table tbody tr:hover td {
    background-color: rgba(theme.$link-color, 0.15) !important;
    color: inherit !important;
}

.table tbody tr:active {
    transform: scale(0.98);
}

.table tbody tr:focus {
    background-color: rgba(theme.$button-color, 0.15) !important;
    color: theme.$button-color !important;
    outline: 2px solid theme.$button-color;
    outline-offset: -2px;
}

/* Ensure cells inherit the row colors */
.table tbody tr:hover td {
    color: inherit;
}

.table tbody tr:focus td {
    color: inherit;
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
    background-color: theme.$link-color;
    padding: 0.5rem;
}

.colloc-cloud-title {
    text-align: center;
    background: theme.$link-color;
    color: #fff;
}

.card-header {
    text-align: center;

    h6 {
        width: 100%;
        font-variant: small-caps;
        font-size: 1rem;
    }
}

.btn-group {
    border: none;
    border-bottom: solid 1px rgba(0, 0, 0, 0.176);
}

.btn-sm {
    font-size: 0.8rem;
    padding: 0.4rem 0.5rem;
    flex: 1;
}

.badge {
    font-size: 0.75rem;
}

:deep(.progress-bar) {
    background-color: theme.$link-color !important;
}
</style>