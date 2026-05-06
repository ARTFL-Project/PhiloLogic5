<template>
    <div>
        <!-- Year-interval controls -->
        <div class="card shadow-sm mx-2 p-3" style="border-top-width: 0;">
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
                @click="runEvolution()">{{ $t('collocation.searchEvolution') }}</button>
        </div>

        <!-- Period grid -->
        <div class="mx-2 my-3">
            <div v-if="searching" role="status" aria-live="polite" aria-atomic="true"
                :aria-label="$t('common.loading')">
                {{ $t('collocation.similarCollocGatheringMessage') }}...
            </div>
            <div v-if="collocationTimePeriods.length > 0" class="row" role="region"
                :aria-label="$t('collocation.timeSeriesResults')">
                <h2 class="visually-hidden">{{ $t('collocation.timeSeriesResults') }}</h2>
                <div class="col-12 col-md-6" v-for="(period, index) in collocationTimePeriods" :key="period.year"
                    :id="`period-${period.periodYear}`">
                    <article class="card mb-3" v-if="period.done" role="article"
                        :aria-labelledby="`period-${index}-title`">
                        <header class="card-header p-2 text-center">
                            <h3 class="mb-0 period-title" :id="`period-${index}-title`">
                                {{ period.periodYear }}
                            </h3>
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
                            <word-cloud
                                :word-weights="period.showDistinctive ? period.distinctive : period.frequent"
                                :label="period.periodYear"
                                :click-handler="onCollocateTimeSeriesClick(period.periodYear)">
                            </word-cloud>
                        </div>
                    </article>
                    <div style="margin-top: 5em; width: 100%; text-align: center" v-else>
                        <p class="mb-1" aria-hidden="true">{{ $t('collocation.gatheringTimeSeriesPeriod') }}...</p>
                        <progress-spinner :message="$t('collocation.gatheringTimeSeriesPeriod')" />
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import { storeToRefs } from "pinia";
import { inject, ref } from "vue";
import { useRouter } from "vue-router";
import { useMainStore } from "../../stores/main";
import {
    collocateCleanup,
    debug,
    extractSurfaceFromCollocate,
    paramsFilter,
    paramsToRoute,
} from "../../utils.js";
import BibliographyCriteria from "../BibliographyCriteria";
import ProgressSpinner from "../ProgressSpinner";
import WordCloud from "../WordCloud.vue";

defineProps({
    biblio: { type: Object, required: true },
    resultsLength: { type: Number, default: 0 },
});

const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const router = useRouter();
const store = useMainStore();
const { formData, searching } = storeToRefs(store);

const timeSeriesInterval = ref(10);
const collocationTimePeriods = ref([]);

function runEvolution(intervalOverride = null) {
    if (intervalOverride !== null && intervalOverride !== undefined) {
        timeSeriesInterval.value = intervalOverride;
    }
    collocationTimePeriods.value = [];
    searching.value = true;
    const interval = parseInt(timeSeriesInterval.value);
    const params = {
        ...paramsFilter(formData.value),
        time_series_interval: interval,
        map_field: "year",
    };

    // Expand year range by one interval on each side for context
    if (formData.value.year) {
        const yearParts = formData.value.year.split("-");
        if (yearParts.length === 2) {
            const [startYear, endYear] = yearParts;
            if (startYear && endYear) {
                params.year = `${parseInt(startYear) - interval}-${parseInt(endYear) + interval}`;
            } else if (startYear) {
                params.year = `${parseInt(startYear) - interval}-`;
            } else if (endYear) {
                params.year = `-${parseInt(endYear) + interval}`;
            }
        } else if (yearParts[0]) {
            const year = parseInt(yearParts[0]);
            params.year = `${year - interval}-${year + interval}`;
        }
    }

    $http.get(`${$dbUrl}/reports/collocation.py`, { params }).then((response) => {
        searching.value = false;
        runCollocationTimeSeries(response.data.file_path, 0);
    }).catch((error) => {
        searching.value = false;
        debug({ $options: { name: "collocation-evolution" } }, error);
    });
}

function runCollocationTimeSeries(filePath, periodNumber) {
    collocationTimePeriods.value[periodNumber] = { year: periodNumber, done: false };
    $http.get(`${$dbUrl}/scripts/collocation_time_series.py`, {
        params: {
            file_path: filePath,
            year_interval: timeSeriesInterval.value,
            period_number: periodNumber,
        },
    }).then((response) => {
        if (response.data.period) {
            const period = response.data.period;
            const year = period.year;
            const interval = parseInt(timeSeriesInterval.value);
            collocationTimePeriods.value[periodNumber] = {
                year,
                frequent: extractSurfaceFromCollocate(period.collocates.frequent || []),
                distinctive: extractSurfaceFromCollocate(period.collocates.distinctive || []),
                periodYear: `${year}-${year + interval}`,
                showDistinctive: true,
                done: true,
            };
        }
        if (!response.data.done) {
            runCollocationTimeSeries(filePath, periodNumber + 1);
        }
    }).catch((error) => {
        debug({ $options: { name: "collocation-evolution" } }, error);
    });
}

function onCollocateTimeSeriesClick(period) {
    return (item) => {
        const method = formData.value.colloc_within === "n" ? "proxy_unordered" : "sentence_unordered";
        router.push(
            paramsToRoute({
                ...formData.value,
                report: "concordance",
                q: collocateCleanup(item, formData.value.q),
                method,
                year: period,
            })
        );
    };
}

function getInterval() {
    return timeSeriesInterval.value;
}

function reset() {
    collocationTimePeriods.value = [];
}

defineExpose({ runEvolution, getInterval, reset });
</script>

<style lang="scss" scoped>
.period-title {
    font-size: 1rem;
    font-variant: small-caps;
    width: 100%;
}
</style>
