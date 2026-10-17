<template>
    <div class="container-fluid">
        <div id="time-series-container" class="mt-4">
            <results-summary :description="results.description"></results-summary>
            <div class="card mt-4 mx-2" id="time-series">
                <div class="btn-group d-inline-block" role="group" :aria-label="$t('timeSeries.frequencyTypeSelection')">
                    <button type="button" class="btn btn-secondary"
                        :class="{ active: frequencyType == 'absolute_time' }" @click="toggleFrequency('absolute_time')"
                        :aria-pressed="frequencyType == 'absolute_time'" :disabled="searching">
                        {{ $t("common.absoluteFrequency") }}
                    </button>
                    <button type="button" class="btn btn-secondary"
                        :class="{ active: frequencyType == 'relative_time' }" @click="toggleFrequency('relative_time')"
                        :aria-pressed="frequencyType == 'relative_time'" :disabled="searching">
                        {{ $t("common.relativeFrequency") }}
                    </button>
                </div>
                <div class="visually-hidden" role="status">{{ frequencyStatus }}</div>

                <!-- Chart container -->
                <div class="chart-wrapper p-3 mt-4">
                    <div v-if="searching" class="text-center p-5">
                        <div class="spinner-border" role="status" aria-live="polite" aria-atomic="true">
                            <span class="visually-hidden">{{ $t("common.loading") }}</span>
                        </div>
                    </div>
                    <div v-else-if="dateLabels.length > 0" class="chart-container">
                        <h3 class="visually-hidden">
                            {{ $t("timeSeries.chartTitle") }} - {{ currentFrequencyLabel }}
                        </h3>

                        <time-series-chart id="time-series-chart" :values="counts" :labels="dateLabels"
                            :ranges="dateRanges" :value-text="valueText" :x-title="$t('timeSeries.year')"
                            :y-title="currentFrequencyLabel" :label="chartAriaLabel" describedby="chart-instructions"
                            :hint="$t('timeSeries.keyboardHint')" @select="navigateToYear" />

                        <div class="visually-hidden" id="chart-instructions">
                            {{ $t("timeSeries.chartInstructions") }}
                        </div>

                        <!-- Hidden data table for screen readers -->
                        <div class="visually-hidden">
                            <table role="table" :aria-label="$t('timeSeries.dataTable')">
                                <caption>{{ $t("timeSeries.dataTableCaption") }}</caption>
                                <thead>
                                    <tr>
                                        <th scope="col">{{ $t("timeSeries.year") }}</th>
                                        <th scope="col">{{ currentFrequencyLabel }}</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr v-for="(label, index) in dateLabels" :key="label">
                                        <td>{{ dateRanges[index] }}</td>
                                        <td>{{ valueText(counts[index]) }}</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                    <div v-else-if="!searching" class="text-center p-5 text-muted" role="status" aria-live="polite">
                        {{ $t("common.noDataAvailable") }}
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import { computed, inject, onMounted, provide, ref } from "vue";
import { useRouter } from "vue-router";
import { storeToRefs } from "pinia";
import { useI18n } from "vue-i18n";
import { useReport } from "../composables/useReport";
import { useMainStore } from "../stores/main";
import { copyObject, mergeResults, paramsFilter, paramsToRoute } from "../utils.js";
import ResultsSummary from "./ResultsSummary";
import TimeSeriesChart from "./TimeSeriesChart.vue";

const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const philoConfig = inject("$philoConfig");
const router = useRouter();
const { t, locale } = useI18n();
const store = useMainStore();
const { formData, searching } = storeToRefs(store);

const frequencyType = ref("absolute_time");
const absoluteCounts = ref([]);
const relativeCounts = ref([]);
const dateLabels = ref([]);
const startDate = ref("");
const endDate = ref("");
const results = ref([]);
const dateCounts = ref({});
const frequencyStatus = ref(""); // said when the frequency shown changes

const currentFrequencyLabel = computed(() =>
    frequencyType.value === "absolute_time"
        ? t("common.absoluteFrequency")
        : t("common.relativeFrequency")
);

const chartAriaLabel = computed(() => {
    const dataPoints = dateLabels.value.length;
    const range = dataPoints > 0
        ? t("timeSeries.periodRange", { start: dateLabels.value[0], end: dateLabels.value[dataPoints - 1] })
        : "";
    return t("timeSeries.chartAriaLabel", {
        type: currentFrequencyLabel.value,
        range,
        count: dataPoints,
    });
});

const counts = computed(() => (frequencyType.value === "absolute_time" ? absoluteCounts.value : relativeCounts.value));
// A count as said: "1 occurrence", "1,234 occurrences", "1.3 per 10,000 words"
function valueText(value) {
    const n = value.toLocaleString(locale.value, { maximumFractionDigits: 2 });
    return frequencyType.value === "absolute_time"
        ? t("timeSeries.occurrences", { n }, value)
        : t("timeSeries.per10000Words", { n });
}

// The periods' years, the last one's stopping at the end date
const dateRanges = computed(() =>
    dateLabels.value.map((start) => {
        const end = Math.min(start + parseInt(formData.value.year_interval) - 1, endDate.value);
        return end === start ? `${start}` : `${start}-${end}`;
    })
);

function navigateToYear(index) {
    store.updateFormDataField({ key: "year", value: dateRanges.value[index] });
    router.push(
        paramsToRoute({
            ...formData.value,
            report: "concordance",
            start_date: "",
            end_date: "",
            year_interval: "",
        })
    );
}

const { run: fetchResults } = useReport("time_series", async (signal) => {
    // An interval below 1 made the loop below endless, freezing the page: as one that isn't a number, the default
    const interval = parseInt(formData.value.year_interval);
    formData.value.year_interval = interval >= 1 ? interval : parseInt(philoConfig.time_series_interval) || 1;
    frequencyType.value = "absolute_time";
    frequencyStatus.value = "";
    const defaults = philoConfig.time_series_start_end_date;
    const start = parseInt(formData.value.start_date);
    const end = parseInt(formData.value.end_date);
    startDate.value = isNaN(start) ? parseInt(defaults.start_date) : start;
    endDate.value = isNaN(end) ? parseInt(defaults.end_date) : end;
    store.updateStartEndDate({ startDate: startDate.value, endDate: endDate.value });

    const dateList = [];
    const zeros = [];
    for (let i = startDate.value; i <= endDate.value; i += formData.value.year_interval) {
        dateList.push(i);
        zeros.push(0);
    }
    dateLabels.value = dateList;
    absoluteCounts.value = copyObject(zeros);
    relativeCounts.value = copyObject(zeros);
    dateCounts.value = {};

    const { data } = await $http.get(`${$dbUrl}/reports/time_series.py`, {
        params: {
            ...paramsFilter({ ...formData.value }),
            start_date: startDate.value,
            end_date: endDate.value,
            year_interval: formData.value.year_interval,
        },
        signal,
    });
    results.value = data;
    for (const date in data.results.date_count) {
        dateCounts.value[date] = data.results.date_count[date];
    }
    renderTimeSeries(data.results.absolute_count);
});

function renderTimeSeries(absoluteCount) {
    const allResults = mergeResults(undefined, absoluteCount, "label");
    for (let i = 0; i < allResults.sorted.length; i += 1) {
        const date = allResults.sorted[i].label;
        const value = allResults.sorted[i].count;
        absoluteCounts.value[i] = value;
        const relative = Math.round((value / dateCounts.value[date]) * 10000 * 100) / 100;
        relativeCounts.value[i] = isNaN(relative) ? 0 : relative;
    }
}

function toggleFrequency(newType) {
    if (searching.value) return;
    frequencyType.value = newType;
    frequencyStatus.value = t("timeSeries.chartUpdated", { type: currentFrequencyLabel.value });
}

onMounted(fetchResults);

provide("results", computed(() => results.value.results));
</script>

<style scoped lang="scss">
#description {
    position: relative;
}

#export-results {
    position: absolute;
    right: 0;
    padding: 0.125rem 0.25rem;
    font-size: 0.8rem !important;
}

.chart-wrapper {
    min-height: 500px;
    max-height: 800px;
    position: relative;
}

@media (max-width: 768px) {
    .chart-wrapper {
        min-height: 300px;
        max-height: 500px;
    }
}
</style>
