<template>
    <div class="container-fluid">
        <div id="time-series-container" class="mt-4">
            <results-summary :description="results.description"></results-summary>
            <div class="card mt-4 mx-2" id="time-series">
                <div class="btn-group d-inline-block" role="group" aria-label="Frequency type selection">
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

                        <div class="chart-with-tooltip">
                            <Bar ref="chartComponent" id="time-series-chart" :data="chartData" :options="chartOptions"
                                role="img" :aria-label="chartAriaLabel" aria-describedby="chart-instructions"
                                tabindex="0" @keydown="handleChartKeydown" @focus="highlightBar(selectedBarIndex)" />

                            <!-- Custom HTML tooltip -->
                            <div v-if="customTooltip.visible" class="custom-tooltip" :style="customTooltip.style"
                                @mouseenter="handleTooltipMouseEnter" @mouseleave="handleTooltipMouseLeave">
                                <div class="tooltip-title">{{ customTooltip.title }}</div>
                                <div class="tooltip-value">{{ customTooltip.value }}</div>
                            </div>
                        </div>

                        <div class="visually-hidden" id="chart-instructions">
                            Use arrow keys to navigate between time periods, Enter or Space to view detailed results for
                            selected period, Escape to dismiss tooltip. A complete data table is available below for
                            screen readers.
                        </div>

                        <!-- Hidden data table for screen readers -->
                        <div class="visually-hidden">
                            <table role="table" aria-label="Time series data table">
                                <caption>Time series data showing frequency over time periods</caption>
                                <thead>
                                    <tr>
                                        <th scope="col">Year</th>
                                        <th scope="col">{{ currentFrequencyLabel }}</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr v-for="(label, index) in dateLabels" :key="label">
                                        <td>{{ formatDateRange(label, index) }}</td>
                                        <td>{{ getCurrentData(index) }} {{ getCurrentUnit() }}</td>
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
import {
    BarElement,
    CategoryScale,
    Chart as ChartJS,
    Legend,
    LinearScale,
    Title,
    Tooltip,
} from "chart.js";
import { Bar } from "vue-chartjs";  // eslint-disable-line no-unused-vars

import { computed, inject, nextTick, onBeforeUnmount, onMounted, provide, reactive, ref, useTemplateRef, watch } from "vue";
import { useRouter } from "vue-router";
import { storeToRefs } from "pinia";
import { useI18n } from "vue-i18n";
import cssVariables from "../assets/styles/theme.module.scss";
import { useMainStore } from "../stores/main";
import { copyObject, debug, mergeResults, paramsFilter, paramsToRoute } from "../utils.js";
import ResultsSummary from "./ResultsSummary";  // eslint-disable-line no-unused-vars

ChartJS.register(Title, Tooltip, Legend, BarElement, CategoryScale, LinearScale);

const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const philoConfig = inject("$philoConfig");
const router = useRouter();
const { t } = useI18n();
const store = useMainStore();
const { formData, currentReport, searching, urlUpdate } = storeToRefs(store);

const chartComponent = useTemplateRef("chartComponent");

const frequencyType = ref("absolute_time");
const absoluteCounts = ref([]);
const relativeCounts = ref([]);
const dateLabels = ref([]);
const startDate = ref("");
const endDate = ref("");
const results = ref([]);
const dateCounts = ref({});
const selectedBarIndex = ref(0);
const customTooltip = reactive({
    visible: false,
    title: "",
    value: "",
    style: {},
});
let tooltipHoverTimeout = null;
let isMouseOverTooltip = false;

const currentFrequencyLabel = computed(() =>
    frequencyType.value === "absolute_time"
        ? t("common.absoluteFrequency")
        : t("common.relativeFrequency")
);

const chartAriaLabel = computed(() => {
    const dataPoints = dateLabels.value.length;
    const range = dataPoints > 0
        ? `${dateLabels.value[0]} to ${dateLabels.value[dataPoints - 1]}`
        : "";
    return t("timeSeries.chartAriaLabel", {
        type: currentFrequencyLabel.value,
        range,
        count: dataPoints,
    });
});

const chartData = computed(() => {
    const backgroundColor = cssVariables.color || "#8e3232";
    return {
        labels: dateLabels.value,
        datasets: [{
            label: currentFrequencyLabel.value,
            backgroundColor,
            hoverBackgroundColor: hexToRGBA(backgroundColor),
            borderWidth: 1,
            data: frequencyType.value === "absolute_time"
                ? absoluteCounts.value
                : relativeCounts.value,
        }],
    };
});

const chartOptions = computed(() => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: { display: false },
        tooltip: {
            enabled: false,
            external: (context) => externalTooltipHandler(context),
        },
    },
    scales: {
        x: {
            grid: { display: false },
            title: { display: true, text: t("timeSeries.year") },
        },
        y: {
            beginAtZero: true,
            title: { display: true, text: currentFrequencyLabel.value },
        },
    },
    onClick: (event, elements) => {
        if (elements.length > 0) navigateToYear(elements[0].index);
    },
}));

function formatDateRange(label) {
    if (formData.value.year_interval == 1) return label;
    return `${label}-${parseInt(label) + parseInt(formData.value.year_interval) - 1}`;
}

function getCurrentData(index) {
    return frequencyType.value === "absolute_time"
        ? absoluteCounts.value[index] || 0
        : relativeCounts.value[index] || 0;
}

function getCurrentUnit() {
    return frequencyType.value === "absolute_time"
        ? t("timeSeries.occurrences")
        : t("timeSeries.per1000Words");
}

function dismissTooltip() {
    if (tooltipHoverTimeout) {
        clearTimeout(tooltipHoverTimeout);
        tooltipHoverTimeout = null;
    }
    customTooltip.visible = false;
    isMouseOverTooltip = false;
}

function handleGlobalEscape(event) {
    if (event.key === "Escape" && customTooltip.visible) {
        dismissTooltip();
        event.preventDefault();
    }
}

function externalTooltipHandler(context) {
    const { chart, tooltip } = context;

    // Chart.js wants to hide tooltip (mouse left bar)
    if (tooltip.opacity === 0) {
        if (!isMouseOverTooltip) {
            if (tooltipHoverTimeout) clearTimeout(tooltipHoverTimeout);
            // Give user time to move mouse to tooltip
            tooltipHoverTimeout = setTimeout(() => {
                if (!isMouseOverTooltip) customTooltip.visible = false;
                tooltipHoverTimeout = null;
            }, 250);
        }
        return;
    }

    if (tooltipHoverTimeout) {
        clearTimeout(tooltipHoverTimeout);
        tooltipHoverTimeout = null;
    }

    if (tooltip.body) {
        const dataIndex = tooltip.dataPoints[0].dataIndex;
        customTooltip.title = formatDateRange(dateLabels.value[dataIndex]);
        customTooltip.value = `${getCurrentData(dataIndex)} ${getCurrentUnit()}`;
        customTooltip.visible = true;

        const position = chart.canvas.getBoundingClientRect();
        customTooltip.style = {
            left: position.left + window.pageXOffset + tooltip.caretX + "px",
            top: position.top + window.pageYOffset + tooltip.caretY + "px",
        };
    }
}

function handleTooltipMouseEnter() {
    isMouseOverTooltip = true;
    if (tooltipHoverTimeout) {
        clearTimeout(tooltipHoverTimeout);
        tooltipHoverTimeout = null;
    }
}

function handleTooltipMouseLeave() {
    isMouseOverTooltip = false;
    tooltipHoverTimeout = setTimeout(() => {
        if (!isMouseOverTooltip) customTooltip.visible = false;
        tooltipHoverTimeout = null;
    }, 200);
}

function highlightBar(index) {
    // vue-chartjs exposes the chart instance under different paths across versions
    const chartInstance =
        chartComponent.value?.chart ||
        chartComponent.value?.$data?.chart ||
        chartComponent.value?.chartInstance;
    if (!chartInstance) return;

    chartInstance.setActiveElements([{ datasetIndex: 0, index }]);
    chartInstance.tooltip.setActiveElements([{ datasetIndex: 0, index }]);
    chartInstance.update("none");
}

function handleChartKeydown(event) {
    if (!dateLabels.value.length) return;

    let newIndex = selectedBarIndex.value;
    let shouldNavigate = false;

    switch (event.key) {
        case "ArrowLeft":
            if (newIndex > 0) { newIndex--; shouldNavigate = true; }
            event.preventDefault();
            break;
        case "ArrowRight":
            if (newIndex < dateLabels.value.length - 1) { newIndex++; shouldNavigate = true; }
            event.preventDefault();
            break;
        case "Enter":
        case " ":
            navigateToYear(selectedBarIndex.value);
            event.preventDefault();
            break;
        case "Home":
            newIndex = 0; shouldNavigate = true;
            event.preventDefault();
            break;
        case "End":
            newIndex = dateLabels.value.length - 1; shouldNavigate = true;
            event.preventDefault();
            break;
        case "Escape":
            dismissTooltip();
            event.preventDefault();
            break;
    }

    if (shouldNavigate) {
        selectedBarIndex.value = newIndex;
        highlightBar(newIndex);
    }
}

function navigateToYear(index) {
    const start = parseInt(dateLabels.value[index]);
    const end = start + parseInt(formData.value.year_interval) - 1;
    const year = start === end ? start.toString() : `${start}-${end}`;

    store.updateFormDataField({ key: "year", value: year });
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

function fetchResults() {
    if (formData.value.year_interval === "") {
        formData.value.year_interval = philoConfig.time_series.interval;
    }
    formData.value.year_interval = parseInt(formData.value.year_interval);
    frequencyType.value = "absolute_time";
    searching.value = true;
    startDate.value = parseInt(formData.value.start_date || philoConfig.time_series_start_end_date.start_date);
    endDate.value = parseInt(formData.value.end_date || philoConfig.time_series_start_end_date.end_date);
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
    selectedBarIndex.value = 0;

    $http
        .get(`${$dbUrl}/reports/time_series.py`, {
            params: {
                ...paramsFilter({ ...formData.value }),
                start_date: startDate.value,
                year_interval: formData.value.year_interval,
            },
        })
        .then((response) => {
            const data = response.data;
            results.value = data;
            for (const date in data.results.date_count) {
                dateCounts.value[date] = data.results.date_count[date];
            }
            renderTimeSeries(data.results.absolute_count);
            searching.value = false;
        })
        .catch((error) => {
            debug({ $options: { name: "timeSeries" } }, error);
            searching.value = false;
        });
}

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

function announceToScreenReader(message) {
    const announcer = document.createElement("div");
    announcer.setAttribute("aria-live", "polite");
    announcer.className = "visually-hidden";
    announcer.textContent = message;
    document.body.appendChild(announcer);
    setTimeout(() => document.body.removeChild(announcer), 1000);
}

function toggleFrequency(newType) {
    if (searching.value) return;
    frequencyType.value = newType;
    nextTick(() => {
        announceToScreenReader(`Chart updated to show ${currentFrequencyLabel.value}`);
    });
}

function hexToRGBA(h) {
    if (!h) h = cssVariables.color || "#8e3232";

    let r = 0, g = 0, b = 0;
    if (h.length === 4) {
        r = "0x" + h[1] + h[1];
        g = "0x" + h[2] + h[2];
        b = "0x" + h[3] + h[3];
    } else if (h.length === 7) {
        r = "0x" + h[1] + h[2];
        g = "0x" + h[3] + h[4];
        b = "0x" + h[5] + h[6];
    }
    return `rgba(${+r}, ${+g}, ${+b}, .7)`;
}

watch(urlUpdate, () => {
    if (formData.value.report === "time_series") fetchResults();
});

onMounted(() => {
    formData.value.report = "time_series";
    currentReport.value = "time_series";
    fetchResults();
    document.addEventListener("keydown", handleGlobalEscape);
});

onBeforeUnmount(() => {
    document.removeEventListener("keydown", handleGlobalEscape);
});

provide("results", computed(() => results.value.results));
</script>

<style scoped lang="scss">
@use "../assets/styles/theme.module.scss" as theme;

#description {
    position: relative;
}

#export-results {
    position: absolute;
    right: 0;
    padding: 0.125rem 0.25rem;
    font-size: 0.8rem !important;
}

/* Fix chart height issues */
.chart-wrapper {
    min-height: 500px;
    max-height: 800px;
    position: relative;
}

.chart-container {
    height: 100%;
    min-height: 450px;
    position: relative;
}

#time-series-chart {
    cursor: pointer;
    height: 450px !important;
    min-height: 450px;
}

/* Ensure chart is focusable and has visible focus indicator */
#time-series-chart:focus {
    outline: 2px solid #007bff;
    outline-offset: 2px;
}

/* Custom HTML tooltip */
.chart-with-tooltip {
    position: relative;
}

.custom-tooltip {
    position: fixed;
    background: theme.$button-color;
    color: white;
    border: 1px solid theme.$button-color;
    border-radius: 4px;
    padding: 0.2rem 0.5rem;
    pointer-events: auto;
    z-index: 1000;
    transform: translate(-50%, -100%);
    margin-top: 0px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3), 0 2px 4px rgba(0, 0, 0, 0.2);
    font-size: 14px;
    line-height: 1.5;
    white-space: nowrap;
    transition: opacity 0.15s ease;
}

.custom-tooltip::after {
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border: 8px solid transparent;
    border-top-color: theme.$button-color;
}

/* Create an invisible hover bridge between tooltip and chart */
.custom-tooltip::before {
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    width: 40px;
    height: 20px;
    background: transparent;
}

.tooltip-title {
    font-weight: bold;
    margin-bottom: 0px;
}

.tooltip-value {
    font-size: 13px;
}

/* Responsive chart sizing */
@media (max-width: 768px) {
    .chart-wrapper {
        min-height: 300px;
        max-height: 500px;
    }

    .chart-container {
        min-height: 280px;
    }

    #time-series-chart {
        height: 280px !important;
        min-height: 280px;
    }
}
</style>
