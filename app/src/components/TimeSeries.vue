<template>
    <div class="container-fluid">
        <div id="time-series-container" class="mt-4">
            <results-summary :description="results.description" :running-total="runningTotal"></results-summary>
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
                        <div class="mt-2" aria-live="polite">
                            {{ $t("timeSeries.loadingProgress", { current: runningTotal, total: resultsLength || 0 }) }}
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

<script>
import {
    BarElement,
    CategoryScale,
    Chart as ChartJS,
    Legend,
    LinearScale,
    Title,
    Tooltip
} from 'chart.js';
import { Bar } from 'vue-chartjs';

import { mapStores, mapWritableState } from "pinia";
import cssVariables from "../assets/styles/theme.module.scss";
import { useMainStore } from "../stores/main";
import ResultsSummary from "./ResultsSummary";

ChartJS.register(Title, Tooltip, Legend, BarElement, CategoryScale, LinearScale);

export default {
    name: "timeSeries",
    components: {
        ResultsSummary,
        Bar,
    },
    inject: ["$http"],
    provide() {
        return {
            results: this.results.results,
        };
    },
    computed: {
        ...mapWritableState(useMainStore, [
            "formData",
            "currentReport",
            "searching",
            "resultsLength",
            "urlUpdate",
            "accessAuthorized"
        ]),
        ...mapStores(useMainStore),

        // Accessibility computed properties
        chartAriaLabel() {
            const dataPoints = this.dateLabels.length;
            const range = this.dateLabels.length > 0 ?
                `${this.dateLabels[0]} to ${this.dateLabels[this.dateLabels.length - 1]}` : '';
            return this.$t("timeSeries.chartAriaLabel", {
                type: this.currentFrequencyLabel,
                range: range,
                count: dataPoints
            });
        },

        currentFrequencyLabel() {
            return this.frequencyType === 'absolute_time' ?
                this.$t("common.absoluteFrequency") :
                this.$t("common.relativeFrequency");
        },

        // Chart data - computed property for reactivity
        chartData() {
            // Use theme color with fallback
            const backgroundColor = cssVariables.color || '#8e3232';

            return {
                labels: this.dateLabels,
                datasets: [{
                    label: this.currentFrequencyLabel,
                    backgroundColor: backgroundColor,
                    hoverBackgroundColor: this.hexToRGBA(backgroundColor),
                    borderWidth: 1,
                    data: this.frequencyType === 'absolute_time' ?
                        this.absoluteCounts :
                        this.relativeCounts,
                }]
            };
        },

        // Chart options - computed property
        chartOptions() {
            return {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false,
                    },
                    tooltip: {
                        enabled: false, // Disable default canvas tooltip
                        external: (context) => this.externalTooltipHandler(context),
                    }
                },
                scales: {
                    x: {
                        grid: {
                            display: false
                        },
                        title: {
                            display: true,
                            text: this.$t("timeSeries.year")
                        }
                    },
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: this.currentFrequencyLabel
                        }
                    }
                },
                onClick: (event, elements) => {
                    if (elements.length > 0) {
                        this.navigateToYear(elements[0].index);
                    }
                }
            };
        }
    },

    data() {
        return {
            frequencyType: "absolute_time",
            totalResults: 100,
            globalQuery: "",
            localQuery: "",
            absoluteCounts: [],
            dateCounter: [],
            relativeCounts: [],
            dateLabels: [],
            moreResults: false,
            done: false,
            startDate: "",
            endDate: "",
            results: [],
            runningTotal: 0,
            dateCounts: {},
            selectedBarIndex: 0, // For keyboard navigation
            customTooltip: {
                visible: false,
                title: '',
                value: '',
                style: {}
            },
            tooltipHoverTimeout: null,
            isMouseOverTooltip: false,
        };
    },
    mounted() {
        this.formData.report = "time_series";
        this.currentReport = "time_series";
        this.fetchResults();

        // Add global escape key listener for dismissing tooltip
        document.addEventListener('keydown', this.handleGlobalEscape);
    },
    beforeUnmount() {
        // Clean up global escape key listener
        document.removeEventListener('keydown', this.handleGlobalEscape);
    },
    watch: {
        urlUpdate() {
            if (this.formData.report == "time_series") {
                this.fetchResults();
            }
        },
    },
    methods: {
        // Accessibility helper methods
        formatDateRange(label, index) {
            if (this.formData.year_interval == 1) {
                return label;
            } else {
                return `${label}-${parseInt(label) + parseInt(this.formData.year_interval) - 1}`;
            }
        },

        getCurrentData(index) {
            return this.frequencyType === 'absolute_time' ?
                this.absoluteCounts[index] || 0 :
                this.relativeCounts[index] || 0;
        },

        getCurrentUnit() {
            return this.frequencyType === 'absolute_time' ?
                this.$t("timeSeries.occurrences") :
                this.$t("timeSeries.per1000Words");
        },

        // Keyboard navigation for chart
        handleChartKeydown(event) {
            if (!this.dateLabels.length) return;

            let newIndex = this.selectedBarIndex;
            let shouldNavigate = false;

            switch (event.key) {
                case 'ArrowLeft':
                    if (newIndex > 0) {
                        newIndex--;
                        shouldNavigate = true;
                    }
                    event.preventDefault();
                    break;
                case 'ArrowRight':
                    if (newIndex < this.dateLabels.length - 1) {
                        newIndex++;
                        shouldNavigate = true;
                    }
                    event.preventDefault();
                    break;
                case 'Enter':
                case ' ':
                    this.navigateToYear(this.selectedBarIndex);
                    event.preventDefault();
                    break;
                case 'Home':
                    newIndex = 0;
                    shouldNavigate = true;
                    event.preventDefault();
                    break;
                case 'End':
                    newIndex = this.dateLabels.length - 1;
                    shouldNavigate = true;
                    event.preventDefault();
                    break;
                case 'Escape':
                    this.dismissTooltip();
                    event.preventDefault();
                    break;
            }

            if (shouldNavigate) {
                this.selectedBarIndex = newIndex;
                this.highlightBar(newIndex);
            }
        },

        dismissTooltip() {
            // Clear any pending hover timeout
            if (this.tooltipHoverTimeout) {
                clearTimeout(this.tooltipHoverTimeout);
                this.tooltipHoverTimeout = null;
            }
            this.customTooltip.visible = false;
            this.isMouseOverTooltip = false;
        },

        handleGlobalEscape(event) {
            if (event.key === 'Escape' && this.customTooltip.visible) {
                this.dismissTooltip();
                event.preventDefault();
            }
        },

        externalTooltipHandler(context) {
            const { chart, tooltip } = context;

            // If Chart.js wants to hide tooltip (mouse left bar area)
            if (tooltip.opacity === 0) {
                // Keep tooltip visible if mouse is over it, otherwise start delayed hide
                if (!this.isMouseOverTooltip) {
                    // Clear any existing timeout
                    if (this.tooltipHoverTimeout) {
                        clearTimeout(this.tooltipHoverTimeout);
                    }
                    // Give user time to move mouse to tooltip
                    this.tooltipHoverTimeout = setTimeout(() => {
                        // Only hide if mouse is still not over tooltip
                        if (!this.isMouseOverTooltip) {
                            this.customTooltip.visible = false;
                        }
                        this.tooltipHoverTimeout = null;
                    }, 250);
                }
                return;
            }

            // Cancel any pending hide timeout when showing new tooltip
            if (this.tooltipHoverTimeout) {
                clearTimeout(this.tooltipHoverTimeout);
                this.tooltipHoverTimeout = null;
            }

            // Set tooltip data
            if (tooltip.body) {
                const dataIndex = tooltip.dataPoints[0].dataIndex;
                const title = this.formatDateRange(this.dateLabels[dataIndex], dataIndex);
                const value = this.getCurrentData(dataIndex);
                const unit = this.getCurrentUnit();

                this.customTooltip.title = title;
                this.customTooltip.value = `${value} ${unit}`;
                this.customTooltip.visible = true;

                // Position the tooltip
                const position = chart.canvas.getBoundingClientRect();
                this.customTooltip.style = {
                    left: position.left + window.pageXOffset + tooltip.caretX + 'px',
                    top: position.top + window.pageYOffset + tooltip.caretY + 'px',
                };
            }
        },

        handleTooltipMouseEnter() {
            // Mark that mouse is over tooltip
            this.isMouseOverTooltip = true;

            // Cancel any pending hide timeout when mouse enters tooltip
            if (this.tooltipHoverTimeout) {
                clearTimeout(this.tooltipHoverTimeout);
                this.tooltipHoverTimeout = null;
            }
        },

        handleTooltipMouseLeave() {
            // Mark that mouse left tooltip
            this.isMouseOverTooltip = false;

            // Hide tooltip after delay only if mouse is not back on bar
            this.tooltipHoverTimeout = setTimeout(() => {
                if (!this.isMouseOverTooltip) {
                    this.customTooltip.visible = false;
                }
                this.tooltipHoverTimeout = null;
            }, 200);
        },

        highlightBar(index) {
            // Try different ways to access the chart instance from vue-chartjs
            let chartInstance = this.$refs.chartComponent?.chart;

            if (!chartInstance) {
                chartInstance = this.$refs.chartComponent?.$data?.chart;
            }

            if (!chartInstance) {
                chartInstance = this.$refs.chartComponent?.chartInstance;
            }

            if (!chartInstance) {
                return;
            }

            // Set active elements to show tooltip and hover effect
            chartInstance.setActiveElements([{
                datasetIndex: 0,
                index: index
            }]);

            chartInstance.tooltip.setActiveElements([{
                datasetIndex: 0,
                index: index
            }]);

            chartInstance.update('none'); // Update without animation
        },



        navigateToYear(index) {
            const startDate = parseInt(this.dateLabels[index]);
            const endDate = startDate + parseInt(this.formData.year_interval) - 1;
            const year = startDate === endDate ? startDate.toString() : `${startDate}-${endDate}`;

            this.mainStore.updateFormDataField({
                key: "year",
                value: year,
            });
            this.$router.push(
                this.paramsToRoute({
                    ...this.formData,
                    report: "concordance",
                    start_date: "",
                    end_date: "",
                    year_interval: "",
                })
            );
        },
        fetchResults() {
            this.runningTotal = 0;
            if (this.formData.year_interval == "") {
                this.formData.year_interval = this.$philoConfig.time_series.interval;
            }
            this.formData.year_interval = parseInt(this.formData.year_interval);
            this.frequencyType = "absolute_time";
            this.searching = true;
            this.startDate = parseInt(this.formData.start_date || this.$philoConfig.time_series_start_end_date.start_date);
            this.endDate = parseInt(this.formData.end_date || this.$philoConfig.time_series_start_end_date.end_date);
            this.mainStore.updateStartEndDate({
                startDate: this.startDate,
                endDate: this.endDate,
            });

            this.globalQuery = this.copyObject(this.formData);
            this.localQuery = this.copyObject(this.globalQuery);

            var dateList = [];
            var zeros = [];
            for (let i = this.startDate; i <= this.endDate; i += this.formData.year_interval) {
                dateList.push(i);
                zeros.push(0);
            }

            this.dateLabels = dateList;
            this.absoluteCounts = this.copyObject(zeros);
            this.relativeCounts = this.copyObject(zeros);
            this.dateCounts = {};
            this.selectedBarIndex = 0; // Reset selection

            var fullResults;
            this.updateTimeSeries(fullResults);
        },

        updateTimeSeries(fullResults) {
            this.$http
                .get(`${this.$dbUrl}/reports/time_series.py`, {
                    params: {
                        ...this.paramsFilter({ ...this.formData }),
                        start_date: this.startDate,
                        max_time: 5,
                        year_interval: this.formData.year_interval,
                    },
                })
                .then((results) => {
                    this.searching = false;
                    var timeSeriesResults = results.data;
                    this.results = results.data;
                    this.runningTotal += timeSeriesResults.results_length;
                    this.moreResults = timeSeriesResults.more_results;
                    this.startDate = timeSeriesResults.new_start_date;
                    for (let date in timeSeriesResults.results.date_count) {
                        this.dateCounts[date] = timeSeriesResults.results.date_count[date];
                    }
                    this.sortAndRenderTimeSeries(fullResults, timeSeriesResults);
                })
                .catch((response) => {
                    this.debug(this, response);
                    this.searching = false;
                });
        },

        sortAndRenderTimeSeries(fullResults, timeSeriesResults) {
            var allResults = this.mergeResults(fullResults, timeSeriesResults.results["absolute_count"], "label");
            fullResults = allResults.unsorted;

            for (let i = 0; i < allResults.sorted.length; i += 1) {
                var date = allResults.sorted[i].label;
                var value = allResults.sorted[i].count;
                this.absoluteCounts[i] = value;
                this.relativeCounts[i] = Math.round((value / this.dateCounts[date]) * 10000 * 100) / 100;
                if (isNaN(this.relativeCounts[i])) {
                    this.relativeCounts[i] = 0;
                }
            }

            if (this.formData.report === "time_series" && this.deepEqual(this.globalQuery, this.localQuery)) {
                if (this.moreResults) {
                    this.searching = true;
                    this.updateTimeSeries(fullResults);
                } else {
                    this.runningTotal = this.resultsLength;
                    this.done = true;
                    this.searching = false;
                }
            }
        },

        toggleFrequency(frequencyType) {
            if (this.searching) return; // Prevent toggle during loading

            this.frequencyType = frequencyType;
            this.$nextTick(() => {
                const announcement = `Chart updated to show ${this.currentFrequencyLabel}`;
                this.announceToScreenReader(announcement);
            });
        },

        announceToScreenReader(message) {
            const announcer = document.createElement('div');
            announcer.setAttribute('aria-live', 'polite');
            announcer.className = 'visually-hidden';
            announcer.textContent = message;
            document.body.appendChild(announcer);

            setTimeout(() => {
                document.body.removeChild(announcer);
            }, 1000);
        },

        hexToRGBA(h) {
            // Use theme color if h is undefined
            if (!h) {
                h = cssVariables.color || '#8e3232'; // Use theme color with fallback
            }

            let r = 0, g = 0, b = 0;
            if (h.length == 4) {
                r = "0x" + h[1] + h[1];
                g = "0x" + h[2] + h[2];
                b = "0x" + h[3] + h[3];
            } else if (h.length == 7) {
                r = "0x" + h[1] + h[2];
                g = "0x" + h[3] + h[4];
                b = "0x" + h[5] + h[6];
            }
            return "rgba(" + +r + "," + +g + "," + +b + ", .7)";
        },
    },
};
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
