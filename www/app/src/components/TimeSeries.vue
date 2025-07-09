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
                        <div class="spinner-border" role="status" aria-live="polite">
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

                        <Bar id="time-series-chart" :data="chartData" :options="chartOptions" role="img"
                            :aria-label="chartAriaLabel" aria-describedby="chart-instructions" tabindex="0"
                            @keydown="handleChartKeydown" />

                        <div class="visually-hidden" id="chart-instructions">
                            Use arrow keys to navigate between time periods, Enter or Space to view detailed results for
                            selected period.
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

import { mapFields } from "vuex-map-fields";
import cssVariables from "../assets/styles/theme.module.scss";
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
        ...mapFields({
            report: "formData.report",
            interval: "formData.year_interval",
            start_date: "formData.start_date",
            end_date: "formData.end_date",
            currentReport: "currentReport",
            searching: "searching",
            resultsLength: "resultsLength",
            urlUpdate: "urlUpdate",
            accessAuthorized: "accessAuthorized",
        }),

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
            return {
                labels: this.dateLabels,
                datasets: [{
                    label: this.currentFrequencyLabel,
                    backgroundColor: cssVariables.color,
                    hoverBackgroundColor: this.hexToRGBA(cssVariables.color),
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
                        callbacks: {
                            title: (tooltipItems) => {
                                const item = tooltipItems[0];
                                return this.formatDateRange(item.label, tooltipItems[0].dataIndex);
                            },
                            label: (tooltipItem) => {
                                const value = tooltipItem.parsed.y;
                                const unit = this.getCurrentUnit();
                                return `${value} ${unit}`;
                            },
                        },
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
        };
    },
    mounted() {
        this.report = "time_series";
        this.currentReport = "time_series";
        this.fetchResults();
    },
    watch: {
        urlUpdate() {
            if (this.report == "time_series") {
                this.fetchResults();
            }
        },
    },
    methods: {
        // Accessibility helper methods
        formatDateRange(label, index) {
            if (this.interval == 1) {
                return label;
            } else {
                return `${label}-${parseInt(label) + parseInt(this.interval) - 1}`;
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
            }

            if (shouldNavigate) {
                this.selectedBarIndex = newIndex;
                this.announceSelection();
            }
        },

        announceSelection() {
            const year = this.formatDateRange(this.dateLabels[this.selectedBarIndex], this.selectedBarIndex);
            const value = this.getCurrentData(this.selectedBarIndex);
            const unit = this.getCurrentUnit();

            // Create announcement for screen readers
            const announcement = this.$t("timeSeries.selectionAnnouncement", {
                year: year,
                value: value,
                unit: unit
            });

            // Announce to screen readers
            this.$nextTick(() => {
                const announcer = document.createElement('div');
                announcer.setAttribute('aria-live', 'assertive');
                announcer.setAttribute('aria-atomic', 'true');
                announcer.className = 'visually-hidden';
                announcer.textContent = announcement;
                document.body.appendChild(announcer);

                setTimeout(() => {
                    document.body.removeChild(announcer);
                }, 1000);
            });
        },

        navigateToYear(index) {
            const startDate = parseInt(this.dateLabels[index]);
            const endDate = startDate + parseInt(this.interval) - 1;
            const year = startDate === endDate ? startDate.toString() : `${startDate}-${endDate}`;

            this.$store.commit("updateFormDataField", {
                key: "year",
                value: year,
            });
            this.$router.push(
                this.paramsToRoute({
                    ...this.$store.state.formData,
                    report: "concordance",
                    start_date: "",
                    end_date: "",
                    year_interval: "",
                })
            );
        },
        fetchResults() {
            this.runningTotal = 0;
            if (this.interval == "") {
                this.interval = this.$philoConfig.time_series.interval;
            }
            this.interval = parseInt(this.interval);
            this.frequencyType = "absolute_time";
            this.searching = true;
            this.startDate = parseInt(this.start_date || this.$philoConfig.time_series_start_end_date.start_date);
            this.endDate = parseInt(this.end_date || this.$philoConfig.time_series_start_end_date.end_date);
            this.$store.dispatch("updateStartEndDate", {
                startDate: this.startDate,
                endDate: this.endDate,
            });

            this.globalQuery = this.copyObject(this.$store.state.formData);
            this.localQuery = this.copyObject(this.globalQuery);

            var dateList = [];
            var zeros = [];
            for (let i = this.startDate; i <= this.endDate; i += this.interval) {
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
                        ...this.paramsFilter({ ...this.$store.state.formData }),
                        start_date: this.startDate,
                        max_time: 5,
                        year_interval: this.interval,
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

            if (this.report === "time_series" && this.deepEqual(this.globalQuery, this.localQuery)) {
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

<style scoped>
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
