<template>
    <div>
        <!-- Detection auto-runs on mount; the min-words knob below triggers
             re-runs. No standalone Detect button is needed — its only role
             was redundant re-runs. -->

        <div v-if="loading && !result" class="text-center py-5">
            <progress-spinner :lg="true" :message="$t('threads.detecting')" />
        </div>

        <div v-if="result && result.threads && result.threads.length > 0" class="mx-2 my-3">
            <!-- Everything (count, explanation, controls, streamgraph) lives
                 inside the result card as one self-contained unit. -->
            <div class="card shadow-sm p-3 mb-3 overview-card">
                <p class="text-muted mb-1">
                    {{ $t('threads.summary', { threads: result.threads.length }) }}
                    <span class="info-tip" tabindex="0"
                        :data-tip="$t('threads.overviewSubtitle')"
                        :aria-label="$t('threads.overviewSubtitle')">ⓘ</span>
                </p>
                <p class="small text-muted mb-3">{{ $t('threads.overviewHint') }}</p>

                <svg :viewBox="`0 0 ${chartWidth} ${overviewHeight}`" class="overview-stream"
                    preserveAspectRatio="none" :aria-label="$t('threads.streamgraph')">
                    <path v-for="(layer, i) in streamLayers" :key="layer.id"
                        :d="layer.path" :fill="threadColor(i, 0.85)" stroke="#fff" stroke-width="0.3"
                        @click="onViewPassages(layer.thread)" class="stream-band"
                        :aria-label="`${$t('threads.threadN', { n: layer.thread.id })}: ${layer.thread.label}`">
                        <title>{{ $t('threads.threadN', { n: layer.thread.id }) }}: {{ layer.thread.label }}</title>
                    </path>
                </svg>
                <div class="d-flex justify-content-between small text-muted mt-1 mb-2 px-1">
                    <span>{{ result.year_range[0] }}</span>
                    <span>{{ result.year_range[1] }}</span>
                </div>
                <div class="d-flex flex-wrap gap-1 px-1">
                    <button v-for="thread in result.threads" :key="`leg-${thread.id}`" type="button"
                        class="btn btn-sm legend-chip"
                        :style="{ borderColor: threadColor(thread.id - 1, 1), color: threadColor(thread.id - 1, 1) }"
                        @click="onViewPassages(thread)"
                        :title="thread.words.slice(0, 10).map((w) => w.word).join(', ')">
                        <span class="legend-swatch" :style="{ backgroundColor: threadColor(thread.id - 1, 1) }"></span>
                        T{{ thread.id }}: {{ thread.label }}
                    </button>
                </div>
                <!-- Clustering-grain control sits with the legend at the bottom of
                     the card, grouped with the other below-the-chart affordances. -->
                <div class="control-stack mt-3 px-1">
                    <label class="small text-muted d-block mb-1" for="theme-count-time">
                        {{ $t('threads.themeCount') }}
                    </label>
                    <select id="theme-count-time" v-model.number="themeCount"
                        @change="onGrainChange"
                        class="form-select form-select-sm" style="max-width: 12rem;">
                        <option v-for="n in (result?.available_theme_counts || [])" :key="n" :value="n">{{ n }}</option>
                    </select>
                    <p class="control-hint">{{ $t('threads.themeCountHint') }}</p>
                </div>
            </div>

            <!-- Per-thread detail cards -->
            <div class="row">
                <div v-for="thread in result.threads" :key="thread.id" class="col-12 mb-3">
                    <article class="card thread-card shadow-sm">
                        <div class="card-header thread-header py-2 d-flex justify-content-between align-items-center flex-wrap"
                            :style="{ borderLeft: `4px solid ${threadColor(thread.id - 1, 1)}` }">
                            <strong>{{ $t('threads.threadN', { n: thread.id }) }}</strong>
                            <button type="button" class="btn btn-sm btn-link p-0"
                                @click="onViewPassages(thread)">
                                {{ $t('threads.viewPassages') }} →
                            </button>
                        </div>
                        <div class="card-body py-2">
                            <div class="mb-2">
                                <span v-for="w in thread.words.slice(0, 15)" :key="w.word" class="word-chip">
                                    {{ w.word }}
                                </span>
                                <span v-if="thread.words.length > 15" class="text-muted small ms-1">
                                    +{{ thread.words.length - 15 }}
                                </span>
                            </div>
                            <svg :viewBox="`0 0 ${chartWidth} ${cardChartHeight}`" class="thread-spark"
                                preserveAspectRatio="none" :aria-label="$t('threads.intensityChart')">
                                <path :d="cardSparkPath(thread)" :fill="threadColor(thread.id - 1, 0.25)" stroke="none" />
                                <path :d="cardSparkLine(thread)" :stroke="threadColor(thread.id - 1, 1)" stroke-width="1" fill="none" />
                            </svg>
                            <div class="d-flex justify-content-between small text-muted">
                                <span>{{ result.year_range[0] }}</span>
                                <span>{{ result.year_range[1] }}</span>
                            </div>
                        </div>
                    </article>
                </div>
            </div>
        </div>

        <div v-else-if="result && (!result.threads || result.threads.length === 0)"
            class="text-center py-5 text-muted">
            {{ $t('threads.noResults') }}
        </div>

        <DistinctivePassagesModal
            :group-name="modal.groupName" :signature="modal.signature"
            :passages="modal.passages" :loading="modal.loading"
            :has-more="modal.hasMore" :view-all-url="modal.viewAllUrl"
            @load-more="loadMorePassages" @view-all="onViewAllPassages" />
    </div>
</template>

<script setup>
import { computed, inject, ref } from "vue";
import { storeToRefs } from "pinia";
import { useRouter } from "vue-router";
import { Modal } from "bootstrap";
import { useMainStore } from "../../stores/main";
import { concordanceMethod, debug, paramsFilter, paramsToRoute } from "../../utils.js";
import DistinctivePassagesModal from "../DistinctivePassagesModal.vue";
import ProgressSpinner from "../ProgressSpinner";

const emit = defineEmits(["filterList"]);

const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const router = useRouter();
const store = useMainStore();
const { formData } = storeToRefs(store);

const loading = ref(false);
const result = ref(null);
// Number of themes to display (top-N senses by mass). Default 4.
const themeCount = ref(4);
let fetchToken = 0;

const modal = ref({
    groupName: "", yearRange: "", signature: [], passages: [],
    loading: false, hasMore: false, offset: 0, pageSize: 25, viewAllUrl: "",
});
let modalInstance = null;
let passagesFetchToken = 0;

// ---- Layout constants ----
const chartWidth = 1000;
const cardChartHeight = 60;
const streamHeight = 140;
const overviewHeight = streamHeight;


const threadHues = [205, 30, 145, 280, 0, 90, 165, 235, 50, 315, 120, 260, 15, 60, 200, 320];
function threadColor(i, alpha) {
    const h = threadHues[i % threadHues.length];
    return `hsla(${h}, 55%, 50%, ${alpha})`;
}

// ---- Card sparkline paths ----
function cardSparkPath(thread) {
    const intensities = thread.intensity || [];
    const maxV = thread.max_intensity || 1;
    if (intensities.length === 0 || maxV === 0) return "";
    const n = intensities.length;
    const xStep = chartWidth / Math.max(1, n - 1);
    let d = `M 0 ${cardChartHeight} `;
    for (let i = 0; i < n; i++) {
        const x = i * xStep;
        const y = cardChartHeight - (intensities[i] / maxV) * (cardChartHeight - 2);
        d += `L ${x.toFixed(2)} ${y.toFixed(2)} `;
    }
    d += `L ${chartWidth} ${cardChartHeight} Z`;
    return d;
}

function cardSparkLine(thread) {
    const intensities = thread.intensity || [];
    const maxV = thread.max_intensity || 1;
    if (intensities.length === 0 || maxV === 0) return "";
    const n = intensities.length;
    const xStep = chartWidth / Math.max(1, n - 1);
    let d = "";
    for (let i = 0; i < n; i++) {
        const x = i * xStep;
        const y = cardChartHeight - (intensities[i] / maxV) * (cardChartHeight - 2);
        d += (i === 0 ? "M" : " L") + ` ${x.toFixed(2)} ${y.toFixed(2)}`;
    }
    return d;
}

// ---- Streamgraph layers (centered-baseline streamgraph) ----
const streamLayers = computed(() => {
    if (!result.value || !result.value.threads || result.value.threads.length === 0) return [];
    const threads = result.value.threads;
    const n = threads[0].intensity.length;
    // Per-year totals across threads
    const totals = new Array(n).fill(0);
    for (const t of threads) for (let i = 0; i < n; i++) totals[i] += t.intensity[i];
    const maxTotal = Math.max(...totals, 0.0001);
    const xStep = chartWidth / Math.max(1, n - 1);
    const layers = [];
    const offsets = new Array(n).fill(0);
    // Center stream around midline
    const heights = threads.map((t) => t.intensity.map((v) => v));
    for (let li = 0; li < threads.length; li++) {
        const thread = threads[li];
        const top = new Array(n);
        const bottom = new Array(n);
        for (let i = 0; i < n; i++) {
            const half = totals[i] / 2 / maxTotal;
            const stackBefore = offsets[i] / maxTotal;
            const v = heights[li][i] / maxTotal;
            const yMid = streamHeight / 2;
            bottom[i] = yMid - (half - stackBefore) * (streamHeight / 2);
            top[i] = yMid - (half - stackBefore - v) * (streamHeight / 2);
            offsets[i] += heights[li][i];
        }
        let d = `M 0 ${bottom[0].toFixed(2)} `;
        for (let i = 0; i < n; i++) d += `L ${(i * xStep).toFixed(2)} ${bottom[i].toFixed(2)} `;
        for (let i = n - 1; i >= 0; i--) d += `L ${(i * xStep).toFixed(2)} ${top[i].toFixed(2)} `;
        d += "Z";
        layers.push({ id: thread.id, path: d, thread });
    }
    return layers;
});

// ---- Fetch ----
function runDetection(opts = {}) {
    const myToken = ++fetchToken;
    loading.value = true;
    if (!opts.keepResult) result.value = null;
    const params = paramsFilter(formData.value);
    params.n_clusters = themeCount.value;
    $http.get(`${$dbUrl}/scripts/get_threads.py`, { params }).then((resp) => {
        if (myToken !== fetchToken) return;
        result.value = resp.data;
        // Keep the dropdown selection valid on thin queries that yield fewer
        // senses than requested (backend already truncated; reflect it here).
        const avail = resp.data?.available_theme_counts || [];
        if (avail.length && !avail.includes(themeCount.value)) {
            themeCount.value = avail[avail.length - 1];
        }
        emit("filterList", resp.data?.filter_list || []);
    }).catch((error) => {
        debug({ $options: { name: "thread-timeline" } }, error);
    }).finally(() => {
        if (myToken === fetchToken) loading.value = false;
    });
}

// Re-run when the user overrides the clustering grain.
function onGrainChange() {
    runDetection({ keepResult: true });
}

function reset() {
    result.value = null;
    fetchToken++;
}

// ---- Passage modal ----
function onViewPassages(thread) {
    // Use the full result year range; the thread's signature tokens do the
    // semantic filtering, so a broader window surfaces more relevant hits
    // than a tight peak-centric slice would.
    const [yMin, yMax] = result.value.year_range;
    const yearRange = `${yMin}-${yMax}`;
    modal.value = {
        groupName: `${formData.value.q} · ${thread.label}`,
        yearRange,
        signature: thread.words.slice(0, 20).map((w) => ({ word: w.word, z: null })),
        passages: [],
        loading: true,
        hasMore: false,
        offset: 0,
        pageSize: 25,
        viewAllUrl: paramsToRoute({
            ...formData.value,
            report: "concordance",
            method: concordanceMethod(formData.value.colloc_within),
            year: yearRange,
        }),
    };
    showModal();
    fetchPassages({ append: false });
}

function showModal() {
    const el = document.getElementById("distinctive-passages-modal");
    if (!el) return;
    if (!modalInstance) modalInstance = new Modal(el);
    modalInstance.show();
}

async function fetchPassages({ append }) {
    const myToken = ++passagesFetchToken;
    modal.value.loading = true;
    try {
        const params = {
            ...paramsFilter(formData.value),
            restrict_to_field: "year",
            restrict_to_value: modal.value.yearRange,
            signature_tokens: modal.value.signature.map((s) => s.word).join(","),
            top_n: modal.value.pageSize,
            offset: modal.value.offset,
        };
        const resp = await $http.get(`${$dbUrl}/scripts/get_representative_passages.py`, { params });
        if (myToken !== passagesFetchToken) return;
        const data = resp.data || {};
        modal.value.passages = append
            ? modal.value.passages.concat(data.passages || [])
            : (data.passages || []);
        modal.value.hasMore = !!data.has_more;
    } catch (error) {
        debug({ $options: { name: "thread-timeline" } }, error);
    } finally {
        if (myToken === passagesFetchToken) modal.value.loading = false;
    }
}

function loadMorePassages() {
    modal.value.offset += modal.value.pageSize;
    fetchPassages({ append: true });
}

function onViewAllPassages() {
    const dest = modal.value.viewAllUrl;
    if (modalInstance) modalInstance.hide();
    if (dest) router.push(dest);
}

defineExpose({ runDetection, reset });
</script>

<style lang="scss" scoped>
@use "../../assets/styles/theme.module.scss" as theme;

.thread-card {
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.thread-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08) !important;
}

.thread-header {
    background-color: #fafafa;
    border-bottom: 1px solid #eee;
}

.word-chip {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    margin: 0.15rem 0.25rem 0.15rem 0;
    background-color: rgba(theme.$link-color, 0.08);
    border: 1px solid rgba(theme.$link-color, 0.2);
    border-radius: 0.5rem;
    color: theme.$link-color;
    font-size: 0.85rem;
}

.thread-spark {
    width: 100%;
    height: 60px;
    background-color: #fafafa;
    border: 1px solid #eee;
    border-radius: 0.25rem;
}

.overview-card {
    background-color: #fff;
}

.overview-stream {
    width: 100%;
    height: 140px;
    display: block;
}

.stream-band {
    cursor: pointer;
    transition: opacity 0.15s ease;
}

.stream-band:hover {
    opacity: 0.85;
}

.legend-chip {
    border: 1px solid;
    background-color: #fff;
    padding: 0.15rem 0.5rem;
    font-size: 0.75rem;
    line-height: 1.3;
}

.control-stack {
    padding-top: 0.75rem;
    border-top: 1px solid #eee;
}

.control-hint {
    margin: 0.25rem 0 0;
    font-size: 0.7rem;
    line-height: 1.3;
    color: #888;
}

.legend-swatch {
    display: inline-block;
    width: 0.65rem;
    height: 0.65rem;
    border-radius: 0.15rem;
    margin-right: 0.3rem;
    vertical-align: middle;
}

.info-tip {
    position: relative;
    display: inline-block;
    cursor: help;
    color: #999;
    margin-left: 0.25rem;
    font-size: 0.95em;
    user-select: none;
    outline: none;
}

.info-tip:hover::after,
.info-tip:focus::after {
    content: attr(data-tip);
    position: absolute;
    top: calc(100% + 6px);
    left: 0;
    z-index: 1000;
    background: rgba(33, 37, 41, 0.95);
    color: #fff;
    padding: 0.5rem 0.75rem;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    line-height: 1.4;
    width: max-content;
    max-width: 320px;
    white-space: normal;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
    pointer-events: none;
}
</style>
