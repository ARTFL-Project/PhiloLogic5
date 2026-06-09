<template>
    <div>
        <!-- Detection auto-runs on mount; the controls below trigger re-runs. -->

        <div v-if="loading && !result" class="text-center py-5">
            <progress-spinner :lg="true" :message="$t('usagePatterns.detecting')" />
        </div>

        <div v-if="result && result.patterns && result.patterns.length > 0" class="mx-2 my-3">
            <!-- Everything (count, explanation, controls, streamgraph) lives
                 inside the result card as one self-contained unit. -->
            <div class="card shadow-sm p-3 mb-3 overview-card">
                <p class="text-muted mb-3 d-flex align-items-center flex-wrap gap-1">
                    <i18n-t keypath="usagePatterns.summary" tag="span">
                        <template #patterns>
                            <select v-model.number="patternCount" @change="onGrainChange"
                                class="form-select form-select-sm summary-dropdown"
                                :aria-label="$t('usagePatterns.patternCount')">
                                <option v-for="n in (result?.available_pattern_counts || [])" :key="n" :value="n">{{ n }}</option>
                            </select>
                        </template>
                    </i18n-t>
                    <span class="info-tip" tabindex="0"
                        :data-tip="$t('usagePatterns.overviewSubtitle')"
                        :aria-label="$t('usagePatterns.overviewSubtitle')">ⓘ</span>
                </p>

                <div class="d-flex flex-wrap gap-1 px-1 mb-3">
                    <button v-for="pattern in result.patterns" :key="`leg-${pattern.id}`" type="button"
                        class="btn btn-sm legend-chip"
                        :style="{ borderColor: patternColor(pattern.id - 1, 1), color: patternColor(pattern.id - 1, 1) }"
                        @click="onViewPassages(pattern)"
                        :title="pattern.words.slice(0, 10).map((w) => w.word).join(', ')">
                        <span class="legend-swatch" :style="{ backgroundColor: patternColor(pattern.id - 1, 1) }"></span>
                        {{ pattern.label }}
                    </button>
                </div>
                <svg :viewBox="`0 0 ${chartWidth} ${overviewHeight}`" class="overview-stream"
                    preserveAspectRatio="none" :aria-label="$t('usagePatterns.streamgraph')">
                    <path v-for="(layer, i) in streamLayers" :key="layer.id"
                        :d="layer.path" :fill="patternColor(i, 0.85)" stroke="#fff" stroke-width="0.3"
                        @click="onViewPassages(layer.pattern)" class="stream-band"
                        :aria-label="`${$t('usagePatterns.patternN', { n: layer.pattern.id })}: ${layer.pattern.label}`">
                        <title>{{ $t('usagePatterns.patternN', { n: layer.pattern.id }) }}: {{ layer.pattern.label }}</title>
                    </path>
                </svg>
                <div class="year-axis small text-muted mt-1">
                    <span v-for="t in yearTicks" :key="t.year" class="year-tick"
                        :style="{ left: t.pct + '%', transform: tickShift(t.pct) }">{{ t.year }}</span>
                </div>
            </div>

            <!-- Per-pattern detail cards -->
            <div class="row">
                <div v-for="pattern in result.patterns" :key="pattern.id" class="col-12 mb-3">
                    <article class="card pattern-card shadow-sm">
                        <div class="card-header pattern-header py-2 d-flex justify-content-between align-items-center flex-wrap"
                            :style="{ borderLeft: `4px solid ${patternColor(pattern.id - 1, 1)}` }">
                            <strong>{{ $t('usagePatterns.patternN', { n: pattern.id }) }}</strong>
                            <button type="button" class="btn btn-sm btn-link p-0"
                                @click="onViewPassages(pattern)">
                                {{ $t('usagePatterns.viewPassages') }} →
                            </button>
                        </div>
                        <div class="card-body py-2">
                            <div class="mb-2">
                                <span v-for="w in pattern.words.slice(0, 15)" :key="w.word" class="word-chip">
                                    {{ w.word }}
                                </span>
                                <span v-if="pattern.words.length > 15" class="text-muted small ms-1">
                                    +{{ pattern.words.length - 15 }}
                                </span>
                            </div>
                            <svg :viewBox="`0 0 ${chartWidth} ${cardChartHeight}`" class="pattern-spark"
                                preserveAspectRatio="none" :aria-label="$t('usagePatterns.intensityChart')">
                                <path :d="cardSparkPath(pattern)" :fill="patternColor(pattern.id - 1, 0.25)" stroke="none" />
                                <path :d="cardSparkLine(pattern)" :stroke="patternColor(pattern.id - 1, 1)" stroke-width="1" fill="none" />
                            </svg>
                            <div class="year-axis small text-muted">
                                <span v-for="t in yearTicks" :key="t.year" class="year-tick"
                                    :style="{ left: t.pct + '%', transform: tickShift(t.pct) }">{{ t.year }}</span>
                            </div>
                        </div>
                    </article>
                </div>
            </div>
        </div>

        <div v-else-if="result && (!result.patterns || result.patterns.length === 0)"
            class="text-center py-5 text-muted">
            {{ $t('usagePatterns.noResults') }}
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
// Number of patterns to display. Seeded at 4, but for a fresh query we let the
// backend pick its smart default (the hub-strength knee) and adopt that count
// here; only once the user picks from the dropdown do we send an explicit count.
const patternCount = ref(4);
let userChoseCount = false;
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


const patternHues = [205, 30, 145, 280, 0, 90, 165, 235, 50, 315, 120, 260, 15, 60, 200, 320];
function patternColor(i, alpha) {
    const h = patternHues[i % patternHues.length];
    return `hsla(${h}, 55%, 50%, ${alpha})`;
}

// ---- X-axis date ticks ----
// Round-number year ticks across the range (~6-8), positioned as a percent of
// the span so they line up with the full-width charts. Shared by the overview
// and the per-pattern cards.
const yearTicks = computed(() => {
    const range = result.value?.year_range;
    if (!range) return [];
    const [min, max] = range;
    const span = max - min;
    if (span <= 0) return [{ year: min, pct: 0 }];
    const steps = [10, 20, 25, 50, 100, 200, 250, 500];
    let step = steps[steps.length - 1];
    for (const s of steps) { if (span / s <= 9) { step = s; break; } }
    const ticks = [];
    for (let y = Math.ceil(min / step) * step; y <= max; y += step) {
        ticks.push({ year: y, pct: ((y - min) / span) * 100 });
    }
    return ticks;
});
// Keep edge labels from overflowing the chart width.
function tickShift(pct) {
    if (pct < 5) return "translateX(0)";
    if (pct > 95) return "translateX(-100%)";
    return "translateX(-50%)";
}

// ---- Per-pattern share over time ----
// Each card shows its pattern's SHARE of the composition over time — derived
// from the same share_weight that drives the overview, normalized per year — so
// a card mirrors its overview band (rising/falling). Auto-scaled to each
// pattern's own peak share so its trajectory is legible.
const shareByPattern = computed(() => {
    const patterns = result.value?.patterns || [];
    if (!patterns.length) return {};
    const weightOf = (t) => t.share_weight || t.intensity;
    const n = weightOf(patterns[0]).length;
    const totals = new Array(n).fill(0);
    for (const t of patterns) { const w = weightOf(t); for (let i = 0; i < n; i++) totals[i] += w[i]; }
    const out = {};
    for (const t of patterns) {
        const w = weightOf(t);
        const vals = new Array(n);
        let mx = 0;
        for (let i = 0; i < n; i++) {
            const s = totals[i] > 0 ? w[i] / totals[i] : 0;
            vals[i] = s;
            if (s > mx) mx = s;
        }
        out[t.id] = { values: vals, max: mx || 1 };
    }
    return out;
});

// ---- Card sparkline paths (per-pattern share) ----
function cardSparkPath(pattern) {
    const series = shareByPattern.value[pattern.id];
    if (!series) return "";
    const vals = series.values, maxV = series.max, n = vals.length;
    const xStep = chartWidth / Math.max(1, n - 1);
    let d = `M 0 ${cardChartHeight} `;
    for (let i = 0; i < n; i++) {
        const x = i * xStep;
        const y = cardChartHeight - (vals[i] / maxV) * (cardChartHeight - 2);
        d += `L ${x.toFixed(2)} ${y.toFixed(2)} `;
    }
    d += `L ${chartWidth} ${cardChartHeight} Z`;
    return d;
}

function cardSparkLine(pattern) {
    const series = shareByPattern.value[pattern.id];
    if (!series) return "";
    const vals = series.values, maxV = series.max, n = vals.length;
    const xStep = chartWidth / Math.max(1, n - 1);
    let d = "";
    for (let i = 0; i < n; i++) {
        const x = i * xStep;
        const y = cardChartHeight - (vals[i] / maxV) * (cardChartHeight - 2);
        d += (i === 0 ? "M" : " L") + ` ${x.toFixed(2)} ${y.toFixed(2)}`;
    }
    return d;
}

// ---- Overview streamgraph: 100%-normalized composition ----
// The overview answers "which sense dominates when", so each year is normalized
// to that year's total (bands sum to full height). Normalizing per year is what
// lets the bands move relative to each other (one sense widening as another
// narrows) rather than tracking the word's overall volume. Years with no data
// (all-zero, masked by the backend) collapse to the midline so the ribbon
// pinches there rather than showing a false even split.
const streamLayers = computed(() => {
    if (!result.value || !result.value.patterns || result.value.patterns.length === 0) return [];
    const patterns = result.value.patterns;
    // Overview is composition: use the proportional share_weight when present
    // (older backends only send intensity, so fall back to it).
    const weightOf = (t) => t.share_weight || t.intensity;
    const n = weightOf(patterns[0]).length;
    // Per-year totals across patterns (the normalizer).
    const totals = new Array(n).fill(0);
    for (const t of patterns) { const w = weightOf(t); for (let i = 0; i < n; i++) totals[i] += w[i]; }
    const xStep = chartWidth / Math.max(1, n - 1);
    const layers = [];
    const offsets = new Array(n).fill(0);
    const heights = patterns.map((t) => weightOf(t));
    for (let li = 0; li < patterns.length; li++) {
        const pattern = patterns[li];
        const top = new Array(n);
        const bottom = new Array(n);
        for (let i = 0; i < n; i++) {
            const tot = totals[i];
            if (tot <= 0) {
                bottom[i] = top[i] = streamHeight / 2; // no data → pinch to midline
                continue;
            }
            const cumBefore = offsets[i] / tot; // share stacked below this band
            const frac = heights[li][i] / tot; // this band's share of the year
            bottom[i] = (1 - cumBefore) * streamHeight;
            top[i] = (1 - cumBefore - frac) * streamHeight;
            offsets[i] += heights[li][i];
        }
        let d = `M 0 ${bottom[0].toFixed(2)} `;
        for (let i = 0; i < n; i++) d += `L ${(i * xStep).toFixed(2)} ${bottom[i].toFixed(2)} `;
        for (let i = n - 1; i >= 0; i--) d += `L ${(i * xStep).toFixed(2)} ${top[i].toFixed(2)} `;
        d += "Z";
        layers.push({ id: pattern.id, path: d, pattern });
    }
    return layers;
});

// ---- Fetch ----
function runDetection(opts = {}) {
    // A fresh run (mount, new query) defers to the backend's smart default; only
    // a user dropdown change sends an explicit count.
    if (!opts.grainChange) userChoseCount = false;
    const myToken = ++fetchToken;
    loading.value = true;
    if (!opts.keepResult) result.value = null;
    const params = paramsFilter(formData.value);
    if (userChoseCount) params.n_clusters = patternCount.value;
    $http.get(`${$dbUrl}/scripts/get_usage_patterns.py`, { params }).then((resp) => {
        if (myToken !== fetchToken) return;
        result.value = resp.data;
        const avail = resp.data?.available_pattern_counts || [];
        if (!userChoseCount) {
            // Adopt the backend's smart default so the dropdown reflects it.
            const n = resp.data?.patterns?.length;
            if (n) patternCount.value = n;
        } else if (avail.length && !avail.includes(patternCount.value)) {
            // Keep the selection valid on thin queries that yield fewer senses
            // than requested (backend already truncated; reflect it here).
            patternCount.value = avail[avail.length - 1];
        }
        emit("filterList", resp.data?.filter_list || []);
    }).catch((error) => {
        debug({ $options: { name: "pattern-timeline" } }, error);
    }).finally(() => {
        if (myToken === fetchToken) loading.value = false;
    });
}

// Re-run when the user overrides the clustering grain.
function onGrainChange() {
    userChoseCount = true;
    runDetection({ keepResult: true, grainChange: true });
}

function reset() {
    result.value = null;
    fetchToken++;
}

// ---- Passage modal ----
function onViewPassages(pattern) {
    // Use the full result year range; the pattern's signature tokens do the
    // semantic filtering, so a broader window surfaces more relevant hits
    // than a tight peak-centric slice would.
    const [yMin, yMax] = result.value.year_range;
    const yearRange = `${yMin}-${yMax}`;
    modal.value = {
        groupName: `${formData.value.q} · ${pattern.label}`,
        yearRange,
        signature: pattern.words.slice(0, 20).map((w) => ({ word: w.word, z: null })),
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
        debug({ $options: { name: "pattern-timeline" } }, error);
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

.pattern-card {
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.pattern-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08) !important;
}

.pattern-header {
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

.pattern-spark {
    width: 100%;
    height: 60px;
    background-color: #fafafa;
    border: 1px solid #eee;
    border-radius: 0.25rem;
}

.overview-card {
    background-color: #fff;
}

.year-axis {
    position: relative;
    height: 1.1em;
}

.year-tick {
    position: absolute;
    top: 0;
    white-space: nowrap;
}

.summary-dropdown {
    display: inline-block;
    width: auto;
    min-width: 3.2rem;
    padding: 0.1rem 1.5rem 0.1rem 0.4rem;
    font-weight: 600;
    vertical-align: baseline;
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
