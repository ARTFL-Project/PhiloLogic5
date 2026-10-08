<template>
    <div>
        <div v-if="loading && !result" class="text-center py-5">
            <progress-spinner :lg="true" :message="$t('wordSenses.detecting')" />
        </div>
        <div v-if="failed" class="alert alert-warning mx-2 my-3" role="alert">{{ $t("wordSenses.failed") }}</div>

        <div v-if="result && result.n_sentences" class="mx-2 my-3">
            <!-- Summary, the rule, the detail shown, and the overview over time -->
            <div class="card shadow-sm p-3 mb-3">
                <p class="mb-1 fw-semibold">
                    {{ shown.length > 1
                        ? $t("wordSenses.summary", {
                            senses: shown.length,
                            sentences: result.n_sentences.toLocaleString(),
                            works: result.n_works.toLocaleString(),
                        })
                        : $t(splits.length ? "wordSenses.oneShown" : "wordSenses.noSplit", {
                            sentences: result.n_sentences.toLocaleString(),
                            works: result.n_works.toLocaleString(),
                        }) }}
                </p>
                <p class="text-muted small mb-2">
                    {{ $t("wordSenses.rule", { folds: result.folds, word: queryWord, finest: percent(result.finest) }) }}
                    <span v-if="result.seconds !== undefined">{{ $t("wordSenses.seconds", { s: result.seconds }) }}</span>
                </p>

                <div v-if="splits.length" class="detail mb-3">
                    <label for="word-senses-detail" class="form-label small fw-semibold mb-0">
                        {{ $t("wordSenses.detail") }}
                    </label>
                    <div class="d-flex align-items-center gap-2 flex-wrap">
                        <span class="small text-muted" aria-hidden="true">{{ $t("wordSenses.coarser") }}</span>
                        <input id="word-senses-detail" v-model.number="position" type="range" class="form-range detail-range"
                            min="0" :max="splits.length" step="1" :aria-valuetext="detailText"
                            aria-describedby="word-senses-detail-text" />
                        <span class="small text-muted" aria-hidden="true">{{ $t("wordSenses.finer") }}</span>
                    </div>
                    <p id="word-senses-detail-text" class="small mb-0" aria-live="polite">{{ detailText }}</p>
                </div>

                <template v-if="time && shown.length > 1">
                    <div class="d-flex align-items-center flex-wrap gap-2 mb-2" role="group"
                        :aria-label="$t('wordSenses.viewLabel')">
                        <div class="btn-group btn-group-sm" role="group">
                            <button type="button" class="btn btn-secondary" :class="{ active: view === 'share' }"
                                :aria-pressed="view === 'share'" @click="view = 'share'">{{ $t("wordSenses.viewShare") }}</button>
                            <button v-if="time.rate" type="button" class="btn btn-secondary"
                                :class="{ active: view === 'rate' }"
                                :aria-pressed="view === 'rate'" @click="view = 'rate'">{{ $t("wordSenses.viewRate") }}</button>
                        </div>
                        <span class="small text-muted">{{ view === "share" ? $t("wordSenses.shareExplained")
                            : $t("wordSenses.rateExplained") }}</span>
                    </div>
                    <div class="chart-frame">
                        <span class="y-max small text-muted">{{ formatValue(overviewMax) }}</span>
                        <svg :viewBox="`0 0 ${chartWidth} ${overviewHeight}`" class="overview-chart" preserveAspectRatio="none"
                            role="img" :aria-label="$t('wordSenses.overviewChart')"
                            @mousemove="onHover($event)" @mouseleave="hoverIndex = null">
                            <path v-for="layer in overviewLayers" :key="layer.id" :d="layer.path"
                                :fill="senseColor(layer.number, 0.85)" stroke="#fff" stroke-width="0.3" />
                            <line v-if="hoverIndex !== null" :x1="xOf(hoverIndex)" :x2="xOf(hoverIndex)" y1="0"
                                :y2="overviewHeight" stroke="#333" stroke-width="1" vector-effect="non-scaling-stroke" />
                        </svg>
                    </div>
                    <div class="year-axis small text-muted mt-1">
                        <span v-for="tick in yearTicks" :key="tick.year" class="year-tick"
                            :style="{ left: tick.pct + '%', transform: tickShift(tick.pct) }">{{ tick.year }}</span>
                    </div>
                    <p class="readout small mb-1 mt-2" aria-live="polite">
                        <template v-if="hoverIndex !== null">
                            <strong>{{ time.start + hoverIndex }}</strong>
                            ({{ $t("wordSenses.sentencesThatYear", { n: time.sentences[hoverIndex], works: time.works[hoverIndex] }) }}):
                            <span v-for="node in shown" :key="node.id" class="me-2">
                                <span class="legend-swatch" :style="{ backgroundColor: senseColor(numberOf[node.id], 1) }"></span>
                                {{ formatValue(curve(node).value[hoverIndex]) }}
                            </span>
                        </template>
                        <span v-else class="text-muted">{{ $t("wordSenses.hoverHint") }}</span>
                    </p>
                </template>
                <p v-else-if="!time" class="text-muted small mb-0">{{ $t("wordSenses.noTime") }}</p>

                <div v-if="shown.length > 1" class="d-flex flex-wrap gap-1 mt-2">
                    <button v-for="node in shown" :key="`chip-${node.id}`" type="button" class="btn btn-sm legend-chip"
                        :style="{ borderColor: senseColor(numberOf[node.id], 1), color: senseColor(numberOf[node.id], 1) }"
                        @click="scrollToSense(node)">
                        <span class="legend-swatch" :style="{ backgroundColor: senseColor(numberOf[node.id], 1) }"></span>
                        {{ $t("wordSenses.senseN", { n: numberOf[node.id] }) }}: {{ label(node) }}
                    </button>
                </div>
            </div>

            <!-- The word around the query in general: the whole query, compared with the corpus -->
            <div class="card shadow-sm p-3 mb-3">
                <h3 class="h6 mb-2">{{ $t("wordSenses.general", { word: queryWord }) }}</h3>
                <WordList :node="root" :query-word="queryWord" :against-corpus="true" />
                <button v-if="shown.length === 1" type="button" class="btn btn-sm btn-link p-0 mt-2"
                    @click="openExamples(root)">{{ $t("wordSenses.examples") }} &rarr;</button>
            </div>

            <!-- The tree, to the detail asked for: splits, and the senses they end in -->
            <template v-if="shown.length > 1">
                <div v-for="row in rows" :key="`${row.kind}-${row.node.id}`" :style="{ marginLeft: row.depth * 1.25 + 'rem' }">
                    <p v-if="row.kind === 'split'" class="split-row small mb-2">
                        <span class="split-mark" aria-hidden="true">&#9662;</span>
                        {{ $t("wordSenses.splitInto", { senses: senseRange(row.node), share: percent(row.node.share) }) }}
                        <span class="text-muted">&middot; {{ $t("wordSenses.split", {
                            seed: seedText(row.node.split.seed),
                            importance: percent(row.node.split.relative),
                            z: row.node.split.z.toFixed(1),
                        }) }}</span>
                    </p>
                    <section v-else :id="`sense-card-${row.node.id}`" class="card shadow-sm mb-3 sense-card"
                        :style="{ borderLeft: `4px solid ${senseColor(numberOf[row.node.id], 1)}` }"
                        :aria-labelledby="`sense-title-${row.node.id}`">
                        <div class="sense-header px-3 py-2 d-flex justify-content-between align-items-center flex-wrap gap-2">
                            <h3 class="h6 mb-0" :id="`sense-title-${row.node.id}`">
                                <span class="legend-swatch" :style="{ backgroundColor: senseColor(numberOf[row.node.id], 1) }"></span>
                                {{ $t("wordSenses.senseN", { n: numberOf[row.node.id] }) }}
                                <span class="fw-normal text-muted">&middot; {{ $t("wordSenses.shareOfSentences",
                                    { share: percent(row.node.share) }) }}
                                    &middot; {{ $t("wordSenses.fromWorks", { n: row.node.works }) }}</span>
                                <span v-if="row.node.top_work_share >= 0.5" class="one-work ms-1">
                                    {{ $t("wordSenses.mostlyOneWork", { share: percent(row.node.top_work_share) }) }}
                                </span>
                            </h3>
                            <button type="button" class="btn btn-sm btn-link p-0"
                                @click="openExamples(row.node)">{{ $t("wordSenses.examples") }} &rarr;</button>
                        </div>
                        <div class="card-body py-2">
                            <WordList :node="row.node" :query-word="queryWord" :against-corpus="false" />
                            <template v-if="time">
                                <div class="chart-frame mt-2">
                                    <span class="y-max small text-muted">{{ formatValue(cardMax(row.node)) }}</span>
                                    <svg :viewBox="`0 0 ${chartWidth} ${cardHeight}`" class="sense-chart"
                                        preserveAspectRatio="none" role="img"
                                        :aria-label="$t('wordSenses.senseChart', { n: numberOf[row.node.id] })"
                                        @mousemove="onHover($event)" @mouseleave="hoverIndex = null">
                                        <path :d="bandPath(row.node)" :fill="senseColor(numberOf[row.node.id], 0.18)" stroke="none" />
                                        <path :d="linePath(row.node)" :stroke="senseColor(numberOf[row.node.id], 1)"
                                            stroke-width="1.5" fill="none" vector-effect="non-scaling-stroke" />
                                        <line v-if="hoverIndex !== null" :x1="xOf(hoverIndex)" :x2="xOf(hoverIndex)" y1="0"
                                            :y2="cardHeight" stroke="#333" stroke-width="1" vector-effect="non-scaling-stroke" />
                                    </svg>
                                </div>
                                <div class="year-axis small text-muted">
                                    <span v-for="tick in yearTicks" :key="tick.year" class="year-tick"
                                        :style="{ left: tick.pct + '%', transform: tickShift(tick.pct) }">{{ tick.year }}</span>
                                </div>
                                <p v-if="hoverIndex !== null" class="readout small mb-0 mt-1">
                                    <strong>{{ time.start + hoverIndex }}</strong>:
                                    {{ formatValue(curve(row.node).value[hoverIndex]) }}
                                    ({{ $t("wordSenses.bandRange", {
                                        low: formatValue(curve(row.node).low[hoverIndex]),
                                        high: formatValue(curve(row.node).high[hoverIndex]),
                                    }) }})
                                </p>
                            </template>
                            <p class="small text-muted mb-0 mt-2">{{ notSplitReason(row.node) }}</p>
                        </div>
                    </section>
                </div>
                <p v-if="time" class="small text-muted mx-1">{{ $t("wordSenses.bandExplained", { n: result.resamples }) }}</p>
            </template>
        </div>

        <div v-else-if="result" class="text-center py-5 text-muted">{{ $t("wordSenses.noResults") }}</div>

        <DistinctivePassagesModal :group-name="modal.groupName" :signature="modal.signature" :passages="modal.passages"
            :loading="modal.loading" :has-more="modal.hasMore" :ordering-note="$t('wordSenses.examplesNote')"
            @load-more="loadMoreExamples" />
    </div>
</template>

<script setup>
import { Modal } from "bootstrap";
import { storeToRefs } from "pinia";
import { computed, defineComponent, h, inject, ref } from "vue";
import { useI18n } from "vue-i18n";
import { useMainStore } from "../../stores/main";
import { debug, paramsFilter } from "../../utils.js";
import DistinctivePassagesModal from "../DistinctivePassagesModal.vue";
import ProgressSpinner from "../ProgressSpinner";

const emit = defineEmits(["filterList", "stopwordsMissing"]);
const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const store = useMainStore();
const { formData } = storeToRefs(store);
const { t } = useI18n();

const loading = ref(false);
const failed = ref(false);
const result = ref(null);
const view = ref("share"); // "share" of the year's sentences, or uses per million words ("rate")
const hoverIndex = ref(null); // the year (from time.start) the pointer is on, in every chart at once
const position = ref(0); // how many distinctions are shown, the most important first
let fetchToken = 0;

const chartWidth = 1000;
const overviewHeight = 140;
const cardHeight = 60;
const PAGE = 25;

// The query's word as it reads in a sentence: "lemma:raison" is raison
const queryWord = computed(() => (formData.value.q || "").replace(/lemma:/g, "").replace(/"/g, "").trim());
const nodes = computed(() => result.value?.nodes || []);
const root = computed(() => nodes.value[0]);
const time = computed(() => result.value?.time || null);

// The distinctions (splits), the most important first
const splits = computed(() =>
    nodes.value.filter((n) => n.children.length).sort((a, b) => b.split.relative - a.split.relative || a.id - b.id)
);

function runDetection() {
    const myToken = ++fetchToken;
    loading.value = true;
    failed.value = false;
    result.value = null;
    view.value = "share";
    $http
        .get(`${$dbUrl}/scripts/get_word_senses.py`, { params: paramsFilter(formData.value) })
        .then((response) => {
            if (myToken !== fetchToken) return;
            result.value = response.data;
            emit("filterList", response.data?.filter_list || []); // the collocation filter's words, left out here too
            emit("stopwordsMissing", Boolean(response.data?.stopwords_missing));
            // At first, the distinctions at least as important as the default detail
            const detail = response.data?.detail ?? 1;
            position.value = splits.value.filter((n) => n.split.relative >= detail).length;
        })
        .catch((error) => {
            if (myToken !== fetchToken) return;
            failed.value = true;
            debug({ $options: { name: "word-senses" } }, error);
        })
        .finally(() => {
            if (myToken === fetchToken) loading.value = false;
        });
}

function reset() {
    result.value = null;
    fetchToken++;
}

defineExpose({ runDetection, reset });

// ---- The tree, to the detail shown ----
// The splits shown: the most important ones, with the splits they are within
const revealed = computed(() => {
    const ids = new Set();
    for (const node of splits.value.slice(0, position.value)) {
        for (let id = node.id; id !== null && !ids.has(id); id = nodes.value[id].parent) ids.add(id);
    }
    return ids;
});

// The senses shown, in order: the nodes the shown splits end in
const shown = computed(() => {
    const out = [];
    const walk = (node) => {
        if (revealed.value.has(node.id)) node.children.forEach((c) => walk(nodes.value[c]));
        else out.push(node);
    };
    if (root.value) walk(root.value);
    return out;
});

const numberOf = computed(() => Object.fromEntries(shown.value.map((node, i) => [node.id, i + 1])));

// The tree as rows: a shown split above the senses it splits into, indented by depth
const rows = computed(() => {
    const out = [];
    const walk = (node, depth) => {
        if (revealed.value.has(node.id)) {
            out.push({ kind: "split", node, depth });
            node.children.forEach((c) => walk(nodes.value[c], depth + 1));
        } else {
            out.push({ kind: "sense", node, depth });
        }
    };
    if (root.value) walk(root.value, 0);
    return out;
});

const detailText = computed(() => {
    if (!position.value) return t("wordSenses.detailNone");
    if (position.value === 1) return t("wordSenses.detailMain");
    const faintest = splits.value[position.value - 1];
    return t("wordSenses.detailText", { senses: shown.value.length, importance: percent(faintest.split.relative) });
});

function sensesUnder(node) {
    return revealed.value.has(node.id) ? node.children.flatMap((c) => sensesUnder(nodes.value[c])) : [numberOf.value[node.id]];
}

function senseRange(node) {
    const numbers = sensesUnder(node);
    const first = Math.min(...numbers);
    const last = Math.max(...numbers);
    return first === last ? `${first}` : `${first}-${last}`;
}

function seedText(seed) {
    return wordText(seed);
}

// A word of a group as it reads: those next to the query with it
function wordText(w, group = w.group) {
    if (group === 1) return `${w.word} ${queryWord.value}`;
    if (group === 2) return `${queryWord.value} ${w.word}`;
    return w.word;
}

// A sense's label: its strongest marker words, else the words next to the query, else what it lacks
function label(node) {
    const words = node.words.slice(0, 4).map((w) => w.word);
    if (words.length < 2) {
        words.push(...node.before.slice(0, 1).map((w) => wordText(w, 1)));
        words.push(...node.after.slice(0, 1).map((w) => wordText(w, 2)));
    }
    if (words.length) return words.join(", ");
    if (node.rarer?.length) return t("wordSenses.without", { words: node.rarer.slice(0, 3).map((w) => wordText(w)).join(", ") });
    return queryWord.value;
}

function notSplitReason(node) {
    if (node.children.length) {
        return t("wordSenses.finerSplit", { importance: percent(node.split.relative), seed: seedText(node.split.seed) });
    }
    if (node.reason === "no_gain" && node.split) return t("wordSenses.notSplitWorse", { z: node.split.z.toFixed(1) });
    if (node.reason === "faint") return t("wordSenses.notSplitFaint", { finest: percent(result.value.finest) });
    return t("wordSenses.notSplitInseparable");
}

function percent(share) {
    return `${Math.round(share * 100)}%`;
}

// Same hues as the usage patterns, so colors read alike across tabs
const hues = [205, 30, 145, 280, 0, 90, 165, 235, 50, 315, 120, 260, 15, 60, 200, 320];
function senseColor(number, alpha) {
    return `hsla(${hues[((number || 1) - 1) % hues.length]}, 55%, 50%, ${alpha})`;
}

function scrollToSense(node) {
    document.getElementById(`sense-card-${node.id}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---- Charts: each sense's curve is its node's ----
function curve(node) {
    const c = time.value[view.value];
    return { value: c.value[node.id], low: c.low[node.id], high: c.high[node.id] };
}

function formatValue(v) {
    if (v === null || v === undefined || Number.isNaN(v)) return "-";
    if (view.value === "share") return `${Math.round(Math.max(0, v) * 100)}%`;
    const x = Math.max(0, v);
    return t("wordSenses.perMillion", { n: x < 10 ? x.toFixed(2) : x.toFixed(1) });
}

const nYears = computed(() => (time.value ? time.value.end - time.value.start + 1 : 0));

function xOf(i) {
    return (i / Math.max(1, nYears.value - 1)) * chartWidth;
}

// Stacked: shares renormalized to fill each year (the smoothed ones need not sum to one exactly), rates as they are
const overviewStack = computed(() => {
    if (!time.value) return { scaled: null, max: 1 };
    const n = nYears.value;
    const values = shown.value.map((node) => curve(node).value.map((v) => Math.max(0, v)));
    const totals = new Array(n).fill(0);
    values.forEach((vals) => vals.forEach((v, i) => (totals[i] += v)));
    const scaled = view.value === "share"
        ? values.map((vals) => vals.map((v, i) => (totals[i] > 0 ? v / totals[i] : 0)))
        : values;
    const max = view.value === "share" ? 1 : Math.max(1e-9, ...totals);
    return { scaled, max };
});

const overviewMax = computed(() => overviewStack.value.max);

const overviewLayers = computed(() => {
    const { scaled, max } = overviewStack.value;
    if (!scaled) return [];
    const n = nYears.value;
    const below = new Array(n).fill(0);
    return shown.value.map((node, k) => {
        const bottom = below.map((b) => overviewHeight - (b / max) * overviewHeight);
        scaled[k].forEach((v, i) => (below[i] += v));
        const top = below.map((b) => overviewHeight - (b / max) * overviewHeight);
        let d = `M ${xOf(0)} ${bottom[0].toFixed(2)} `;
        for (let i = 0; i < n; i++) d += `L ${xOf(i).toFixed(2)} ${bottom[i].toFixed(2)} `;
        for (let i = n - 1; i >= 0; i--) d += `L ${xOf(i).toFixed(2)} ${top[i].toFixed(2)} `;
        return { id: node.id, number: k + 1, path: `${d}Z` };
    });
});

function cardMax(node) {
    const c = curve(node);
    return Math.max(1e-9, ...c.high.map((v) => (Number.isNaN(v) ? 0 : v)), ...c.value);
}

function yOf(v, max) {
    return cardHeight - (Math.max(0, Number.isNaN(v) ? 0 : v) / max) * (cardHeight - 2);
}

function linePath(node) {
    const c = curve(node);
    const max = cardMax(node);
    return c.value.map((v, i) => `${i ? "L" : "M"} ${xOf(i).toFixed(2)} ${yOf(v, max).toFixed(2)}`).join(" ");
}

function bandPath(node) {
    const c = curve(node);
    const max = cardMax(node);
    const n = c.value.length;
    let d = "";
    for (let i = 0; i < n; i++) d += `${i ? "L" : "M"} ${xOf(i).toFixed(2)} ${yOf(c.high[i], max).toFixed(2)} `;
    for (let i = n - 1; i >= 0; i--) d += `L ${xOf(i).toFixed(2)} ${yOf(c.low[i], max).toFixed(2)} `;
    return `${d}Z`;
}

function onHover(event) {
    const rect = event.currentTarget.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    hoverIndex.value = Math.round(frac * (nYears.value - 1));
}

// Round-number year ticks (about 6 to 9) across the span
const yearTicks = computed(() => {
    if (!time.value) return [];
    const { start, end } = time.value;
    const span = end - start;
    if (span <= 0) return [{ year: start, pct: 0 }];
    const steps = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500];
    const step = steps.find((s) => span / s <= 9) || 1000;
    const ticks = [];
    for (let y = Math.ceil(start / step) * step; y <= end; y += step) ticks.push({ year: y, pct: ((y - start) / span) * 100 });
    return ticks;
});

function tickShift(pct) {
    if (pct < 5) return "translateX(0)";
    if (pct > 95) return "translateX(-100%)";
    return "translateX(-50%)";
}

// ---- Examples: the sentences most clearly in a sense ----
const modal = ref({ groupName: "", signature: [], passages: [], loading: false, hasMore: false, examples: [], offset: 0 });
let modalInstance = null;
let examplesToken = 0;

function evidenceText(e) {
    return wordText(e);
}

function openExamples(node) {
    if (!node) return;
    modal.value = {
        groupName: shown.value.length > 1
            ? `${queryWord.value} · ${t("wordSenses.senseN", { n: numberOf.value[node.id] })}: ${label(node)}`
            : queryWord.value,
        signature: node.words.slice(0, 12).map((w) => ({ word: w.word, z: null })),
        passages: [],
        loading: true,
        hasMore: false,
        examples: node.examples,
        offset: 0,
    };
    const el = document.getElementById("distinctive-passages-modal");
    if (el) {
        if (!modalInstance) modalInstance = new Modal(el);
        modalInstance.show();
    }
    fetchExamples(false);
}

async function fetchExamples(append) {
    const myToken = ++examplesToken;
    const page = modal.value.examples.slice(modal.value.offset, modal.value.offset + PAGE);
    modal.value.loading = true;
    try {
        const response = await $http.get(`${$dbUrl}/scripts/get_sense_passages.py`, {
            params: { ...paramsFilter(formData.value), hits: page.map((e) => e.hit).join(",") },
        });
        if (myToken !== examplesToken) return;
        const byHit = Object.fromEntries(page.map((e) => [e.hit, e]));
        const passages = (response.data?.passages || []).map((p) => {
            const e = byHit[p.hit];
            const note = e
                ? `${t("wordSenses.inThisSense", { p: percent(e.p) })} · ${t("wordSenses.decidedBy")}: ${e.evidence.map(evidenceText).join(", ")}`
                : "";
            return { ...p, note };
        });
        modal.value.passages = append ? modal.value.passages.concat(passages) : passages;
        modal.value.hasMore = modal.value.offset + PAGE < modal.value.examples.length;
    } catch (error) {
        debug({ $options: { name: "word-senses" } }, error);
    } finally {
        if (myToken === examplesToken) modal.value.loading = false;
    }
}

function loadMoreExamples() {
    modal.value.offset += PAGE;
    fetchExamples(true);
}

// ---- A sense's words: its marker words, those markedly more frequent in it than in the sense it was split from ----
const SHOWN = 12;
const WordList = defineComponent({
    props: { node: Object, queryWord: String, againstCorpus: Boolean },
    setup(props) {
        const open = ref(false);
        const ratio = (w) => {
            const r = props.againstCorpus ? w.lift : w.ratio;
            return r >= 10 ? r.toFixed(0) : r.toFixed(1);
        };
        const chip = (w, text, extra = "") =>
            h("span", {
                class: ["word-chip", extra],
                title: t(props.againstCorpus ? "wordSenses.chipTitleCorpus" : "wordSenses.chipTitle", {
                    n: Math.round(w.n), ratio: ratio(w),
                }),
            }, [text, h("span", { class: "chip-ratio" }, ` ${String.fromCharCode(215)}${ratio(w)}`)]);
        return () => {
            const lines = [];
            // Words that go together in the sense's sentences, around hub words, then the others
            const groups = props.node.groups || [];
            const grouped = new Set(groups.flatMap((g) => [g.hub, ...g.members]));
            for (const g of groups) {
                const hub = props.node.words[g.hub];
                lines.push(h("div", { class: "word-group mb-1", title: t("wordSenses.groupTitle", { hub: hub.word }) }, [
                    chip(hub, hub.word, "hub-chip"),
                    ...g.members.map((m) => chip(props.node.words[m], props.node.words[m].word)),
                ]));
            }
            const words = props.node.words.filter((_, i) => !grouped.has(i));
            const listed = open.value ? words : words.slice(0, SHOWN);
            if (groups.length && words.length) lines.push(h("span", { class: "small text-muted me-1" }, `${t("wordSenses.otherWords")}:`));
            if (words.length) {
                const items = listed.map((w) => chip(w, w.word));
                if (words.length > SHOWN) {
                    items.push(h("button", {
                        type: "button", class: "btn btn-sm btn-link p-0 ms-1 align-baseline",
                        "aria-expanded": open.value, onClick: () => (open.value = !open.value),
                    }, open.value ? t("wordSenses.fewerWords") : t("wordSenses.moreWords", { n: words.length - SHOWN })));
                }
                lines.push(h("div", { class: "mb-1" }, items));
            } else if (!groups.length) {
                lines.push(h("div", { class: "small text-muted mb-1" }, t("wordSenses.noWords")));
            }
            const near = [];
            if (props.node.before.length) {
                near.push(h("span", { class: "small text-muted me-1" }, `${t("wordSenses.before")}:`));
                near.push(...props.node.before.slice(0, 6).map((w) => chip(w, `${w.word} ${props.queryWord}`)));
            }
            if (props.node.after.length) {
                near.push(h("span", { class: "small text-muted ms-2 me-1" }, `${t("wordSenses.after")}:`));
                near.push(...props.node.after.slice(0, 6).map((w) => chip(w, `${props.queryWord} ${w.word}`)));
            }
            if (near.length) lines.push(h("div", {}, near));
            const rarer = props.againstCorpus ? [] : props.node.rarer || [];
            if (rarer.length) {
                const times = (w) => {
                    const r = 1 / Math.max(w.ratio, 1e-9);
                    return r >= 10 ? r.toFixed(0) : r.toFixed(1);
                };
                const text = (w) => (w.group === 1 ? `${w.word} ${props.queryWord}` : w.group === 2 ? `${props.queryWord} ${w.word}` : w.word);
                lines.push(h("div", { class: "mt-1" }, [
                    h("span", { class: "small text-muted me-1" }, `${t("wordSenses.rarer")}:`),
                    ...rarer.slice(0, 8).map((w) => h("span", {
                        class: "word-chip rarer-chip",
                        title: t("wordSenses.chipTitleRarer", { n: Math.round(w.n), ratio: times(w) }),
                    }, [text(w), h("span", { class: "chip-ratio" }, ` ${String.fromCharCode(247)}${times(w)}`)])),
                ]));
            }
            return h("div", {}, lines);
        };
    },
});
</script>

<style lang="scss" scoped>
@use "../../assets/styles/theme.module.scss" as theme;

.sense-header {
    background-color: #fafafa;
    border-bottom: 1px solid #eee;
}

:deep(.word-chip) {
    display: inline-block;
    padding: 0.1rem 0.45rem;
    margin: 0.15rem 0.25rem 0.15rem 0;
    background-color: rgba(theme.$link-color, 0.08);
    border: 1px solid rgba(theme.$link-color, 0.2);
    border-radius: 0.5rem;
    color: theme.$link-color;
    font-size: 0.85rem;
}

:deep(.hub-chip) {
    font-weight: 600;
}

:deep(.word-group) {
    border-left: 2px solid rgba(theme.$link-color, 0.25);
    padding-left: 0.4rem;
}

:deep(.rarer-chip) {
    background-color: #f8f9fa;
    border: 1px dashed #adb5bd;
    color: #495057;
}

:deep(.chip-ratio) {
    color: #6c757d;
    font-size: 0.75rem;
}

.chart-frame {
    position: relative;
}

.y-max {
    position: absolute;
    top: 2px;
    left: 4px;
    pointer-events: none;
}

.overview-chart {
    width: 100%;
    height: 140px;
    display: block;
    background-color: #fafafa;
    border: 1px solid #eee;
}

.sense-chart {
    width: 100%;
    height: 60px;
    display: block;
    background-color: #fafafa;
    border: 1px solid #eee;
    border-radius: 0.25rem;
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

.legend-chip {
    border: 1px solid;
    background-color: #fff;
    padding: 0.15rem 0.5rem;
    font-size: 0.75rem;
    line-height: 1.3;
}

.legend-swatch {
    display: inline-block;
    width: 0.65rem;
    height: 0.65rem;
    border-radius: 0.15rem;
    margin-right: 0.3rem;
    vertical-align: middle;
}

.one-work {
    font-size: 0.8rem;
    font-weight: normal;
    color: #8a5300;
    background-color: #fff4e0;
    border-radius: 0.25rem;
    padding: 0 0.3rem;
}

.split-row {
    color: #495057;
}

.detail-range {
    max-width: 28rem;
}

.split-mark {
    color: #adb5bd;
    margin-right: 0.25rem;
}
</style>
