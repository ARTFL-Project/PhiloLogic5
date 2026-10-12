<template>
    <div>
        <div v-if="loading && !result" class="text-center py-5">
            <progress-spinner :lg="true" :message="$t('wordSenses.detecting')" />
        </div>
        <div v-if="failed" class="alert alert-warning mx-2 my-3" role="alert">{{ $t("wordSenses.failed") }}</div>

        <div v-if="result && result.n_sentences" class="mx-2 my-3">
            <div class="card shadow-sm p-3 mb-3">
                <p class="mb-2 fw-semibold">
                    {{ $t("senseOverview.summary", {
                        word: queryWord,
                        sentences: result.n_sentences.toLocaleString(),
                        works: result.n_works.toLocaleString(),
                    }) }}
                    <InfoButton :label="$t('senseOverview.howToRead')">
                        <span class="d-block mb-1">{{ $t("senseOverview.treeExplained") }}</span>
                        <span class="d-block mb-1">{{ $t("senseOverview.timeExplained") }}</span>
                        <span class="d-block">{{ $t("senseOverview.periodsExplained") }}</span>
                    </InfoButton>
                </p>

                <div v-if="splits.length" class="d-flex align-items-center gap-2 flex-wrap mb-1">
                    <label for="sense-overview-detail" class="small fw-semibold mb-0">{{ $t("wordSenses.detail") }}</label>
                    <span class="small text-muted" aria-hidden="true">{{ $t("wordSenses.coarser") }}</span>
                    <input id="sense-overview-detail" v-model.number="position" type="range" class="form-range detail-range"
                        min="0" :max="splits.length" step="1" :aria-valuetext="detailText" />
                    <span class="small text-muted" aria-hidden="true">{{ $t("wordSenses.finer") }}</span>
                    <button type="button" class="btn btn-sm btn-link p-0" :disabled="position === defaultPosition"
                        @click="position = defaultPosition">{{ $t("senseTree.firstView") }}</button>
                    <span class="small text-muted ms-2" aria-live="polite">{{ detailText }}</span>
                    <span class="ms-auto d-flex align-items-center gap-1">
                        <label for="sense-overview-find" class="small fw-semibold mb-0">{{ $t("wordSenses.findWord") }}</label>
                        <input id="sense-overview-find" v-model="search" type="search" class="form-control form-control-sm find-input"
                            list="sense-overview-words" :placeholder="$t('wordSenses.findPlaceholder')" />
                        <datalist id="sense-overview-words">
                            <option v-for="w in searchable" :key="w" :value="w" />
                        </datalist>
                    </span>
                </div>
                <p v-if="search.trim()" class="small mb-1" aria-live="polite">{{ foundText }}</p>
                <p v-if="!time" class="small text-muted mb-1">{{ $t("wordSenses.noTime") }}</p>

                <svg :viewBox="`0 0 ${W} ${tree.height}`" class="sense-overview" role="group"
                    :aria-label="$t('senseOverview.chart', { word: queryWord })">
                    <!-- The years, above the senses' rows -->
                    <g v-if="time" class="axis" aria-hidden="true">
                        <template v-for="year in yearTicks" :key="year">
                            <line :x1="xYear(year)" :x2="xYear(year)" :y1="TOP - 8" :y2="tree.height - 6" />
                            <text :x="xYear(year)" :y="TOP - 12" text-anchor="middle">{{ year }}</text>
                        </template>
                    </g>
                    <!-- The tree: each sense's branch, as thick as its share, forking where it is split -->
                    <path :d="`M 4 ${tree.rootY} H ${tree.rootX}`" class="branch" :stroke-width="thickness(1)"
                        stroke="hsla(0, 0%, 62%, 1)" aria-hidden="true" />
                    <path v-for="b in tree.segments" :key="`branch-${b.node.id}`" class="branch"
                        :d="`M ${b.x0} ${b.y0} V ${b.y1} H ${b.x1}`" :stroke="senseColor(b.node.sense, 1)"
                        :stroke-width="thickness(b.node.share)" aria-hidden="true" @click="select(b.node)">
                        <title>{{ title(b.node) }}</title>
                    </path>
                    <!-- Each sense shown on its row: its name, then its share of the sentences over the years, cut into
                         its periods, each with its strongest new word -->
                    <g v-for="row in tree.rows" :key="`row-${row.node.id}`" class="sense-row" role="button" tabindex="0"
                        :class="{ picked: selected && selected.id === row.node.id, found: isFound(row.node) }"
                        :aria-label="title(row.node)" :aria-pressed="Boolean(selected && selected.id === row.node.id)"
                        @click="select(row.node)" @keydown.enter.prevent="select(row.node)"
                        @keydown.space.prevent="select(row.node)">
                        <rect class="row-hit" :x="TREE_RIGHT" :y="row.y - ROW / 2 + 2" :width="W - TREE_RIGHT - 4"
                            :height="ROW - 4" rx="4" />
                        <text :x="TREE_RIGHT + 8" :y="row.y + 4" :fill="senseTextColor(row.node.sense, shades)">
                            <tspan font-weight="600">{{ row.node.sense ? $t("wordSenses.senseN", { n: row.node.sense })
                                : $t("senseTree.restShort") }}</tspan>
                            <tspan fill="#495057"> · {{ percent(row.node.share) }}{{ row.node.sense ? ` · ${shortLabel(row.node)}` : "" }}</tspan>
                        </text>
                        <template v-if="time">
                            <path :d="bandPath(row.node, row.y)" :fill="senseColor(row.node.sense, 0.55)"
                                :stroke="senseColor(row.node.sense, 1)" stroke-width="0.8" />
                            <template v-for="mark in periodMarks(row.node)" :key="mark.key">
                                <line v-if="mark.tick !== null" class="period-tick" :x1="mark.tick" :x2="mark.tick"
                                    :y1="row.y - BAND + 2" :y2="row.y + 2" />
                                <text v-if="mark.word" class="period-word" :class="{ start: mark.start }" :x="mark.x"
                                    :y="row.y + 15" text-anchor="middle">{{ mark.word }}</text>
                            </template>
                            <text v-if="row.node.sense && periodsById[row.node.id] === undefined" class="period-word"
                                :x="T1" :y="row.y + 15" text-anchor="end">…</text>
                        </template>
                    </g>
                </svg>
            </div>

            <!-- A sense picked: its words, and its periods in full -->
            <div v-if="selected" class="card shadow-sm p-3 mb-3 selected-sense"
                :style="{ borderLeft: `4px solid ${senseColor(selected.sense, 1)}` }">
                <h3 class="h6 mb-2">
                    <span class="legend-swatch" :style="{ backgroundColor: senseColor(selected.sense, 1) }"></span>
                    <span :style="{ color: senseTextColor(selected.sense, shades) }">{{ selected.sense
                        ? $t("wordSenses.senseN", { n: selected.sense }) : $t("senseTree.restShort") }}</span>
                    {{ " " }}<span class="fw-normal text-muted">&middot; {{ $t("wordSenses.shareOfSentences",
                        { share: percent(selected.share) }) }}</span>
                </h3>
                <p v-if="selected.words.length" class="mb-1">
                    <button v-for="w in selected.words.slice(0, 12)" :key="w.word" type="button" class="word-chip"
                        :disabled="!w.hits?.length" @click="openSenseWord(selected, w, 0)">{{ w.word }}</button>
                </p>
                <p v-if="selected.before.length || selected.after.length" class="mb-2">
                    <button v-for="w in selected.before.slice(0, 4)" :key="`b-${w.word}`" type="button" class="word-chip"
                        :disabled="!w.hits?.length" @click="openSenseWord(selected, w, 1)">{{ wordText(w, 1) }}</button>
                    <button v-for="w in selected.after.slice(0, 4)" :key="`a-${w.word}`" type="button" class="word-chip"
                        :disabled="!w.hits?.length" @click="openSenseWord(selected, w, 2)">{{ wordText(w, 2) }}</button>
                </p>
                <p v-if="!selected.sense" class="small text-muted mb-0">{{ $t("senseTree.restExplained") }}</p>
                <template v-else-if="time">
                    <p v-if="periodsById[selected.id] === undefined" class="small text-muted mb-0">{{ $t("senseOverview.periodsLoading") }}</p>
                    <p v-else-if="periodsById[selected.id] === null" class="small text-muted mb-0">{{ $t("senseOverview.noPeriods") }}</p>
                    <template v-else>
                        <p class="small text-muted mb-1">{{ periodsById[selected.id].periods.length > 1
                            ? $t("sensePeriods.nPeriods", { n: periodsById[selected.id].periods.length })
                            : $t("sensePeriods.onePeriod") }}</p>
                        <PeriodColumns :periods="periodsById[selected.id].periods" :sense="selected.sense" :shades="shades"
                            :query-word="queryWord" @word="(p, w) => openPeriodWord(selected, p, w)"
                            @work="(p, w) => openPeriodWork(selected, p, w)" />
                    </template>
                </template>
            </div>
        </div>

        <div v-else-if="result" class="text-center py-5 text-muted">{{ $t("wordSenses.noResults") }}</div>

        <DistinctivePassagesModal :group-name="modal.groupName" :signature="[]" :passages="modal.passages"
            :loading="modal.loading" :has-more="false" :ordering-note="modal.note" />
    </div>
</template>

<script setup>
import { Modal } from "bootstrap";
import { storeToRefs } from "pinia";
import { computed, inject, onMounted, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import { useMainStore } from "../../stores/main";
import { debug, paramsFilter } from "../../utils.js";
import DistinctivePassagesModal from "../DistinctivePassagesModal.vue";
import ProgressSpinner from "../ProgressSpinner";
import InfoButton from "./InfoButton.vue";
import PeriodColumns, { periodWord, periodYears } from "./PeriodColumns.vue";
import { senseColor as colorOf, senseShades, senseTextColor } from "./senseColors.js";
import { findWord, inSense, searchableWords } from "./senseSearch.js";

const emit = defineEmits(["filterList", "stopwordsMissing"]);
const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const store = useMainStore();
const { formData } = storeToRefs(store);
const { t } = useI18n();

const loading = ref(false);
const failed = ref(false);
const result = ref(null);
const position = ref(0); // how many distinctions are shown, the most important first
const selected = ref(null);
const search = ref("");
const periodsById = ref({}); // each sense's periods once fetched (null: too few works to cut)
let fetchToken = 0;

// The tree on the left, the senses' names in the middle, their years on the right
const W = 1300;
const LEFT = 20;
const TREE_RIGHT = 300;
const T0 = 640;
const T1 = W - 12;
const TOP = 34;
const ROW = 58;
const BAND = 34;

const queryWord = computed(() => (formData.value.q || "").replace(/lemma:/g, "").replace(/"/g, "").trim());
const nodes = computed(() => result.value?.nodes || []);
const root = computed(() => nodes.value[0]);
const time = computed(() => result.value?.time || null);

const splits = computed(() =>
    nodes.value.filter((n) => n.children.length).sort((a, b) => b.split.relative - a.split.relative || a.id - b.id)
);
const defaultPosition = computed(() => splits.value.filter((n) => n.split.relative >= (result.value?.detail ?? 1)).length);

function runDetection() {
    const myToken = ++fetchToken;
    loading.value = true;
    failed.value = false;
    result.value = null;
    selected.value = null;
    periodsById.value = {};
    $http
        .get(`${$dbUrl}/scripts/get_word_senses.py`, { params: paramsFilter(formData.value) })
        .then((response) => {
            if (myToken !== fetchToken) return;
            result.value = response.data;
            emit("filterList", response.data?.filter_list || []);
            emit("stopwordsMissing", Boolean(response.data?.stopwords_missing));
            position.value = defaultPosition.value;
        })
        .catch((error) => {
            if (myToken !== fetchToken) return;
            failed.value = true;
            debug({ $options: { name: "sense-overview" } }, error);
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

onMounted(() => {
    if (!result.value && !loading.value) runDetection();
});

// ---- The periods of the senses shown: fetched once each, after the tree is drawn ----
let periodsTimer = null;
let periodsToken = 0;

function fetchPeriods() {
    if (!time.value) return;
    const wanted = tree.value.rows.map((row) => row.node).filter((n) => n.sense && periodsById.value[n.id] === undefined);
    if (!wanted.length) return;
    const myToken = ++periodsToken;
    const query = fetchToken;
    $http
        .get(`${$dbUrl}/scripts/get_sense_periods.py`, {
            params: { ...paramsFilter(formData.value), nodes: wanted.map((n) => n.id).join(",") },
        })
        .then((response) => {
            if (query !== fetchToken) return;
            const got = { ...periodsById.value };
            for (const n of wanted) got[n.id] = null; // asked for, not cut: too few works
            for (const s of response.data?.senses || []) got[s.id] = s;
            periodsById.value = got;
        })
        .catch((error) => {
            if (myToken === periodsToken) debug({ $options: { name: "sense-overview" } }, error);
        });
}

// When the detail changes, the senses newly shown are fetched, once the slider rests
watch([result, position], () => {
    clearTimeout(periodsTimer);
    periodsTimer = setTimeout(fetchPeriods, result.value && Object.keys(periodsById.value).length ? 300 : 0);
});

// ---- The tree, as the sense tree tab draws it ----
const heightOf = computed(() => {
    const out = {};
    const walk = (node) => {
        let h = node.children.length ? node.split.relative : 0;
        for (const c of node.children) h = Math.max(h, walk(nodes.value[c]));
        out[node.id] = h;
        return h;
    };
    if (root.value) walk(root.value);
    return out;
});

const lowest = computed(() => Math.min(result.value?.finest ?? 0.1, ...splits.value.map((n) => n.split.relative)));

function xOf(r) {
    const span = Math.log(1 / lowest.value) || 1;
    return LEFT + (Math.log(1 / Math.max(r, lowest.value)) / span) * (TREE_RIGHT - LEFT - 20);
}

function thickness(share) {
    return 1.5 + 12 * share;
}

function firstRank(node) {
    if (node.children.length) return Math.min(...node.children.map((c) => firstRank(nodes.value[c])));
    return node.sense === null ? Infinity : node.rank;
}

const revealed = computed(() => {
    const ids = new Set();
    for (const node of splits.value.slice(0, position.value)) {
        for (let id = node.id; id !== null && !ids.has(id); id = nodes.value[id].parent) ids.add(id);
    }
    return ids;
});

const tree = computed(() => {
    const rows = [];
    const segments = [];
    if (!root.value) return { rows, segments, rootX: LEFT, rootY: TOP, height: TOP + ROW + 10 };
    const layout = (node) => {
        if (!revealed.value.has(node.id)) {
            const y = TOP + rows.length * ROW + ROW / 2;
            rows.push({ node, y });
            return y;
        }
        const x = xOf(heightOf.value[node.id]);
        const kids = node.children.map((c) => nodes.value[c]).sort((a, b) => firstRank(a) - firstRank(b));
        const ys = kids.map(layout);
        const y = (ys[0] + ys[ys.length - 1]) / 2;
        kids.forEach((kid, i) => {
            const x1 = revealed.value.has(kid.id) ? xOf(heightOf.value[kid.id]) : TREE_RIGHT;
            segments.push({ node: kid, x0: x, y0: y, y1: ys[i], x1 });
        });
        return y;
    };
    const rootY = layout(root.value);
    const rootX = revealed.value.has(root.value.id) ? xOf(heightOf.value[root.value.id]) : TREE_RIGHT;
    return { rows, segments, rootX, rootY, height: TOP + rows.length * ROW + 10 };
});

const detailText = computed(() => {
    if (!position.value) return t("wordSenses.detailNone");
    const senses = tree.value.rows.filter((row) => row.node.sense !== null).length;
    return senses === 1 ? t("wordSenses.oneSense") : t("wordSenses.nSenses", { n: senses });
});

function select(node) {
    selected.value = selected.value && selected.value.id === node.id ? null : node;
}

// ---- The years: each sense's share of the sentences, on one axis for all ----
function xYear(year) {
    const span = Math.max(1, time.value.end - time.value.start);
    return T0 + ((year - time.value.start) / span) * (T1 - T0);
}

const yearTicks = computed(() => {
    if (!time.value) return [];
    const span = time.value.end - time.value.start;
    const step = [10, 25, 50, 100, 200].find((s) => span / s <= 8) || 500;
    const out = [];
    for (let y = Math.ceil(time.value.start / step) * step; y <= time.value.end; y += step) out.push(y);
    return out;
});

// The same height for the same share in every row, the highest share shown filling the band
const bandMax = computed(() => {
    if (!time.value) return 1;
    let max = 1e-9;
    for (const row of tree.value.rows) for (const v of time.value.share.value[row.node.id] || []) max = Math.max(max, v);
    return max;
});

function bandPath(node, y) {
    const values = time.value.share.value[node.id] || [];
    const base = y + 2;
    if (!values.length) return "";
    const pts = values.map((v, i) => `${xYear(time.value.start + i).toFixed(1)} ${(base - (Math.max(0, v) / bandMax.value) * BAND).toFixed(1)}`);
    return `M ${xYear(time.value.start).toFixed(1)} ${base} L ${pts.join(" L ")} L ${xYear(time.value.end).toFixed(1)} ${base} Z`;
}

// Its periods' edges, and the strongest new word of each where it fits (the first: what set it apart)
function periodMarks(node) {
    const entry = periodsById.value[node.id];
    if (!entry || entry.periods.length < 2) return [];
    const out = [];
    let lastRight = -Infinity;
    entry.periods.forEach((p, i) => {
        const tick = i === 0 ? null : xYear((entry.periods[i - 1].end + p.start) / 2);
        const first = (i === 0 ? p.characteristic : p.new)?.[0];
        const word = first ? periodWord(first, queryWord.value) : "";
        const x = xYear((p.start + p.end) / 2);
        const half = (word.length * 5.4) / 2;
        const fits = word && x - half > lastRight + 4;
        if (fits) lastRight = x + half;
        out.push({ key: `${p.start}`, tick, x, word: fits ? word : "", start: i === 0 });
    });
    return out;
}

// ---- A word looked for, as in the sense tree tab ----
const searchable = computed(() => searchableWords(nodes.value));
const found = computed(() => findWord(nodes.value, search.value, queryWord.value));

function isFound(node) {
    return Boolean(search.value.trim() && node.sense && inSense(node, search.value, queryWord.value));
}

const foundText = computed(() => {
    const term = search.value.trim();
    if (!found.value.length) return t("wordSenses.notFound", { word: term });
    const list = found.value.map((f) => t("wordSenses.senseN", { n: f.sense }));
    return t("wordSenses.foundIn", { word: term, senses: list.join(", ") });
});

// ---- Words and senses ----
function wordText(w, group = w.group) {
    return periodWord({ ...w, group }, queryWord.value);
}

function label(node) {
    const words = node.words.slice(0, 3).map((w) => w.word);
    if (words.length < 2) {
        words.push(...node.before.slice(0, 1).map((w) => wordText(w, 1)));
        words.push(...node.after.slice(0, 1).map((w) => wordText(w, 2)));
    }
    return words.join(", ") || queryWord.value;
}

function shortLabel(node) {
    const text = label(node);
    return text.length > 34 ? `${text.slice(0, 32)}…` : text;
}

function title(node) {
    const name = node.sense ? t("wordSenses.senseN", { n: node.sense }) : t("senseTree.restShort");
    return `${name}: ${label(node)} · ${t("wordSenses.shareOfSentences", { share: percent(node.share) })}`;
}

function percent(share) {
    return `${Math.round(share * 100)}%`;
}

const shades = computed(() => senseShades(nodes.value));
function senseColor(sense, alpha) {
    return colorOf(sense, alpha, shades.value);
}

// ---- Reading passages ----
const modal = ref({ groupName: "", passages: [], loading: false, note: "" });
let modalInstance = null;

async function openPassages(groupName, hits, note) {
    modal.value = { groupName, passages: [], loading: true, note };
    const el = document.getElementById("distinctive-passages-modal");
    if (el) {
        if (!modalInstance) modalInstance = new Modal(el);
        modalInstance.show();
    }
    try {
        const response = await $http.get(`${$dbUrl}/scripts/get_sense_passages.py`, {
            params: { ...paramsFilter(formData.value), hits: hits.join(",") },
        });
        modal.value.passages = response.data?.passages || [];
    } catch (error) {
        debug({ $options: { name: "sense-overview" } }, error);
    } finally {
        modal.value.loading = false;
    }
}

function openSenseWord(node, w, group) {
    if (!w.hits?.length) return;
    const text = wordText(w, group);
    openPassages(`${queryWord.value} · ${t("wordSenses.senseN", { n: node.sense })} · “${text}”`, w.hits,
        t("wordSenses.wordExamplesNote", { word: text }));
}

function openPeriodWord(node, p, w) {
    if (!w.hits?.length) return;
    const name = `${t("wordSenses.senseN", { n: node.sense })} · ${periodYears(p)}`;
    openPassages(`${queryWord.value} · ${name} · “${wordText(w)}”`, w.hits,
        t("sensePeriods.wordNote", { word: wordText(w), years: periodYears(p) }));
}

function openPeriodWork(node, p, w) {
    const name = `${t("wordSenses.senseN", { n: node.sense })} · ${periodYears(p)}`;
    const work = [w.author, w.title].filter(Boolean).join(", ");
    openPassages(`${queryWord.value} · ${name} · ${work}`, w.hits, t("sensePeriods.workNote", { years: periodYears(p) }));
}
</script>

<style scoped>
.sense-overview {
    width: 100%;
    height: auto;
}

.detail-range {
    max-width: 16rem;
}

.find-input {
    width: 12rem;
}

.branch {
    fill: none;
    stroke-linecap: round;
    cursor: pointer;
}

.axis line {
    stroke: #e9ecef;
}

.axis text {
    font-size: 10px;
    fill: #868e96;
}

.sense-row {
    cursor: pointer;
}

.row-hit {
    fill: transparent;
}

.sense-row:hover .row-hit,
.sense-row:focus-visible .row-hit {
    fill: rgba(13, 110, 253, 0.05);
}

.sense-row.picked .row-hit {
    fill: rgba(13, 110, 253, 0.1);
}

.sense-row.found .row-hit {
    fill: rgba(255, 193, 7, 0.22);
}

.sense-row:focus {
    outline: none;
}

.sense-row text {
    font-size: 13px;
}

.period-tick {
    stroke: #343a40;
    stroke-width: 1;
    stroke-dasharray: 2 2;
}

.sense-row text.period-word {
    font-size: 10px;
    fill: #343a40;
}

.sense-row text.period-word.start {
    font-style: italic;
    fill: #6c757d;
}

.legend-swatch {
    display: inline-block;
    width: 0.75rem;
    height: 0.75rem;
    border-radius: 2px;
    margin-right: 0.3rem;
    vertical-align: -0.05rem;
}

.word-chip {
    display: inline-block;
    padding: 0.05rem 0.45rem;
    margin: 0.15rem 0.25rem 0.15rem 0;
    border: 1px solid #ced4da;
    border-radius: 0.5rem;
    background: #fff;
    font-size: 0.85rem;
}

.word-chip:disabled {
    color: #212529;
    opacity: 1;
}
</style>
