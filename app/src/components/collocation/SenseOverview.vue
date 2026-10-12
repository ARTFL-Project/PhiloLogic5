<template>
    <div>
        <div v-if="loading && !result" class="text-center py-5">
            <progress-spinner :lg="true" :message="$t('wordSenses.detecting')" />
        </div>
        <div v-if="failed" class="alert alert-warning mx-2 my-3" role="alert">{{ $t("wordSenses.failed") }}</div>

        <div v-if="result && result.n_sentences" class="mx-2 my-3">
            <div class="card shadow-sm p-3 mb-3">
                <h2 class="h6 mb-2 fw-semibold">
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
                </h2>

                <div v-if="splits.length" class="d-flex align-items-center gap-2 flex-wrap mb-1">
                    <label for="sense-overview-detail" class="small fw-semibold mb-0">{{ $t("wordSenses.detail") }}</label>
                    <span class="small text-muted" aria-hidden="true">{{ $t("wordSenses.coarser") }}</span>
                    <input id="sense-overview-detail" v-model.number="position" type="range" class="form-range detail-range"
                        min="0" :max="splits.length" step="1" :aria-valuetext="detailText" />
                    <span class="small text-muted" aria-hidden="true">{{ $t("wordSenses.finer") }}</span>
                    <button type="button" class="btn btn-sm btn-link p-0" :disabled="position === defaultPosition"
                        @click="position = defaultPosition">{{ $t("senseTree.firstView") }}</button>
                    <span class="small text-muted" aria-live="polite">{{ detailText }}</span>
                    <span class="find d-flex align-items-center gap-1 flex-wrap">
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
                <p class="visually-hidden" aria-live="polite">{{ periodsStatus }}</p>

                <!-- The senses, one on each row: the tree on the left (wide screens), each sense's name, and its share of
                     the sentences over the years, cut into its periods. On narrow screens the rows stack. -->
                <div ref="box" class="overview" :class="{ wide }" :style="{ '--tree-width': `${treeWidth}px` }">
                    <!-- Its width follows the text's font and spacing, so the words are placed again when the reader changes them -->
                    <span ref="probe" class="probe" aria-hidden="true">{{ queryWord }} 1900</span>
                    <div v-if="time" class="years-row" aria-hidden="true">
                        <div v-if="wide" class="tree-cell"></div>
                        <div v-if="wide" class="name-cell"></div>
                        <div class="years">
                            <span v-for="year in yearTicks" :key="year" class="year" :style="{ left: `${xYear(year)}px` }">{{ year }}</span>
                        </div>
                    </div>
                    <ul ref="list" class="sense-rows" :aria-label="$t('senseOverview.chart', { word: queryWord })">
                        <li v-for="row in rows" :key="row.id" class="sense-row"
                            :class="{ picked: selected && selected.id === row.id, found: isFound(row) }"
                            :style="{ '--depth': depth(row), '--sense-color': senseColor(row.sense, 1),
                                '--sense-dark': senseTextColor(row.sense, shades) }">
                            <div v-if="wide" class="tree-cell"></div>
                            <button type="button" class="sense-name" :aria-expanded="Boolean(selected && selected.id === row.id)"
                                :aria-controls="selected && selected.id === row.id ? 'sense-overview-panel' : null"
                                :aria-describedby="time ? `sense-overview-about-${row.id}` : null"
                                @click="select(row)">
                                <span class="swatch" aria-hidden="true"></span>
                                <span class="name">{{ row.sense ? $t("wordSenses.senseN", { n: row.sense }) : $t("senseTree.restShort") }}</span>
                                <span class="what"> · {{ percent(row.share) }}<template v-if="row.sense"> · {{ label(row) }}</template></span>
                            </button>
                            <div v-if="time" class="sense-time">
                                <svg v-if="timeWidth" class="band" :width="timeWidth" :height="bandHeight" aria-hidden="true">
                                    <line v-for="year in yearTicks" :key="year" class="grid" :x1="xYear(year)" :x2="xYear(year)"
                                        y1="0" :y2="bandHeight" />
                                    <path :d="bandPath(row)" class="band-area" />
                                    <line v-for="tick in periodTicks(row)" :key="tick" class="period-tick" :x1="tick" :x2="tick"
                                        y1="0" :y2="bandHeight" />
                                </svg>
                                <div class="period-words" aria-hidden="true">
                                    <span v-for="w in periodWords(row)" :key="w.key" class="period-word" :class="{ start: w.start }"
                                        :style="{ left: `${w.left}px` }">{{ w.text }}</span>
                                </div>
                                <span :id="`sense-overview-about-${row.id}`" class="visually-hidden">{{ about(row) }}</span>
                            </div>
                        </li>
                    </ul>
                    <!-- The tree, beside the rows, from their measured places (it draws what the rows' names say) -->
                    <svg v-if="wide && treeLayout" class="tree" :width="treeWidth" :height="listHeight"
                        :style="{ top: `${listTop}px` }" aria-hidden="true">
                        <path :d="`M 2 ${treeLayout.rootY} H ${treeLayout.rootX}`" class="trunk" :stroke-width="thickness(1)" />
                        <g v-for="b in treeLayout.segments" :key="`branch-${b.node.id}`">
                            <path class="branch-casing" :d="branchPath(b)" :stroke="senseTextColor(b.node.sense, shades)"
                                :stroke-width="thickness(b.node.share) + 2" />
                            <path class="branch" :d="branchPath(b)" :stroke="senseColor(b.node.sense, 1)"
                                :stroke-width="thickness(b.node.share)" @click="select(b.node)" />
                        </g>
                    </svg>
                </div>
            </div>

            <!-- A sense picked: its words, and its periods in full -->
            <section v-if="selected" id="sense-overview-panel" class="card shadow-sm p-3 mb-3 selected-sense"
                :style="{ borderLeft: `4px solid ${senseColor(selected.sense, 1)}` }" aria-labelledby="sense-overview-panel-title">
                <h3 id="sense-overview-panel-title" class="h6 mb-2">
                    <span class="legend-swatch" :style="{ backgroundColor: senseColor(selected.sense, 1) }" aria-hidden="true"></span>
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
            </section>
        </div>

        <div v-else-if="result" class="text-center py-5 text-muted">{{ $t("wordSenses.noResults") }}</div>

        <DistinctivePassagesModal :group-name="modal.groupName" :signature="[]" :passages="modal.passages"
            :loading="modal.loading" :has-more="false" :ordering-note="modal.note" />
    </div>
</template>

<script setup>
import { Modal } from "bootstrap";
import { storeToRefs } from "pinia";
import { computed, inject, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
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

// ---- The senses shown: those the detail reveals, in the tree's order ----
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

function kids(node) {
    return node.children.map((c) => nodes.value[c]).sort((a, b) => firstRank(a) - firstRank(b));
}

const rows = computed(() => {
    const out = [];
    const walk = (node) => (revealed.value.has(node.id) ? kids(node).forEach(walk) : out.push(node));
    if (root.value) walk(root.value);
    return out;
});

function depth(node) {
    return node.sense ? node.sense.split(".").length - 1 : 0;
}

const detailText = computed(() => {
    if (!position.value) return t("wordSenses.detailNone");
    const senses = rows.value.filter((n) => n.sense !== null).length;
    return senses === 1 ? t("wordSenses.oneSense") : t("wordSenses.nSenses", { n: senses });
});

function select(node) {
    selected.value = selected.value && selected.value.id === node.id ? null : node;
}

// ---- The page's measures: the drawings are made at the size they show, so their text is the page's ----
const box = ref(null);
const list = ref(null);
const probe = ref(null);
const width = ref(0);
const timeWidth = ref(0);
const centers = ref([]); // the middle of each row's name, from the top of the list
const listHeight = ref(0);
const listTop = ref(0);
const fontSize = ref(16);
// The text's font and spacing, as the reader may set them (a text spacing stylesheet): the words are placed again
const textStyle = ref({ font: "16px sans-serif", letter: 0, word: 0 });
let observer = null;
let frame = null;

// Wide enough for the tree beside the rows; narrower, the rows stack, the senses' numbers telling the tree
const wide = computed(() => width.value >= 46 * fontSize.value);
const treeWidth = computed(() => Math.round(Math.min(Math.max(width.value * 0.18, 9 * fontSize.value), 17 * fontSize.value)));
const bandHeight = computed(() => Math.round(2.2 * fontSize.value));

function measure() {
    frame = null;
    if (!box.value) return;
    const style = getComputedStyle(box.value);
    fontSize.value = parseFloat(style.fontSize) || 16;
    const font = `${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
    const letter = parseFloat(style.letterSpacing) || 0;
    const word = parseFloat(style.wordSpacing) || 0;
    const was = textStyle.value;
    if (font !== was.font || letter !== was.letter || word !== was.word) textStyle.value = { font, letter, word };
    width.value = box.value.clientWidth;
    if (!list.value) return;
    listTop.value = list.value.offsetTop;
    listHeight.value = list.value.offsetHeight;
    centers.value = [...list.value.children].map((li) => {
        const name = li.querySelector(".sense-name");
        return li.offsetTop + (name ? name.offsetTop + name.offsetHeight / 2 : li.offsetHeight / 2);
    });
    const cell = list.value.querySelector(".sense-time");
    timeWidth.value = cell ? cell.clientWidth : 0;
}

function remeasure() {
    if (!frame) frame = requestAnimationFrame(measure);
}

onMounted(() => {
    if (!result.value && !loading.value) runDetection();
    if (typeof ResizeObserver !== "undefined") observer = new ResizeObserver(remeasure);
});
onBeforeUnmount(() => {
    observer?.disconnect();
    if (frame) cancelAnimationFrame(frame);
});

// Measured again when the rows change, and whenever the box or the list change size (zoom, text spacing, the window)
watch([rows, box, list, () => periodsById.value], async () => {
    await nextTick();
    if (observer) {
        observer.disconnect();
        for (const el of [box.value, list.value, probe.value]) if (el) observer.observe(el);
    }
    measure();
});

// ---- The tree, beside the rows ----
const lowest = computed(() => Math.min(result.value?.finest ?? 0.1, ...splits.value.map((n) => n.split.relative)));
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

function xTree(r) {
    const span = Math.log(1 / lowest.value) || 1;
    return 4 + (Math.log(1 / Math.max(r, lowest.value)) / span) * (treeWidth.value - 16);
}

function thickness(share) {
    return 1.5 + 9 * share;
}

const treeLayout = computed(() => {
    if (!root.value || centers.value.length !== rows.value.length) return null;
    const yOf = new Map(rows.value.map((n, i) => [n.id, centers.value[i]]));
    const segments = [];
    const layout = (node) => {
        if (!revealed.value.has(node.id)) return yOf.get(node.id);
        const x = xTree(heightOf.value[node.id]);
        const children = kids(node);
        const ys = children.map(layout);
        children.forEach((child, i) => {
            const x1 = revealed.value.has(child.id) ? xTree(heightOf.value[child.id]) : treeWidth.value;
            segments.push({ node: child, x0: x, y0: (ys[0] + ys[ys.length - 1]) / 2, y1: ys[i], x1 });
        });
        return (ys[0] + ys[ys.length - 1]) / 2;
    };
    const rootY = layout(root.value);
    const rootX = revealed.value.has(root.value.id) ? xTree(heightOf.value[root.value.id]) : treeWidth.value;
    return { segments, rootY, rootX };
});

function branchPath(b) {
    return `M ${b.x0} ${b.y0} V ${b.y1} H ${b.x1}`;
}

// ---- The years: each sense's share of the sentences, on one axis and one scale for all ----
function xYear(year) {
    const span = Math.max(1, time.value.end - time.value.start);
    return ((year - time.value.start) / span) * timeWidth.value;
}

const yearTicks = computed(() => {
    if (!time.value || !timeWidth.value) return [];
    const span = time.value.end - time.value.start;
    const room = Math.max(1, Math.floor(timeWidth.value / (5 * fontSize.value))); // a year's label needs about 5em
    const step = [10, 25, 50, 100, 200, 500].find((s) => span / s <= room) || 1000;
    const out = [];
    for (let y = Math.ceil(time.value.start / step) * step; y <= time.value.end; y += step) out.push(y);
    return out;
});

const bandMax = computed(() => {
    if (!time.value) return 1;
    let max = 1e-9;
    for (const n of rows.value) for (const v of time.value.share.value[n.id] || []) max = Math.max(max, v);
    return max;
});

function bandPath(node) {
    const values = time.value.share.value[node.id] || [];
    if (!values.length || !timeWidth.value) return "";
    const base = bandHeight.value - 1;
    const top = 3;
    const pts = values.map((v, i) => `${xYear(time.value.start + i).toFixed(1)} ${(base - (Math.max(0, v) / bandMax.value) * (base - top)).toFixed(1)}`);
    return `M 0 ${base} L ${pts.join(" L ")} L ${timeWidth.value.toFixed(1)} ${base} Z`;
}

function periodTicks(node) {
    const entry = periodsById.value[node.id];
    if (!entry || entry.periods.length < 2) return [];
    return entry.periods.slice(1).map((p, i) => xYear((entry.periods[i].end + p.start) / 2));
}

// Under each period, its strongest new word where it fits (the first: what set it apart), at the page's size
let canvas = null;
function textWidth(words) {
    const { font, letter, word } = textStyle.value;
    const spaced = letter * [...words].length + word * (words.split(" ").length - 1);
    let ctx = null;
    try {
        canvas ||= document.createElement("canvas");
        ctx = canvas.getContext("2d");
    } catch {
        ctx = null;
    }
    if (!ctx) return words.length * fontSize.value * 0.55 + spaced;
    ctx.font = font;
    return ctx.measureText(words).width + spaced;
}

function periodWords(node) {
    const entry = periodsById.value[node.id];
    if (!entry || entry.periods.length < 2 || !timeWidth.value) return [];
    const out = [];
    let lastRight = -Infinity;
    const gap = fontSize.value * 0.6;
    entry.periods.forEach((p, i) => {
        const first = (i === 0 ? p.characteristic : p.new)?.[0];
        if (!first) return;
        const text = periodWord(first, queryWord.value);
        const w = textWidth(text);
        const center = xYear((p.start + p.end) / 2);
        const left = Math.min(Math.max(center - w / 2, 0), timeWidth.value - w);
        if (left < lastRight + gap) return;
        lastRight = left + w;
        out.push({ key: `${p.start}`, text, left, start: i === 0 });
    });
    return out;
}

// What the drawing of a row tells, in words: when the sense is most used, and its periods with their new words
function about(node) {
    const values = time.value?.share.value[node.id] || [];
    const parts = [];
    if (values.length) {
        let best = 0;
        values.forEach((v, i) => {
            if (v > values[best]) best = i;
        });
        parts.push(t("senseOverview.rowPeak", { year: time.value.start + best, share: percent(values[best]) }));
    }
    const entry = periodsById.value[node.id];
    if (entry && entry.periods.length > 1) {
        const list = entry.periods.map((p, i) => {
            const first = (i === 0 ? p.characteristic : p.new)?.[0];
            return first ? `${periodYears(p)} (${periodWord(first, queryWord.value)})` : periodYears(p);
        });
        parts.push(t("senseOverview.rowPeriods", { periods: list.join("; ") }));
    } else if (entry && entry.periods.length === 1) {
        parts.push(t("sensePeriods.onePeriod"));
    }
    return parts.join(" ");
}

// ---- The periods of the senses shown: fetched once each, after the rows are drawn ----
let periodsTimer = null;
const periodsPending = ref(false);

function fetchPeriods() {
    if (!time.value) return;
    const wanted = rows.value.filter((n) => n.sense && periodsById.value[n.id] === undefined);
    if (!wanted.length) return;
    const query = fetchToken;
    periodsPending.value = true;
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
        .catch((error) => debug({ $options: { name: "sense-overview" } }, error))
        .finally(() => {
            if (query === fetchToken) periodsPending.value = false;
        });
}

const periodsStatus = computed(() => (periodsPending.value ? t("senseOverview.periodsLoading") : ""));

watch([result, position], () => {
    clearTimeout(periodsTimer);
    periodsTimer = setTimeout(fetchPeriods, result.value && Object.keys(periodsById.value).length ? 300 : 0);
});

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
.detail-range {
    max-width: 16rem;
}

.find {
    margin-left: auto;
}

.find-input {
    width: 12rem;
    max-width: 100%;
}

/* ---- The rows ---- */
.overview {
    position: relative;
}

.sense-rows {
    position: relative;
    list-style: none;
    margin: 0;
    padding: 0;
}

.sense-row {
    position: relative;
    padding: 0.35rem 0.25rem 0.35rem 0.5rem;
    border-left: 4px solid var(--sense-color);
    margin-bottom: 0.4rem;
    border-radius: 0.25rem;
}

/* Narrow: the tree is told by the senses' numbers and by indenting their names (not their years, which share an axis) */
.overview:not(.wide) .sense-name {
    margin-left: calc(var(--depth) * 1rem);
}

.overview:not(.wide) .years-row {
    padding: 0 0.25rem 0 calc(4px + 0.5rem);
}

.overview.wide .years-row,
.overview.wide .sense-row {
    display: grid;
    grid-template-columns: var(--tree-width) minmax(12rem, 24rem) 1fr;
    column-gap: 0.75rem;
    align-items: start;
}

.overview.wide .sense-row {
    padding: 0.35rem 0.25rem;
    border-left: none;
    margin-bottom: 0;
}

.sense-row.picked {
    background: #e7f1ff;
}

.sense-row.found {
    background: #fff3cd;
}

.sense-name {
    display: inline-flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0 0.25rem;
    text-align: left;
    background: none;
    border: none;
    padding: 0.1rem 0.25rem;
    color: #212529;
    font: inherit;
    border-radius: 0.25rem;
}

.sense-name:hover .name {
    text-decoration: underline;
}

.sense-name:focus-visible {
    outline: 3px solid #0d6efd;
    outline-offset: 1px;
}

.sense-name .name {
    font-weight: 600;
    color: var(--sense-dark);
}

.sense-name .what {
    color: #495057;
}

.swatch {
    display: inline-block;
    width: 0.8em;
    height: 0.8em;
    border-radius: 2px;
    background: var(--sense-color);
    border: 1px solid var(--sense-dark);
    align-self: center;
}

.sense-time {
    position: relative;
    min-width: 0;
    margin-top: 0.25rem;
}

.overview.wide .sense-time {
    margin-top: 0;
}

.band {
    display: block;
}

.band-area {
    fill: color-mix(in srgb, var(--sense-color) 55%, white);
    stroke: var(--sense-dark);
    stroke-width: 1.25;
}

.grid {
    stroke: #dee2e6;
}

.period-tick {
    stroke: #212529;
    stroke-width: 1.5;
    stroke-dasharray: 3 3;
}

.probe {
    position: absolute;
    visibility: hidden;
    white-space: nowrap;
    pointer-events: none;
}

.period-words {
    position: relative;
    height: 1.5em;
    height: 1lh;
}

.period-word {
    position: absolute;
    top: 0;
    white-space: nowrap;
    color: #212529;
}

.period-word.start {
    font-style: italic;
    color: #495057;
}

/* ---- The years above the rows ---- */
.years-row {
    margin-bottom: 0.2rem;
}

.years {
    position: relative;
    height: 1.5em;
    height: 1lh;
}

.year {
    position: absolute;
    transform: translateX(-50%);
    color: #495057;
    white-space: nowrap;
}

/* ---- The tree, beside the rows ---- */
.tree {
    position: absolute;
    left: 0;
    pointer-events: none;
}

.tree .branch {
    pointer-events: stroke;
    cursor: pointer;
}

.trunk,
.branch,
.branch-casing {
    fill: none;
    stroke-linecap: butt;
}

.trunk {
    stroke: #6c757d;
}

/* In forced colors, the drawings keep the system's colors */
@media (forced-colors: active) {
    .band-area {
        fill: Canvas;
        stroke: CanvasText;
    }

    .period-tick,
    .trunk,
    .branch,
    .branch-casing {
        stroke: CanvasText;
    }

    .swatch {
        forced-color-adjust: none;
    }
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
    color: #212529;
}

.word-chip:disabled {
    color: #212529;
    opacity: 1;
}
</style>
