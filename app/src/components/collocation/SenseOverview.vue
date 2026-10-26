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
                        <span class="d-block">{{ $t("senseOverview.clickExplained") }}</span>
                    </InfoButton>
                </h2>

                <div v-if="splits.length" class="d-flex align-items-center gap-2 flex-wrap mb-1">
                    <label for="sense-overview-detail" class="small fw-semibold mb-0">{{ $t("wordSenses.detail") }}</label>
                    <span class="small text-muted" aria-hidden="true">{{ $t("wordSenses.coarser") }}</span>
                    <input id="sense-overview-detail" v-model.number="position" type="range" class="form-range detail-range"
                        min="0" :max="splits.length" step="1" :aria-valuetext="detailText" />
                    <span class="small text-muted" aria-hidden="true">{{ $t("wordSenses.finer") }}</span>
                    <button type="button" class="btn btn-sm btn-link p-0" :disabled="position === defaultPosition"
                        @click="position = defaultPosition">{{ $t("senseOverview.firstView") }}</button>
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

                <!-- On a narrow screen, the tree and the years take turns -->
                <div v-if="!wide && time" class="nav nav-tabs mb-2" role="tablist" :aria-label="$t('senseOverview.views')">
                    <button v-for="v in VIEWS" :id="`sense-overview-tab-${v}`" :key="v" type="button" role="tab"
                        class="nav-link" :class="{ active: view === v }" :aria-selected="view === v"
                        :tabindex="view === v ? 0 : -1" aria-controls="sense-overview-chart" @click="view = v"
                        @keydown="tabKey">{{ $t(`senseOverview.${v}Tab`) }}</button>
                </div>

                <!-- The senses, one on each row: the tree and the senses' names on the left half; on the right half, how the
                     word's sentences divide between them over the years, each sense a layer in the rows' order -->
                <div id="sense-overview-chart" ref="box" class="overview"
                    :class="{ wide, 'with-tree': showTree, 'time-only': showTime && !showTree }"
                    :role="wide || !time ? null : 'tabpanel'"
                    :aria-labelledby="wide || !time ? null : `sense-overview-tab-${view}`">
                    <div v-if="showTime" class="years-row" aria-hidden="true">
                        <div v-if="wide" class="tree-cell"></div>
                        <div v-if="wide" class="name-cell"></div>
                        <div class="years">
                            <span v-for="year in yearTicks" :key="year" class="year" :style="{ left: `${xYear(year)}px` }">{{ year }}</span>
                        </div>
                    </div>
                    <ul ref="list" class="sense-rows" :aria-label="$t('senseOverview.chart', { word: queryWord })">
                        <li v-for="row in rows" :key="row.id" class="sense-row"
                            :class="{ picked: selected && selected.id === row.id, found: isFound(row), pointed: pointed === row.id }"
                            :style="{ '--depth': depth(row), '--sense-color': senseColor(row.sense, 1),
                                '--sense-dark': senseTextColor(row.sense, shades) }">
                            <div v-if="showTree" class="tree-cell"></div>
                            <button type="button" class="sense-name" :class="{ rest: !row.sense }"
                                :aria-expanded="Boolean(selected && selected.id === row.id)"
                                :aria-controls="selected && selected.id === row.id ? 'sense-overview-panel' : null"
                                :aria-describedby="time ? `sense-overview-about-${row.id}` : null"
                                @click="select(row)" @mouseenter="pointed = row.id" @mouseleave="pointed = null"
                                @focus="pointed = row.id" @blur="pointed = null">
                                <span class="swatch" aria-hidden="true"></span>
                                <span class="name">{{ row.sense ? $t("wordSenses.senseN", { n: row.sense }) : $t("senseOverview.restShort") }}</span>
                                <span class="sep"> · </span>
                                <!-- Beside the tree, the share is on the sense's branch: here it is for screen readers -->
                                <span class="what"><span class="share">{{ percent(row.share) }}<span v-if="row.sense"
                                    class="share-sep"> · </span></span><span v-if="row.sense" class="label">{{ label(row) }}</span></span>
                            </button>
                            <div v-if="wide" class="sense-time"></div>
                            <span v-if="time" :id="`sense-overview-about-${row.id}`" class="visually-hidden">{{ about(row) }}</span>
                        </li>
                    </ul>
                    <!-- The senses' layers, beside the rows (on a narrow screen, under them): each year, the word's sentences
                         divided between them -->
                    <svg v-if="showTime && stack" class="stack" :width="timeWidth" :height="stackHeight"
                        :style="{ top: `${listTop}px`, left: `${timeLeft}px` }" aria-hidden="true">
                        <path v-for="layer in stack.layers" :key="layer.node.id" class="layer" :d="layer.path"
                            :fill="senseColor(layer.node.sense, 1)" @click="select(layer.node)"
                            @mouseenter="pointed = layer.node.id" @mouseleave="pointed = null" />
                        <template v-if="stack.marked">
                            <path class="layer-mark-halo" :d="stack.marked.path" />
                            <path class="layer-mark" :d="stack.marked.path" />
                        </template>
                    </svg>
                    <!-- The tree, beside the rows, from their measured places (it draws what the rows' names say) -->
                    <svg v-if="showTree && treeLayout" class="tree" :width="treeWidth" :height="listHeight"
                        :style="{ top: `${listTop}px`, left: `${treeLeft}px` }" aria-hidden="true">
                        <path :d="`M 2 ${treeLayout.rootY} H ${treeLayout.rootX}`" class="trunk" :stroke-width="thickness(1)" />
                        <g v-for="b in treeLayout.segments" :key="`branch-${b.node.id}`">
                            <path class="branch-casing" :d="branchPath(b)" :stroke="senseTextColor(b.node.sense, shades)"
                                :stroke-width="thickness(b.node.share) + 2" />
                            <path class="branch" :d="branchPath(b)" :stroke="senseColor(b.node.sense, 1)"
                                :stroke-width="thickness(b.node.share)" @click="select(b.node)" />
                        </g>
                        <!-- Each branch's share of the sentences, centered above it: what each division leaves to each side -->
                        <text v-for="b in treeLayout.segments" :key="`share-${b.node.id}`" class="branch-share"
                            :x="(b.x0 + b.x1) / 2" :y="b.y1 - thickness(b.node.share) / 2 - 4">{{ percent(b.node.share) }}</text>
                    </svg>
                </div>
            </div>

            <!-- A sense picked: its words, and its clearest example sentences -->
            <section v-if="selected" id="sense-overview-panel" class="card shadow-sm p-3 mb-3 selected-sense"
                :style="{ borderLeft: `4px solid ${senseColor(selected.sense, 1)}` }" aria-labelledby="sense-overview-panel-title">
                <h3 id="sense-overview-panel-title" class="h6 mb-2">
                    <span class="legend-swatch" :style="{ backgroundColor: senseColor(selected.sense, 1) }" aria-hidden="true"></span>
                    <span :style="{ color: senseTextColor(selected.sense, shades) }">{{ selected.sense
                        ? $t("wordSenses.senseN", { n: selected.sense }) : $t("senseOverview.restShort") }}</span>
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
                <p v-if="!selected.sense" class="small text-muted mb-0">{{ $t("senseOverview.restExplained") }}</p>
                <p v-if="selected.sense" class="mb-0">
                    <button type="button" class="btn btn-sm btn-outline-secondary" :disabled="!selected.examples?.length"
                        @click="openExamples(selected)">{{ $t("wordSenses.examples") }} &rarr;</button>
                </p>
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
import { computed, inject, nextTick, onActivated, onBeforeUnmount, onDeactivated, onMounted, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import { useMainStore } from "../../stores/main";
import { debug, paramsFilter } from "../../utils.js";
import DistinctivePassagesModal from "../DistinctivePassagesModal.vue";
import ProgressSpinner from "../ProgressSpinner";
import InfoButton from "./InfoButton.vue";
import { senseColor as colorOf, senseShades, senseTextColor } from "./senseColors.js";
import { findWord, inSense, searchableWords } from "./senseSearch.js";
import { useKeptTab } from "./keptTab.js";

const emit = defineEmits(["filterList", "stopwordsMissing"]);
const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const store = useMainStore();
const { formData } = storeToRefs(store);
const { t } = useI18n();

const loading = ref(false);
const failed = ref(false);
const result = ref(null);
const { wanted, giveFilters } = useKeptTab(emit, result, loading);
const position = ref(0); // how many distinctions are shown, the most important first
const selected = ref(null);
const search = ref("");
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
    const params = paramsFilter(formData.value);
    if (!wanted(params)) return;
    const myToken = ++fetchToken;
    loading.value = true;
    failed.value = false;
    result.value = null;
    selected.value = null;
    $http
        .get(`${$dbUrl}/scripts/get_word_senses.py`, { params })
        .then((response) => {
            if (myToken !== fetchToken) return;
            result.value = response.data;
            giveFilters(response.data);
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
const width = ref(0);
const timeWidth = ref(0);
const timeLeft = ref(0);
const pointed = ref(null); // the sense under the pointer or the keyboard's focus: its row and its layer are marked
const centers = ref([]); // the middle of each row's name (its first line), from the top of the list
const listHeight = ref(0);
const listTop = ref(0);
const fontSize = ref(16);
let observer = null;
let frame = null;

// Wide enough, the tree and the years side by side; narrower, they take turns, the tree first
const VIEWS = ["tree", "time"];
const view = ref("tree");
const wide = computed(() => width.value >= 46 * fontSize.value);
const showTree = computed(() => wide.value || !time.value || view.value === "tree");
const showTime = computed(() => Boolean(time.value) && (wide.value || view.value === "time"));
const treeWidth = ref(0); // the tree's column, as wide as the names leave it
const treeLeft = ref(0);

// The tabs' keys: the arrows, Home and End move between them
function tabKey(event) {
    const i = VIEWS.indexOf(view.value);
    const next = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: VIEWS.length - 1 }[event.key];
    if (next === undefined) return;
    event.preventDefault();
    view.value = VIEWS[(next + VIEWS.length) % VIEWS.length];
    nextTick(() => document.getElementById(`sense-overview-tab-${view.value}`)?.focus());
}

function measure() {
    frame = null;
    if (!box.value) return;
    const style = getComputedStyle(box.value);
    fontSize.value = parseFloat(style.fontSize) || 16;
    width.value = box.value.clientWidth;
    if (!list.value) return;
    listTop.value = list.value.offsetTop;
    listHeight.value = list.value.offsetHeight;
    centers.value = [...list.value.children].map((li) => {
        const name = li.querySelector(".sense-name .name");
        return li.offsetTop + (name ? name.offsetTop + name.offsetHeight / 2 : li.offsetHeight / 2);
    });
    // The years' room: beside the rows, their column; under them (a narrow screen), all the width
    const cell = list.value.querySelector(".sense-time");
    timeWidth.value = cell ? cell.clientWidth : showTime.value ? box.value.clientWidth : 0;
    timeLeft.value = cell ? cell.getBoundingClientRect().left - box.value.getBoundingClientRect().left : 0;
    const treeCell = list.value.querySelector(".tree-cell")?.getBoundingClientRect();
    treeWidth.value = treeCell ? Math.round(treeCell.width) : 0;
    treeLeft.value = treeCell ? treeCell.left - box.value.getBoundingClientRect().left : 0;
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
// Kept out of the page while another tab is shown, where it would measure nothing: as it was until it is back
onDeactivated(() => {
    observer?.disconnect();
    if (frame) cancelAnimationFrame(frame);
    frame = null;
});
onActivated(() => {
    for (const el of [box.value, list.value]) if (el) observer?.observe(el);
});

// Measured again when the rows or the view change, and whenever the box or the list change size (zoom, text spacing,
// the window)
watch([rows, box, list, wide, view], async () => {
    await nextTick();
    if (observer) {
        observer.disconnect();
        for (const el of [box.value, list.value]) if (el) observer.observe(el);
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

// Beside the rows, as tall as they are; under them, two lines of text for each, as the rows beside it would be
const stackHeight = computed(() => (wide.value ? listHeight.value : Math.round(rows.value.length * 2.2 * fontSize.value)));

// Stacked as the Word senses tab stacks them: each year, the rows' shares renormalized to fill it (the smoothed ones
// need not sum to one exactly), the first row's layer on top
const stack = computed(() => {
    const height = stackHeight.value;
    if (!time.value || !timeWidth.value || !height) return null;
    const { start, end } = time.value;
    const n = end - start + 1;
    const values = rows.value.map((node) => {
        const v = time.value.share.value[node.id] || [];
        return Array.from({ length: n }, (_, i) => Math.max(0, v[i] || 0));
    });
    const totals = new Array(n).fill(0);
    values.forEach((vals) => vals.forEach((v, i) => (totals[i] += v)));
    const xs = Array.from({ length: n }, (_, i) => xYear(start + i).toFixed(1));
    const above = new Array(n).fill(0);
    const layers = rows.value.map((node, k) => {
        const top = above.map((a) => a * height);
        values[k].forEach((v, i) => (above[i] += totals[i] > 0 ? v / totals[i] : 0));
        const bottom = above.map((a) => a * height);
        const down = xs.map((x, i) => `${x} ${top[i].toFixed(1)}`);
        const up = xs.map((x, i) => `${x} ${bottom[i].toFixed(1)}`).reverse();
        return { node, top, bottom, path: `M ${down.join(" L ")} L ${up.join(" L ")} Z` };
    });
    const markedId = pointed.value ?? selected.value?.id;
    return { layers, marked: layers.find((l) => l.node.id === markedId) || null };
});

// What the drawing of a row tells, in words: when the sense is most used
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
    return parts.join(" ");
}

// ---- A word looked for, as in the Word senses tab ----
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
    if (w.phrase) return w.phrase;
    if (group === 1) return `${w.word} ${queryWord.value}`;
    if (group === 2) return `${queryWord.value} ${w.word}`;
    return w.word;
}

// A sense's name in the tree: its first four words; with fewer, the words just before and after the query word make up
// the rest, then those the filter hides
const LABEL_WORDS = 4;

function label(node) {
    const out = [];
    const seen = new Set();
    const add = (w, text) => {
        if (out.length >= LABEL_WORDS || seen.has(w.word)) return;
        seen.add(w.word);
        out.push(text);
    };
    node.words.forEach((w) => add(w, w.word));
    for (let i = 0; i < Math.max(node.before.length, node.after.length); i++) {
        if (node.before[i]) add(node.before[i], wordText(node.before[i], 1));
        if (node.after[i]) add(node.after[i], wordText(node.after[i], 2));
    }
    (node.hidden || []).forEach((w) => add(w, w.word));
    return out.join(", ") || queryWord.value;
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

// As many examples as the Word senses tab shows at a time
const EXAMPLES = 25;

function openSenseWord(node, w, group) {
    if (!w.hits?.length) return;
    const text = wordText(w, group);
    openPassages(`${queryWord.value} · ${t("wordSenses.senseN", { n: node.sense })} · “${text}”`, w.hits,
        t("wordSenses.wordExamplesNote", { word: text }));
}

// The sense's sentences, the most clearly in it first
function openExamples(node) {
    if (!node.examples?.length) return;
    openPassages(`${queryWord.value} · ${t("wordSenses.senseN", { n: node.sense })}`,
        node.examples.slice(0, EXAMPLES).map((e) => e.hit), t("senseOverview.examplesNote"));
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
    padding: 0.2rem 0;
    border-radius: 0.25rem;
    align-items: center;
}

/* With the tree: one grid for all the rows (each row a subgrid). The names take what they need, up to half the
   tree's side, and the tree all the rest */
.overview.with-tree {
    display: grid;
    grid-template-columns: minmax(0, 1fr) fit-content(50%);
    column-gap: 0.75rem;
}

/* Wide: the tree and the names on the left half, the years on the right half */
.overview.wide {
    grid-template-columns: minmax(0, 1fr) fit-content(25%) minmax(0, 50%);
}

.overview.with-tree .years-row,
.overview.with-tree .sense-rows,
.overview.with-tree .sense-row {
    grid-column: 1 / -1;
    display: grid;
    grid-template-columns: subgrid;
}

/* Over time on a narrow screen: the senses' names first, in the layers' order, as the chart's key; then the years and
   the chart, all the width */
.overview.time-only {
    display: flex;
    flex-direction: column;
}

.overview.time-only .sense-rows {
    order: -1;
    display: flex;
    flex-wrap: wrap;
    gap: 0 0.5rem;
    margin-bottom: 0.5rem;
}

.overview.time-only .sense-name .label,
.overview.time-only .sense-name .share-sep {
    display: none;
}

/* Beside the tree, the shares are on its branches: the names keep them for screen readers only */
.overview.with-tree .sense-name .share,
.overview.with-tree .sense-name.rest .sep {
    position: absolute;
    width: 1px;
    height: 1px;
    margin: -1px;
    padding: 0;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
}

/* Room above the first branch for its share, where no years are above it */
.overview.with-tree:not(.wide) .sense-rows {
    padding-top: 1em;
}

.overview.time-only .stack {
    position: static;
}

.sense-row.picked {
    background: #e7f1ff;
}

.sense-row.found {
    background: #fff3cd;
}

.sense-row.pointed {
    background: #f1f3f5;
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

/* The dot stays after the name when the rest goes to the next line */
.sense-name .sep,
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

/* Each row's room in the layers: as tall as two lines of text at least */
.sense-time {
    min-height: 2.2em;
}

/* ---- The layers ---- */
.stack {
    position: absolute;
}

.layer {
    stroke: #fff;
    stroke-width: 1;
    cursor: pointer;
}

/* The sense pointed at or opened: a dark outline on a white one, seen on any color */
.layer-mark,
.layer-mark-halo {
    fill: none;
    pointer-events: none;
}

.layer-mark-halo {
    stroke: #fff;
    stroke-width: 5;
}

.layer-mark {
    stroke: #212529;
    stroke-width: 2;
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
    overflow: visible;
    pointer-events: none;
}

/* On a white halo, so that it reads where it crosses a branch */
.branch-share {
    text-anchor: middle;
    fill: #495057;
    stroke: #fff;
    stroke-width: 3px;
    stroke-linejoin: round;
    paint-order: stroke;
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
    .layer {
        fill: Canvas;
        stroke: CanvasText;
    }

    .layer-mark {
        stroke: Highlight;
    }

    .layer-mark-halo {
        stroke: Canvas;
    }

    .branch-share {
        fill: CanvasText;
        stroke: Canvas;
    }

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
