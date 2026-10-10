<template>
    <div>
        <div v-if="loading && !result" class="text-center py-5">
            <progress-spinner :lg="true" :message="$t('wordSenses.detecting')" />
        </div>
        <div v-if="failed" class="alert alert-warning mx-2 my-3" role="alert">{{ $t("wordSenses.failed") }}</div>

        <div v-if="result && result.n_sentences" class="mx-2 my-3">
            <div class="card shadow-sm p-3 mb-3">
                <p class="mb-1 fw-semibold">
                    {{ $t("senseTree.summary", {
                        word: queryWord,
                        sentences: result.n_sentences.toLocaleString(),
                        works: result.n_works.toLocaleString(),
                    }) }}
                </p>
                <div class="btn-group btn-group-sm mb-2" role="group" :aria-label="$t('senseTree.viewLabel')">
                    <button type="button" class="btn btn-secondary" :class="{ active: view === 'branches' }"
                        :aria-pressed="view === 'branches'" @click="view = 'branches'">{{ $t("senseTree.viewBranches") }}</button>
                    <button type="button" class="btn btn-secondary" :class="{ active: view === 'blocks' }"
                        :aria-pressed="view === 'blocks'" @click="view = 'blocks'">{{ $t("senseTree.viewBlocks") }}</button>
                </div>
                <p class="text-muted small mb-2">{{ $t(view === "branches" ? "senseTree.explainedBranches"
                    : "senseTree.explained", { finest: percent(result.finest) }) }}</p>

                <div v-if="splits.length" class="d-flex align-items-center gap-2 flex-wrap">
                    <label for="sense-tree-cut" class="small fw-semibold mb-0">{{ $t("senseTree.cut") }}</label>
                    <span class="small text-muted" aria-hidden="true">{{ $t("wordSenses.coarser") }}</span>
                    <input id="sense-tree-cut" v-model.number="position" type="range" class="form-range cut-range" min="0"
                        :max="splits.length" step="1" :aria-valuetext="cutText" aria-describedby="sense-tree-cut-text" />
                    <span class="small text-muted" aria-hidden="true">{{ $t("wordSenses.finer") }}</span>
                    <button type="button" class="btn btn-sm btn-link p-0" :disabled="position === defaultPosition"
                        @click="position = defaultPosition">{{ $t("senseTree.firstView") }}</button>
                </div>
                <p id="sense-tree-cut-text" class="small mb-2" aria-live="polite">{{ cutText }}</p>

                <svg v-if="view === 'branches'" :viewBox="`0 0 ${W} ${branches.height}`" class="sense-tree branches"
                    role="img" :aria-label="$t('senseTree.chart', { word: queryWord })" @click="cutAtX($event)">
                    <!-- The importance axis, relative to the main distinction, from the left -->
                    <g class="axis">
                        <template v-for="tick in ticks" :key="tick">
                            <line :x1="xOf(tick)" :x2="xOf(tick)" :y1="B_TOP - 6" :y2="branches.height - 18" />
                            <text :x="xOf(tick)" :y="branches.height - 4" text-anchor="middle">{{ percent(tick) }}</text>
                        </template>
                    </g>
                    <text class="root-label" :x="4" :y="14">{{ $t("senseTree.all", {
                        word: queryWord, sentences: result.n_sentences.toLocaleString() }) }}</text>
                    <!-- The trunk, and each sense's branch: as thick as its share, forking where it is split -->
                    <path :d="`M 4 ${branches.rootY} H ${branches.rootX}`" class="branch" :stroke-width="thickness(1)"
                        stroke="hsla(0, 0%, 62%, 1)" />
                    <path v-for="b in branches.segments" :key="`branch-${b.node.id}`" :id="`sense-branch-${b.node.id}`"
                        class="branch" :d="`M ${b.x0} ${b.y0} V ${b.y1} H ${b.x1}`" :stroke="senseColor(b.node.rank, 1)"
                        :stroke-width="thickness(b.node.share)" @click.stop="select(b.node)">
                        <title>{{ blockTitle(b.node) }}</title>
                    </path>
                    <!-- The first view, and the cut -->
                    <line v-if="splits.length && position !== defaultPosition" class="first-view" :x1="cutX(defaultPosition)"
                        :x2="cutX(defaultPosition)" :y1="B_TOP - 6" :y2="branches.height - 18" />
                    <text v-if="splits.length && position !== defaultPosition" class="line-label first-view-label"
                        :x="cutX(defaultPosition)" :y="B_TOP - 10" text-anchor="middle">{{ $t("senseTree.firstViewLine") }}</text>
                    <line class="cut" :x1="cutX(position)" :x2="cutX(position)" :y1="B_TOP - 6" :y2="branches.height - 18" />
                    <text class="line-label" :x="cutX(position)" :y="B_TOP - 10" text-anchor="middle">{{
                        position === defaultPosition && splits.length ? $t("senseTree.cutFirstView") : $t("senseTree.cutLine") }}</text>
                    <!-- The senses at the cut, each on its row -->
                    <g v-for="row in branches.rows" :key="`row-${row.node.id}`" class="row-label"
                        :class="{ picked: selected && selected.id === row.node.id }" @click.stop="select(row.node)">
                        <text :id="`sense-row-${row.node.id}`" :x="B_RIGHT + 10" :y="row.y + 4"
                            :fill="row.node.sense ? senseColor(row.node.rank, 1) : '#6c757d'">
                            <tspan font-weight="600">{{ row.node.sense ? $t("wordSenses.senseN", { n: row.node.sense })
                                : $t("senseTree.restShort") }}</tspan>
                            <tspan fill="#495057"> · {{ percent(row.node.share) }} · {{ label(row.node) }}</tspan>
                        </text>
                    </g>
                </svg>

                <svg v-else :viewBox="`0 0 ${W} ${H}`" class="sense-tree" role="img"
                    :aria-label="$t('senseTree.chart', { word: queryWord })" @click="cutAt($event)">
                    <!-- The importance axis, relative to the main distinction -->
                    <g class="axis">
                        <template v-for="tick in ticks" :key="tick">
                            <line :x1="LEFT" :x2="RIGHT" :y1="yOf(tick)" :y2="yOf(tick)" />
                            <text :x="LEFT - 6" :y="yOf(tick) + 4" text-anchor="end">{{ percent(tick) }}</text>
                        </template>
                        <text :x="4" :y="TOP - 6">{{ $t("senseTree.axis") }}</text>
                    </g>
                    <!-- All the sentences, above; then each sense, as wide as its share, as tall as it is shown -->
                    <rect class="block" :class="{ shown: !position }" :x="LEFT" :y="4" :width="RIGHT - LEFT"
                        :height="TOP - 20" fill="hsla(0, 0%, 88%, 1)" />
                    <rect v-for="b in blocks" :id="`sense-block-${b.node.id}`" :key="b.node.id" class="block"
                        :class="{ shown: shownIds.has(b.node.id) }" :x="b.x + 0.5" :y="b.y" :width="Math.max(0, b.w - 1)"
                        :height="b.h" :fill="senseColor(b.node.rank, shownIds.has(b.node.id) ? 0.8 : 0.2)"
                        :stroke="selected && selected.id === b.node.id ? '#000' : '#fff'"
                        :stroke-width="selected && selected.id === b.node.id ? 2 : 1" @click.stop="select(b.node)">
                        <title>{{ blockTitle(b.node) }}</title>
                    </rect>
                    <!-- The first view, and the cut, under the labels; their names in the margin -->
                    <line v-if="splits.length && position !== defaultPosition" class="first-view" :x1="LEFT" :x2="RIGHT"
                        :y1="cutY(defaultPosition)" :y2="cutY(defaultPosition)" />
                    <text v-if="splits.length && position !== defaultPosition" class="line-label first-view-label"
                        :x="RIGHT + 6" :y="cutY(defaultPosition) + 4">{{ $t("senseTree.firstViewLine") }}</text>
                    <line class="cut" :x1="LEFT" :x2="RIGHT" :y1="cutY(position)" :y2="cutY(position)" />
                    <text class="line-label" :x="RIGHT + 6" :y="cutY(position) + 4">{{
                        position === defaultPosition && splits.length ? $t("senseTree.cutFirstView") : $t("senseTree.cutLine") }}</text>
                    <text :x="LEFT + 6" :y="TOP - 20">{{ $t("senseTree.all", {
                        word: queryWord, sentences: result.n_sentences.toLocaleString() }) }}</text>
                    <template v-for="b in blocks" :key="`label-${b.node.id}`">
                        <text v-if="b.label" class="block-label" :x="b.x + 5" :y="b.y + 14"
                            :fill="shownIds.has(b.node.id) ? '#fff' : '#333'">{{ b.label }}</text>
                    </template>
                </svg>
            </div>

            <!-- The senses the cut crosses, in the order of their numbers, and the rest -->
            <div class="card shadow-sm p-3 mb-3">
                <h3 class="h6 mb-2">{{ $t("senseTree.atCut") }}</h3>
                <ul class="list-unstyled mb-0">
                    <li v-for="node in atCut" :key="node.id" class="mb-1">
                        <button type="button" class="btn btn-sm btn-link p-0 text-start at-cut" @click="select(node)">
                            <span class="legend-swatch" :style="{ backgroundColor: senseColor(node.rank, 1) }"></span>
                            <template v-if="node.sense">{{ $t("wordSenses.senseN", { n: node.sense }) }}:</template>
                            {{ label(node) }}
                            <span class="text-muted">&middot; {{ $t("wordSenses.shareOfSentences",
                                { share: percent(node.share) }) }}</span>
                        </button>
                    </li>
                </ul>
            </div>

            <!-- A sense picked on the chart or in the list -->
            <div v-if="selected" class="card shadow-sm p-3 mb-3 selected-sense">
                <h3 class="h6 mb-2">
                    <span class="legend-swatch" :style="{ backgroundColor: senseColor(selected.rank, 1) }"></span>
                    {{ selected.sense ? $t("wordSenses.senseN", { n: selected.sense }) : $t("senseTree.restShort") }}
                    <span class="fw-normal text-muted">&middot; {{ $t("wordSenses.shareOfSentences",
                        { share: percent(selected.share) }) }} &middot; {{ $t("wordSenses.fromWorks",
                        { n: selected.works }) }}</span>
                </h3>
                <p v-if="selected.words.length" class="mb-1">
                    <span v-for="w in selected.words.slice(0, 12)" :key="w.word" class="word-chip">{{ w.word }}</span>
                </p>
                <p v-if="near(selected).length" class="mb-1">
                    <span v-for="w in near(selected)" :key="w" class="word-chip">{{ w }}</span>
                </p>
                <p v-if="selected.rarer && selected.rarer.length" class="small text-muted mb-0">
                    {{ $t("wordSenses.rarer") }}: {{ selected.rarer.slice(0, 6).map((w) => wordText(w)).join(", ") }}
                </p>
                <p v-if="!selected.sense" class="small text-muted mb-0">{{ $t("senseTree.restExplained") }}</p>
            </div>
        </div>

        <div v-else-if="result" class="text-center py-5 text-muted">{{ $t("wordSenses.noResults") }}</div>
    </div>
</template>

<script setup>
import { storeToRefs } from "pinia";
import { computed, inject, onMounted, ref } from "vue";
import { useI18n } from "vue-i18n";
import { useMainStore } from "../../stores/main";
import { debug, paramsFilter } from "../../utils.js";
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
const position = ref(0); // how many distinctions the cut reveals, the most important first
const selected = ref(null);
const view = ref("branches"); // the tree as "branches" forking from the left, or as nested "blocks"
let fetchToken = 0;

const W = 1000;
const H = 440;
const TOP = 40;
const BOTTOM = H - 16;
const LEFT = 46;
const RIGHT = 900; // the plot's right edge: the margin beyond names the lines
// The branches: importance from the left; the senses at the cut on rows, named in the column on the right
const B_LEFT = 60;
const B_RIGHT = 560;
const B_TOP = 44;
const ROW = 28;

const queryWord = computed(() => (formData.value.q || "").replace(/lemma:/g, "").replace(/"/g, "").trim());
const nodes = computed(() => result.value?.nodes || []);
const root = computed(() => nodes.value[0]);

// The distinctions, the most important first, as the senses tab orders them
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
            debug({ $options: { name: "sense-tree" } }, error);
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

// Loaded on demand: the page's call may come before the component is there, so it fetches on its own when it mounts
onMounted(() => {
    if (!result.value && !loading.value) runDetection();
});

// ---- The tree's geometry ----
// A sense's height: the importance at which the detail first shows its split, its own or a finer one's within it
// (showing a split shows those it is within)
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

// Importance on a log scale, from the main distinction (100%) at the top to the finest detail at the bottom
function yOf(r) {
    const span = Math.log(1 / lowest.value) || 1;
    return TOP + (Math.log(1 / Math.max(r, lowest.value)) / span) * (BOTTOM - TOP - 20);
}

const ticks = computed(() => [1, 0.5, 0.25, result.value?.finest ?? 0.1].filter((r) => r >= lowest.value));

// The cut: between the last distinction it reveals and the next, so that it crosses the senses shown
function cutY(p) {
    if (!p) return TOP - 6;
    const above = yOf(splits.value[p - 1].split.relative);
    const below = p < splits.value.length ? yOf(splits.value[p].split.relative) : BOTTOM;
    return (above + below) / 2;
}

// In the order of their senses' numbers, the rest of the sentences last
function firstRank(node) {
    if (node.children.length) return Math.min(...node.children.map((c) => firstRank(nodes.value[c])));
    return node.sense === null ? Infinity : node.rank;
}

const blocks = computed(() => {
    const out = [];
    if (!root.value) return out;
    const place = (node, x0, x1, top) => {
        const bottom = node.children.length ? yOf(heightOf.value[node.id]) : BOTTOM;
        if (node !== root.value) {
            const w = x1 - x0;
            const h = Math.max(0, bottom - top);
            const text = `${node.sense ? `${node.sense} ` : ""}${label(node)}`;
            const room = Math.floor((w - 8) / 6.4);
            const label_ = w >= 34 && h >= 16 ? (text.length > room ? `${text.slice(0, Math.max(1, room - 1))}…` : text) : "";
            out.push({ node, x: x0, w, y: top, h, label: label_ });
        }
        if (!node.children.length) return;
        const kids = node.children.map((c) => nodes.value[c]).sort((a, b) => firstRank(a) - firstRank(b));
        const total = kids.reduce((s, k) => s + k.share, 0) || 1;
        let x = x0;
        for (const kid of kids) {
            const w = ((x1 - x0) * kid.share) / total;
            place(kid, x, x + w, bottom);
            x += w;
        }
    };
    place(root.value, LEFT, RIGHT, TOP);
    return out;
});

// The splits the cut reveals (with those they are within), and the senses it crosses: as in the senses tab
const revealed = computed(() => {
    const ids = new Set();
    for (const node of splits.value.slice(0, position.value)) {
        for (let id = node.id; id !== null && !ids.has(id); id = nodes.value[id].parent) ids.add(id);
    }
    return ids;
});

const atCut = computed(() => {
    const out = [];
    const walk = (node) => {
        if (revealed.value.has(node.id)) node.children.forEach((c) => walk(nodes.value[c]));
        else out.push(node);
    };
    if (root.value && position.value) walk(root.value);
    return out.sort((a, b) => (a.rank ?? Infinity) - (b.rank ?? Infinity));
});

const shownIds = computed(() => new Set(atCut.value.map((n) => n.id)));

const cutText = computed(() => {
    if (!position.value) return t("senseTree.cutNone");
    const senses = atCut.value.filter((n) => n.sense !== null).length;
    return t("senseTree.cutText", {
        senses: senses === 1 ? t("wordSenses.oneSense") : t("wordSenses.nSenses", { n: senses }),
        importance: percent(splits.value[position.value - 1].split.relative),
    });
});

// ---- The branches ----
function xOf(r) {
    const span = Math.log(1 / lowest.value) || 1;
    return B_LEFT + (Math.log(1 / Math.max(r, lowest.value)) / span) * (B_RIGHT - B_LEFT - 30);
}

function cutX(p) {
    if (!p) return B_LEFT - 6;
    const left = xOf(splits.value[p - 1].split.relative);
    const right = p < splits.value.length ? xOf(splits.value[p].split.relative) : B_RIGHT;
    return (left + right) / 2;
}

function thickness(share) {
    return 1.5 + 14 * share;
}

// The tree down to the cut: the senses at the cut on rows, in the tree's order; a split forks at the height of its
// importance, halfway between the rows of its two halves
const branches = computed(() => {
    const rows = [];
    const segments = [];
    if (!root.value) return { rows, segments, rootX: B_LEFT, rootY: B_TOP, height: B_TOP + ROW + 24 };
    const layout = (node) => {
        if (!revealed.value.has(node.id)) {
            const y = B_TOP + rows.length * ROW + ROW / 2;
            rows.push({ node, y });
            return y;
        }
        const x = xOf(heightOf.value[node.id]);
        const kids = node.children.map((c) => nodes.value[c]).sort((a, b) => firstRank(a) - firstRank(b));
        const ys = kids.map(layout);
        const y = (ys[0] + ys[ys.length - 1]) / 2;
        kids.forEach((kid, i) => {
            const x1 = revealed.value.has(kid.id) ? xOf(heightOf.value[kid.id]) : B_RIGHT;
            segments.push({ node: kid, x0: x, y0: y, y1: ys[i], x1 });
        });
        return y;
    };
    const rootY = layout(root.value);
    const rootX = revealed.value.has(root.value.id) ? xOf(heightOf.value[root.value.id]) : B_RIGHT;
    return { rows, segments, rootX, rootY, height: B_TOP + rows.length * ROW + 24 };
});

// A click on the branches moves the cut to the importance there
function cutAtX(event) {
    if (!splits.value.length) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / Math.max(1, rect.width)) * W;
    const span = Math.log(1 / lowest.value) || 1;
    const r = Math.exp(-((x - B_LEFT) / (B_RIGHT - B_LEFT - 30)) * span);
    position.value = splits.value.filter((n) => n.split.relative >= r).length;
}

// A click on the chart moves the cut to the importance there
function cutAt(event) {
    if (!splits.value.length) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const y = ((event.clientY - rect.top) / Math.max(1, rect.height)) * H;
    const span = Math.log(1 / lowest.value) || 1;
    const r = Math.exp(-((y - TOP) / (BOTTOM - TOP - 20)) * span);
    position.value = splits.value.filter((n) => n.split.relative >= r).length;
}

function select(node) {
    selected.value = node;
}

// ---- Words ----
function wordText(w, group = w.group) {
    if (w.phrase) return w.phrase;
    if (group === 1) return `${w.word} ${queryWord.value}`;
    if (group === 2) return `${queryWord.value} ${w.word}`;
    return w.word;
}

function near(node) {
    return [...node.before.slice(0, 4).map((w) => wordText(w, 1)), ...node.after.slice(0, 4).map((w) => wordText(w, 2))];
}

// A sense's label, as in the senses tab: its strongest marker words, else the words next to the query
function label(node) {
    if (node.sense === null) return t("senseTree.restShort");
    const words = node.words.slice(0, 3).map((w) => w.word);
    if (words.length < 2) words.push(...near(node).slice(0, 2));
    if (!words.length) words.push(...(node.hidden || []).slice(0, 3).map((w) => w.word));
    return words.join(", ") || queryWord.value;
}

function blockTitle(node) {
    const name = node.sense ? t("wordSenses.senseN", { n: node.sense }) : t("senseTree.restShort");
    return `${name}: ${label(node)} · ${t("wordSenses.shareOfSentences", { share: percent(node.share) })}`;
}

function percent(share) {
    return `${Math.round(share * 100)}%`;
}

// The senses tab's colors, by the rank of a sense's number, the rest grey
const hues = [205, 30, 145, 280, 0, 90, 165, 235, 50, 315, 120, 260, 15, 60, 200, 320];
function senseColor(rank, alpha) {
    if (rank === null || rank === undefined) return `hsla(0, 0%, 62%, ${alpha})`;
    return `hsla(${hues[(rank - 1) % hues.length]}, 55%, 50%, ${alpha})`;
}
</script>

<style scoped>
.sense-tree {
    width: 100%;
    height: auto;
    cursor: crosshair;
}

.sense-tree text {
    font-size: 11px;
    pointer-events: none;
}

.axis line {
    stroke: #e3e3e3;
    stroke-width: 1;
}

.axis text {
    fill: #6c757d;
}

.block {
    cursor: pointer;
}

.cut {
    stroke: #212529;
    stroke-width: 2;
}

.first-view {
    stroke: #6c757d;
    stroke-width: 1.5;
    stroke-dasharray: 2 4;
}

.first-view-label {
    fill: #6c757d;
}

.line-label {
    font-weight: 600;
}

.branch {
    fill: none;
    stroke-linecap: butt;
    cursor: pointer;
}

.root-label {
    fill: #495057;
}

.row-label {
    cursor: pointer;
}

.row-label.picked text {
    text-decoration: underline;
}

.sense-tree.branches text {
    font-size: 13px;
}

.cut-range {
    max-width: 22rem;
}

.legend-swatch {
    display: inline-block;
    width: 0.75rem;
    height: 0.75rem;
    border-radius: 2px;
    margin-right: 0.3rem;
    vertical-align: -0.05rem;
}

.at-cut {
    text-decoration: none;
}

.word-chip {
    display: inline-block;
    border: 1px solid #ced4da;
    border-radius: 0.8rem;
    padding: 0 0.5rem;
    margin: 0 0.25rem 0.25rem 0;
    font-size: 0.85rem;
}
</style>
