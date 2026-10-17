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
                <p class="text-muted small mb-2">{{ $t("senseTree.explained", { finest: percent(result.finest) }) }}</p>

                <div v-if="splits.length" class="d-flex align-items-center gap-2 flex-wrap">
                    <label for="sense-tree-detail" class="small fw-semibold mb-0">{{ $t("wordSenses.detail") }}</label>
                    <span class="small text-muted" aria-hidden="true">{{ $t("wordSenses.coarser") }}</span>
                    <input id="sense-tree-detail" v-model.number="position" type="range" class="form-range detail-range"
                        min="0" :max="splits.length" step="1" :aria-valuetext="detailText"
                        aria-describedby="sense-tree-detail-text" />
                    <span class="small text-muted" aria-hidden="true">{{ $t("wordSenses.finer") }}</span>
                    <button type="button" class="btn btn-sm btn-link p-0" :disabled="position === defaultPosition"
                        @click="position = defaultPosition">{{ $t("senseTree.firstView") }}</button>
                </div>
                <p id="sense-tree-detail-text" class="small mb-2" aria-live="polite">{{ detailText }}</p>

                <div v-if="splits.length" class="mb-2">
                    <label for="sense-tree-find" class="form-label small fw-semibold mb-0">{{ $t("wordSenses.findWord") }}</label>
                    <input id="sense-tree-find" v-model="search" type="search" class="form-control form-control-sm find-input"
                        list="sense-tree-words" :placeholder="$t('wordSenses.findPlaceholder')" />
                    <datalist id="sense-tree-words">
                        <option v-for="w in searchable" :key="w" :value="w" />
                    </datalist>
                    <p v-if="search.trim()" class="small mb-0 mt-1" aria-live="polite">{{ foundText }}</p>
                </div>

                <svg :viewBox="`0 0 ${W} ${tree.height}`" class="sense-tree" role="group"
                    :aria-label="$t('senseTree.chart', { word: queryWord })">
                    <!-- The importance axis, relative to the main distinction, from the left -->
                    <g class="axis" aria-hidden="true">
                        <template v-for="tick in ticks" :key="tick">
                            <line :x1="xOf(tick)" :x2="xOf(tick)" :y1="TOP - 6" :y2="tree.height - 18" />
                            <text :x="xOf(tick)" :y="tree.height - 4" text-anchor="middle">{{ percent(tick) }}</text>
                        </template>
                    </g>
                    <text class="root-label" :x="4" :y="14">{{ $t("senseTree.all", {
                        word: queryWord, sentences: result.n_sentences.toLocaleString() }) }}</text>
                    <!-- The trunk, and each sense's branch: as thick as its share, forking where it is split -->
                    <path :d="`M 4 ${tree.rootY} H ${tree.rootX}`" class="branch" :stroke-width="thickness(1)"
                        stroke="hsla(0, 0%, 62%, 1)" aria-hidden="true" />
                    <path v-for="b in tree.segments" :key="`branch-${b.node.id}`" :id="`sense-branch-${b.node.id}`"
                        class="branch" :d="`M ${b.x0} ${b.y0} V ${b.y1} H ${b.x1}`" :stroke="senseColor(b.node.sense, 1)"
                        :stroke-width="thickness(b.node.share)" aria-hidden="true" @click="select(b.node)">
                        <title>{{ title(b.node) }}</title>
                    </path>
                    <!-- The senses shown, each on its row: a click (or Enter) shows its words -->
                    <g v-for="row in tree.rows" :key="`row-${row.node.id}`" class="row-label" role="button" tabindex="0"
                        :class="{ picked: selected && selected.id === row.node.id, found: isFound(row.node) }"
                        :aria-label="title(row.node)"
                        :aria-pressed="Boolean(selected && selected.id === row.node.id)" @click="select(row.node)"
                        @keydown.enter.prevent="select(row.node)" @keydown.space.prevent="select(row.node)">
                        <rect v-if="isFound(row.node)" class="found-row" :x="RIGHT + 4" :y="row.y - ROW / 2 + 3"
                            :width="W - RIGHT - 6" :height="ROW - 6" rx="4" />
                        <text :id="`sense-row-${row.node.id}`" :x="RIGHT + 10" :y="row.y + 4"
                            :fill="senseTextColor(row.node.sense, shades)">
                            <tspan font-weight="600">{{ row.node.sense ? $t("wordSenses.senseN", { n: row.node.sense })
                                : $t("senseTree.restShort") }}</tspan>
                            <tspan fill="#495057"> · {{ percent(row.node.share) }}{{ row.node.sense ? ` · ${label(row.node)}` : "" }}</tspan>
                        </text>
                    </g>
                </svg>
            </div>

            <!-- A sense picked in the tree -->
            <div v-if="selected" class="card shadow-sm p-3 mb-3 selected-sense">
                <h3 class="h6 mb-2">
                    <span class="legend-swatch" :style="{ backgroundColor: senseColor(selected.sense, 1) }"></span>
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
const search = ref(""); // a word looked for among the senses' words
let fetchToken = 0;

// Importance from the left; the senses shown on rows, named in the column on the right
const W = 1000;
const LEFT = 60;
const RIGHT = 560;
const TOP = 44;
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

// ---- The tree ----
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
const ticks = computed(() => [1, 0.5, 0.25, result.value?.finest ?? 0.1].filter((r) => r >= lowest.value));

// Importance on a log scale, from the main distinction (100%) on the left to the finest detail
function xOf(r) {
    const span = Math.log(1 / lowest.value) || 1;
    return LEFT + (Math.log(1 / Math.max(r, lowest.value)) / span) * (RIGHT - LEFT - 30);
}

function thickness(share) {
    return 1.5 + 14 * share;
}

// In the order of their senses' numbers, the rest of the sentences last
function firstRank(node) {
    if (node.children.length) return Math.min(...node.children.map((c) => firstRank(nodes.value[c])));
    return node.sense === null ? Infinity : node.rank;
}

// The splits the detail reveals, with those they are within: as in the senses tab
const revealed = computed(() => {
    const ids = new Set();
    for (const node of splits.value.slice(0, position.value)) {
        for (let id = node.id; id !== null && !ids.has(id); id = nodes.value[id].parent) ids.add(id);
    }
    return ids;
});

// The tree down to the detail: the senses shown on rows, in the tree's order; a split forks at the height of its
// importance, halfway between the rows of its two halves
const tree = computed(() => {
    const rows = [];
    const segments = [];
    if (!root.value) return { rows, segments, rootX: LEFT, rootY: TOP, height: TOP + ROW + 24 };
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
            const x1 = revealed.value.has(kid.id) ? xOf(heightOf.value[kid.id]) : RIGHT;
            segments.push({ node: kid, x0: x, y0: y, y1: ys[i], x1 });
        });
        return y;
    };
    const rootY = layout(root.value);
    const rootX = revealed.value.has(root.value.id) ? xOf(heightOf.value[root.value.id]) : RIGHT;
    return { rows, segments, rootX, rootY, height: TOP + rows.length * ROW + 24 };
});

const detailText = computed(() => {
    if (!position.value) return t("wordSenses.detailNone");
    const senses = tree.value.rows.filter((row) => row.node.sense !== null).length;
    return t("wordSenses.detailText", {
        senses: senses === 1 ? t("wordSenses.oneSense") : t("wordSenses.nSenses", { n: senses }),
        importance: percent(splits.value[position.value - 1].split.relative),
    });
});

function select(node) {
    selected.value = node;
}

// ---- A word looked for: the senses it marks, at the detail shown or another (as in the senses tab) ----
const searchable = computed(() => searchableWords(nodes.value));
const found = computed(() => findWord(nodes.value, search.value, queryWord.value));

function isFound(node) {
    return Boolean(search.value.trim() && node.sense && inSense(node, search.value, queryWord.value));
}

const foundText = computed(() => {
    const term = search.value.trim();
    if (!found.value.length) return t("wordSenses.notFound", { word: term });
    const shownOf = new Map(tree.value.rows.map((row) => [row.node.sense, row.node]));
    const list = found.value.map((f) => {
        const node = shownOf.get(f.sense);
        const elsewhere = !node || !inSense(node, term, queryWord.value);
        const times = `${String.fromCharCode(215)}${f.ratio >= 10 ? f.ratio.toFixed(0) : f.ratio.toFixed(1)}`;
        return `${t("wordSenses.senseN", { n: f.sense })} (${times}${elsewhere ? `, ${t("wordSenses.otherDetail")}` : ""})`;
    });
    return t("wordSenses.foundIn", { word: term, senses: list.join(", ") });
});

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

function title(node) {
    const name = node.sense ? t("wordSenses.senseN", { n: node.sense }) : t("senseTree.restShort");
    return `${name}: ${label(node)} · ${t("wordSenses.shareOfSentences", { share: percent(node.share) })}`;
}

function percent(share) {
    return `${Math.round(share * 100)}%`;
}

// The senses tab's colors: a hue for each sense at the top, shades of it for those carved out of it (senseColors.js)
const shades = computed(() => senseShades(nodes.value));
function senseColor(sense, alpha) {
    return colorOf(sense, alpha, shades.value);
}
</script>

<style scoped>
.sense-tree {
    width: 100%;
    height: auto;
}

.sense-tree text {
    font-size: 13px;
}

.axis line {
    stroke: #e3e3e3;
    stroke-width: 1;
}

.axis text {
    fill: #6c757d;
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

.row-label:focus {
    outline: none;
}

.row-label.picked text,
.row-label:focus-visible text {
    text-decoration: underline;
}

.find-input {
    max-width: 22rem;
}

.found-row {
    fill: rgba(255, 193, 7, 0.4);
}

.row-label.found text {
    font-weight: 600;
}

.detail-range {
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

.word-chip {
    display: inline-block;
    border: 1px solid #ced4da;
    border-radius: 0.8rem;
    padding: 0 0.5rem;
    margin: 0 0.25rem 0.25rem 0;
    font-size: 0.85rem;
}
</style>
