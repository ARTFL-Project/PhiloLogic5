<template>
    <div>
        <div v-if="loading && !result" class="text-center py-5">
            <progress-spinner :lg="true" :message="$t('wordSenses.detecting')" />
        </div>
        <div v-if="failed" class="alert alert-warning mx-2 my-3" role="alert">{{ $t("wordSenses.failed") }}</div>

        <div v-if="result && result.n_sentences" class="mx-2 my-3">
            <div class="card shadow-sm p-3 mb-3">
                <p class="mb-1 fw-semibold">
                    {{ $t("senseMap.summary", {
                        word: queryWord,
                        sentences: result.n_sentences.toLocaleString(),
                        works: result.n_works.toLocaleString(),
                    }) }}
                </p>
                <p class="text-muted small mb-2">{{ $t("senseMap.explained", { word: queryWord }) }}</p>

                <div v-if="splits.length" class="d-flex align-items-center gap-2 flex-wrap">
                    <label for="sense-map-detail" class="small fw-semibold mb-0">{{ $t("wordSenses.detail") }}</label>
                    <span class="small text-muted" aria-hidden="true">{{ $t("wordSenses.coarser") }}</span>
                    <input id="sense-map-detail" v-model.number="position" type="range" class="form-range detail-range"
                        min="1" :max="splits.length" step="1" :aria-valuetext="detailText" />
                    <span class="small text-muted" aria-hidden="true">{{ $t("wordSenses.finer") }}</span>
                    <button type="button" class="btn btn-sm btn-link p-0" :disabled="position === defaultPosition"
                        @click="position = defaultPosition">{{ $t("senseTree.firstView") }}</button>
                </div>
                <p class="small mb-2" aria-live="polite">
                    {{ detailText }}
                    <span v-if="shown.length > 2" class="text-muted">{{ $t("senseMap.kept", { kept: percent(map.kept) }) }}</span>
                </p>

                <svg v-if="shown.length > 1" :viewBox="`0 0 ${W} ${H}`" class="sense-map" role="img"
                    :aria-label="$t('senseMap.chart', { word: queryWord })" @click="selected = null">
                    <!-- From the sense picked, its distance to each other one -->
                    <template v-if="selected">
                        <template v-for="p in map.points" :key="`to-${p.node.id}`">
                            <template v-if="p.node.id !== selected.id">
                                <line :x1="pointOf(selected).x" :y1="pointOf(selected).y" :x2="p.x" :y2="p.y" class="link" />
                                <text class="distance" :x="(pointOf(selected).x + p.x) / 2" :y="(pointOf(selected).y + p.y) / 2 - 4"
                                    text-anchor="middle">{{ distance(selected, p.node).toFixed(2) }}</text>
                            </template>
                        </template>
                    </template>
                    <g v-for="p in map.points" :key="p.node.id" class="point" @click.stop="selected = p.node">
                        <circle :id="`sense-point-${p.node.id}`" :cx="p.x" :cy="p.y" :r="p.r"
                            :fill="senseColor(p.node.rank, 0.75)" :stroke="selected && selected.id === p.node.id ? '#000' : '#fff'"
                            :stroke-width="selected && selected.id === p.node.id ? 2.5 : 1.5">
                            <title>{{ pointTitle(p.node) }}</title>
                        </circle>
                        <line v-if="Math.abs(p.labelY - p.y) > 6" class="leader" :x1="p.x + (p.left ? -p.r : p.r)" :y1="p.y"
                            :x2="p.labelX" :y2="p.labelY - 4" />
                        <text :x="p.labelX" :y="p.labelY" :text-anchor="p.left ? 'end' : 'start'"
                            :fill="p.node.sense ? senseColor(p.node.rank, 1) : '#6c757d'">
                            <tspan font-weight="600">{{ p.node.sense ? p.node.sense : $t("senseTree.restShort") }}</tspan>
                            <tspan fill="#495057" dx="5">{{ label(p.node) }}</tspan>
                        </text>
                    </g>
                </svg>
                <p v-else class="text-muted small mb-0">{{ $t("senseMap.oneSense") }}</p>
            </div>

            <!-- The distances themselves -->
            <div v-if="shown.length > 1" class="card shadow-sm p-3 mb-3">
                <h3 class="h6 mb-2">{{ $t("senseMap.table") }}</h3>
                <div class="table-responsive">
                    <table class="table table-sm distance-table mb-0">
                        <thead>
                            <tr>
                                <th scope="col"></th>
                                <th v-for="b in shown" :key="`h-${b.id}`" scope="col">{{ nameOf(b) }}</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr v-for="a in shown" :key="`r-${a.id}`" :class="{ picked: selected && selected.id === a.id }">
                                <th scope="row">
                                    <button type="button" class="btn btn-sm btn-link p-0 text-start" @click="selected = a">
                                        <span class="legend-swatch" :style="{ backgroundColor: senseColor(a.rank, 1) }"></span>
                                        {{ nameOf(a) }}: {{ label(a) }}
                                    </button>
                                </th>
                                <td v-for="b in shown" :key="`c-${a.id}-${b.id}`"
                                    :style="{ backgroundColor: a.id === b.id ? 'transparent' : shade(distance(a, b)) }">
                                    {{ a.id === b.id ? "" : distance(a, b).toFixed(2) }}
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                <p class="small text-muted mb-0 mt-2">{{ $t("senseMap.tableExplained") }}</p>
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
const position = ref(1); // how many distinctions are shown, the most important first
const selected = ref(null);
let fetchToken = 0;

const W = 1000;
const H = 560;
const MARGIN = 70;

const queryWord = computed(() => (formData.value.q || "").replace(/lemma:/g, "").replace(/"/g, "").trim());
const nodes = computed(() => result.value?.nodes || []);
const root = computed(() => nodes.value[0]);

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
            position.value = Math.max(1, defaultPosition.value);
        })
        .catch((error) => {
            if (myToken !== fetchToken) return;
            failed.value = true;
            debug({ $options: { name: "sense-map" } }, error);
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

// ---- The senses shown, as in the senses tab: the splits revealed, and what they end in ----
const revealed = computed(() => {
    const ids = new Set();
    for (const node of splits.value.slice(0, position.value)) {
        for (let id = node.id; id !== null && !ids.has(id); id = nodes.value[id].parent) ids.add(id);
    }
    return ids;
});

const shown = computed(() => {
    const out = [];
    const walk = (node) => {
        if (revealed.value.has(node.id)) node.children.forEach((c) => walk(nodes.value[c]));
        else out.push(node);
    };
    if (root.value) walk(root.value);
    return out.sort((a, b) => (a.rank ?? Infinity) - (b.rank ?? Infinity));
});

const detailText = computed(() => {
    const n = shown.value.filter((node) => node.sense !== null).length;
    const senses = n === 1 ? t("wordSenses.oneSense") : t("wordSenses.nSenses", { n });
    if (!splits.value.length) return senses;
    return t("senseTree.cutText", { senses, importance: percent(splits.value[position.value - 1].split.relative) });
});

function distance(a, b) {
    return result.value.distances[a.id][b.id];
}

// ---- The map: the distances laid out in two dimensions, by classical multidimensional scaling ----
// The eigenvalues and eigenvectors of a small symmetric matrix (Jacobi's method)
function eigen(A) {
    const n = A.length;
    const a = A.map((row) => row.slice());
    const v = A.map((_, i) => A.map((__, j) => (i === j ? 1 : 0)));
    for (let sweep = 0; sweep < 100; sweep++) {
        let off = 0;
        for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
        if (off < 1e-20) break;
        for (let p = 0; p < n; p++) {
            for (let q = p + 1; q < n; q++) {
                if (Math.abs(a[p][q]) < 1e-30) continue;
                const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
                const tt = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
                const c = 1 / Math.sqrt(tt * tt + 1);
                const s = tt * c;
                for (let k = 0; k < n; k++) {
                    const akp = a[k][p];
                    const akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for (let k = 0; k < n; k++) {
                    const apk = a[p][k];
                    const aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for (let k = 0; k < n; k++) {
                    const vkp = v[k][p];
                    const vkq = v[k][q];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    return { values: a.map((row, i) => row[i]), vectors: v };
}

function scaling(D) {
    const n = D.length;
    const D2 = D.map((row) => row.map((d) => d * d));
    const rowMean = D2.map((row) => row.reduce((s, x) => s + x, 0) / n);
    const all = rowMean.reduce((s, x) => s + x, 0) / n;
    const B = D2.map((row, i) => row.map((d, j) => -0.5 * (d - rowMean[i] - rowMean[j] + all)));
    const { values, vectors } = eigen(B);
    const order = values.map((_, i) => i).sort((i, j) => values[j] - values[i]);
    const positive = values.filter((x) => x > 1e-12).reduce((s, x) => s + x, 0) || 1;
    const axes = order.slice(0, 2).map((k) => ({ value: Math.max(0, values[k]), column: vectors.map((row) => row[k]) }));
    while (axes.length < 2) axes.push({ value: 0, column: new Array(n).fill(0) });
    // The same way round every time: the first sense on the left, and above
    axes.forEach((axis) => {
        if (axis.column[0] > 0) axis.column = axis.column.map((x) => -x);
    });
    const xy = D.map((_, i) => axes.map((axis) => axis.column[i] * Math.sqrt(axis.value)));
    return { xy, kept: (axes[0].value + axes[1].value) / positive };
}

const map = computed(() => {
    const list = shown.value;
    if (list.length < 2 || !result.value?.distances) return { points: [], kept: 1 };
    const { xy, kept } = scaling(list.map((a) => list.map((b) => distance(a, b))));
    const xs = xy.map((p) => p[0]);
    const ys = xy.map((p) => p[1]);
    const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 1e-9);
    const scale = Math.min(W - 2 * MARGIN - 260, H - 2 * MARGIN) / span; // the same scale both ways: distances hold
    const cx = (Math.max(...xs) + Math.min(...xs)) / 2;
    const cy = (Math.max(...ys) + Math.min(...ys)) / 2;
    const points = list.map((node, i) => {
        const x = W / 2 - 100 + (xy[i][0] - cx) * scale;
        const r = 6 + 34 * Math.sqrt(node.share);
        const left = x > W / 2 + 60;
        const width = 7.2 * (String(node.sense ?? t("senseTree.restShort")).length + label(node).length + 1);
        return { node, x, y: H / 2 + (xy[i][1] - cy) * scale, r, left, labelX: x + (left ? -r - 6 : r + 6), labelY: 0, width };
    });
    // Labels apart: those that overlap move up and down, the circles staying where the distances put them
    points.forEach((p) => (p.labelY = p.y + 4));
    const span_ = (p) => (p.left ? [p.labelX - p.width, p.labelX] : [p.labelX, p.labelX + p.width]);
    for (let pass = 0; pass < 60; pass++) {
        let moved = false;
        for (const a of points) {
            for (const b of points) {
                if (a === b) continue;
                const [a0, a1] = span_(a);
                const [b0, b1] = span_(b);
                const overlap = 17 - Math.abs(a.labelY - b.labelY);
                if (a1 > b0 && b1 > a0 && overlap > 0) {
                    const down = a.labelY > b.labelY || (a.labelY === b.labelY && a.node.id > b.node.id) ? 1 : -1;
                    a.labelY += (down * overlap) / 2;
                    moved = true;
                }
            }
        }
        if (!moved) break;
    }
    return { points, kept };
});

function pointOf(node) {
    return map.value.points.find((p) => p.node.id === node.id) || { x: 0, y: 0 };
}

// Closer, darker
function shade(d) {
    const all = shown.value.flatMap((a) => shown.value.filter((b) => b.id !== a.id).map((b) => distance(a, b)));
    const lo = Math.min(...all);
    const hi = Math.max(...all);
    const near = hi > lo ? (hi - d) / (hi - lo) : 1;
    return `hsla(210, 60%, 50%, ${(0.08 + 0.4 * near).toFixed(3)})`;
}

// ---- Words ----
function wordText(w, group = w.group) {
    if (w.phrase) return w.phrase;
    if (group === 1) return `${w.word} ${queryWord.value}`;
    if (group === 2) return `${queryWord.value} ${w.word}`;
    return w.word;
}

function label(node) {
    if (node.sense === null) return "";
    const words = node.words.slice(0, 3).map((w) => w.word);
    if (words.length < 2) {
        words.push(...node.before.slice(0, 1).map((w) => wordText(w, 1)));
        words.push(...node.after.slice(0, 1).map((w) => wordText(w, 2)));
    }
    return words.join(", ") || queryWord.value;
}

function nameOf(node) {
    return node.sense ? t("wordSenses.senseN", { n: node.sense }) : t("senseTree.restShort");
}

function pointTitle(node) {
    return `${nameOf(node)}: ${label(node)} · ${t("wordSenses.shareOfSentences", { share: percent(node.share) })}`;
}

function percent(share) {
    return `${Math.round(share * 100)}%`;
}

const hues = [205, 30, 145, 280, 0, 90, 165, 235, 50, 315, 120, 260, 15, 60, 200, 320];
function senseColor(rank, alpha) {
    if (rank === null || rank === undefined) return `hsla(0, 0%, 62%, ${alpha})`;
    return `hsla(${hues[(rank - 1) % hues.length]}, 55%, 50%, ${alpha})`;
}
</script>

<style scoped>
.sense-map {
    width: 100%;
    height: auto;
}

.sense-map text {
    font-size: 13px;
}

.point {
    cursor: pointer;
}

.leader {
    stroke: #adb5bd;
    stroke-width: 1;
}

.link {
    stroke: #6c757d;
    stroke-width: 1;
    stroke-dasharray: 3 3;
}

.distance {
    fill: #343a40;
    font-size: 12px;
    paint-order: stroke;
    stroke: #fff;
    stroke-width: 3px;
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

.distance-table td {
    text-align: center;
    min-width: 4rem;
}

.distance-table th,
.distance-table thead th {
    font-weight: normal;
    background-color: transparent !important;
    color: #212529 !important;
    border-color: #dee2e6;
}

.distance-table tr.picked th {
    font-weight: 600;
}
</style>
