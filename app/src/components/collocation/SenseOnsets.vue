<template>
    <div>
        <div v-if="loading && !result" class="text-center py-5">
            <progress-spinner :lg="true" :message="$t('wordSenses.detecting')" />
        </div>
        <div v-if="failed" class="alert alert-warning mx-2 my-3" role="alert">{{ $t("wordSenses.failed") }}</div>

        <div v-if="result && result.n_sentences" class="mx-2 my-3">
            <div class="card shadow-sm p-3 mb-3">
                <p class="mb-1 fw-semibold">
                    {{ $t("senseOnsets.summary", {
                        word: queryWord,
                        sentences: result.n_sentences.toLocaleString(),
                        works: result.n_works.toLocaleString(),
                    }) }}
                </p>
                <p class="text-muted small mb-2">{{ $t("senseOnsets.explained") }}</p>
                <div v-if="splits.length" class="d-flex align-items-center gap-2 flex-wrap">
                    <label for="sense-onsets-detail" class="small fw-semibold mb-0">{{ $t("wordSenses.detail") }}</label>
                    <span class="small text-muted" aria-hidden="true">{{ $t("wordSenses.coarser") }}</span>
                    <input id="sense-onsets-detail" v-model.number="position" type="range" class="form-range detail-range"
                        min="0" :max="splits.length" step="1" :aria-valuetext="detailText" />
                    <span class="small text-muted" aria-hidden="true">{{ $t("wordSenses.finer") }}</span>
                    <button type="button" class="btn btn-sm btn-link p-0" :disabled="position === defaultPosition"
                        @click="position = defaultPosition">{{ $t("senseTree.firstView") }}</button>
                </div>
                <p class="small mb-0" aria-live="polite">{{ detailText }}</p>
                <p v-if="!time" class="text-muted small mb-0 mt-2">{{ $t("wordSenses.noTime") }}</p>
            </div>

            <!-- The senses shown, those that take off first, then those that do not -->
            <section v-for="node in ordered" :key="node.id" :id="`sense-onset-${node.id}`" class="card shadow-sm mb-3 onset-card"
                :style="{ borderLeft: `4px solid ${senseColor(node.sense, 1)}` }">
                <div class="card-body py-2">
                    <h3 class="h6 mb-1">
                        <span class="legend-swatch" :style="{ backgroundColor: senseColor(node.sense, 1) }"></span>
                        <span :style="{ color: senseTextColor(node.sense, shades) }">{{ nameOf(node) }}</span>
                        <span class="fw-normal">{{ label(node) }}</span>
                        <span class="fw-normal text-muted">&middot; {{ $t("wordSenses.shareOfSentences",
                            { share: percent(node.share) }) }}</span>
                    </h3>
                    <p v-if="node.onset" class="mb-1 onset-line">
                        {{ $t("senseOnsets.rises", {
                            year: node.onset.year, before: percent(node.onset.before, 1), after: percent(node.onset.after, 1),
                        }) }}
                    </p>
                    <p v-else class="mb-1 text-muted">{{ $t("senseOnsets.noRise") }}</p>
                    <svg v-if="time" :viewBox="`0 0 ${W} ${H}`" class="onset-chart" role="img"
                        :aria-label="$t('senseOnsets.chart', { sense: nameOf(node) })" preserveAspectRatio="none">
                        <path :d="linePath(node)" :stroke="senseColor(node.sense, 1)" stroke-width="1.5" fill="none"
                            vector-effect="non-scaling-stroke" />
                        <template v-if="node.onset">
                            <line :x1="xOf(node.onset.year)" :x2="xOf(node.onset.year)" y1="0" :y2="H" class="onset-mark"
                                vector-effect="non-scaling-stroke" />
                            <circle v-for="(p, i) in node.onset.precursors" :key="i" :cx="xOf(p.year)" :cy="H - 4" r="2.5"
                                class="precursor" />
                        </template>
                    </svg>
                    <div v-if="time" class="year-axis small text-muted">
                        <span>{{ time.start }}</span><span>{{ time.end }}</span>
                    </div>
                    <p v-if="node.onset && node.onset.sentences_before" class="small mb-0 mt-1">
                        {{ $t("senseOnsets.before", {
                            year: node.onset.year, sentences: node.onset.sentences_before, works: node.onset.works_before,
                        }) }}
                        <span class="precursor-years">{{ node.onset.precursors.map((p) => p.year).join(", ") }}</span>
                        <button type="button" class="btn btn-sm btn-link p-0 ms-1 align-baseline"
                            @click="openPrecursors(node)">{{ $t("senseOnsets.readEarliest") }} &rarr;</button>
                    </p>
                </div>
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
import { computed, inject, onMounted, ref } from "vue";
import { useI18n } from "vue-i18n";
import { useMainStore } from "../../stores/main";
import { debug, paramsFilter } from "../../utils.js";
import DistinctivePassagesModal from "../DistinctivePassagesModal.vue";
import ProgressSpinner from "../ProgressSpinner";
import { senseColor as colorOf, senseShades, senseTextColor } from "./senseColors.js";

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
let fetchToken = 0;

const W = 600;
const H = 60;

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
            debug({ $options: { name: "sense-onsets" } }, error);
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

// ---- The senses at the detail asked for, as in the senses tab ----
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
    return out.filter((node) => node.sense !== null);
});

// Those that take off first, then those that do not, in the order of their numbers
const ordered = computed(() =>
    [...shown.value].sort((a, b) => (a.onset?.year ?? Infinity) - (b.onset?.year ?? Infinity) || a.rank - b.rank)
);

const detailText = computed(() => {
    const n = shown.value.length;
    const senses = n === 1 ? t("wordSenses.oneSense") : t("wordSenses.nSenses", { n });
    const rising = shown.value.filter((node) => node.onset).length;
    return t("senseOnsets.detailText", { senses, rising });
});

// ---- Each sense's share over time, its rise and its precursors ----
function xOf(year) {
    const span = Math.max(1, time.value.end - time.value.start);
    return ((year - time.value.start) / span) * W;
}

function linePath(node) {
    const values = time.value.share.value[node.id];
    const max = Math.max(1e-9, ...values);
    return values
        .map((v, i) => `${i ? "L" : "M"} ${((i / Math.max(1, values.length - 1)) * W).toFixed(1)} ${(H - 8 - (Math.max(0, v) / max) * (H - 12)).toFixed(1)}`)
        .join(" ");
}

// ---- The earliest sentences of a sense, before its rise ----
const modal = ref({ groupName: "", passages: [], loading: false, note: "" });
let modalInstance = null;

async function openPrecursors(node) {
    const precursors = node.onset.precursors;
    modal.value = {
        groupName: `${queryWord.value} · ${nameOf(node)}: ${label(node)}`,
        passages: [],
        loading: true,
        note: t("senseOnsets.earliestNote", { year: node.onset.year }),
    };
    const el = document.getElementById("distinctive-passages-modal");
    if (el) {
        if (!modalInstance) modalInstance = new Modal(el);
        modalInstance.show();
    }
    try {
        const response = await $http.get(`${$dbUrl}/scripts/get_sense_passages.py`, {
            params: { ...paramsFilter(formData.value), hits: precursors.map((p) => p.hit).join(",") },
        });
        const yearOf = Object.fromEntries(precursors.map((p) => [p.hit, p]));
        modal.value.passages = (response.data?.passages || []).map((p) => ({
            ...p,
            note: yearOf[p.hit] ? t("wordSenses.inThisSense", { p: percent(yearOf[p.hit].p) }) : "",
        }));
    } catch (error) {
        debug({ $options: { name: "sense-onsets" } }, error);
    } finally {
        modal.value.loading = false;
    }
}

// ---- Words and colors, as in the senses tab ----
function wordText(w, group) {
    if (w.phrase) return w.phrase;
    if (group === 1) return `${w.word} ${queryWord.value}`;
    if (group === 2) return `${queryWord.value} ${w.word}`;
    return w.word;
}

function label(node) {
    const words = node.words.slice(0, 4).map((w) => w.word);
    if (words.length < 2) {
        words.push(...node.before.slice(0, 1).map((w) => wordText(w, 1)));
        words.push(...node.after.slice(0, 1).map((w) => wordText(w, 2)));
    }
    return words.join(", ") || queryWord.value;
}

function nameOf(node) {
    return t("wordSenses.senseN", { n: node.sense });
}

function percent(share, digits = 0) {
    return `${(share * 100).toFixed(digits)}%`;
}

const shades = computed(() => senseShades(nodes.value));
function senseColor(sense, alpha) {
    return colorOf(sense, alpha, shades.value);
}
</script>

<style scoped>
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

.onset-line {
    font-weight: 600;
}

.onset-chart {
    width: 100%;
    max-width: 40rem;
    height: 60px;
    display: block;
}

.onset-mark {
    stroke: #212529;
    stroke-width: 1;
    stroke-dasharray: 3 3;
}

.precursor {
    fill: #212529;
    opacity: 0.6;
}

.year-axis {
    display: flex;
    justify-content: space-between;
    max-width: 40rem;
}

.precursor-years {
    color: #495057;
}
</style>
