<template>
    <div>
        <div v-if="loading && !result" class="text-center py-5">
            <progress-spinner :lg="true" :message="$t('risingWords.detecting')" />
        </div>
        <div v-if="failed" class="alert alert-warning mx-2 my-3" role="alert">{{ $t("wordSenses.failed") }}</div>

        <div v-if="result && result.n_sentences" class="mx-2 my-3">
            <div class="card shadow-sm p-3 mb-3">
                <p class="mb-1 fw-semibold">
                    {{ $t("risingWords.summary", {
                        word: queryWord,
                        sentences: result.n_sentences.toLocaleString(),
                        works: result.n_works.toLocaleString(),
                    }) }}
                </p>
                <p class="text-muted small mb-2">{{ $t("risingWords.explained") }}</p>
                <p v-if="!result.time" class="text-muted small mb-0">{{ $t("wordSenses.noTime") }}</p>
                <p v-else class="small mb-0">
                    {{ $t("risingWords.counts", {
                        rising: result.rising || 0, candidates: result.candidates || 0, uses: grouped.length, alone: alone.length,
                    }) }}
                </p>
            </div>

            <!-- The emerging uses: rising words that go together, the earliest first -->
            <section v-for="u in grouped" :key="u.index" :id="`rising-use-${u.index}`" class="card shadow-sm mb-3 use-card">
                <div class="card-body py-2">
                    <p class="mb-1 onset-line">
                        {{ u.use.rises
                            ? $t("risingWords.rises", { year: u.use.onset, before: percent(u.use.before), after: percent(u.use.after) })
                            : $t("risingWords.firstAppears", { year: u.use.onset }) }}
                        <span class="fw-normal text-muted">&middot; {{ $t("risingWords.sentences", {
                            sentences: u.use.sentences, works: u.use.works }) }}</span>
                    </p>
                    <p class="mb-1">
                        <span v-for="w in u.use.words" :key="`${w.group}-${w.word}`" class="rising-chip"
                            :title="$t('risingWords.wordTitle', {
                                year: w.onset, before: percent(w.before), after: percent(w.after),
                                corpusBefore: w.corpus_before, corpusAfter: w.corpus_after,
                            })">
                            {{ wordText(w) }}
                            <span v-if="w.new_word" class="badge new-badge ms-1">{{ $t("risingWords.newWord") }}</span>
                        </span>
                        <span v-if="u.use.n_words > u.use.words.length" class="small text-muted">
                            {{ $t("risingWords.more", { n: u.use.n_words - u.use.words.length }) }}</span>
                    </p>
                    <svg v-if="result.time" :viewBox="`0 0 ${W} ${H}`" class="use-chart" role="img"
                        :aria-label="$t('risingWords.chart')" preserveAspectRatio="none">
                        <path :d="linePath(u.index)" stroke="#0d6efd" stroke-width="1.5" fill="none"
                            vector-effect="non-scaling-stroke" />
                        <line :x1="xOf(u.use.onset)" :x2="xOf(u.use.onset)" y1="0" :y2="H" class="onset-mark"
                            vector-effect="non-scaling-stroke" />
                        <circle v-for="(source, i) in u.use.sources" :key="i" :cx="xOf(source.year)" :cy="H - 4" r="2.5"
                            class="source-dot" />
                    </svg>
                    <div v-if="result.time" class="year-axis small text-muted">
                        <span>{{ result.time.start }}</span><span>{{ result.time.end }}</span>
                    </div>
                    <p v-if="u.use.sources.length" class="small mb-0 mt-1">
                        {{ $t("risingWords.earliest") }}
                        <span class="source-years">{{ u.use.sources.map((s) => s.year).join(", ") }}</span>
                        <button type="button" class="btn btn-sm btn-link p-0 ms-1 align-baseline"
                            @click="openSources(u.use)">{{ $t("senseOnsets.readEarliest") }} &rarr;</button>
                    </p>
                </div>
            </section>

            <!-- The words that rise alone -->
            <div v-if="alone.length" class="card shadow-sm p-3 mb-3">
                <h3 class="h6 mb-2">{{ $t("risingWords.alone") }}</h3>
                <p class="mb-0">
                    <button v-for="u in alone" :key="u.index" type="button" class="btn btn-sm rising-chip alone-chip"
                        :title="$t('risingWords.readTitle')" @click="openSources(u.use)">
                        {{ wordText(u.use.words[0]) }} <span class="text-muted">{{ u.use.onset }}</span>
                        <span v-if="u.use.words[0].new_word" class="badge new-badge ms-1">{{ $t("risingWords.newWord") }}</span>
                    </button>
                </p>
            </div>

            <p v-if="result.time && !result.uses.length" class="text-muted mx-1">{{ $t("risingWords.none") }}</p>
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

const emit = defineEmits(["filterList", "stopwordsMissing"]);
const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const store = useMainStore();
const { formData } = storeToRefs(store);
const { t } = useI18n();

const loading = ref(false);
const failed = ref(false);
const result = ref(null);
let fetchToken = 0;

const W = 600;
const H = 50;

const queryWord = computed(() => (formData.value.q || "").replace(/lemma:/g, "").replace(/"/g, "").trim());

function runDetection() {
    const myToken = ++fetchToken;
    loading.value = true;
    failed.value = false;
    result.value = null;
    $http
        .get(`${$dbUrl}/scripts/get_rising_words.py`, { params: paramsFilter(formData.value) })
        .then((response) => {
            if (myToken !== fetchToken) return;
            result.value = response.data;
            emit("filterList", response.data?.filter_list || []);
            emit("stopwordsMissing", Boolean(response.data?.stopwords_missing));
        })
        .catch((error) => {
            if (myToken !== fetchToken) return;
            failed.value = true;
            debug({ $options: { name: "rising-words" } }, error);
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

// The uses (their index into the report's courses over time), those of several words apart from the words alone
const indexed = computed(() => (result.value?.uses || []).map((use, index) => ({ use, index })));
const grouped = computed(() => indexed.value.filter((u) => u.use.n_words > 1));
const alone = computed(() => indexed.value.filter((u) => u.use.n_words === 1));

function wordText(w) {
    if (w.phrase) return w.phrase;
    if (w.group === 1) return `${w.word} ${queryWord.value}`;
    if (w.group === 2) return `${queryWord.value} ${w.word}`;
    return w.word;
}

function percent(share) {
    return share === null || share === undefined ? "–" : `${(share * 100).toFixed(1)}%`;
}

// ---- Each use's share of the sentences over time, its rise and its earliest sentences ----
function xOf(year) {
    const time = result.value.time;
    return ((year - time.start) / Math.max(1, time.end - time.start)) * W;
}

function linePath(index) {
    const values = result.value.time.share.value[index];
    const max = Math.max(1e-9, ...values);
    return values
        .map((v, i) => `${i ? "L" : "M"} ${((i / Math.max(1, values.length - 1)) * W).toFixed(1)} ${(H - 8 - (Math.max(0, v) / max) * (H - 12)).toFixed(1)}`)
        .join(" ");
}

const modal = ref({ groupName: "", passages: [], loading: false, note: "" });
let modalInstance = null;

async function openSources(use) {
    modal.value = {
        groupName: `${queryWord.value} · ${use.words.slice(0, 4).map(wordText).join(", ")}`,
        passages: [],
        loading: true,
        note: t("risingWords.sourcesNote"),
    };
    const el = document.getElementById("distinctive-passages-modal");
    if (el) {
        if (!modalInstance) modalInstance = new Modal(el);
        modalInstance.show();
    }
    try {
        const response = await $http.get(`${$dbUrl}/scripts/get_sense_passages.py`, {
            params: { ...paramsFilter(formData.value), hits: use.sources.map((s) => s.hit).join(",") },
        });
        modal.value.passages = response.data?.passages || [];
    } catch (error) {
        debug({ $options: { name: "rising-words" } }, error);
    } finally {
        modal.value.loading = false;
    }
}
</script>

<style scoped>
.onset-line {
    font-weight: 600;
}

.rising-chip {
    display: inline-block;
    padding: 0.1rem 0.45rem;
    margin: 0.15rem 0.25rem 0.15rem 0;
    border: 1px solid #ced4da;
    border-radius: 0.5rem;
    font-size: 0.85rem;
}

.alone-chip {
    line-height: 1.3;
}

.new-badge {
    background-color: #fff3cd;
    color: #664d03;
    font-weight: normal;
}

.use-chart {
    width: 100%;
    max-width: 40rem;
    height: 50px;
    display: block;
}

.onset-mark {
    stroke: #212529;
    stroke-width: 1;
    stroke-dasharray: 3 3;
}

.source-dot {
    fill: #212529;
    opacity: 0.6;
}

.year-axis {
    display: flex;
    justify-content: space-between;
    max-width: 40rem;
}

.source-years {
    color: #495057;
}
</style>
