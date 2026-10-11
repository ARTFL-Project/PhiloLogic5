<template>
    <div>
        <div v-if="loading && !result" class="text-center py-5">
            <progress-spinner :lg="true" :message="$t('aheadOfTime.detecting')" />
        </div>
        <div v-if="failed" class="alert alert-warning mx-2 my-3" role="alert">{{ $t("wordSenses.failed") }}</div>

        <div v-if="result && result.n_sentences" class="mx-2 my-3">
            <div class="card shadow-sm p-3 mb-3">
                <p class="mb-1 fw-semibold">
                    {{ $t("aheadOfTime.summary", {
                        word: queryWord,
                        sentences: result.n_sentences.toLocaleString(),
                        works: result.n_works.toLocaleString(),
                    }) }}
                </p>
                <p class="text-muted small mb-2">{{ $t("aheadOfTime.explained") }}</p>
                <p v-if="!result.tested" class="text-muted small mb-0">{{ $t("aheadOfTime.noTest") }}</p>
                <template v-else>
                    <p class="small mb-2">
                        {{ $t("aheadOfTime.counts", {
                            tested: result.tested.toLocaleString(), span: result.span, clear: result.n_clear,
                            perhaps: result.n_perhaps, chance: Math.round(result.by_chance),
                        }) }}
                    </p>
                    <!-- When the works ahead of their time were written: clearly (dark) or perhaps (light), period by period -->
                    <svg :viewBox="`0 0 ${W} ${H}`" class="period-chart" role="img" :aria-label="periodsText"
                        preserveAspectRatio="none">
                        <g v-for="(p, i) in result.periods" :key="p.start">
                            <title>{{ periodTitle(p) }}</title>
                            <rect :x="i * barWidth" :y="H - barHeight(p.clear + p.perhaps)" :width="Math.max(1, barWidth - 1)"
                                :height="barHeight(p.clear + p.perhaps)" class="bar-perhaps" />
                            <rect :x="i * barWidth" :y="H - barHeight(p.clear)" :width="Math.max(1, barWidth - 1)"
                                :height="barHeight(p.clear)" class="bar-clear" />
                        </g>
                    </svg>
                    <div class="year-axis small text-muted">
                        <span>{{ result.periods[0].start }}</span><span>{{ result.periods[result.periods.length - 1].end }}</span>
                    </div>
                </template>
            </div>

            <template v-if="result.tested">
                <h3 class="h6 mx-1">{{ $t("aheadOfTime.clearHeading") }}</h3>
                <p v-if="!result.clear.length" class="text-muted small mx-1">{{ $t("aheadOfTime.noneClear") }}</p>
                <AheadWork v-for="w in result.clear" :key="`clear-${w.doc}`" :work="w" :query-word="queryWord"
                    @read="openSources" />

                <div v-if="result.perhaps.length" class="mb-3">
                    <button type="button" class="btn btn-sm btn-outline-secondary" :aria-expanded="showPerhaps"
                        aria-controls="ahead-perhaps" @click="showPerhaps = !showPerhaps">
                        {{ showPerhaps ? $t("aheadOfTime.hidePerhaps") : $t("aheadOfTime.showPerhaps", { n: result.n_perhaps }) }}
                    </button>
                    <div v-if="showPerhaps" id="ahead-perhaps" class="mt-2">
                        <p class="small text-muted mx-1">
                            {{ $t("aheadOfTime.perhapsNote", { n: result.n_perhaps, chance: Math.round(result.by_chance) }) }}
                            <span v-if="result.n_perhaps > result.perhaps.length">
                                {{ $t("aheadOfTime.firstOnly", { shown: result.perhaps.length }) }}</span>
                        </p>
                        <AheadWork v-for="w in result.perhaps" :key="`perhaps-${w.doc}`" :work="w" :query-word="queryWord"
                            perhaps @read="openSources" />
                    </div>
                </div>
            </template>
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
import AheadWork from "./AheadWork.vue";

const emit = defineEmits(["filterList", "stopwordsMissing"]);
const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const store = useMainStore();
const { formData } = storeToRefs(store);
const { t } = useI18n();

const loading = ref(false);
const failed = ref(false);
const result = ref(null);
const showPerhaps = ref(false);
let fetchToken = 0;

const W = 600;
const H = 40;

const queryWord = computed(() => (formData.value.q || "").replace(/lemma:/g, "").replace(/"/g, "").trim());

function runDetection() {
    const myToken = ++fetchToken;
    loading.value = true;
    failed.value = false;
    result.value = null;
    showPerhaps.value = false;
    $http
        .get(`${$dbUrl}/scripts/get_ahead_of_time.py`, { params: paramsFilter(formData.value) })
        .then((response) => {
            if (myToken !== fetchToken) return;
            result.value = response.data;
            emit("filterList", response.data?.filter_list || []);
            emit("stopwordsMissing", Boolean(response.data?.stopwords_missing));
            // With no work clearly ahead, the works perhaps ahead are what there is to read
            showPerhaps.value = !response.data?.clear?.length;
        })
        .catch((error) => {
            if (myToken !== fetchToken) return;
            failed.value = true;
            debug({ $options: { name: "ahead-of-time" } }, error);
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

// ---- When the works ahead were written ----
const barWidth = computed(() => W / Math.max(1, result.value?.periods?.length || 1));
const most = computed(() => Math.max(1, ...(result.value?.periods || []).map((p) => p.clear + p.perhaps)));

function barHeight(n) {
    return n ? Math.max(2, (n / most.value) * (H - 2)) : 0;
}

function periodTitle(p) {
    return t("aheadOfTime.periodTitle", { start: p.start, end: p.end, clear: p.clear, perhaps: p.perhaps, tested: p.tested });
}

const periodsText = computed(() => {
    const found = (result.value?.periods || []).filter((p) => p.clear + p.perhaps);
    return t("aheadOfTime.periodsText", {
        periods: found.map((p) => `${p.start}–${p.end}: ${p.clear + p.perhaps}`).join(", ") || "–",
    });
});

// ---- The sentences of a work ahead of its time most like the works after it ----
function workName(w) {
    return [w.author, w.title].filter(Boolean).join(", ") || t("aheadOfTime.untitled");
}

const modal = ref({ groupName: "", passages: [], loading: false, note: "" });
let modalInstance = null;

async function openSources(w) {
    modal.value = {
        groupName: `${queryWord.value} · ${w.year} · ${workName(w)}`,
        passages: [],
        loading: true,
        note: t("aheadOfTime.sourcesNote"),
    };
    const el = document.getElementById("distinctive-passages-modal");
    if (el) {
        if (!modalInstance) modalInstance = new Modal(el);
        modalInstance.show();
    }
    try {
        const response = await $http.get(`${$dbUrl}/scripts/get_sense_passages.py`, {
            params: { ...paramsFilter(formData.value), hits: w.sources.map((s) => s.hit).join(",") },
        });
        modal.value.passages = response.data?.passages || [];
    } catch (error) {
        debug({ $options: { name: "ahead-of-time" } }, error);
    } finally {
        modal.value.loading = false;
    }
}
</script>

<style scoped>
.period-chart {
    width: 100%;
    max-width: 40rem;
    height: 40px;
    display: block;
}

.bar-perhaps {
    fill: #9ec5fe;
}

.bar-clear {
    fill: #0d6efd;
}

.year-axis {
    display: flex;
    justify-content: space-between;
    max-width: 40rem;
}
</style>
