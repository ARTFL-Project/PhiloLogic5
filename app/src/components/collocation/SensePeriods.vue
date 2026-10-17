<template>
    <div>
        <div v-if="loading && !result" class="text-center py-5">
            <progress-spinner :lg="true" :message="$t('wordSenses.detecting')" />
        </div>
        <div v-if="failed" class="alert alert-warning mx-2 my-3" role="alert">{{ $t("wordSenses.failed") }}</div>

        <div v-if="result && result.n_sentences" class="mx-2 my-3">
            <div class="card shadow-sm p-3 mb-3">
                <p class="mb-1 fw-semibold">
                    {{ $t("sensePeriods.summary", {
                        word: queryWord,
                        sentences: result.n_sentences.toLocaleString(),
                        works: result.n_works.toLocaleString(),
                    }) }}
                </p>
                <p class="text-muted small mb-0">{{ $t("sensePeriods.explained") }}</p>
                <p v-if="!result.senses.length" class="text-muted small mb-0 mt-2">{{ $t("wordSenses.noTime") }}</p>
            </div>

            <!-- Each sense: its periods from left to right -->
            <section v-for="s in result.senses" :key="s.id" class="card shadow-sm mb-3 period-card"
                :style="{ borderLeft: `4px solid ${senseColor(s.sense, 1)}` }" :aria-labelledby="`sense-periods-${s.id}`">
                <div class="card-body py-2">
                    <h3 :id="`sense-periods-${s.id}`" class="h6 mb-2">
                        <span class="legend-swatch" :style="{ backgroundColor: senseColor(s.sense, 1) }"></span>
                        <span :style="{ color: senseTextColor(s.sense, shades) }">{{ $t("wordSenses.senseN", { n: s.sense }) }}:</span>
                        {{ " " }}<span class="fw-normal">{{ label(nodes[s.id]) }}</span>
                        {{ " " }}<span class="fw-normal text-muted">&middot; {{ $t("wordSenses.shareOfSentences",
                            { share: percent(nodes[s.id].share) }) }} &middot;
                            {{ s.periods.length > 1 ? $t("sensePeriods.nPeriods", { n: s.periods.length })
                                : $t("sensePeriods.onePeriod") }}</span>
                    </h3>
                    <PeriodColumns :periods="s.periods" :sense="s.sense" :shades="shades" :query-word="queryWord"
                        @word="(p, w) => openWord(s, p, w)" @work="(p, w) => openWork(s, p, w)" />
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
import PeriodColumns, { periodWord, periodYears } from "./PeriodColumns.vue";
import { senseColor as colorOf, senseShades, senseTextColor } from "./senseColors.js";
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
let fetchToken = 0;

const queryWord = computed(() => (formData.value.q || "").replace(/lemma:/g, "").replace(/"/g, "").trim());
const nodes = computed(() => result.value?.nodes || []);

function runDetection() {
    const params = paramsFilter(formData.value);
    if (!wanted(params)) return;
    const myToken = ++fetchToken;
    loading.value = true;
    failed.value = false;
    result.value = null;
    $http
        .get(`${$dbUrl}/scripts/get_sense_periods.py`, { params })
        .then((response) => {
            if (myToken !== fetchToken) return;
            result.value = response.data;
            giveFilters(response.data);
        })
        .catch((error) => {
            if (myToken !== fetchToken) return;
            failed.value = true;
            debug({ $options: { name: "sense-periods" } }, error);
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

// ---- Words, senses and works, as the senses tab names them ----
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

function workName(w) {
    return [w.author, w.title].filter(Boolean).join(", ") || t("sensePeriods.untitled");
}

function percent(share) {
    return `${Math.round(share * 100)}%`;
}

const shades = computed(() => senseShades(nodes.value));
function senseColor(sense, alpha) {
    return colorOf(sense, alpha, shades.value);
}

// ---- Reading: a word's sentences in a period, a work's most typical ----
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
        debug({ $options: { name: "sense-periods" } }, error);
    } finally {
        modal.value.loading = false;
    }
}

function years(p) {
    return periodYears(p);
}

function openWord(s, p, w) {
    if (!w.hits?.length) return;
    const name = `${t("wordSenses.senseN", { n: s.sense })} · ${years(p)}`;
    openPassages(`${queryWord.value} · ${name} · “${wordText(w)}”`, w.hits, t("sensePeriods.wordNote", { word: wordText(w), years: years(p) }));
}

function openWork(s, p, w) {
    const name = `${t("wordSenses.senseN", { n: s.sense })} · ${years(p)}`;
    openPassages(`${queryWord.value} · ${name} · ${workName(w)}`, w.hits, t("sensePeriods.workNote", { years: years(p) }));
}
</script>

<style scoped>
.legend-swatch {
    display: inline-block;
    width: 0.75rem;
    height: 0.75rem;
    border-radius: 2px;
    margin-right: 0.3rem;
    vertical-align: -0.05rem;
}
</style>
