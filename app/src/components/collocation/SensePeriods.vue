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
                    <ol class="periods" :aria-label="$t('sensePeriods.periodsOf', { sense: s.sense })">
                        <li v-for="(p, i) in s.periods" :key="p.start" class="period">
                            <div class="period-years">{{ p.start === p.end ? p.start : `${p.start}–${p.end}` }}</div>
                            <div class="period-size">{{ $t("sensePeriods.size", { sentences: p.sentences, authors: p.authors }) }}</div>

                            <template v-if="i === 0 && p.characteristic">
                                <div class="period-label">{{ $t("sensePeriods.atStart") }}</div>
                                <div v-if="!p.characteristic.length" class="fading">{{ $t("sensePeriods.nothingApart") }}</div>
                                <div class="chips">
                                    <button v-for="w in p.characteristic" :key="`${w.group}-${w.word}`" type="button"
                                        class="chip chip-start" :title="startTitle(w)"
                                        :style="{ borderColor: senseColor(s.sense, 0.8) }"
                                        @click="openWord(s, p, w)">{{ wordText(w) }}</button>
                                </div>
                            </template>
                            <template v-else-if="p.new">
                                <div class="period-label">{{ $t("sensePeriods.new") }}</div>
                                <div class="chips">
                                    <button v-for="w in p.new" :key="`${w.group}-${w.word}`" type="button" class="chip"
                                        :title="newTitle(w)" :style="{ backgroundColor: senseColor(s.sense, 0.18) }"
                                        @click="openWord(s, p, w)">{{ wordText(w) }}</button>
                                </div>
                                <div v-if="p.fading.length" class="fading">
                                    <span class="period-label">{{ $t("sensePeriods.fading") }}</span>
                                    {{ p.fading.map(wordText).join(", ") }}
                                </div>
                            </template>

                            <div v-if="p.representative?.length" class="works">
                                <div class="period-label">{{ $t("sensePeriods.representative") }}</div>
                                <button v-for="w in p.representative" :key="w.hits[0]" type="button"
                                    class="btn btn-link btn-sm p-0 work-link" @click="openWork(s, p, w)">
                                    {{ shortAuthor(w.author) }}<template v-if="w.author">, </template><i>{{ shortTitle(w.title) }}</i>
                                    {{ " " }}<span class="text-muted">({{ w.year }})</span>
                                </button>
                            </div>
                        </li>
                    </ol>
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
let fetchToken = 0;

const queryWord = computed(() => (formData.value.q || "").replace(/lemma:/g, "").replace(/"/g, "").trim());
const nodes = computed(() => result.value?.nodes || []);

function runDetection() {
    const myToken = ++fetchToken;
    loading.value = true;
    failed.value = false;
    result.value = null;
    $http
        .get(`${$dbUrl}/scripts/get_sense_periods.py`, { params: paramsFilter(formData.value) })
        .then((response) => {
            if (myToken !== fetchToken) return;
            result.value = response.data;
            emit("filterList", response.data?.filter_list || []);
            emit("stopwordsMissing", Boolean(response.data?.stopwords_missing));
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
    if (w.phrase) return w.phrase;
    if (group === 1) return `${w.word} ${queryWord.value}`;
    if (group === 2) return `${queryWord.value} ${w.word}`;
    return w.word;
}

function label(node) {
    const words = node.words.slice(0, 3).map((w) => w.word);
    if (words.length < 2) {
        words.push(...node.before.slice(0, 1).map((w) => wordText(w, 1)));
        words.push(...node.after.slice(0, 1).map((w) => wordText(w, 2)));
    }
    return words.join(", ") || queryWord.value;
}

// The catalogue's author and title, short enough for a column: the name without its dates, the title's first words
function shortAuthor(author) {
    const parts = (author || "").split(",").map((p) => p.trim());
    const name = parts.slice(0, 2).filter((p, i) => i === 0 || !/\d/.test(p));
    return name.join(", ");
}

function shortTitle(title) {
    const clean = (title || "").replace(/\s+/g, " ").trim();
    return clean.length > 48 ? `${clean.slice(0, 46).replace(/[\s,.;:]+\S*$/, "")}…` : clean;
}

function workName(w) {
    return [w.author, w.title].filter(Boolean).join(", ") || t("sensePeriods.untitled");
}

function percent(share) {
    return `${Math.round(share * 100)}%`;
}

function times(r) {
    return r >= 10 ? Math.round(r) : r.toFixed(1);
}

function corpusText(w) {
    return w.corpus === null ? "" : ` ${t("sensePeriods.corpusTitle", { corpus: times(w.corpus) })}`;
}

function newTitle(w) {
    const rate = w.ratio === null ? t("sensePeriods.notBefore") : t("sensePeriods.newTitle", { ratio: times(w.ratio) });
    return `${rate}${corpusText(w)}`;
}

function startTitle(w) {
    const rate = w.ratio === null ? t("sensePeriods.notAfter") : t("sensePeriods.startTitle", { ratio: times(w.ratio) });
    return rate;
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
    return p.start === p.end ? `${p.start}` : `${p.start}–${p.end}`;
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

.periods {
    display: flex;
    gap: 0;
    overflow-x: auto;
    list-style: none;
    padding: 0 0 0.5rem 0;
    margin: 0;
}

.period {
    flex: 1 0 10.5rem;
    max-width: 16rem;
    padding: 0.25rem 0.6rem 0.25rem 0.6rem;
    border-left: 1px solid #dee2e6;
}

.period:first-child {
    border-left: none;
    padding-left: 0;
}

.period-years {
    font-weight: 600;
}

.period-size {
    font-size: 0.75rem;
    color: #6c757d;
    margin-bottom: 0.3rem;
}

.period-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    color: #6c757d;
    margin-top: 0.3rem;
}

.chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
    margin-top: 0.15rem;
}

.chip {
    border: 1px solid transparent;
    border-radius: 0.5rem;
    padding: 0.05rem 0.4rem;
    font-size: 0.85rem;
    line-height: 1.3;
    color: #212529;
    background: #f1f3f5;
}

.chip:hover,
.chip:focus-visible {
    border-color: #495057;
}

.chip-start {
    background: #fff;
    border-style: dashed;
}

.fading {
    font-size: 0.8rem;
    color: #6c757d;
    margin-top: 0.3rem;
}

.works {
    margin-top: 0.3rem;
}

.work-link {
    display: block;
    text-align: left;
    font-size: 0.78rem;
    line-height: 1.25;
    margin-top: 0.15rem;
}
</style>
