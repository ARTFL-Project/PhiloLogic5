<template>
    <div>
        <!-- Controls -->
        <div class="card shadow-sm mx-2 p-3" style="border-top-width: 0;">
            <bibliography-criteria :biblio="biblio" :query-report="formData.report"
                :results-length="resultsLength" />
            <p class="text-muted small mb-2 mt-2">
                {{ $t('phaseShifts.description') }}
            </p>
            <button type="button" class="btn btn-secondary mt-2" style="width: fit-content"
                :disabled="loading" @click="runDetection">
                {{ loading ? $t('common.loading') : $t('phaseShifts.detect') }}
            </button>
        </div>

        <!-- Initial loading -->
        <div v-if="loading && !result" class="text-center py-5">
            <progress-spinner :lg="true" :message="$t('phaseShifts.detecting')" />
        </div>

        <!-- Results -->
        <div v-if="result && result.phases.length > 0" class="mx-2 my-3">
            <!-- Summary -->
            <p class="text-muted small mb-3">
                {{ $t('phaseShifts.summary', {
                    hits: result.n_total_hits,
                    start: result.year_range[0],
                    end: result.year_range[1],
                }) }}
            </p>

            <!-- Frequency anchor: per-year hit counts -->
            <div v-if="result.frequency && result.frequency.length > 1" class="mb-3">
                <svg :viewBox="`0 0 ${freqWidth} ${freqHeight}`" class="frequency-chart"
                    preserveAspectRatio="none" :aria-label="$t('phaseShifts.frequencyChart')">
                    <!-- Phase background bands -->
                    <rect v-for="(phase, i) in result.phases" :key="`band-${i}`"
                        :x="yearToX(phase.start_year)"
                        :y="0"
                        :width="yearToX(phase.end_year + 1) - yearToX(phase.start_year)"
                        :height="freqHeight"
                        :fill="phaseColor(i, 0.08)" />
                    <!-- Frequency bars -->
                    <rect v-for="([year, n], i) in result.frequency" :key="`f-${i}`"
                        :x="yearToX(year)" :y="freqHeight - barHeight(n)"
                        :width="Math.max(1, freqWidth / (result.year_range[1] - result.year_range[0] + 1) - 0.5)"
                        :height="barHeight(n)" fill="#666" />
                </svg>
                <div class="d-flex justify-content-between text-muted small">
                    <span>{{ result.year_range[0] }}</span>
                    <span>{{ result.year_range[1] }}</span>
                </div>
            </div>

            <!-- Slider -->
            <div class="d-flex align-items-center mb-3 gap-3">
                <label for="n-phases-slider" class="form-label mb-0">
                    <strong>{{ $t('phaseShifts.nPhasesLabel') }}:</strong> {{ nPhases }}
                </label>
                <input type="range" id="n-phases-slider" class="form-range flex-grow-1"
                    :min="result.min_phases + 1" :max="result.max_phases"
                    v-model.number="nPhases" @change="onSliderChange"
                    :aria-valuemin="result.min_phases + 1" :aria-valuemax="result.max_phases"
                    :aria-valuenow="nPhases" />
                <button type="button" class="btn btn-sm btn-outline-secondary"
                    @click="resetToDefault" :disabled="nPhases === result.default_n_phases">
                    {{ $t('phaseShifts.reset', { n: result.default_n_phases }) }}
                </button>
            </div>

            <!-- Phase cards -->
            <div class="row" :class="{ 'opacity-50': loading }">
                <div v-for="(phase, i) in result.phases" :key="i" class="col-12 mb-3">
                    <article class="card phase-card shadow-sm">
                        <div class="card-header phase-header py-2 d-flex justify-content-between align-items-center"
                            :style="{ borderLeft: `4px solid ${phaseColor(i, 1)}` }">
                            <div>
                                <strong>{{ $t('phaseShifts.phaseN', { n: i + 1 }) }}</strong>
                                <span class="ms-2">
                                    {{ phase.start_year }}–{{ phase.end_year }}
                                </span>
                                <span class="text-muted ms-2 small">
                                    ({{ $t('phaseShifts.nHits', { n: phase.n_hits }) }})
                                </span>
                            </div>
                            <button type="button" class="btn btn-sm btn-link p-0"
                                @click="onViewPassages(phase, i)">
                                {{ $t('phaseShifts.viewPassages') }} →
                            </button>
                        </div>
                        <div class="card-body py-2">
                            <span v-for="(w, j) in phase.top_words" :key="j" class="word-chip">
                                {{ w.word }}
                            </span>
                        </div>
                    </article>
                </div>
            </div>
        </div>

        <div v-else-if="result && result.phases.length === 0" class="text-center py-5 text-muted">
            {{ $t('phaseShifts.noResults') }}
        </div>

        <DistinctivePassagesModal
            :group-name="modal.groupName" :signature="modal.signature"
            :passages="modal.passages" :loading="modal.loading"
            :has-more="modal.hasMore" :view-all-url="modal.viewAllUrl"
            @load-more="loadMorePassages" @view-all="onViewAllPassages" />
    </div>
</template>

<script setup>
import { computed, inject, ref } from "vue";
import { storeToRefs } from "pinia";
import { useRouter } from "vue-router";
import { Modal } from "bootstrap";
import { useMainStore } from "../../stores/main";
import { concordanceMethod, debug, paramsFilter, paramsToRoute } from "../../utils.js";
import BibliographyCriteria from "../BibliographyCriteria";
import DistinctivePassagesModal from "../DistinctivePassagesModal.vue";
import ProgressSpinner from "../ProgressSpinner";

defineProps({
    biblio: { type: Object, required: true },
    resultsLength: { type: Number, default: 0 },
});

const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const router = useRouter();
const store = useMainStore();
const { formData } = storeToRefs(store);

const loading = ref(false);
const result = ref(null);
const nPhases = ref(0);
let fetchToken = 0;
let sliderDebounce = null;

const modal = ref({
    groupName: "",
    yearRange: "",
    signature: [],
    passages: [],
    loading: false,
    hasMore: false,
    offset: 0,
    pageSize: 25,
    viewAllUrl: "",
});
let modalInstance = null;
let passagesFetchToken = 0;

const freqWidth = 1000;
const freqHeight = 60;

const yearSpan = computed(() => {
    if (!result.value) return 1;
    const [a, b] = result.value.year_range;
    return Math.max(1, b - a + 1);
});

function yearToX(year) {
    if (!result.value) return 0;
    return ((year - result.value.year_range[0]) / yearSpan.value) * freqWidth;
}

function barHeight(n) {
    if (!result.value || result.value.frequency.length === 0) return 0;
    const max = Math.max(...result.value.frequency.map(([, c]) => c));
    if (max === 0) return 0;
    return Math.max(1, (n / max) * (freqHeight - 2));
}

const phaseHues = [205, 30, 145, 280, 0, 90, 165, 235, 50, 315, 120, 260, 15];
function phaseColor(i, alpha) {
    const h = phaseHues[i % phaseHues.length];
    return `hsla(${h}, 50%, 45%, ${alpha})`;
}

function runDetection() {
    fetchPhases({ n_phases: undefined });
}

function onSliderChange() {
    if (sliderDebounce) clearTimeout(sliderDebounce);
    sliderDebounce = setTimeout(() => fetchPhases({ n_phases: nPhases.value }), 200);
}

function resetToDefault() {
    nPhases.value = result.value.default_n_phases;
    fetchPhases({ n_phases: undefined });
}

async function fetchPhases({ n_phases }) {
    const myToken = ++fetchToken;
    loading.value = true;
    try {
        const params = { ...paramsFilter(formData.value) };
        if (n_phases !== undefined) params.n_phases = n_phases;
        const resp = await $http.get(`${$dbUrl}/scripts/get_phase_shifts.py`, { params });
        if (myToken !== fetchToken) return;
        result.value = resp.data;
        if (n_phases === undefined) {
            nPhases.value = result.value.default_n_phases;
        }
    } catch (error) {
        debug({ $options: { name: "phase-timeline" } }, error);
    } finally {
        if (myToken === fetchToken) loading.value = false;
    }
}

function reset() {
    result.value = null;
    nPhases.value = 0;
    fetchToken++;
}

// ---- Passage modal ----
function onViewPassages(phase, idx) {
    const yearRange = `${phase.start_year}-${phase.end_year}`;
    modal.value = {
        groupName: `${formData.value.q} · ${phase.start_year}–${phase.end_year}`,
        yearRange,
        signature: (phase.top_words || []).map((w) => ({ word: w.word, z: w.z })),
        passages: [],
        loading: true,
        hasMore: false,
        offset: 0,
        pageSize: 25,
        viewAllUrl: paramsToRoute({
            ...formData.value,
            report: "concordance",
            method: concordanceMethod(formData.value.colloc_within),
            year: yearRange,
        }),
    };
    showModal();
    fetchPassages({ append: false });
}

function showModal() {
    const el = document.getElementById("distinctive-passages-modal");
    if (!el) return;
    if (!modalInstance) modalInstance = new Modal(el);
    modalInstance.show();
}

async function fetchPassages({ append }) {
    const myToken = ++passagesFetchToken;
    modal.value.loading = true;
    try {
        const params = {
            ...paramsFilter(formData.value),
            restrict_to_field: "year",
            restrict_to_value: modal.value.yearRange,
            signature_tokens: modal.value.signature.map((s) => s.word).join(","),
            top_n: modal.value.pageSize,
            offset: modal.value.offset,
        };
        const resp = await $http.get(`${$dbUrl}/scripts/get_representative_passages.py`, { params });
        if (myToken !== passagesFetchToken) return;
        const data = resp.data || {};
        modal.value.passages = append
            ? modal.value.passages.concat(data.passages || [])
            : (data.passages || []);
        modal.value.hasMore = !!data.has_more;
    } catch (error) {
        debug({ $options: { name: "phase-timeline" } }, error);
    } finally {
        if (myToken === passagesFetchToken) modal.value.loading = false;
    }
}

function loadMorePassages() {
    modal.value.offset += modal.value.pageSize;
    fetchPassages({ append: true });
}

function onViewAllPassages() {
    const dest = modal.value.viewAllUrl;
    if (modalInstance) modalInstance.hide();
    if (dest) router.push(dest);
}

defineExpose({ runDetection, reset });
</script>

<style lang="scss" scoped>
@use "../../assets/styles/theme.module.scss" as theme;

.frequency-chart {
    width: 100%;
    height: 60px;
    background-color: #fafafa;
    border: 1px solid #eee;
    border-radius: 0.25rem;
}

.phase-card {
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.phase-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08) !important;
}

.phase-header {
    background-color: #fafafa;
    border-bottom: 1px solid #eee;
}

.word-chip {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    margin: 0.15rem 0.25rem 0.15rem 0;
    background-color: rgba(theme.$link-color, 0.08);
    border: 1px solid rgba(theme.$link-color, 0.2);
    border-radius: 0.5rem;
    color: theme.$link-color;
    font-size: 0.85rem;
}
</style>
