<template>
    <div class="row my-3 pe-1" style="padding: 0 0.5rem" v-if="resultsLength">
        <!-- Frequency table -->
        <div class="col-12"
            :class="outlierPanels.length > 0 ? 'col-md-3 col-xl-2' : 'col-md-4 col-xl-3'">
            <div class="card shadow-sm collocate-table-card">
                <table class="table table-borderless caption-top mb-0"
                    :aria-label="$t('collocation.collocatesTable')">
                    <caption class="visually-hidden">
                        {{ $t('collocation.collocatesTableCaption') }}
                    </caption>
                    <thead class="table-header">
                        <tr>
                            <th scope="col" id="collocate-header">{{ $t("collocation.collocate") }}</th>
                            <th scope="col" id="count-header">{{ $t("collocation.count") }}</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="line-height: 1.5rem" v-for="(word, index) in sortedList"
                            :key="word.collocate" :tabindex="0" @click="onCollocateClick(word)"
                            @keydown.enter="onCollocateClick(word)"
                            @keydown.space.prevent="onCollocateClick(word)"
                            :aria-label="`${word.collocate} ${word.count}`">
                            <td class="text-view" :id="`collocate-${index}`">{{ word.collocate }}</td>
                            <td :id="`count-${index}`">{{ word.count }}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Word cloud -->
        <div class="col-12"
            :class="outlierPanels.length > 0 ? 'col-md-5 col-xl-6' : 'col-md-8 col-xl-9'">
            <div class="card shadow-sm">
                <word-cloud v-if="sortedList.length > 0" :word-weights="sortedList"
                    label="frequency" :click-handler="onCollocateClick"></word-cloud>
            </div>
        </div>

        <!-- Distinctive groups -->
        <div v-if="outlierPanels.length > 0" class="col-12 col-md-4 mt-3 mt-md-0">
            <div class="card shadow-sm">
                <div class="card-header p-2">
                    <h3 class="mb-0" aria-live="polite" aria-atomic="true">
                        {{ activeOutlierLabel
                            ? $t('collocation.mostDistinctiveByField', { field: activeOutlierLabel })
                            : $t('collocation.distinctiveGroups') }}
                        <span v-if="hasActiveFilter" class="scope-suffix">
                            {{ $t('collocation.withinFilteredScope') }}
                        </span>
                    </h3>
                </div>
                <ul class="nav nav-tabs" id="distinctive-tabs" role="tablist"
                    :aria-label="$t('collocation.distinctiveGroups')">
                    <li v-for="panel in outlierPanels" :key="panel.field" class="nav-item" role="presentation">
                        <button type="button" class="nav-link"
                            :class="{ active: panel.field === activeOutlierField }"
                            @click="activeOutlierField = panel.field" role="tab"
                            :aria-selected="panel.field === activeOutlierField"
                            :aria-controls="`distinctive-pane-${panel.field}`">
                            {{ panel.label }}
                        </button>
                    </li>
                </ul>

                <!-- Min-hits incrementer -->
                <div class="d-flex align-items-center px-3 pt-3 pb-1" style="gap: 0.5rem">
                    <label for="min-hits-input" class="fw-bold mb-0 text-nowrap">
                        {{ $t('collocation.minHitsLabel', { field: activeOutlierLabel }) }}
                    </label>
                    <input id="min-hits-input" type="number" class="form-control form-control-sm"
                        style="width: 3.75rem" min="1" max="100" step="1" v-model.number="minHits"
                        @input="onMinHitsChange" />
                </div>
                <div class="visually-hidden" role="status" aria-live="polite" aria-atomic="true">
                    {{ outlierStatus }}
                </div>

                <div class="tab-content">
                    <div v-for="panel in outlierPanels" :key="panel.field"
                        v-show="panel.field === activeOutlierField" :id="`distinctive-pane-${panel.field}`"
                        role="tabpanel" :aria-label="$t('collocation.outlierResults', { field: panel.label })">
                        <div v-if="panel.loading"
                            class="d-flex flex-column align-items-center justify-content-center py-4"
                            role="status" aria-live="polite" aria-atomic="true"
                            :aria-label="$t('collocation.gatheringOutliers', { field: panel.label })">
                            <progress-spinner :lg="true"
                                :message="$t('collocation.gatheringOutliers', { field: panel.label })" />
                            <p class="mt-3 mb-0 text-muted small" aria-hidden="true">
                                {{ $t('collocation.gatheringOutliers', { field: panel.label }) }}...
                            </p>
                        </div>
                        <GroupRanking v-else embedded :items="panel.items"
                            :explainer-label="$t('collocation.distinctiveCollocates')"
                            :select-aria-label-prefix="$t('collocation.compareTo')"
                            :score-aria-label="$t('collocation.score')"
                            :region-aria-label="$t('collocation.outlierResults', { field: panel.label })"
                            :format-score="(n) => n.toFixed(1)"
                            :enable-view-passages="true"
                            :view-passages-label="$t('distinctivePassages.viewFor')"
                            @select="(name) => onOutlierSelect(name, panel.field)"
                            @view-passages="(item) => onViewPassages(item, panel.field)" />
                    </div>
                </div>
            </div>
        </div>

        <DistinctivePassagesModal
            :group-name="modal.groupName" :signature="modal.signature"
            :passages="modal.passages" :loading="modal.loading"
            :has-more="modal.hasMore" :view-all-url="modal.viewAllUrl"
            @load-more="loadMorePassages" @view-all="onViewAll" />
    </div>
</template>

<script setup>
import { storeToRefs } from "pinia";
import { computed, inject, onMounted, ref } from "vue";
import { useI18n } from "vue-i18n";
import { useRouter } from "vue-router";
import { useMainStore } from "../../stores/main";
import {
    collocateCleanup,
    concordanceMethod,
    debug,
    extractSurfaceFromCollocate,
    paramsFilter,
    paramsToRoute,
} from "../../utils.js";
import { Modal } from "bootstrap";
import DistinctivePassagesModal from "../DistinctivePassagesModal.vue";
import GroupRanking from "../GroupRanking.vue";
import ProgressSpinner from "../ProgressSpinner";
import WordCloud from "../WordCloud.vue";

const props = defineProps({
    sortedList: { type: Array, required: true },
    resultsLength: { type: Number, default: 0 },
});

const emit = defineEmits(["pivot-to-compare"]);

// Self-trigger on remount when primary results are already in the parent.
// First mount has empty sortedList; parent's runPostFetchModeAction triggers there.
onMounted(() => {
    if (props.sortedList.length > 0) fetchOutliers();
});

const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const philoConfig = inject("$philoConfig");
const router = useRouter();
const { t } = useI18n();
const store = useMainStore();
const { formData } = storeToRefs(store);

const outlierPanels = ref([]);
const activeOutlierField = ref("");
const minHits = ref(10);
const outlierStatus = ref("");
let outlierFetchToken = 0;
let outlierRerankToken = 0;
let minHitsDebounce = null;

// Drives the "(within filtered scope)" header suffix: when a metadata filter
// narrows the query, distinctive-groups compare to peers in scope, not the corpus.
const hasActiveFilter = computed(() => {
    const fields = Object.keys(philoConfig.metadata_aliases || {});
    return fields.some((f) => {
        const v = formData.value[f];
        return v && (typeof v !== "string" || v.trim() !== "");
    });
});

const activeOutlierLabel = computed(() => {
    const panel = outlierPanels.value.find((p) => p.field === activeOutlierField.value);
    return panel ? panel.label : "";
});

function onCollocateClick(item) {
    router.push(
        paramsToRoute({
            ...formData.value,
            report: "concordance",
            q: collocateCleanup(item, formData.value.q),
            method: concordanceMethod(formData.value.colloc_within),
            cooc_order: "no",
        })
    );
}

// Sequential per-field cascade — avoids parallel map_field requests.
async function fetchOutliers() {
    const aliases = philoConfig.metadata_aliases || {};
    // Skip fields already pinned by the primary filters (author=Holbach → Author tab is moot).
    const fields = (philoConfig.collocation_fields_to_compare || []).filter((f) => {
        const v = formData.value[f];
        return !v || (typeof v === "string" && v.trim() === "");
    });
    const myToken = ++outlierFetchToken;

    outlierPanels.value = fields.map((f) => ({
        field: f,
        label: aliases[f] || f,
        items: [],
        loading: true,
        done: false,
        filePath: "",
    }));
    if (fields.length > 0) activeOutlierField.value = fields[0];

    for (let i = 0; i < outlierPanels.value.length; i++) {
        if (myToken !== outlierFetchToken) return;
        const panel = outlierPanels.value[i];
        try {
            const mapResp = await $http.get(`${$dbUrl}/reports/collocation.py`, {
                params: {
                    ...paramsFilter(formData.value),
                    map_field: panel.field,
                },
            });
            if (myToken !== outlierFetchToken) return;
            panel.filePath = mapResp.data.file_path;
            const outResp = await $http.get(`${$dbUrl}/scripts/get_outlier_groups.py`, {
                params: { file_path: panel.filePath, min_hits: minHits.value },
            });
            if (myToken !== outlierFetchToken) return;
            panel.items = (outResp.data.outliers || []).filter(
                (item) => item[0] && String(item[0]).trim() !== ""
            );
        } catch (error) {
            debug({ $options: { name: "collocation-frequency" } }, error);
        } finally {
            if (myToken === outlierFetchToken) {
                panel.loading = false;
                panel.done = true;
            }
        }
    }
}

// Re-rank from the cached .npz; in-flight cascade picks up minHits.value itself.
async function rerankOutliers() {
    const myToken = ++outlierRerankToken;
    for (const panel of outlierPanels.value) {
        if (myToken !== outlierRerankToken) return;
        if (!panel.done || !panel.filePath) continue;
        try {
            const outResp = await $http.get(`${$dbUrl}/scripts/get_outlier_groups.py`, {
                params: { file_path: panel.filePath, min_hits: minHits.value },
            });
            if (myToken !== outlierRerankToken) return;
            panel.items = (outResp.data.outliers || []).filter(
                (item) => item[0] && String(item[0]).trim() !== ""
            );
        } catch (error) {
            debug({ $options: { name: "collocation-frequency" } }, error);
        }
    }
    if (myToken === outlierRerankToken) {
        outlierStatus.value = t("collocation.outliersUpdated", { value: minHits.value });
    }
}

function onMinHitsChange() {
    if (minHitsDebounce) clearTimeout(minHitsDebounce);
    minHitsDebounce = setTimeout(rerankOutliers, 500);
}

function onOutlierSelect(name, field) {
    // Came from a distinctive listing → land on the over-represented tab.
    emit("pivot-to-compare", { field, name, focusDistinctive: true });
}

// Modal state for representative passages
const modal = ref({
    groupName: "",
    field: "",
    value: "",
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

function onViewPassages(item, field) {
    const [name, , explainers] = item;
    const escapedValue = `"${String(name).replace(/"/g, '\\"')}"`;
    modal.value = {
        groupName: name,
        field,
        value: escapedValue,
        // explainers from GroupRanking are bare words (no z); endpoint accepts
        // signature_weights optional and falls back to unweighted coverage if absent.
        signature: (explainers || []).map((w) => ({ word: w, z: null })),
        passages: [],
        loading: true,
        hasMore: false,
        offset: 0,
        pageSize: 25,
        viewAllUrl: paramsToRoute({
            ...formData.value,
            report: "concordance",
            method: concordanceMethod(formData.value.colloc_within),
            [field]: escapedValue,
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

function onViewAll() {
    const dest = modal.value.viewAllUrl;
    if (modalInstance) modalInstance.hide();
    if (dest) router.push(dest);
}

async function fetchPassages({ append }) {
    const myToken = ++passagesFetchToken;
    modal.value.loading = true;
    try {
        const params = {
            ...paramsFilter(formData.value),
            restrict_to_field: modal.value.field,
            restrict_to_value: modal.value.value,
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
        debug({ $options: { name: "collocation-frequency" } }, error);
    } finally {
        if (myToken === passagesFetchToken) modal.value.loading = false;
    }
}

function loadMorePassages() {
    modal.value.offset += modal.value.pageSize;
    fetchPassages({ append: true });
}

function reset() {
    outlierPanels.value = [];
    activeOutlierField.value = "";
    outlierFetchToken++;
}

defineExpose({ fetchOutliers, reset });
</script>

<style lang="scss" scoped>
@use "../../assets/styles/theme.module.scss" as theme;

th {
    font-variant: small-caps;
    background-color: theme.$card-header-color !important;
    color: white !important;
    border-color: theme.$card-header-color !important;
}

.collocate-table-card {
    max-height: clamp(20rem, 65vh, 48rem);
    overflow-y: auto;
}

.collocate-table-card .table-header th {
    position: sticky;
    top: 0;
    z-index: 1;
}

.table th,
.table td {
    padding: 0.45rem 0.75rem;
}

.table tbody tr {
    cursor: pointer;
    transition: all 0.2s ease;
    border: 1px solid transparent;
}

.table tbody tr:hover {
    transform: scale(1.01) !important;
    background-color: rgba(theme.$link-color, 0.15) !important;
    border: 1px solid rgba(theme.$link-color, 0.3) !important;
    box-shadow: inset 0 0 8px rgba(theme.$link-color, 0.1) !important;
    cursor: pointer !important;
    z-index: 1 !important;
}

.table tbody tr:hover td {
    background-color: rgba(theme.$link-color, 0.15) !important;
    color: inherit !important;
}

.table tbody tr:active {
    transform: scale(0.98);
}

.table tbody tr:focus {
    background-color: rgba(theme.$button-color, 0.15) !important;
    color: theme.$button-color !important;
    outline: 2px solid theme.$button-color;
    outline-offset: -2px;
}

.table tbody tr:focus td {
    color: inherit;
}

.nav-link {
    border-bottom: 1px solid #dee2e6;
    background-color: #fff;
    color: theme.$link-color;
}

.nav-link.active {
    color: theme.$link-color;
    font-weight: bold;
}

.card-header {
    text-align: center;

    h3,
    h4 {
        width: 100%;
        font-variant: small-caps;
        font-size: 1rem;
    }
}

.scope-suffix {
    font-variant: normal;
    font-size: 0.8rem;
    font-weight: 400;
    opacity: 0.75;
    margin-left: 0.25rem;
}
</style>
