<template>
    <div>
        <!-- Field selector + criteria -->
        <div class="card mx-2 p-3" style="border-top-width: 0;">
            <div class="d-flex align-items-center flex-wrap mt-2">
                <label for="similarity-field-select" class="me-2 fw-bold">
                    {{ $t("collocation.compareBy") }}
                </label>
                <div class="btn-group" style="width: fit-content;" role="group">
                    <button class="btn btn-secondary dropdown-toggle" type="button" id="similarity-field-select"
                        data-bs-toggle="dropdown" aria-expanded="false"
                        :aria-label="`${$t('collocation.compareBy')}: ${similarFieldSelected || $t('collocation.selectField')}`">
                        {{ similarFieldSelected || $t('collocation.selectField') }}
                    </button>
                    <ul class="dropdown-menu" aria-labelledby="similarity-field-select">
                        <li v-for="field in fieldsToCompare" :key="field.value">
                            <button type="button" class="dropdown-item" @click="runSimilar(field.value)">
                                {{ field.label }}
                            </button>
                        </li>
                    </ul>
                </div>
                <span class="ms-2">{{ $t("collocation.mostSimilarUsage") }}</span>
            </div>

            <bibliography-criteria class="ms-2 mt-3" :biblio="biblio" :query-report="formData.report"
                :results-length="resultsLength" :hide-criteria-string="true"></bibliography-criteria>

            <div class="mt-2" style="display: flex; align-items: center;" v-if="similarSearching">
                <div class="alert alert-info p-1 mb-0 d-inline-block" style="width: fit-content" role="alert"
                    aria-live="polite" aria-atomic="true">
                    {{ similarSearchProgress }}...
                </div>
                <progress-spinner class="px-2" />
            </div>
        </div>

        <!-- Similar usage results -->
        <div class="mx-2" style="margin-bottom: 6rem;">
            <GroupRanking :title="$t('collocation.similarUsagePattern')" :items="similarDistributions"
                :explainer-label="$t('collocation.sharedCollocates')"
                :select-aria-label-prefix="$t('collocation.compareTo')"
                :score-aria-label="$t('collocation.count')"
                :region-aria-label="$t('collocation.similarUsageResults')"
                :hidden-header="$t('collocation.similarUsageResults')" @select="onSimilarSelect" />
        </div>
    </div>
</template>

<script setup>
import { storeToRefs } from "pinia";
import { inject, ref } from "vue";
import { useI18n } from "vue-i18n";
import { useMainStore } from "../../stores/main";
import {
    debug,
    extractSurfaceFromCollocate,
} from "../../utils.js";
import BibliographyCriteria from "../BibliographyCriteria";
import GroupRanking from "../GroupRanking.vue";
import ProgressSpinner from "../ProgressSpinner";

const props = defineProps({
    biblio: { type: Object, required: true },
    resultsLength: { type: Number, default: 0 },
    collocatesFilePath: { type: String, default: "" },
    fieldsToCompare: { type: Array, required: true },
});

const emit = defineEmits(["pivot-to-compare", "field-selected"]);

const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const { t } = useI18n();
const store = useMainStore();
const { formData } = storeToRefs(store);

const similarDistributions = ref([]);
const cachedDistributions = ref("");
const similarFieldSelected = ref("");
const similarSearchProgress = ref("");
const similarSearching = ref(false);

function runSimilar(fieldValue) {
    if (!fieldValue) return;
    similarFieldSelected.value = fieldValue;
    emit("field-selected", fieldValue);
    similarSearching.value = true;
    similarSearchProgress.value = t("collocation.similarCollocGatheringMessage");
    similarDistributions.value = [];
    $http.get(`${$dbUrl}/reports/collocation.py`, {
        params: {
            q: formData.value.q,
            colloc_filter_choice: formData.value.colloc_filter_choice,
            colloc_within: formData.value.colloc_within,
            filter_frequency: formData.value.filter_frequency,
            map_field: fieldValue,
            q_attribute: formData.value.q_attribute || "",
            q_attribute_value: formData.value.q_attribute_value || "",
        },
    }).then((response) => {
        getMostSimilarCollocDistribution(response.data.file_path);
    }).catch((error) => {
        similarSearching.value = false;
        debug({ $options: { name: "collocation-similar" } }, error);
    });
}

function getMostSimilarCollocDistribution(filePath) {
    similarSearchProgress.value = t("collocation.similarCollocCompareMessage");
    $http.get(`${$dbUrl}/scripts/get_similar_collocate_distributions.py`, {
        params: {
            primary_file_path: props.collocatesFilePath,
            file_path: filePath,
        },
    }).then((response) => {
        similarDistributions.value = response.data.similar || [];
        cachedDistributions.value = filePath;
        similarSearching.value = false;
    }).catch((error) => {
        similarSearching.value = false;
        debug({ $options: { name: "collocation-similar" } }, error);
    });
}

function onSimilarSelect(name) {
    // Fetch the chosen group's collocate distribution, then ask the parent to
    // pivot into compare mode using this prebuilt file path. Saves the parent
    // a redundant compare-side primary fetch.
    $http.get(`${$dbUrl}/scripts/get_collocate_distribution.py`, {
        params: { file_path: cachedDistributions.value, field: name },
    }).then((response) => {
        emit("pivot-to-compare", {
            field: similarFieldSelected.value,
            name,
            otherFilePath: response.data.file_path,
            otherCollocates: extractSurfaceFromCollocate(response.data.collocates),
        });
    }).catch((error) => {
        debug({ $options: { name: "collocation-similar" } }, error);
    });
}

function reset() {
    similarDistributions.value = [];
    similarFieldSelected.value = "";
}

defineExpose({ runSimilar, reset });
</script>
