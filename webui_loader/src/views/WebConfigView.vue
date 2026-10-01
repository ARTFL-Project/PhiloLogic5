<template>
    <div class="medium mx-auto" v-if="config && schema">
        <div class="d-flex flex-wrap align-items-center gap-2 mb-2">
            <h1 class="h3 mb-0">{{ $t("webConfig.title") }} <span class="font-monospace">{{ name }}</span></h1>
            <a class="btn btn-sm btn-outline-secondary ms-auto" :href="databaseUrl" target="_blank" rel="noopener"><i class="bi bi-box-arrow-up-right me-1"></i>{{ $t("databases.open") }}</a>
        </div>
        <p class="small text-body-secondary">{{ $t("webConfig.help") }}</p>
        <div v-if="!config.writable" class="alert alert-secondary"><i class="bi bi-lock me-2"></i>{{ $t("webConfig.readOnly") }}: {{ config.reason }}</div>
        <div v-if="config.error" class="alert alert-danger">{{ config.error }}</div>
        <div v-if="saved" class="alert alert-success py-2">
            <i class="bi bi-check-circle me-2"></i>{{ $t("webConfig.saved") }} <span class="small mono">{{ saved }}</span>
        </div>
        <div v-if="error" class="alert alert-danger py-2">{{ error }}</div>
        <ul class="nav nav-tabs mb-3">
            <li class="nav-item" v-for="group in schema.groups" :key="group">
                <button class="nav-link" :class="{ active: group === current }" type="button" @click="current = group">
                    {{ $t(`webConfig.groups.${group}`) }}<span v-if="groupChanged(group)" class="text-primary"> &bull;</span>
                </button>
            </li>
        </ul>
        <div v-for="option in groupOptions" :key="option.key" class="mb-4" :class="{ changed: isChanged(option.key) }">
            <div class="d-flex align-items-baseline gap-2">
                <label class="form-label option-label mb-1 mono" :for="`web-${option.key}`">{{ option.key }}</label>
                <span v-if="option.read_only" class="badge text-bg-secondary">{{ $t("webConfig.notEditable") }}</span>
                <span class="ms-auto">
                    <button v-if="STRUCTURED.includes(option.kind)" type="button" class="btn btn-link btn-sm p-0 me-2" @click="asJson[option.key] = !asJson[option.key]">
                        {{ asJson[option.key] ? $t("webConfig.asForm") : $t("webConfig.asJson") }}
                    </button>
                    <template v-if="editable(option)">
                        <button v-if="isChanged(option.key)" type="button" class="btn btn-link btn-sm p-0 me-2" @click="undo(option.key)">{{ $t("webConfig.undo") }}</button>
                        <button v-if="!deepEqual(values[option.key], option.default)" type="button" class="btn btn-link btn-sm p-0" @click="values[option.key] = clone(option.default)">{{ $t("webConfig.default") }}</button>
                    </template>
                </span>
            </div>
            <div class="form-text mt-0 mb-1">{{ option.help }}</div>
            <pre v-if="config.code[option.key]" class="bg-light p-2 rounded mono">{{ config.code[option.key] }}</pre>
            <WebConfigField
                v-else
                :id="`web-${option.key}`"
                :option="option"
                v-model="values[option.key]"
                :readonly="!editable(option)"
                :metadata-fields="metadataFields"
                :citations="values.citations || {}"
                :word-attributes="wordAttributes"
                :character-names="config.character_names || {}"
                :as-json="!!asJson[option.key]"
            />
        </div>
        <div v-if="current === 'general' && Object.keys(config.other).length" class="mb-3">
            <h2 class="h6">{{ $t("webConfig.other") }}</h2>
            <JsonView class="bg-light p-2 rounded" :value="config.other" />
        </div>
        <div class="sticky-actions" v-if="config.writable">
            <div v-if="reviewing && changes.length" class="mb-2 review">
                <h2 class="h6">{{ $t("webConfig.changes") }}</h2>
                <div v-for="key in changes" :key="key" class="small mb-2">
                    <span class="mono fw-semibold">{{ key }}</span>
                    <JsonDiff :before="original[key]" :after="values[key]" />
                </div>
            </div>
            <div class="d-flex gap-2 align-items-center">
                <span class="small">{{ $t("webConfig.changeCount", { count: changes.length }) }}</span>
                <button class="btn btn-outline-secondary btn-sm ms-auto" type="button" :disabled="!changes.length" @click="reviewing = !reviewing">{{ reviewing ? $t("webConfig.hideChanges") : $t("webConfig.showChanges") }}</button>
                <button class="btn btn-outline-secondary btn-sm" type="button" :disabled="!changes.length" @click="undoAll">{{ $t("webConfig.undoAll") }}</button>
                <button class="btn btn-secondary" type="button" :disabled="!changes.length || saving" @click="save">
                    <span v-if="saving" class="spinner-border spinner-border-sm me-1"></span>{{ $t("webConfig.save") }}
                </button>
            </div>
        </div>
    </div>
    <div v-else-if="error" class="alert alert-danger">{{ error }}</div>
    <div v-else class="text-center my-5"><span class="spinner-border"></span></div>
</template>

<script setup>
import { computed, onMounted, reactive, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import { onBeforeRouteLeave } from "vue-router";
import { api } from "../api";
import JsonDiff from "../components/JsonDiff.vue";
import JsonView from "../components/JsonView.vue";
import WebConfigField from "../components/WebConfigField.vue";
import { session } from "../session";
import { clone, deepEqual } from "../utils";
import { STRUCTURED, followCitations } from "../webConfigForms";

const props = defineProps({ name: { type: String, required: true } });
const { t } = useI18n();
const schema = ref(null);
const config = ref(null);
const values = reactive({});
const original = ref({});
const current = ref("general");
const error = ref(null);
const saved = ref(null);
const saving = ref(false);
const reviewing = ref(false);
const asJson = reactive({});

const databaseUrl = computed(() => `${(session.server.url_root || "").replace(/\/$/, "")}/${props.name}`);
const groupOptions = computed(() => schema.value.options.filter((option) => option.group === current.value));
const changes = computed(() => schema.value.options.map((option) => option.key).filter((key) => isChanged(key)));
const isChanged = (key) => !deepEqual(values[key], original.value[key]);
const groupChanged = (group) => schema.value.options.some((option) => option.group === group && isChanged(option.key));
const editable = (option) => config.value.writable && !option.read_only && !config.value.code[option.key];
// The fields offered by the editors: not the ids of objects
const metadataFields = computed(() => config.value.metadata_fields.filter((field) => !field.startsWith("philo_")));
const wordAttributes = computed(() => ["lemma", ...Object.keys(values.word_attributes || {}).filter((attribute) => attribute !== "lemma")]);

// The lists of citations follow the changes of the named citations they use
watch(
    () => clone(values.citations),
    (after, before) => {
        if (config.value && before !== undefined) {
            followCitations(values, before, after, (key) => editable(schema.value.options.find((option) => option.key === key) || { read_only: true }));
        }
    },
);

function load(result) {
    config.value = result;
    original.value = clone(result.values);
    for (const [key, value] of Object.entries(result.values)) {
        values[key] = clone(value);
    }
}

function undo(key) {
    values[key] = clone(original.value[key]);
}

function undoAll() {
    for (const key of changes.value) {
        undo(key);
    }
}

async function save() {
    saving.value = true;
    error.value = null;
    saved.value = null;
    const payload = {};
    for (const key of changes.value) {
        payload[key] = values[key];
    }
    try {
        const result = await api.put(`databases/${props.name}/web_config`, { changes: payload, hash: config.value.hash });
        load(result);
        saved.value = result.backup ? t("webConfig.backup", { path: result.backup }) : "";
        reviewing.value = false;
    } catch (requestError) {
        error.value = requestError.message;
    } finally {
        saving.value = false;
    }
}

onBeforeRouteLeave(() => !changes.value.length || window.confirm(t("webConfig.leave")));

onMounted(async () => {
    try {
        const [schemaResult, configResult] = await Promise.all([api.get("web_config_options"), api.get(`databases/${props.name}/web_config`)]);
        schema.value = schemaResult;
        load(configResult);
    } catch (requestError) {
        error.value = requestError.message;
    }
});
</script>
