<template>
    <div>
        <div v-if="inline.length" class="d-flex flex-wrap gap-3 align-items-end">
            <div v-for="field in inline" :key="field.key" :class="{ 'form-check mb-1': field.type === 'bool' }">
                <template v-if="field.type === 'bool'">
                    <input :id="`${id}-${field.key}`" class="form-check-input" type="checkbox" :checked="!!record[field.key]" :disabled="readonly" @change="set(field, $event.target.checked)" />
                    <label class="form-check-label small" :for="`${id}-${field.key}`">{{ label(field) }}</label>
                </template>
                <template v-else>
                    <label class="form-label small mb-0" :for="`${id}-${field.key}`">{{ label(field) }}</label>
                    <select v-if="field.type === 'level'" :id="`${id}-${field.key}`" class="form-select form-select-sm" :value="record[field.key]" :disabled="readonly" @change="set(field, $event.target.value)">
                        <option v-for="level in levels(record[field.key])" :key="level" :value="level">{{ level }}</option>
                    </select>
                    <input
                        v-else-if="field.type === 'year'"
                        :id="`${id}-${field.key}`"
                        type="number"
                        class="form-control form-control-sm year"
                        :value="record[field.key]"
                        :readonly="readonly"
                        @change="set(field, $event.target.value === '' ? '' : parseInt($event.target.value, 10))"
                    />
                    <input
                        v-else
                        :id="`${id}-${field.key}`"
                        class="form-control form-control-sm"
                        :class="{ mono: field.type !== 'text' }"
                        :list="field.type.endsWith('field') ? fieldsList : null"
                        :value="record[field.key] === null || record[field.key] === undefined ? '' : record[field.key]"
                        :placeholder="field.type === 'optional_field' ? $t('form.none') : ''"
                        :readonly="readonly"
                        @change="setText(field, $event.target.value)"
                    />
                </template>
                <div v-if="help(field)" class="form-text mt-0">{{ help(field) }}</div>
            </div>
        </div>
        <div v-for="field in blocks" :key="field.key" class="mt-2">
            <div class="small">{{ label(field) }}</div>
            <div v-if="help(field)" class="form-text mt-0 mb-1">{{ help(field) }}</div>
            <CitationListEditor v-if="field.type === 'citations'" :model-value="record[field.key] || []" :citations="citations" :metadata-fields="metadataFields" :readonly="readonly" :help="false" @update:model-value="set(field, $event)" />
            <ListEditor v-else-if="field.type === 'strings'" :model-value="record[field.key] || []" :readonly="readonly" @update:model-value="set(field, $event)" />
            <DictEditor v-else-if="field.type === 'string_map'" :model-value="record[field.key] || {}" :readonly="readonly" @update:model-value="set(field, $event)" />
        </div>
        <datalist :id="fieldsList">
            <option v-for="field in metadataFields" :key="field" :value="field"></option>
        </datalist>
    </div>
</template>

<script setup>
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { LEVELS } from "../webConfigForms";
import CitationListEditor from "./CitationListEditor.vue";
import DictEditor from "./DictEditor.vue";
import ListEditor from "./ListEditor.vue";

// An object edited with a form of its fields (see FORMS in webConfigForms.js), keeping the keys the form doesn't know
const props = defineProps({
    modelValue: { type: Object, default: () => ({}) },
    fields: { type: Array, required: true },
    citations: { type: Object, default: () => ({}) },
    metadataFields: { type: Array, default: () => [] },
    readonly: Boolean,
});
const emit = defineEmits(["update:modelValue"]);
const { t, te } = useI18n();
const id = `record-${Math.random().toString(36).slice(2)}`;
const fieldsList = `${id}-fields`;
const BLOCKS = ["citations", "strings", "string_map"];
const record = computed(() => props.modelValue || {});
const shown = computed(() => props.fields.filter((field) => !field.when || record.value[field.when]));
const inline = computed(() => shown.value.filter((field) => !BLOCKS.includes(field.type)));
const blocks = computed(() => shown.value.filter((field) => BLOCKS.includes(field.type)));

const label = (field) => (te(`form.${field.key}`) ? t(`form.${field.key}`) : field.key);
const help = (field) => (te(`formHelp.${field.key}`) ? t(`formHelp.${field.key}`) : "");
const levels = (value) => (!value || LEVELS.includes(value) ? LEVELS : [...LEVELS, value]);

function set(field, value) {
    const changed = { ...record.value, [field.key]: value };
    if (field.clears && (value === null || value === "")) {
        changed[field.clears] = null;
    } else if (field.clears && changed[field.clears] === null) {
        changed[field.clears] = [];
    }
    emit("update:modelValue", changed);
}

function setText(field, value) {
    value = field.type === "text" ? value : value.trim();
    set(field, field.type === "optional_field" && value === "" ? null : value);
}
</script>

<style scoped>
.year {
    width: 8rem;
}
</style>
