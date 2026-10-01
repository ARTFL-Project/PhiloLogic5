<template>
    <div>
        <div class="d-flex flex-wrap gap-2 align-items-end">
            <div>
                <label class="form-label small mb-0" :for="`${id}-field`">{{ $t("citations.field") }}</label>
                <input :id="`${id}-field`" class="form-control form-control-sm mono" :list="`${id}-fields`" :value="citation.field" :readonly="readonly" @change="set('field', $event.target.value.trim())" />
                <datalist :id="`${id}-fields`">
                    <option v-for="field in metadataFields" :key="field" :value="field"></option>
                </datalist>
            </div>
            <div>
                <label class="form-label small mb-0" :for="`${id}-level`">{{ $t("citations.level") }}</label>
                <select :id="`${id}-level`" class="form-select form-select-sm" :value="citation.object_level" :disabled="readonly" @change="set('object_level', $event.target.value)">
                    <option v-for="level in levels" :key="level" :value="level">{{ level }}</option>
                </select>
            </div>
            <div>
                <label class="form-label small mb-0" :for="`${id}-prefix`">{{ $t("citations.prefix") }}</label>
                <input :id="`${id}-prefix`" class="form-control form-control-sm mono" :value="citation.prefix" :readonly="readonly" @change="set('prefix', $event.target.value)" />
            </div>
            <div>
                <label class="form-label small mb-0" :for="`${id}-suffix`">{{ $t("citations.suffix") }}</label>
                <input :id="`${id}-suffix`" class="form-control form-control-sm mono" :value="citation.suffix" :readonly="readonly" @change="set('suffix', $event.target.value)" />
            </div>
            <div class="form-check mb-1">
                <input :id="`${id}-link`" class="form-check-input" type="checkbox" :checked="citation.link" :disabled="readonly" @change="set('link', $event.target.checked)" />
                <label class="form-check-label small" :for="`${id}-link`">{{ $t("citations.link") }}</label>
            </div>
        </div>
        <div class="form-text">{{ $t("citations.affixHelp") }}</div>
        <div class="mt-2">
            <div class="small">{{ $t("citations.style") }}</div>
            <DictEditor :model-value="citation.style || {}" :key-suggestions="CSS_PROPERTIES" :key-label="$t('citations.property')" :value-label="$t('editors.value')" :readonly="readonly" @update:model-value="set('style', $event)" />
        </div>
    </div>
</template>

<script setup>
import { computed } from "vue";
import { LEVELS } from "../webConfigForms";
import DictEditor from "./DictEditor.vue";

// A citation: a metadata field of an object level, with what precedes and follows it, whether it links, and its CSS
// style. Its other keys, if any, are kept.
const props = defineProps({
    modelValue: { type: Object, required: true },
    metadataFields: { type: Array, default: () => [] },
    readonly: Boolean,
});
const emit = defineEmits(["update:modelValue"]);
const CSS_PROPERTIES = ["font-variant", "font-style", "font-weight", "font-size", "color", "text-transform", "text-decoration"];
const id = `citation-${Math.random().toString(36).slice(2)}`;
const citation = computed(() => props.modelValue || {});
const levels = computed(() => (LEVELS.includes(citation.value.object_level) || !citation.value.object_level ? LEVELS : [...LEVELS, citation.value.object_level]));

function set(key, value) {
    emit("update:modelValue", { ...citation.value, [key]: value });
}
</script>
