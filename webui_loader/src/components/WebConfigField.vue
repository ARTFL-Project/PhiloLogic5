<template>
    <div>
        <JsonEditor v-if="asJson || !fits || option.kind === 'json'" :model-value="modelValue" :readonly="readonly" @update:model-value="update" />
        <div v-else-if="option.kind === 'bool'" class="form-check form-switch">
            <input :id="id" class="form-check-input" type="checkbox" :checked="modelValue" :disabled="readonly" @change="update($event.target.checked)" />
        </div>
        <input v-else-if="option.kind === 'int'" :id="id" type="number" class="form-control w-auto" :value="modelValue" :readonly="readonly" @change="update(parseInt($event.target.value, 10) || 0)" />
        <input v-else-if="option.kind === 'string'" :id="id" class="form-control" :value="modelValue" :readonly="readonly" @change="update($event.target.value)" />
        <template v-else-if="option.kind === 'choice' || option.kind === 'field'">
            <input :id="id" class="form-control mono w-auto" :list="`${id}-choices`" :value="modelValue" :readonly="readonly" @change="update($event.target.value)" />
            <datalist :id="`${id}-choices`">
                <option v-for="choice in option.kind === 'field' ? metadataFields : option.choices" :key="choice" :value="choice"></option>
            </datalist>
        </template>
        <OrderedChoice v-else-if="option.kind === 'reports'" :model-value="modelValue" :choices="option.choices" :readonly="readonly" @update:model-value="update" />
        <OrderedChoice v-else-if="option.kind === 'fields'" :model-value="modelValue" :choices="option.key === 'autocomplete' ? ['q', ...metadataFields] : metadataFields" allow-custom :readonly="readonly" @update:model-value="update" />
        <DictEditor v-else-if="option.kind === 'field_map'" :model-value="modelValue" :value-choices="option.choices" :key-suggestions="metadataFields" :key-label="$t('webConfig.field')" :readonly="readonly" @update:model-value="update" />
        <DictEditor v-else-if="option.kind === 'string_map'" :model-value="modelValue" :key-suggestions="wordAttributes" :key-label="$t('webConfig.wordAttribute')" :value-label="$t('webConfig.shownAs')" :readonly="readonly" @update:model-value="update" />
        <DictListEditor v-else-if="option.kind === 'string_lists'" :model-value="modelValue || {}" :readonly="readonly" @update:model-value="update" />
        <ListEditor v-else-if="option.kind === 'string_list'" :model-value="modelValue" :readonly="readonly" @update:model-value="update" />
        <ReplacementsEditor v-else-if="option.kind === 'replacements'" :model-value="modelValue" :query="option.key === 'query_parser_regex'" :readonly="readonly" @update:model-value="update" />
        <CitationsEditor v-else-if="option.kind === 'citations'" :model-value="modelValue" :metadata-fields="metadataFields" :readonly="readonly" @update:model-value="update" />
        <CitationListEditor v-else-if="option.kind === 'citation_list'" :model-value="modelValue" :citations="citations" :metadata-fields="metadataFields" :readonly="readonly" @update:model-value="update" />
        <SortOrdersEditor v-else-if="option.kind === 'sort_orders'" :model-value="modelValue" :metadata-fields="metadataFields" :readonly="readonly" @update:model-value="update" />
        <ChoiceValuesEditor v-else-if="option.kind === 'choice_values'" :model-value="modelValue" :metadata-fields="metadataFields" :readonly="readonly" @update:model-value="update" />
        <RecordEditor v-else-if="option.kind === 'record' && FORMS[option.key]" :model-value="modelValue" :fields="FORMS[option.key]" :citations="citations" :metadata-fields="metadataFields" :readonly="readonly" @update:model-value="update" />
        <RecordListEditor
            v-else-if="option.kind === 'records' && FORMS[option.key]"
            :model-value="modelValue"
            :fields="FORMS[option.key]"
            :new-item="NEW_ITEMS[option.key]"
            :title-key="ITEM_TITLES[option.key]"
            :citations="citations"
            :metadata-fields="metadataFields"
            :readonly="readonly"
            @update:model-value="update"
        />
        <JsonEditor v-else :model-value="modelValue" :readonly="readonly" @update:model-value="update" />
    </div>
</template>

<script setup>
import { computed } from "vue";
import { FORMS, ITEM_TITLES, NEW_ITEMS, STRUCTURED } from "../webConfigForms";
import ChoiceValuesEditor from "./ChoiceValuesEditor.vue";
import CitationListEditor from "./CitationListEditor.vue";
import CitationsEditor from "./CitationsEditor.vue";
import DictEditor from "./DictEditor.vue";
import DictListEditor from "./DictListEditor.vue";
import JsonEditor from "./JsonEditor.vue";
import ListEditor from "./ListEditor.vue";
import OrderedChoice from "./OrderedChoice.vue";
import RecordEditor from "./RecordEditor.vue";
import RecordListEditor from "./RecordListEditor.vue";
import ReplacementsEditor from "./ReplacementsEditor.vue";
import SortOrdersEditor from "./SortOrdersEditor.vue";

const props = defineProps({
    id: String,
    option: { type: Object, required: true },
    modelValue: { default: null },
    readonly: Boolean,
    metadataFields: { type: Array, default: () => [] },
    // The named citations (the citations option), which lists of citations use
    citations: { type: Object, default: () => ({}) },
    // The word attributes of the database, and lemma
    wordAttributes: { type: Array, default: () => [] },
    // Options with a form of their own (STRUCTURED) can also be edited as JSON, for what the form can't show
    asJson: Boolean,
});
const emit = defineEmits(["update:modelValue"]);
// As JSON too if their value isn't what the form expects
const structured = computed(() => STRUCTURED.includes(props.option.kind));
const isObject = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
const FITS = {
    string_map: isObject,
    string_lists: (value) => isObject(value) && Object.values(value).every(Array.isArray),
    replacements: (value) => Array.isArray(value) && value.every((pair) => Array.isArray(pair) && pair.length === 2),
    citations: (value) => isObject(value) && Object.values(value).every(isObject),
    citation_list: (value) => Array.isArray(value) && value.every(isObject),
    sort_orders: (value) => Array.isArray(value) && value.every(Array.isArray),
    choice_values: (value) => isObject(value) && Object.values(value).every(Array.isArray),
    record: isObject,
    records: (value) => Array.isArray(value) && value.every(isObject),
};
const fits = computed(() => !structured.value || FITS[props.option.kind](props.modelValue));

function update(value) {
    emit("update:modelValue", value);
}
</script>
