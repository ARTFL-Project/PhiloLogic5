<template>
    <div>
        <div v-if="option.kind === 'bool'" class="form-check form-switch">
            <input :id="id" class="form-check-input" type="checkbox" :checked="modelValue" :disabled="readonly" @change="emit('update:modelValue', $event.target.checked)" />
        </div>
        <input v-else-if="option.kind === 'int'" :id="id" type="number" class="form-control w-auto" :value="modelValue" :readonly="readonly" @change="emit('update:modelValue', parseInt($event.target.value, 10) || 0)" />
        <input v-else-if="option.kind === 'string'" :id="id" class="form-control" :value="modelValue" :readonly="readonly" @change="emit('update:modelValue', $event.target.value)" />
        <template v-else-if="option.kind === 'choice' || option.kind === 'field'">
            <input :id="id" class="form-control mono w-auto" :list="`${id}-choices`" :value="modelValue" :readonly="readonly" @change="emit('update:modelValue', $event.target.value)" />
            <datalist :id="`${id}-choices`">
                <option v-for="choice in option.kind === 'field' ? metadataFields : option.choices" :key="choice" :value="choice"></option>
            </datalist>
        </template>
        <OrderedChoice v-else-if="option.kind === 'reports'" :model-value="modelValue" :choices="option.choices" :readonly="readonly" @update:model-value="emit('update:modelValue', $event)" />
        <OrderedChoice v-else-if="option.kind === 'fields'" :model-value="modelValue" :choices="option.key === 'autocomplete' ? ['q', ...metadataFields] : metadataFields" allow-custom :readonly="readonly" @update:model-value="emit('update:modelValue', $event)" />
        <DictEditor v-else-if="option.kind === 'field_map'" :model-value="modelValue" :value-choices="option.choices" :key-suggestions="metadataFields" :key-label="$t('webConfig.field')" :readonly="readonly" @update:model-value="emit('update:modelValue', $event)" />
        <ListEditor v-else-if="option.kind === 'string_list'" :model-value="modelValue" :readonly="readonly" @update:model-value="emit('update:modelValue', $event)" />
        <JsonEditor v-else :model-value="modelValue" :readonly="readonly" @update:model-value="emit('update:modelValue', $event)" />
    </div>
</template>

<script setup>
import DictEditor from "./DictEditor.vue";
import JsonEditor from "./JsonEditor.vue";
import ListEditor from "./ListEditor.vue";
import OrderedChoice from "./OrderedChoice.vue";

defineProps({
    id: String,
    option: { type: Object, required: true },
    modelValue: { default: null },
    readonly: Boolean,
    metadataFields: { type: Array, default: () => [] },
});
const emit = defineEmits(["update:modelValue"]);
</script>
