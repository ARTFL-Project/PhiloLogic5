<template>
    <div class="mb-3" :class="{ changed }">
        <div class="d-flex align-items-baseline gap-2">
            <label class="form-label option-label mb-1 mono" :for="id">{{ option.key }}</label>
            <span v-if="option.cli_flag" class="badge text-bg-light">{{ option.cli_flag }}</span>
            <span v-if="readonlySource" class="badge text-bg-secondary">{{ $t("options.setByCode") }}</span>
            <button v-if="changed && !readonlySource" type="button" class="btn btn-link btn-sm p-0 ms-auto" @click="reset">{{ $t("options.reset") }}</button>
        </div>
        <div class="form-text mt-0 mb-1">{{ option.help }}</div>
        <pre v-if="readonlySource" class="bg-light p-2 rounded mono mb-0">{{ readonlySource }}</pre>
        <template v-else-if="option.kind === 'bool'">
            <div class="form-check form-switch">
                <input :id="id" class="form-check-input" type="checkbox" :checked="modelValue" @change="emit('update:modelValue', $event.target.checked)" />
            </div>
        </template>
        <input v-else-if="option.kind === 'int'" :id="id" type="number" class="form-control w-auto" :class="{ 'is-invalid': error }" :min="option.minimum" :value="modelValue" @change="emit('update:modelValue', parseInt($event.target.value, 10))" />
        <select v-else-if="option.kind === 'choice'" :id="id" class="form-select w-auto" :value="modelValue" @change="emit('update:modelValue', $event.target.value)">
            <option v-for="choice in option.choices" :key="choice" :value="choice">{{ choice }}</option>
        </select>
        <div v-else-if="option.kind === 'multichoice'">
            <div v-for="choice in option.choices" :key="choice" class="form-check form-check-inline">
                <input :id="`${id}-${choice}`" class="form-check-input" type="checkbox" :checked="(modelValue || []).includes(choice)" @change="toggle(choice, $event.target.checked)" />
                <label class="form-check-label mono" :for="`${id}-${choice}`">{{ choice }}</label>
            </div>
        </div>
        <template v-else-if="option.key === 'spacy_model'">
            <input :id="id" class="form-control mono" :list="`${id}-models`" :value="modelValue || ''" :placeholder="$t('options.none')" @change="emit('update:modelValue', $event.target.value.trim() || null)" />
            <datalist :id="`${id}-models`">
                <option v-for="model in spacyModels" :key="model" :value="model"></option>
            </datalist>
        </template>
        <input v-else-if="['string', 'regex', 'path'].includes(option.kind)" :id="id" class="form-control" :class="{ mono: option.kind !== 'string', 'is-invalid': error }" :value="modelValue || ''" :placeholder="option.kind === 'path' ? $t('options.pathPlaceholder') : ''" @change="setText($event.target.value)" />
        <ListEditor v-else-if="option.kind === 'list'" :model-value="modelValue || []" @update:model-value="emit('update:modelValue', $event)" />
        <DictEditor v-else-if="option.kind === 'dict'" :model-value="modelValue || {}" :value-choices="option.value_choices" @update:model-value="emit('update:modelValue', $event)" />
        <DictListEditor v-else-if="option.kind === 'dict_list'" :model-value="modelValue || {}" @update:model-value="emit('update:modelValue', $event)" />
        <div v-if="error" class="text-danger small mt-1"><i class="bi bi-x-octagon me-1"></i>{{ error }}</div>
    </div>
</template>

<script setup>
import { computed } from "vue";
import { clone, deepEqual } from "../utils";
import DictEditor from "./DictEditor.vue";
import DictListEditor from "./DictListEditor.vue";
import ListEditor from "./ListEditor.vue";

const props = defineProps({
    option: { type: Object, required: true },
    modelValue: { default: null },
    error: { type: String, default: null },
    readonlySource: { type: String, default: null },
    spacyModels: { type: Array, default: () => [] },
});
const emit = defineEmits(["update:modelValue"]);
const id = `option-${props.option.key}`;
const empty = (value) => value === null || value === undefined || value === "";
const changed = computed(() => {
    if (["path", "string"].includes(props.option.kind) && empty(props.modelValue) && empty(props.option.default)) {
        return false;
    }
    return !deepEqual(props.modelValue, props.option.default);
});

function reset() {
    emit("update:modelValue", clone(props.option.default));
}

function toggle(choice, checked) {
    const selected = new Set(props.modelValue || []);
    if (checked) {
        selected.add(choice);
    } else {
        selected.delete(choice);
    }
    emit("update:modelValue", props.option.choices.filter((item) => selected.has(item)));
}

function setText(value) {
    emit("update:modelValue", props.option.kind === "path" && value.trim() === "" ? props.option.default : value);
}
</script>
