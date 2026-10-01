<template>
    <div>
        <div v-for="([field, choices], index) in entries" :key="index" class="border rounded p-2 mb-2">
            <div class="d-flex align-items-end gap-2 mb-2">
                <div>
                    <label class="form-label small mb-0" :for="`${id}-${index}`">{{ $t("form.field") }}</label>
                    <input :id="`${id}-${index}`" class="form-control form-control-sm mono" :list="`${id}-fields`" :value="field" :readonly="readonly" @change="rename(index, $event.target)" />
                </div>
                <button v-if="!readonly" class="btn btn-sm btn-outline-danger ms-auto" type="button" @click="remove(index)"><i class="bi bi-trash me-1"></i>{{ $t("editors.remove") }}</button>
            </div>
            <table class="table table-sm align-middle mb-1" v-if="choices.length">
                <thead>
                    <tr>
                        <th>{{ $t("choiceValues.label") }}</th>
                        <th>{{ $t("choiceValues.value") }}</th>
                        <th v-if="!readonly"></th>
                    </tr>
                </thead>
                <tbody>
                    <tr v-for="(choice, position) in choices" :key="position">
                        <td><input class="form-control form-control-sm" :value="choice.label" :readonly="readonly" :aria-label="$t('choiceValues.label')" @change="setChoice(index, position, 'label', $event.target.value)" /></td>
                        <td><input class="form-control form-control-sm mono" :value="choice.value" :readonly="readonly" :aria-label="$t('choiceValues.value')" @change="setChoice(index, position, 'value', $event.target.value)" /></td>
                        <td v-if="!readonly" class="text-end text-nowrap">
                            <button class="btn btn-sm btn-link p-0 me-2" type="button" :disabled="position === 0" @click="moveChoice(index, position, -1)" :aria-label="$t('editors.up')"><i class="bi bi-arrow-up"></i></button>
                            <button class="btn btn-sm btn-link p-0 me-2" type="button" :disabled="position === choices.length - 1" @click="moveChoice(index, position, 1)" :aria-label="$t('editors.down')"><i class="bi bi-arrow-down"></i></button>
                            <button class="btn btn-sm btn-link p-0 text-danger" type="button" @click="removeChoice(index, position)" :aria-label="$t('editors.remove')"><i class="bi bi-x-lg"></i></button>
                        </td>
                    </tr>
                </tbody>
            </table>
            <button v-if="!readonly" class="btn btn-sm btn-link p-0" type="button" @click="addChoice(index)"><i class="bi bi-plus"></i> {{ $t("choiceValues.addChoice") }}</button>
        </div>
        <datalist :id="`${id}-fields`">
            <option v-for="field in metadataFields" :key="field" :value="field"></option>
        </datalist>
        <button v-if="!readonly" class="btn btn-sm btn-outline-secondary" type="button" @click="add"><i class="bi bi-plus"></i> {{ $t("choiceValues.addField") }}</button>
    </div>
</template>

<script setup>
import { computed } from "vue";

// The choices of the dropdowns of metadata fields in the search form: for each field, a label and the value searched
const props = defineProps({
    modelValue: { type: Object, default: () => ({}) },
    metadataFields: { type: Array, default: () => [] },
    readonly: Boolean,
});
const emit = defineEmits(["update:modelValue"]);
const id = `choices-${Math.random().toString(36).slice(2)}`;
const entries = computed(() => Object.entries(props.modelValue || {}).map(([field, choices]) => [field, choices || []]));

function emitEntries(newEntries) {
    emit("update:modelValue", Object.fromEntries(newEntries));
}

function setChoices(index, change) {
    emitEntries(entries.value.map(([field, choices], otherIndex) => (otherIndex === index ? [field, change(choices)] : [field, choices])));
}

function rename(index, input) {
    const field = input.value.trim();
    if (!field || entries.value.some(([other], otherIndex) => other === field && otherIndex !== index)) {
        input.value = entries.value[index][0]; // empty, or a field which has its choices already
        return;
    }
    emitEntries(entries.value.map(([other, choices], otherIndex) => (otherIndex === index ? [field, choices] : [other, choices])));
}

function remove(index) {
    emitEntries(entries.value.filter((entry, otherIndex) => otherIndex !== index));
}

function add() {
    const field = props.metadataFields.find((name) => !(name in (props.modelValue || {}))) || "field";
    emitEntries([...entries.value, [field, [{ label: "", value: "" }]]]);
}

function setChoice(index, position, key, value) {
    setChoices(index, (choices) => choices.map((choice, other) => (other === position ? { ...choice, [key]: value } : choice)));
}

function addChoice(index) {
    setChoices(index, (choices) => [...choices, { label: "", value: "" }]);
}

function removeChoice(index, position) {
    setChoices(index, (choices) => choices.filter((choice, other) => other !== position));
}

function moveChoice(index, position, step) {
    setChoices(index, (choices) => {
        const newChoices = [...choices];
        [newChoices[position], newChoices[position + step]] = [newChoices[position + step], newChoices[position]];
        return newChoices;
    });
}
</script>
