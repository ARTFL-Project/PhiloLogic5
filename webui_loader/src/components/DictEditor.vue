<template>
    <div>
        <table v-if="rows.length" class="table table-sm align-middle mb-1">
            <thead>
                <tr>
                    <th>{{ keyLabel || $t("editors.key") }}</th>
                    <th>{{ valueLabel || $t("editors.value") }}</th>
                    <th v-if="!readonly"></th>
                </tr>
            </thead>
            <tbody>
                <tr v-for="(row, index) in rows" :key="index">
                    <td>
                        <input class="form-control form-control-sm mono" :value="row[0]" :readonly="readonly" :list="keyListId" @change="setKey(index, $event.target.value)" />
                    </td>
                    <td>
                        <select v-if="valueChoices.length" class="form-select form-select-sm" :value="row[1]" :disabled="readonly" @change="setValue(index, $event.target.value)">
                            <option v-for="choice in valueChoices" :key="choice" :value="choice">{{ choice }}</option>
                        </select>
                        <input v-else class="form-control form-control-sm mono" :value="row[1]" :readonly="readonly" @change="setValue(index, $event.target.value)" />
                    </td>
                    <td v-if="!readonly" class="text-end">
                        <button class="btn btn-sm btn-outline-danger" type="button" @click="remove(index)" :aria-label="$t('editors.remove')"><i class="bi bi-trash"></i></button>
                    </td>
                </tr>
            </tbody>
        </table>
        <datalist v-if="keySuggestions.length" :id="keyListId">
            <option v-for="suggestion in keySuggestions" :key="suggestion" :value="suggestion"></option>
        </datalist>
        <button v-if="!readonly" class="btn btn-sm btn-outline-secondary" type="button" @click="add"><i class="bi bi-plus"></i> {{ $t("editors.add") }}</button>
    </div>
</template>

<script setup>
import { computed } from "vue";

// A mapping of strings to strings (or to one of valueChoices), edited as rows
const props = defineProps({
    modelValue: { type: Object, default: () => ({}) },
    valueChoices: { type: Array, default: () => [] },
    keySuggestions: { type: Array, default: () => [] },
    keyLabel: String,
    valueLabel: String,
    readonly: Boolean,
});
const emit = defineEmits(["update:modelValue"]);
const keyListId = `keys-${Math.random().toString(36).slice(2)}`;
const rows = computed(() => Object.entries(props.modelValue || {}));

function emitRows(newRows) {
    const result = {};
    for (const [key, value] of newRows) {
        if (key !== "") {
            result[key] = value;
        }
    }
    emit("update:modelValue", result);
}

function setKey(index, key) {
    const newRows = rows.value.map((row) => [...row]);
    newRows[index][0] = key.trim();
    emitRows(newRows);
}

function setValue(index, value) {
    const newRows = rows.value.map((row) => [...row]);
    newRows[index][1] = value;
    emitRows(newRows);
}

function remove(index) {
    emitRows(rows.value.filter((row, rowIndex) => rowIndex !== index));
}

function add() {
    let key = "new";
    let number = 1;
    while (key in (props.modelValue || {})) {
        number += 1;
        key = `new${number}`;
    }
    emitRows([...rows.value, [key, props.valueChoices[0] || ""]]);
}
</script>
