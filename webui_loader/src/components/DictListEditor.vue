<template>
    <div>
        <div v-for="(row, index) in rows" :key="index" class="border rounded p-2 mb-2">
            <div class="d-flex gap-2 mb-1">
                <input class="form-control form-control-sm mono w-auto" :value="row[0]" :readonly="readonly" @change="setKey(index, $event.target.value)" :aria-label="$t('editors.key')" />
                <button v-if="!readonly" class="btn btn-sm btn-outline-danger ms-auto" type="button" @click="remove(index)" :aria-label="$t('editors.remove')"><i class="bi bi-trash"></i></button>
            </div>
            <ListEditor :model-value="row[1]" :readonly="readonly" @update:model-value="setValues(index, $event)" />
        </div>
        <button v-if="!readonly" class="btn btn-sm btn-outline-secondary" type="button" @click="add"><i class="bi bi-plus"></i> {{ $t("editors.add") }}</button>
    </div>
</template>

<script setup>
import { computed } from "vue";
import ListEditor from "./ListEditor.vue";

// A mapping of strings to lists of strings (such as doc_xpaths)
const props = defineProps({ modelValue: { type: Object, default: () => ({}) }, readonly: Boolean });
const emit = defineEmits(["update:modelValue"]);
const rows = computed(() => Object.entries(props.modelValue || {}));

function emitRows(newRows) {
    const result = {};
    for (const [key, values] of newRows) {
        if (key !== "") {
            result[key] = values;
        }
    }
    emit("update:modelValue", result);
}

function setKey(index, key) {
    const newRows = rows.value.map(([k, v]) => [k, v]);
    newRows[index][0] = key.trim();
    emitRows(newRows);
}

function setValues(index, values) {
    const newRows = rows.value.map(([k, v]) => [k, v]);
    newRows[index][1] = values;
    emitRows(newRows);
}

function remove(index) {
    emitRows(rows.value.filter((row, rowIndex) => rowIndex !== index));
}

function add() {
    let key = "new_field";
    let number = 1;
    while (key in (props.modelValue || {})) {
        number += 1;
        key = `new_field${number}`;
    }
    emitRows([...rows.value, [key, []]]);
}
</script>
