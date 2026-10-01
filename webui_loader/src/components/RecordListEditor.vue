<template>
    <div>
        <div v-for="(item, index) in items" :key="index" class="border rounded px-2 py-1 mb-2">
            <div class="d-flex align-items-start gap-2">
                <div class="flex-grow-1">
                    <div v-if="titleKey" class="mono fw-semibold small mb-1">{{ item[titleKey] || "\u2026" }}</div>
                    <RecordEditor :model-value="item" :fields="fields" :citations="citations" :metadata-fields="metadataFields" :readonly="readonly" @update:model-value="replace(index, $event)" />
                </div>
                <span v-if="!readonly" class="text-nowrap mt-1">
                    <button class="btn btn-sm btn-link p-0 me-2" type="button" :disabled="index === 0" @click="move(index, -1)" :aria-label="$t('editors.up')"><i class="bi bi-arrow-up"></i></button>
                    <button class="btn btn-sm btn-link p-0 me-2" type="button" :disabled="index === items.length - 1" @click="move(index, 1)" :aria-label="$t('editors.down')"><i class="bi bi-arrow-down"></i></button>
                    <button class="btn btn-sm btn-link p-0 text-danger" type="button" @click="remove(index)" :aria-label="$t('editors.remove')"><i class="bi bi-x-lg"></i></button>
                </span>
            </div>
        </div>
        <button v-if="!readonly" class="btn btn-sm btn-outline-secondary" type="button" @click="add"><i class="bi bi-plus"></i> {{ $t("editors.add") }}</button>
    </div>
</template>

<script setup>
import { computed } from "vue";
import { clone } from "../utils";
import RecordEditor from "./RecordEditor.vue";

// An ordered list of objects, each edited with the form of its fields
const props = defineProps({
    modelValue: { type: Array, default: () => [] },
    fields: { type: Array, required: true },
    newItem: { type: Object, default: () => ({}) },
    titleKey: { type: String, default: null },
    citations: { type: Object, default: () => ({}) },
    metadataFields: { type: Array, default: () => [] },
    readonly: Boolean,
});
const emit = defineEmits(["update:modelValue"]);
const items = computed(() => props.modelValue || []);

function replace(index, item) {
    emit("update:modelValue", items.value.map((other, otherIndex) => (otherIndex === index ? item : other)));
}

function remove(index) {
    emit("update:modelValue", items.value.filter((other, otherIndex) => otherIndex !== index));
}

function move(index, step) {
    const newItems = [...items.value];
    [newItems[index], newItems[index + step]] = [newItems[index + step], newItems[index]];
    emit("update:modelValue", newItems);
}

function add() {
    emit("update:modelValue", [...items.value, clone(props.newItem)]);
}
</script>
