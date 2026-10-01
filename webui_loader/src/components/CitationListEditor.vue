<template>
    <div>
        <ul class="list-group mb-2" v-if="items.length">
            <li v-for="(item, index) in items" :key="index" class="list-group-item py-1">
                <div class="d-flex align-items-center gap-2">
                    <span v-if="names[index]" class="mono fw-semibold">{{ names[index] }}</span>
                    <span v-else class="badge text-bg-light">{{ $t("citations.custom") }}</span>
                    <span class="small text-body-secondary mono">{{ summary(item) }}<i v-if="item.link" class="bi bi-link-45deg ms-1" :title="$t('citations.link')"></i></span>
                    <span class="ms-auto text-nowrap" v-if="!readonly">
                        <button v-if="!names[index]" class="btn btn-sm btn-link p-0 me-2" type="button" @click="editing = editing === index ? null : index">{{ editing === index ? $t("citations.done") : $t("citations.edit") }}</button>
                        <button class="btn btn-sm btn-link p-0 me-2" type="button" :disabled="index === 0" @click="move(index, -1)" :aria-label="$t('editors.up')"><i class="bi bi-arrow-up"></i></button>
                        <button class="btn btn-sm btn-link p-0 me-2" type="button" :disabled="index === items.length - 1" @click="move(index, 1)" :aria-label="$t('editors.down')"><i class="bi bi-arrow-down"></i></button>
                        <button class="btn btn-sm btn-link p-0 text-danger" type="button" @click="remove(index)" :aria-label="$t('editors.remove')"><i class="bi bi-x-lg"></i></button>
                    </span>
                </div>
                <CitationForm v-if="editing === index && !names[index]" class="mt-2 mb-1" :model-value="item" :metadata-fields="metadataFields" :readonly="readonly" @update:model-value="replace(index, $event)" />
            </li>
        </ul>
        <div v-if="!readonly" class="d-flex flex-wrap gap-2 align-items-center">
            <div class="input-group input-group-sm w-auto">
                <select class="form-select" v-model="picked" :aria-label="$t('citations.add')">
                    <option value="" disabled>{{ $t("citations.add") }}</option>
                    <option v-for="name in Object.keys(citations || {})" :key="name" :value="name">{{ name }}</option>
                </select>
                <button class="btn btn-outline-secondary" type="button" :disabled="!picked" @click="addNamed"><i class="bi bi-plus"></i> {{ $t("editors.add") }}</button>
            </div>
            <button class="btn btn-sm btn-outline-secondary" type="button" @click="addCustom"><i class="bi bi-plus"></i> {{ $t("citations.addCustom") }}</button>
        </div>
        <div v-if="help" class="form-text">{{ $t("citations.listHelp") }}</div>
    </div>
</template>

<script setup>
import { computed, ref } from "vue";
import { clone } from "../utils";
import { BLANK_CITATION, citationName } from "../webConfigForms";
import CitationForm from "./CitationForm.vue";

// An ordered list of citations: those of citations (by name, edited there), or citations of its own
const props = defineProps({
    modelValue: { type: Array, default: () => [] },
    citations: { type: Object, default: () => ({}) },
    metadataFields: { type: Array, default: () => [] },
    readonly: Boolean,
    help: { type: Boolean, default: true },
});
const emit = defineEmits(["update:modelValue"]);
const picked = ref("");
const editing = ref(null);
const items = computed(() => props.modelValue || []);
const names = computed(() => items.value.map((item) => citationName(item, props.citations)));

function summary(citation) {
    const parts = [`${citation.field || "?"} \u00b7 ${citation.object_level || "?"}`]; // middle dot
    if (citation.prefix || citation.suffix) {
        parts.push(`"${citation.prefix || ""}\u2026${citation.suffix || ""}"`); // the value, between what surrounds it
    }
    return parts.join("  ");
}

function emitItems(newItems) {
    emit("update:modelValue", newItems);
}

function addNamed() {
    emitItems([...items.value, clone(props.citations[picked.value])]);
    picked.value = "";
}

function addCustom() {
    emitItems([...items.value, clone(BLANK_CITATION)]);
    editing.value = items.value.length;
}

function replace(index, citation) {
    emitItems(items.value.map((item, itemIndex) => (itemIndex === index ? citation : item)));
}

function remove(index) {
    editing.value = null;
    emitItems(items.value.filter((item, itemIndex) => itemIndex !== index));
}

function move(index, step) {
    const newItems = [...items.value];
    [newItems[index], newItems[index + step]] = [newItems[index + step], newItems[index]];
    editing.value = editing.value === index ? index + step : editing.value === index + step ? index : editing.value;
    emitItems(newItems);
}
</script>
