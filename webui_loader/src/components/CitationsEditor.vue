<template>
    <div>
        <details v-for="([name, citation], index) in entries" :key="index" class="border rounded mb-2" :open="opened === index" @toggle="toggled(index, $event.target.open)">
            <summary class="px-2 py-1">
                <span class="mono fw-semibold">{{ name }}</span>
                <span class="small text-body-secondary mono ms-2">{{ citation.field }} &middot; {{ citation.object_level }}</span>
            </summary>
            <div class="px-2 pb-2">
                <div class="d-flex flex-wrap gap-2 align-items-end mb-2">
                    <div>
                        <label class="form-label small mb-0" :for="`${id}-${index}`">{{ $t("citations.name") }}</label>
                        <input :id="`${id}-${index}`" class="form-control form-control-sm mono" :value="name" :readonly="readonly" @change="rename(index, $event.target)" />
                    </div>
                    <button v-if="!readonly" class="btn btn-sm btn-outline-danger ms-auto" type="button" @click="remove(index)"><i class="bi bi-trash me-1"></i>{{ $t("editors.remove") }}</button>
                </div>
                <CitationForm :model-value="citation" :metadata-fields="metadataFields" :readonly="readonly" @update:model-value="replace(index, $event)" />
            </div>
        </details>
        <button v-if="!readonly" class="btn btn-sm btn-outline-secondary" type="button" @click="add"><i class="bi bi-plus"></i> {{ $t("citations.new") }}</button>
        <div class="form-text">{{ $t("citations.namedHelp") }}</div>
    </div>
</template>

<script setup>
import { computed, ref } from "vue";
import { clone } from "../utils";
import { BLANK_CITATION } from "../webConfigForms";
import CitationForm from "./CitationForm.vue";

// The named citations, which the lists of citations use (citations["name"] in the web config)
const props = defineProps({
    modelValue: { type: Object, default: () => ({}) },
    metadataFields: { type: Array, default: () => [] },
    readonly: Boolean,
});
const emit = defineEmits(["update:modelValue"]);
const id = `citations-${Math.random().toString(36).slice(2)}`;
const opened = ref(null);
const entries = computed(() => Object.entries(props.modelValue || {}));

function toggled(index, open) {
    if (open) {
        opened.value = index;
    } else if (opened.value === index) {
        opened.value = null;
    }
}

function emitEntries(newEntries) {
    emit("update:modelValue", Object.fromEntries(newEntries));
}

function rename(index, input) {
    const name = input.value.trim();
    if (!name || entries.value.some(([other], otherIndex) => other === name && otherIndex !== index)) {
        input.value = entries.value[index][0]; // empty, or the name of another one
        return;
    }
    emitEntries(entries.value.map(([other, citation], otherIndex) => (otherIndex === index ? [name, citation] : [other, citation])));
}

function replace(index, citation) {
    emitEntries(entries.value.map(([name, other], otherIndex) => (otherIndex === index ? [name, citation] : [name, other])));
}

function remove(index) {
    opened.value = null;
    emitEntries(entries.value.filter((entry, otherIndex) => otherIndex !== index));
}

function add() {
    let name = "new_citation";
    for (let number = 2; name in (props.modelValue || {}); number += 1) {
        name = `new_citation${number}`;
    }
    emitEntries([...entries.value, [name, clone(BLANK_CITATION)]]);
    opened.value = entries.value.length;
}
</script>
