<template>
    <div>
        <div class="json-editor" :class="{ 'is-invalid': error }">
            <pre ref="highlight" class="json-layer json-highlight mono" aria-hidden="true" v-html="highlighted"></pre>
            <textarea class="json-layer json-input mono" :class="{ 'is-invalid': error }" :rows="rows" v-model="text" :readonly="readonly" spellcheck="false" @input="update" @scroll="syncScroll"></textarea>
        </div>
        <div v-if="error" class="invalid-feedback d-block">{{ error }}</div>
        <div v-else class="form-text">{{ $t("editors.json") }}</div>
    </div>
</template>

<script setup>
import { computed, ref, watch } from "vue";
import { highlightJson } from "../utils";

// Any structure, edited as JSON, with its syntax highlighted (a copy of the text under the transparent text of the
// textarea): only valid JSON is passed on
const props = defineProps({ modelValue: { default: null }, readonly: Boolean });
const emit = defineEmits(["update:modelValue"]);
const text = ref(JSON.stringify(props.modelValue, null, 2));
const error = ref(null);
const highlight = ref(null);
let emitted = JSON.stringify(props.modelValue);
const rows = computed(() => Math.min(20, Math.max(3, text.value.split("\n").length)));
// A last line break shows in a textarea, not in a pre
const highlighted = computed(() => highlightJson(text.value) + "\n");

// A value set from outside (a reset) replaces the text
watch(
    () => JSON.stringify(props.modelValue),
    (value) => {
        if (value !== emitted) {
            emitted = value;
            text.value = JSON.stringify(props.modelValue, null, 2);
            error.value = null;
        }
    },
);

function update() {
    try {
        const value = JSON.parse(text.value);
        error.value = null;
        emitted = JSON.stringify(value);
        emit("update:modelValue", value);
    } catch (parseError) {
        error.value = parseError.message;
    }
}

function syncScroll(event) {
    highlight.value.scrollTop = event.target.scrollTop;
    highlight.value.scrollLeft = event.target.scrollLeft;
}
</script>

<style scoped>
.json-editor {
    position: relative;
}
.json-layer {
    display: block;
    width: 100%;
    margin: 0;
    padding: 0.375rem 0.75rem;
    border: var(--bs-border-width) solid var(--bs-border-color);
    border-radius: var(--bs-border-radius);
    font-size: 0.875rem;
    line-height: 1.5;
    white-space: pre;
    overflow: auto;
    tab-size: 2;
}
.json-highlight {
    position: absolute;
    inset: 0;
    overflow: hidden;
    pointer-events: none;
    background: var(--bs-body-bg);
    color: var(--bs-body-color);
}
.json-input {
    position: relative;
    background: transparent;
    color: transparent;
    caret-color: var(--bs-body-color);
    resize: vertical;
}
.json-input::selection {
    background: rgba(13, 110, 253, 0.25);
}
.json-input:focus {
    outline: 0;
    border-color: var(--bs-border-color);
    box-shadow: 0 0 0 0.2rem rgba(128, 128, 128, 0.2);
}
.json-input.is-invalid {
    border-color: var(--bs-form-invalid-border-color);
}
.json-input[readonly] {
    caret-color: transparent;
}
</style>
