<template>
    <div>
        <div ref="host" class="json-editor" :class="{ invalid: error }"></div>
        <pre v-if="!view" class="loading mono">{{ initialText }}</pre>
        <div v-if="error" class="invalid-feedback d-block">{{ error }}</div>
        <div v-else class="form-text">{{ $t("editors.json") }}</div>
    </div>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref, shallowRef, watch } from "vue";

// Any structure, edited as JSON, in a code editor (CodeMirror, loaded with the first one shown): only valid JSON is
// passed on
const props = defineProps({ modelValue: { default: null }, readonly: Boolean });
const emit = defineEmits(["update:modelValue"]);
const host = ref(null);
const view = shallowRef(null);
const error = ref(null);
const initialText = JSON.stringify(props.modelValue, null, 2);
let editor = null;
let emitted = JSON.stringify(props.modelValue);
let replacing = false;

onMounted(async () => {
    editor = await import("../codeEditor.js");
    if (host.value) {
        const doc = JSON.stringify(props.modelValue, null, 2); // which may have changed meanwhile
        view.value = editor.createJsonEditor(host.value, { doc, readonly: props.readonly, onChange });
    }
});
onBeforeUnmount(() => view.value && view.value.destroy());

function onChange(text) {
    if (replacing) {
        return;
    }
    try {
        const value = JSON.parse(text);
        error.value = null;
        emitted = JSON.stringify(value);
        emit("update:modelValue", value);
    } catch (parseError) {
        error.value = parseError.message;
    }
}

// A value set from outside (a reset) replaces the text
watch(
    () => JSON.stringify(props.modelValue),
    (value) => {
        if (value !== emitted) {
            emitted = value;
            error.value = null;
            if (view.value) {
                replacing = true;
                editor.replaceText(view.value, JSON.stringify(props.modelValue, null, 2));
                replacing = false;
            }
        }
    },
);
watch(
    () => props.readonly,
    (readonly) => view.value && editor.setReadonly(view.value, readonly),
);
</script>

<style scoped>
.loading {
    max-height: 24rem;
    overflow: auto;
    padding: 0.25rem 0.5rem;
    border: var(--bs-border-width) solid var(--bs-border-color);
    border-radius: var(--bs-border-radius);
}
.invalid :deep(.cm-editor) {
    border-color: var(--bs-form-invalid-border-color);
}
</style>
