<template>
    <div>
        <textarea class="form-control mono" :class="{ 'is-invalid': error }" :rows="rows" v-model="text" :readonly="readonly" @input="update"></textarea>
        <div v-if="error" class="invalid-feedback">{{ error }}</div>
        <div v-else class="form-text">{{ $t("editors.json") }}</div>
    </div>
</template>

<script setup>
import { computed, ref, watch } from "vue";

// Any structure, edited as JSON: only valid JSON is passed on
const props = defineProps({ modelValue: { default: null }, readonly: Boolean });
const emit = defineEmits(["update:modelValue"]);
const text = ref(JSON.stringify(props.modelValue, null, 2));
const error = ref(null);
let emitted = JSON.stringify(props.modelValue);
const rows = computed(() => Math.min(20, Math.max(3, text.value.split("\n").length)));

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
</script>
