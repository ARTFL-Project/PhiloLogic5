<template>
    <div>
        <textarea class="form-control mono" :rows="rows" :value="text" :readonly="readonly" @input="update($event.target.value)"></textarea>
        <div class="form-text">{{ $t("editors.onePerLine") }}</div>
    </div>
</template>

<script setup>
import { computed } from "vue";

// A list of strings, one per line
const props = defineProps({ modelValue: { type: Array, default: () => [] }, readonly: Boolean });
const emit = defineEmits(["update:modelValue"]);
const text = computed(() => (props.modelValue || []).join("\n"));
const rows = computed(() => Math.min(12, Math.max(3, (props.modelValue || []).length + 1)));

function update(value) {
    emit(
        "update:modelValue",
        value
            .split("\n")
            .map((line) => line.trim())
            .filter((line) => line !== ""),
    );
}
</script>
