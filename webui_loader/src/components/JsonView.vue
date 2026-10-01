<template>
    <pre v-if="html !== null" class="mono" v-html="html"></pre>
    <pre v-else class="mono">{{ text }}</pre>
</template>

<script setup>
import { computed, shallowRef } from "vue";

// A value shown as highlighted JSON (once the highlighter is loaded; as plain text until then)
const props = defineProps({ value: { default: null }, indent: { type: Number, default: 2 } });
const highlight = shallowRef(null);
import("../jsonHighlight.js").then((module) => (highlight.value = module.highlightJsonLines));
const text = computed(() => JSON.stringify(props.value, null, props.indent) ?? "");
const html = computed(() => (highlight.value ? highlight.value(text.value).join("\n") : null));
</script>
