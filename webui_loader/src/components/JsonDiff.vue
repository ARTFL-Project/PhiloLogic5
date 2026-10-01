<template>
    <div class="json-diff mono small rounded border">
        <div v-for="(row, index) in rows" :key="index" class="line" :class="row.type">
            <span class="sign">{{ SIGNS[row.type] }}</span><span v-if="row.type === 'gap'" class="text-body-secondary">{{ row.text }}</span><span v-else-if="highlighted" v-html="highlighted[row.type === 'added' ? 'after' : 'before'][row.line]"></span><span v-else>{{ row.text }}</span>
        </div>
    </div>
</template>

<script setup>
import { computed, shallowRef } from "vue";
import { lineDiff } from "../utils";

// What changed between two values, as the lines of their JSON which differ (with some context), highlighted (each
// text as a whole, so that keys are told from other strings)
const props = defineProps({ before: { default: null }, after: { default: null } });
const SIGNS = { same: " ", removed: "-", added: "+", gap: " " };
const highlight = shallowRef(null);
import("../jsonHighlight.js").then((module) => (highlight.value = module.highlightJsonLines));
const texts = computed(() => ({ before: JSON.stringify(props.before, null, 2) ?? "", after: JSON.stringify(props.after, null, 2) ?? "" }));
const rows = computed(() => lineDiff(texts.value.before, texts.value.after, 3));
const highlighted = computed(() => (highlight.value ? { before: highlight.value(texts.value.before), after: highlight.value(texts.value.after) } : null));
</script>

<style scoped>
.json-diff {
    overflow-x: auto;
    padding: 0.25rem 0;
}
.line {
    white-space: pre;
    padding: 0 0.5rem;
}
.sign {
    display: inline-block;
    width: 1.25rem;
    user-select: none;
}
.removed {
    background: var(--bs-danger-bg-subtle);
}
.added {
    background: var(--bs-success-bg-subtle);
}
</style>
