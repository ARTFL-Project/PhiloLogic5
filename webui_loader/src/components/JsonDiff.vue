<template>
    <div class="json-diff mono small rounded border">
        <div v-for="(row, index) in rows" :key="index" class="line" :class="row.type">
            <span class="sign">{{ SIGNS[row.type] }}</span><span v-if="row.type === 'gap'" class="text-body-secondary">{{ row.text }}</span><span v-else v-html="highlightJson(row.text)"></span>
        </div>
    </div>
</template>

<script setup>
import { computed } from "vue";
import { highlightJson, lineDiff } from "../utils";

// What changed between two values, as the lines of their JSON which differ (with some context), highlighted
const props = defineProps({ before: { default: null }, after: { default: null } });
const SIGNS = { same: " ", removed: "-", added: "+", gap: " " };
const rows = computed(() => lineDiff(JSON.stringify(props.before, null, 2) ?? "", JSON.stringify(props.after, null, 2) ?? "", 3));
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
