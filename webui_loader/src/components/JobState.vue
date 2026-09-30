<template>
    <span class="badge" :class="classes[state] || 'text-bg-secondary'">
        <span v-if="state === 'running'" class="spinner-grow spinner-grow-sm me-1"></span>{{ $t(`jobs.states.${state}`) }}
    </span>
</template>

<script setup>
defineProps({ state: { type: String, required: true } });
const classes = {
    running: "text-bg-secondary",
    succeeded: "text-bg-success",
    failed: "text-bg-danger",
    cancelled: "text-bg-warning",
    interrupted: "text-bg-danger",
};
</script>
