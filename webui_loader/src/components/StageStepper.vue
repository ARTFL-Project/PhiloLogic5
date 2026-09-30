<template>
    <ol class="list-unstyled mb-0">
        <li v-for="stage in stages" :key="stage.name" class="mb-1" :class="`stage-${stage.state}`">
            <i class="bi me-2" :class="icons[stage.state]"></i>
            <span>{{ $t(`stages.${stage.name}`) }}</span>
            <span v-if="stage.start" class="text-body-secondary small ms-2">{{ duration(stage) }}</span>
        </li>
    </ol>
</template>

<script setup>
import { formatDuration } from "../utils";

const props = defineProps({ stages: { type: Array, default: () => [] }, now: { type: Number, default: 0 } });
const icons = {
    pending: "bi-circle",
    running: "bi-arrow-repeat",
    done: "bi-check-circle-fill",
    failed: "bi-x-circle-fill",
    cancelled: "bi-slash-circle",
};

function duration(stage) {
    const end = stage.end || (stage.state === "running" ? props.now : null);
    return end ? formatDuration(end - stage.start) : "";
}
</script>
