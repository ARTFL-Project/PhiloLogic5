<template>
    <div class="small">
        <div class="text-success" v-if="optionCount">
            <i class="bi bi-check2 me-1"></i>{{ $t("base.options", { count: optionCount }) }}
        </div>
        <div v-if="info.custom_code.length" class="alert alert-info py-2 mt-2 mb-0">
            <i class="bi bi-code-slash me-1"></i>{{ $t("base.customCode") }}
            <pre class="mono mb-0 mt-1">{{ info.custom_code.join("\n") }}</pre>
        </div>
        <div v-if="Object.keys(info.code).length" class="mt-1">
            <i class="bi bi-lock me-1"></i>{{ $t("base.code", { names: Object.keys(info.code).join(", ") }) }}
        </div>
        <div v-if="Object.keys(info.run_options).length" class="text-body-secondary mt-1">
            <i class="bi bi-info-circle me-1"></i>{{ $t("base.runOptions", { names: Object.keys(info.run_options).join(", ") }) }}
        </div>
        <div v-if="info.unused.length" class="text-body-secondary mt-1">
            <i class="bi bi-info-circle me-1"></i>{{ $t("base.unused", { names: info.unused.join(", ") }) }}
        </div>
        <div v-if="Object.keys(info.unknown).length" class="text-warning-emphasis mt-1">
            <i class="bi bi-question-circle me-1"></i>{{ $t("base.unknown", { names: Object.keys(info.unknown).join(", ") }) }}
        </div>
        <div v-for="(error, key) in info.errors" :key="key" class="text-danger mt-1">
            <i class="bi bi-x-octagon me-1"></i><span class="mono">{{ key }}</span>: {{ error }}
        </div>
    </div>
</template>

<script setup>
import { computed } from "vue";

// What a load config which a load starts from sets, and what of it is left aside
const props = defineProps({ info: { type: Object, required: true } });
const optionCount = computed(() => Object.keys(props.info.options).length);
</script>
