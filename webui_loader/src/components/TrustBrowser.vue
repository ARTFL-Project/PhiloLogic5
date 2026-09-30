<template>
    <div v-if="days > 0" class="form-check mb-3">
        <input id="trust-browser" class="form-check-input" type="checkbox" :checked="modelValue" @change="emit('update:modelValue', $event.target.checked)" />
        <label class="form-check-label" for="trust-browser">{{ $t("login.trust", { days }) }}</label>
        <div class="form-text">{{ $t("login.trustHelp") }}</div>
    </div>
</template>

<script setup>
import { computed } from "vue";
import { session } from "../session";

// Trust this browser for the second factor: the password is then enough in it, for some days
defineProps({ modelValue: Boolean });
const emit = defineEmits(["update:modelValue"]);
const days = computed(() => session.server.trusted_browser_days || 0);
</script>
