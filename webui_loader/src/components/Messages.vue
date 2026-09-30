<template>
    <div>
        <div v-for="(message, index) in errors" :key="`error-${index}`" class="alert alert-danger py-2 mb-2">
            <i class="bi bi-x-octagon me-2"></i><strong v-if="message.field">{{ label(message.field) }}: </strong>{{ message.message }}
        </div>
        <div v-for="(message, index) in warnings" :key="`warning-${index}`" class="alert alert-warning py-2 mb-2">
            <i class="bi bi-exclamation-triangle me-2"></i><strong v-if="message.field">{{ label(message.field) }}: </strong>{{ message.message }}
        </div>
    </div>
</template>

<script setup>
import { useI18n } from "vue-i18n";

defineProps({ errors: { type: Array, default: () => [] }, warnings: { type: Array, default: () => [] } });
const { t, te } = useI18n();
const label = (field) => (te(`fields.${field}`) ? t(`fields.${field}`) : field);
</script>
