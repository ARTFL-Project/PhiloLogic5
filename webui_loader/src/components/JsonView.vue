<template>
    <pre class="mono" v-html="highlightJson(JSON.stringify(value, null, indent))"></pre>
</template>

<script setup>
import { highlightJson } from "../utils";

// A value shown as highlighted JSON
defineProps({ value: { default: null }, indent: { type: Number, default: 2 } });
</script>
