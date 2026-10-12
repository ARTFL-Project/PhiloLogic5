<template>
    <!-- An explanation kept out of the way: a small button opens it beside the text, Escape or the button closes it -->
    <span ref="root" class="info" @keydown.esc="open = false">
        <button type="button" class="info-button" :aria-expanded="open" :aria-controls="id" :aria-label="label"
            :title="label" @click="open = !open">i</button>
        <span v-if="open" :id="id" class="info-pop shadow" role="note">
            <slot />
            <button type="button" class="btn-close info-close" :aria-label="$t('common.close')" @click="open = false"></button>
        </span>
    </span>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref, useId } from "vue";

defineProps({ label: { type: String, required: true } });

const open = ref(false);
const root = ref(null);
const id = `info-${useId()}`;

// A click anywhere else closes it
function outside(event) {
    if (open.value && root.value && !root.value.contains(event.target)) open.value = false;
}
onMounted(() => document.addEventListener("click", outside));
onBeforeUnmount(() => document.removeEventListener("click", outside));
</script>

<style scoped>
.info {
    position: relative;
    display: inline-block;
}

.info-button {
    width: 1.15rem;
    height: 1.15rem;
    border-radius: 50%;
    border: 1px solid #6c757d;
    background: #fff;
    color: #495057;
    font-size: 0.72rem;
    font-weight: 600;
    font-style: italic;
    font-family: Georgia, serif;
    line-height: 1;
    padding: 0;
    vertical-align: 0.1rem;
}

.info-button:hover,
.info-button:focus-visible,
.info-button[aria-expanded="true"] {
    background: #495057;
    color: #fff;
}

.info-pop {
    position: absolute;
    z-index: 20;
    top: 1.5rem;
    left: 0;
    width: min(32rem, 80vw);
    padding: 0.6rem 2rem 0.6rem 0.75rem;
    background: #fff;
    border: 1px solid #ced4da;
    border-radius: 0.4rem;
    font-size: 0.82rem;
    font-weight: normal;
    color: #343a40;
    line-height: 1.4;
}

.info-close {
    position: absolute;
    top: 0.45rem;
    right: 0.45rem;
    font-size: 0.6rem;
}
</style>
