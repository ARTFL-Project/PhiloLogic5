<template>
    <!-- the field keeps the focus: a click on the list is no blur, which closes it -->
    <div class="autocomplete-popup shadow" @mousedown.prevent>
        <ul :id="id" ref="list" class="autocomplete-results" role="listbox" aria-multiselectable="true"
            :aria-label="$t('searchForm.autocompleteResults')">
            <li v-for="(result, i) in results" :key="i" :id="`${id}-option-${i}`" class="autocomplete-result"
                :class="{ 'is-active': i === activeIndex }" role="option"
                :aria-selected="result.selected ? 'true' : 'false'" @click="emit('toggle', i)">
                <span class="autocomplete-checkbox" :class="{ checked: result.selected }" aria-hidden="true">
                    <i class="bi bi-check-lg" v-if="result.selected"></i>
                </span>
                <span v-html="result.html"></span>
            </li>
        </ul>
        <!-- what the field's description (searchForm.autocompleteInstructions) says, for the eye -->
        <div class="autocomplete-hint" aria-hidden="true">{{ $t("searchForm.autocompleteHint") }}</div>
    </div>
</template>

<script setup>
import { nextTick, ref, watch } from "vue";

const props = defineProps({
    id: { type: String, required: true },
    results: { type: Array, required: true }, // { html, selected }
    activeIndex: { type: Number, default: -1 },
});
const emit = defineEmits(["toggle"]);

const list = ref(null);

// the suggestion moved to with the arrow keys stays in view
watch(
    () => props.activeIndex,
    (index) => {
        if (index < 0) return;
        nextTick(() => list.value?.children[index]?.scrollIntoView?.({ block: "nearest" }));
    }
);
</script>

<style lang="scss" scoped>
@use "../assets/styles/theme.module.scss" as theme;

.autocomplete-popup {
    margin: 3px 0 0 15px;
    border: 1px solid #eeeeee;
    border-top-width: 0px;
    width: 267px;
    position: absolute;
    left: 0;
    background-color: #fff;
    z-index: 100;
    top: 34px;
}

.autocomplete-results {
    padding: 0;
    margin: 0;
    max-height: 216px;
    overflow-y: auto;
}

.autocomplete-result {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    list-style: none;
    text-align: left;
    padding: 4px 12px;
    cursor: pointer;
    font-size: 1.2rem;
}

.autocomplete-result:hover,
.is-active {
    background-color: #ddd;
    color: black;
}

/* the suggestion moved to with the arrow keys: an outline of 3:1 and more on white and on #ddd (WCAG 1.4.11) */
.is-active {
    outline: 2px solid theme.$link-color;
    outline-offset: -2px;
}

.autocomplete-checkbox {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    width: 1.1rem;
    height: 1.1rem;
    border: 1px solid #495057;
    border-radius: 0.25em;
    background-color: #fff;
    font-size: 0.9rem;
}

.autocomplete-checkbox.checked {
    background-color: theme.$link-color;
    border-color: theme.$link-color;
    color: #fff;
}

.autocomplete-hint {
    padding: 2px 12px;
    border-top: 1px solid #dee2e6;
    background-color: #f8f9fa;
    color: #495057;
    font-size: 0.8rem;
}
</style>
