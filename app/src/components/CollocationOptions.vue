<template>
    <div class="mt-1" id="collocation-params" role="group" aria-labelledby="collocation-params-heading">
        <h2 id="collocation-params-heading" class="colloc-heading">{{ $t("searchForm.collocationParams") }}:</h2>

        <div class="input-group colloc-row">
            <label class="btn btn-outline-secondary" for="colloc-within">{{ $t("searchForm.collocCount") }}</label>
            <select class="form-select colloc-select" id="colloc-within" v-model="formData.colloc_within">
                <option value="sent">{{ $t("searchForm.collocInSentence") }}</option>
                <option value="n">{{ $t("searchForm.within") }}</option>
            </select>
            <template v-if="formData.colloc_within === 'n'">
                <input type="number" min="1" class="form-control colloc-n-words" id="collocNWords" name="method_arg"
                    :aria-label="$t('searchForm.collocNWordsLabel')" v-model="formData.method_arg" />
                <span class="input-group-text">{{ $t("searchForm.words") }}</span>
            </template>
        </div>

        <div class="input-group colloc-row colloc-choice" role="radiogroup" aria-labelledby="colloc-lemmas-label"
            v-if="$philoConfig.collocation_lemmas">
            <span class="btn btn-outline-secondary colloc-chip" id="colloc-lemmas-label">
                {{ $t("searchForm.collocCountAs") }}
            </span>
            <input type="radio" class="btn-check" name="colloc_lemmas" id="collocLemmas" value="yes"
                v-model="formData.colloc_lemmas" />
            <label class="btn btn-outline-secondary" for="collocLemmas">{{ $t("searchForm.collocLemmas") }}</label>
            <input type="radio" class="btn-check" name="colloc_lemmas" id="collocForms" value="no"
                v-model="formData.colloc_lemmas" />
            <label class="btn btn-outline-secondary" for="collocForms">{{ $t("searchForm.collocWordForms") }}</label>
        </div>
    </div>

    <div class="mt-3" id="collocate-filters" role="group" aria-labelledby="collocate-filters-heading">
        <h2 id="collocate-filters-heading" class="colloc-heading">{{ $t("searchForm.collocateFilters") }}:</h2>

        <div class="input-group colloc-row">
            <label class="btn btn-outline-secondary" for="colloc-common">{{ $t("searchForm.collocExclude") }}</label>
            <select class="form-select colloc-select" id="colloc-common" v-model="common">
                <option value="nofilter">{{ $t("searchForm.collocLeaveNothing") }}</option>
                <option value="frequency">{{ $t("searchForm.collocLeaveFrequent") }}</option>
                <template v-if="$philoConfig.stopwords">
                    <option value="stopwords">{{ $t("searchForm.collocLeaveStopwords") }}</option>
                    <option value="frequency | stopwords">{{ $t("searchForm.collocLeaveBoth") }}</option>
                </template>
            </select>
            <template v-if="common.includes('frequency')">
                <span class="input-group-text">{{ $t("searchForm.collocInOver") }}</span>
                <div class="form-control colloc-range">
                    <input type="range" class="form-range df-filter-range" id="filter-df-pct" name="filter_df_pct"
                        min="0.5" max="5" step="0.1" v-model="formData.filter_df_pct"
                        :aria-label="$t('searchForm.dfFilterLabel')"
                        :aria-valuetext="$t('searchForm.dfFilterValueText', { pct: formData.filter_df_pct })"
                        aria-describedby="filter-df-pct-desc" />
                </div>
                <span class="input-group-text colloc-pct" aria-hidden="true">
                    {{ formData.filter_df_pct }}{{ $t("searchForm.dfFilterUnit") }}
                </span>
                <span id="filter-df-pct-desc" class="visually-hidden">{{ $t("searchForm.dfFilterDescription") }}</span>
            </template>
        </div>

        <template v-if="propertyNames.length > 0">
            <div class="input-group colloc-row colloc-properties" role="group"
                aria-labelledby="colloc-properties-label" aria-describedby="colloc-properties-help">
                <span class="btn btn-outline-secondary colloc-chip" id="colloc-properties-label">
                    {{ $t("searchForm.collocKeepOnly") }}
                </span>
                <div class="colloc-property" v-for="name in propertyNames" :key="name" :ref="(el) => (wrappers[name] = el)"
                    @keydown.esc.stop="close(name, true)" @focusout="onFocusOut(name, $event)">
                    <button type="button" class="form-select colloc-property-toggle" :id="`colloc-property-${name}-toggle`"
                        :aria-expanded="open === name ? 'true' : 'false'" :aria-controls="`colloc-property-${name}`"
                        @click="toggle(name)">
                        {{ alias(name) }}: <span :class="{ 'colloc-any': kept(name).length === 0 }">{{ summary(name)
                        }}</span>
                    </button>
                    <div class="colloc-popup shadow" :id="`colloc-property-${name}`" tabindex="-1"
                        :hidden="open !== name">
                        <fieldset v-if="open === name">
                            <legend class="visually-hidden">
                                {{ $t("searchForm.collocValuesOf", { property: alias(name) }) }}
                            </legend>
                            <div class="colloc-value-actions">
                                <button type="button" class="btn btn-link btn-sm p-0" @click="setKept(name, values(name))">
                                    {{ $t("searchForm.collocSelectAll") }}
                                    <span class="visually-hidden">{{ alias(name) }}</span>
                                </button>
                                <button type="button" class="btn btn-link btn-sm p-0" @click="setKept(name, [])">
                                    {{ $t("searchForm.collocClear") }}
                                    <span class="visually-hidden">{{ alias(name) }}</span>
                                </button>
                            </div>
                            <div class="colloc-value-list">
                                <div class="form-check colloc-value" v-for="(value, index) in values(name)" :key="value">
                                    <input class="form-check-input" type="checkbox"
                                        :id="`colloc-property-${name}-${index}`" :checked="kept(name).includes(value)"
                                        @change="toggleValue(name, value, $event)" />
                                    <label class="form-check-label" :for="`colloc-property-${name}-${index}`">{{ value
                                        }}</label>
                                </div>
                            </div>
                        </fieldset>
                    </div>
                </div>
            </div>
            <p class="colloc-help" id="colloc-properties-help">{{ $t("searchForm.collocPropertiesHelp") }}</p>
        </template>
    </div>
</template>

<script setup>
// The collocation parameters of the search form: where collocates are counted (in the sentence, within n words), as
// word forms or lemmas, without the common words (the most frequent, the stopwords: either or both), and only with
// some values of word properties (any of a property's ticked values, for each property with values ticked)
import { storeToRefs } from "pinia";
import { computed, inject, nextTick, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import { useMainStore } from "../stores/main";
import { PROPERTY_PARAM, defaultCommonWords, splitValues } from "../utils.js";

const philoConfig = inject("$philoConfig");
const store = useMainStore();
const { formData } = storeToRefs(store);
const { t } = useI18n();

const propertyNames = Object.keys(philoConfig.word_attributes || {});
// The number of words "within" counts in until another is typed
const DEFAULT_WORDS = 5;

watch(
    () => formData.value.colloc_within,
    (within) => {
        if (within === "n" && !String(formData.value.method_arg ?? "").trim()) {
            store.updateFormDataField({ key: "method_arg", value: String(DEFAULT_WORDS) });
        }
    },
    { immediate: true }
);
const open = ref("");  // the property whose values show
const wrappers = {};

// The common words left out, as colloc_filter_choice has them: "frequency", "stopwords", both
// ("frequency | stopwords") or "nofilter". Empty, in a new search, is those it leaves out at first (see SearchForm's
// onSubmit).
const common = computed({
    get() {
        const chosen = splitValues(formData.value.colloc_filter_choice || defaultCommonWords(philoConfig));
        const value = ["frequency", "stopwords"].filter((choice) => chosen.includes(choice)).join(" | ");
        return value || "nofilter";
    },
    set(value) {
        store.updateFormDataField({ key: "colloc_filter_choice", value });
    },
});

function alias(name) {
    return philoConfig.word_property_aliases?.[name] || name;
}
function values(name) {
    return philoConfig.word_attributes[name] || [];
}
function kept(name) {
    return splitValues(formData.value[`${PROPERTY_PARAM}${name}`]);
}
function setKept(name, kept) {
    const order = values(name);
    const sorted = [...kept].sort((a, b) => order.indexOf(a) - order.indexOf(b));
    store.updateFormDataField({ key: `${PROPERTY_PARAM}${name}`, value: sorted.join(" | ") });
}
function toggleValue(name, value, event) {
    const now = kept(name).filter((v) => v !== value);
    if (event.target.checked) now.push(value);
    setKept(name, now);
}
// What the property's button says of it: its values ticked, or any
function summary(name) {
    const values = kept(name);
    return values.length > 0 ? values.join(", ") : t("searchForm.collocAnyValue");
}

// A property's values show below its button, one property's at a time, until Escape, the button again, or the focus
// going elsewhere
function toggle(name) {
    open.value = open.value === name ? "" : name;
    if (open.value) nextTick(() => keepInView(name));
}
// The values below their button, moved left as far as they must to stay on the screen (WCAG 1.4.10)
function keepInView(name) {
    const popup = document.getElementById(`colloc-property-${name}`);
    if (!popup) return;
    popup.style.left = "";
    const margin = 8;
    const box = popup.getBoundingClientRect();
    const over = box.right - (document.documentElement.clientWidth - margin);
    if (over > 0) popup.style.left = `${-Math.min(over, box.left - margin)}px`;
}
function close(name, refocus = false) {
    if (open.value !== name) return;
    open.value = "";
    if (refocus) nextTick(() => document.getElementById(`colloc-property-${name}-toggle`)?.focus());
}
function onFocusOut(name, event) {
    if (!wrappers[name]?.contains(event.relatedTarget)) close(name);
}
</script>

<style lang="scss" scoped>
@use "../assets/styles/theme.module.scss" as theme;

.colloc-heading {
    width: fit-content;
    font-variant: small-caps;
    font-weight: 700;
    font-size: 1.15rem;
}

.colloc-row {
    width: fit-content;
    max-width: 100%;
    margin-bottom: 0.5rem;
}

.colloc-select {
    width: auto;
    min-width: 0;
    flex: 0 1 auto;
}

.colloc-n-words {
    width: 5rem;
    flex: 0 0 auto;
    text-align: center;
}

.colloc-range {
    display: flex;
    align-items: center;
    width: 6.5rem;
    flex: 0 0 auto;
    padding-left: 0.5rem;
    padding-right: 0.5rem;
}

/* The slider in the theme's colors, its thumb smaller than Bootstrap's */
.df-filter-range::-webkit-slider-thumb {
    width: 0.85rem;
    height: 0.85rem;
    margin-top: -0.175rem;
    background-color: theme.$button-color;
}

.df-filter-range::-moz-range-thumb {
    width: 0.85rem;
    height: 0.85rem;
    background-color: theme.$button-color;
}

.df-filter-range::-webkit-slider-runnable-track {
    height: 0.5rem;
}

.df-filter-range::-moz-range-track {
    height: 0.5rem;
}

.colloc-pct {
    white-space: nowrap;
}

/* Visible focus indicator for the document-frequency slider (WCAG 2.4.7). */
.df-filter-range:focus-visible {
    outline: 2px solid theme.$button-color;
    outline-offset: 2px;
}

/* The choice keyboard focus is on: the theme's outline, as the form's buttons have (WCAG 2.4.7) */
.colloc-choice .btn-check:focus-visible + .btn {
    outline: 2px solid theme.$button-color;
    outline-offset: 2px;
    z-index: 5;
}

/* The label of the row, not a button: no pointer */
.colloc-chip {
    cursor: default;
}

.colloc-property {
    position: relative;
    display: flex;
}

/* The property buttons are one group with the row's label, as the controls of the other rows */
.colloc-property-toggle {
    width: auto;
    text-align: start;
    white-space: normal;
    border-radius: 0;
    margin-left: calc(var(--bs-border-width) * -1);
}

.colloc-property:last-child .colloc-property-toggle {
    border-top-right-radius: var(--bs-border-radius);
    border-bottom-right-radius: var(--bs-border-radius);
}

.colloc-any {
    font-style: italic;
}

.colloc-popup {
    position: absolute;
    top: 100%;
    left: 0;
    z-index: 100;
    margin-top: 2px;
    padding: 0.5rem 0.75rem;
    width: max-content;
    min-width: 100%;
    max-width: min(28rem, calc(100vw - 2rem));
    background-color: #fff;
    border: 1px solid #ced4da;
    border-radius: var(--bs-border-radius);
}

.colloc-popup fieldset {
    min-width: 0;
}

.colloc-value-actions {
    display: flex;
    gap: 1rem;
    margin-bottom: 0.25rem;
}

.colloc-value-list {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(6.5rem, 1fr));
    gap: 0 0.75rem;
    width: min(26rem, calc(100vw - 3.5rem));
    max-height: 16rem;
    overflow-y: auto;
}

.colloc-value {
    margin-bottom: 0.125rem;
}

.colloc-help {
    margin-top: -0.25rem;
    font-size: 0.875rem;
    color: #495057;
}
</style>
