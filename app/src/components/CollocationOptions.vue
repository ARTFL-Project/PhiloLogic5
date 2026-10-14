<template>
    <div class="mt-1" id="collocation-params" role="group" aria-labelledby="collocation-params-heading">
        <h2 id="collocation-params-heading">{{ $t("searchForm.collocationParams") }}:</h2>
        <div class="colloc-options">
            <fieldset class="colloc-fieldset">
                <legend>{{ $t("searchForm.collocCountWithin") }}</legend>
                <div class="form-check">
                    <input class="form-check-input" type="radio" name="colloc_within" id="collocSentence" value="sent"
                        v-model="formData.colloc_within" />
                    <label class="form-check-label" for="collocSentence">{{ $t("searchForm.collocSameSentence") }}</label>
                </div>
                <div class="form-check colloc-inline">
                    <input class="form-check-input" type="radio" name="colloc_within" id="collocWithinN" value="n"
                        v-model="formData.colloc_within" aria-labelledby="collocNWords collocWithinN-unit" />
                    <input type="number" min="1" class="form-control form-control-sm colloc-n-words" id="collocNWords"
                        name="method_arg" :aria-label="$t('searchForm.collocNWordsLabel')" v-model="formData.method_arg"
                        @input="formData.colloc_within = 'n'" />
                    <label id="collocWithinN-unit" for="collocWithinN">{{ $t("searchForm.words") }}</label>
                </div>
            </fieldset>

            <fieldset class="colloc-fieldset" v-if="$philoConfig.collocation_lemmas">
                <legend>{{ $t("searchForm.collocCountAs") }}</legend>
                <div class="form-check">
                    <input class="form-check-input" type="radio" name="colloc_lemmas" id="collocForms" value="no"
                        v-model="formData.colloc_lemmas" />
                    <label class="form-check-label" for="collocForms">{{ $t("searchForm.collocWordForms") }}</label>
                </div>
                <div class="form-check">
                    <input class="form-check-input" type="radio" name="colloc_lemmas" id="collocLemmas" value="yes"
                        v-model="formData.colloc_lemmas" />
                    <label class="form-check-label" for="collocLemmas">{{ $t("searchForm.collocLemmas") }}</label>
                </div>
            </fieldset>

            <fieldset class="colloc-fieldset">
                <legend>{{ $t("searchForm.collocCommonWords") }}</legend>
                <div class="form-check colloc-inline">
                    <input class="form-check-input" type="checkbox" id="collocFrequent" v-model="leaveOutFrequent"
                        aria-labelledby="collocFrequent-label filter-df-pct-value" />
                    <label class="form-check-label" id="collocFrequent-label" for="collocFrequent">
                        {{ $t("searchForm.collocFrequentWords") }}
                    </label>
                    <input type="range" class="form-range df-filter-range" id="filter-df-pct" name="filter_df_pct"
                        min="0.5" max="5" step="0.1" v-model="formData.filter_df_pct" @input="leaveOutFrequent = true"
                        :aria-label="$t('searchForm.dfFilterLabel')"
                        :aria-valuetext="$t('searchForm.dfFilterValueText', { pct: formData.filter_df_pct })"
                        aria-describedby="filter-df-pct-desc" />
                    <output for="filter-df-pct" id="filter-df-pct-value" class="colloc-nowrap" aria-hidden="true">
                        {{ formData.filter_df_pct }}{{ $t("searchForm.dfFilterUnit") }}
                    </output>
                    <span id="filter-df-pct-desc" class="visually-hidden">{{ $t("searchForm.dfFilterDescription") }}</span>
                </div>
                <div class="form-check" v-if="$philoConfig.stopwords">
                    <input class="form-check-input" type="checkbox" id="collocStopwords" v-model="leaveOutStopwords" />
                    <label class="form-check-label" for="collocStopwords">{{ $t("searchForm.collocStopwords") }}</label>
                </div>
            </fieldset>

            <fieldset class="colloc-fieldset colloc-properties" v-if="propertyNames.length > 0"
                aria-describedby="colloc-properties-help">
                <legend>{{ $t("searchForm.collocProperties") }}</legend>
                <p class="colloc-help" id="colloc-properties-help">{{ $t("searchForm.collocPropertiesHelp") }}</p>
                <div class="colloc-property" v-for="name in propertyNames" :key="name">
                    <button type="button" class="btn btn-outline-secondary btn-sm colloc-property-toggle"
                        :id="`colloc-property-${name}-toggle`" :aria-expanded="open[name] ? 'true' : 'false'"
                        :aria-controls="`colloc-property-${name}`" @click="open[name] = !open[name]">
                        <span class="colloc-chevron" aria-hidden="true">{{ open[name] ? "▾" : "▸" }}</span>
                        <span>{{ alias(name) }}: </span>
                        <span :class="{ 'colloc-any': kept(name).length === 0 }">{{ summary(name) }}</span>
                    </button>
                    <fieldset class="colloc-values" :id="`colloc-property-${name}`" :hidden="!open[name]">
                        <template v-if="open[name]">
                        <legend class="visually-hidden">{{ $t("searchForm.collocValuesOf", { property: alias(name) }) }}
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
                            <div class="form-check" v-for="(value, index) in values(name)" :key="value">
                                <input class="form-check-input" type="checkbox" :id="`colloc-property-${name}-${index}`"
                                    :checked="kept(name).includes(value)" @change="toggleValue(name, value, $event)" />
                                <label class="form-check-label" :for="`colloc-property-${name}-${index}`">{{ value
                                    }}</label>
                            </div>
                        </div>
                        </template>
                    </fieldset>
                </div>
            </fieldset>
        </div>
    </div>
</template>

<script setup>
// The collocation parameters of the search form: where collocates are counted (in the sentence, within n words), as
// word forms or lemmas, without the common words (the most frequent, the stopwords: either or both), and only with
// some values of word properties (any of a property's ticked values, for each property with values ticked)
import { storeToRefs } from "pinia";
import { computed, inject, reactive } from "vue";
import { useMainStore } from "../stores/main";
import { PROPERTY_PARAM, defaultCommonWords, splitValues } from "../utils.js";
import { useI18n } from "vue-i18n";

const philoConfig = inject("$philoConfig");
const store = useMainStore();
const { formData } = storeToRefs(store);
const { t } = useI18n();

const propertyNames = Object.keys(philoConfig.word_attributes || {});
const open = reactive(Object.fromEntries(propertyNames.map((name) => [name, false])));

// The common words left out: colloc_filter_choice is "frequency", "stopwords", both ("frequency | stopwords") or
// "nofilter". Empty, in a new search, is those it leaves out at first (see SearchForm's onSubmit).
const common = computed(() => {
    const choice = formData.value.colloc_filter_choice || defaultCommonWords(philoConfig);
    return new Set(splitValues(choice));
});
function setCommon(choice, on) {
    const chosen = new Set(common.value);
    chosen.delete("nofilter");
    if (on) chosen.add(choice);
    else chosen.delete(choice);
    const value = ["frequency", "stopwords"].filter((c) => chosen.has(c)).join(" | ");
    store.updateFormDataField({ key: "colloc_filter_choice", value: value || "nofilter" });
}
const leaveOutFrequent = computed({
    get: () => common.value.has("frequency"),
    set: (on) => setCommon("frequency", on),
});
const leaveOutStopwords = computed({
    get: () => common.value.has("stopwords"),
    set: (on) => setCommon("stopwords", on),
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
</script>

<style scoped>
#collocation-params-heading {
    width: fit-content;
    font-variant: small-caps;
    font-weight: 700;
    font-size: 1.15rem;
}

.colloc-options {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 17rem), 1fr));
    gap: 0.75rem 1rem;
    align-items: start;
}

.colloc-fieldset {
    border: 1px solid #ced4da;
    border-radius: 0.375rem;
    padding: 0.5rem 0.75rem 0.75rem;
    min-width: 0;
}

.colloc-fieldset legend {
    float: none;
    width: auto;
    font-size: 1rem;
    font-weight: 600;
    padding: 0 0.25rem;
    margin-bottom: 0.25rem;
}

.colloc-properties {
    grid-column: 1 / -1;
}

.colloc-inline {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.25rem 0.5rem;
}

.colloc-inline .form-check-input {
    flex-shrink: 0;
}

.colloc-n-words {
    width: 5rem;
    text-align: center;
}

.df-filter-range {
    width: 8rem;
}

/* Visible focus indicator for the document-frequency slider (WCAG 2.4.7). */
.df-filter-range:focus-visible {
    outline: 3px solid #0d6efd;
    outline-offset: 2px;
}

.colloc-nowrap {
    white-space: nowrap;
}

.colloc-help {
    margin-bottom: 0.5rem;
}

.colloc-property + .colloc-property {
    margin-top: 0.5rem;
}

.colloc-property-toggle {
    text-align: start;
    white-space: normal;
}

.colloc-chevron {
    display: inline-block;
    width: 1em;
}

.colloc-any {
    font-style: italic;
}

.colloc-values {
    margin: 0.5rem 0 0 1.25rem;
    min-width: 0;
}

.colloc-value-actions {
    display: flex;
    gap: 1rem;
    margin-bottom: 0.25rem;
}

.colloc-value-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem 1.25rem;
    max-height: 14rem;
    overflow-y: auto;
}
</style>
