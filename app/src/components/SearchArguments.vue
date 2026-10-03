<template>
    <div id="search-arguments" class="pb-2">
        <div v-if="currentWordQuery !== ''">
            <div v-if="currentReport !== 'collocation'">
                {{ $t("searchArgs.searchingDbFor") }}
                <span v-if="exactPhrase">
                    {{ $t("searchArgs.exactPhrase") }}<button
                        class="btn rounded-pill btn-outline-secondary btn-sm ms-1">{{ formData.q
                        }}</button>
                </span>
                <span v-else>
                    <span v-if="formData.approximate == 'yes'">
                        {{ $t("searchArgs.termsSimilarTo") }} <b>{{ currentWordQuery }}</b>:
                    </span>
                    <span v-else>{{ $t("searchArgs.terms") }}&nbsp;</span>
                    <span v-if="formData.approximate.length == 0 || formData.approximate == 'no'"></span>

                    <div class="term-groups-container" v-for="(group, index) in wordGroups" :key="index">
                        <button type="button" class="term-group-word" @click="getQueryTerms(group, index, $event)"
                            :aria-label="$t('searchArgs.expandTermGroup', { group: groupLabel(group, index) })"
                            :ref="el => { if (el) termGroupButtons[index] = el }">
                            {{ groupLabel(group, index) }}
                        </button>
                        <button type="button" class="close-pill" @click="removeTerm(index)"
                            :aria-label="$t('searchArgs.removeTerm', { term: group })">
                            <span class="icon-x"></span>
                        </button>
                    </div>
                    {{ termsProximity }}
                </span>
                <div class="card shadow" id="query-terms" v-if="showQueryTerms" role="dialog" aria-modal="true"
                    aria-labelledby="query-terms-title" aria-describedby="query-terms-description"
                    ref="queryTermsDialog" @keydown="handleDialogKeydown">
                    <div class="card-header d-flex align-items-center">
                        <h2 class="h6 mb-0 flex-grow-1" id="query-terms-title">{{ $t("searchArgs.searchTerms") }}</h2>
                        <button type="button" class="btn btn-sm close-box close" @click="closeTermsList()"
                            :aria-label="$t('common.close')" ref="closeButton">
                            <span class="icon-x" aria-hidden="true"></span>
                        </button>
                    </div>
                    <div class="card-body">
                        <p class="query-terms-summary" id="query-terms-description">
                            {{ $t("searchArgs.termCount", { term: selectedTerm, n: words.length }) }}
                            <span v-if="words.length > 100">({{ $t("searchArgs.mostFrequentTerms") }})</span>
                        </p>
                        <ul id="query-terms-list" aria-labelledby="query-terms-title" ref="termsList"
                            @keydown="moveBetweenTerms">
                            <li class="term-chip" v-for="word in words" :key="word">
                                <span class="term-chip-word">{{ bare(word) }}</span>
                                <button type="button" class="term-chip-remove"
                                    @click="removeFromTermsList(word, groupIndexSelected)"
                                    :aria-label="$t('searchArgs.excludeTerm', { term: bare(word) })">
                                    <span class="icon-x" aria-hidden="true"></span>
                                </button>
                            </li>
                        </ul>
                    </div>
                    <div class="card-footer d-flex align-items-center flex-wrap gap-2" v-if="wordListChanged">
                        <span class="flex-grow-1" role="status">
                            {{ $t("searchArgs.termsExcluded", { n: excludedCount }) }}
                        </span>
                        <button type="button" class="btn btn-secondary btn-sm" @click="rerunQuery()" ref="rerunButton">
                            {{ $t("searchArgs.rerunQuery") }}
                        </button>
                    </div>
                </div>
            </div>
            <div v-else>
                {{ $t("searchArgs.searchingCollocates") }} <b>{{ currentWordQuery }}</b>&nbsp;
                <span v-if="formData.colloc_within == 'sent'">
                    {{ $t("searchArgs.sameSentence") }}
                </span>
                <span v-else>
                    {{ proximity() }}</span>
                <div v-if="collocationFilter">
                    {{ $t("searchArgs.collocateFilter") }}:&nbsp; <b>{{
                        $philoConfig.word_property_aliases[collocationFilter.attrib] || collocationFilter.attrib }} = {{
                            collocationFilter.value
                        }}</b>
                </div>
            </div>
            <!-- Terms that matched more word forms than the search takes (regexes with no literal start) -->
            <div class="alert alert-warning py-1 px-2 my-2 d-inline-block" role="note"
                v-for="cut in description.cutTerms || []" :key="`${cut.term}-${cut.not}`">
                {{ $t(cut.not ? "searchArgs.notTermCut" : "searchArgs.termCut", {
                    term: cut.term, n: (description.expansionCap || 0).toLocaleString($i18n.locale),
                }) }}
            </div>
        </div>
        <bibliography-criteria :biblio="queryArgs.biblio" :queryReport="queryReport" :resultsLength="resultsLength"
            :start_date="formData.start_date" :end_date="formData.end_date"
            :removeMetadata="removeMetadata"></bibliography-criteria>
        <div style="margin-top: 10px" v-if="queryReport === 'collocation'">
            {{ $t("searchArgs.collocOccurrences", { n: resultsLength }) }}
        </div>
    </div>
</template>
<script setup>
import { computed, inject, nextTick, reactive, ref, useTemplateRef, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { storeToRefs } from "pinia";
import { useI18n } from "vue-i18n";
import { useMainStore } from "../stores/main";
import {
    buildBiblioCriteria,
    copyObject,
    debug,
    isOnlyFacetChange,
    paramsFilter,
    paramsToRoute,
} from "../utils.js";
import BibliographyCriteria from "./BibliographyCriteria";  // eslint-disable-line no-unused-vars

const props = defineProps(["resultStart", "resultEnd"]);

const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const philoConfig = inject("$philoConfig");
const route = useRoute();
const router = useRouter();
const { t } = useI18n();
const store = useMainStore();
const { formData, currentReport, description, resultsLength } = storeToRefs(store);

const closeButton = useTemplateRef("closeButton");
const queryTermsDialog = useTemplateRef("queryTermsDialog");

const currentWordQuery = ref(typeof route.query.q === "undefined" ? "" : route.query.q);
const queryArgs = reactive({});
const words = ref([]);
const wordListChanged = ref(false);
const queryReport = ref(route.name);
const termGroupsCopy = ref([]);
const showQueryTerms = ref(false);
const groupIndexSelected = ref(null);
const termGroupButtons = ref({});
const triggerButtonIndex = ref(null);

const wordGroups = computed(() => description.value.termGroups);
const excludedCount = ref(0);  // the words taken out of the dialog's list since it opened
const termsList = useTemplateRef("termsList");
const rerunButton = useTemplateRef("rerunButton");
const bare = (word) => word.replace(/"/g, "");  // a word of the list, without the quotes of its search
// What the dialog's list expanded from: the term typed, for an approximate search's similar words
const selectedTerm = computed(() => {
    const index = groupIndexSelected.value;
    return description.value.approximateGroups?.[index]?.term ?? wordGroups.value?.[index] ?? "";
});

// The similar words of a term of an approximate search, folded: liberté (27 similar terms), not the 27 of them
// (152 at 80%), whose list the button's dialog shows
function groupLabel(group, index) {
    const folded = description.value.approximateGroups?.[index];
    return folded ? t("searchArgs.similarTerms", { term: folded.term, n: folded.variants }) : group;
}

// How the term groups are searched together, as the server reads the form (Query.resolve_method): no distance, or 0,
// is a phrase. One group ("a | b", or a word with "NOT") has nothing to describe.
const termsProximity = computed(() => {
    if (!wordGroups.value || wordGroups.value.length < 2) return "";
    const method = formData.value.method || "proxy";
    const distance = parseInt(formData.value.method_arg) || 0;
    let how;
    if (method === "sentence") how = t("searchArgs.sameSentence");
    else if (distance === 0) how = t("searchArgs.adjacent");
    else if (method === "exact_cooc") how = t("searchArgs.withinExactlyProximity", { n: distance });
    else how = t("searchArgs.withinProximity", { n: distance });
    return formData.value.cooc_order === "yes" ? `${how}, ${t("searchArgs.inThisOrder")}` : how;
});

const collocationFilter = computed(() => {
    if (formData.value.colloc_filter_choice === "attribute") {
        return {
            attrib: formData.value.q_attribute,
            value: formData.value.q_attribute_value,
        };
    }
    return false;
});

const exactPhrase = computed(() => {
    const q = formData.value.q;
    const querySplit = q.split(" ");
    if (q.split('"').length - 1 === 2 && querySplit.length > 1) {
        const firstWord = querySplit.shift();
        const lastWord = querySplit.pop();
        return firstWord.startsWith('"') && lastWord.endsWith('"');
    }
    return false;
});

function fetchSearchArgs() {
    queryReport.value = route.name;
    currentWordQuery.value = typeof route.query.q === "undefined" ? "" : route.query.q;
    const queryParams = { ...formData.value };
    queryArgs.queryTerm = "q" in queryParams ? queryParams.q : "";
    queryArgs.biblio = buildBiblioCriteria(philoConfig, route.query, formData.value);

    queryArgs.approximate = queryParams.approximate === "yes";

    $http
        .get(`${$dbUrl}/scripts/get_term_groups.py`, {
            params: paramsFilter({ report: formData.value.report, ...route.query }),
        })
        .then((response) => {
            store.updateDescription({
                ...description.value,
                start: props.resultStart,
                end: props.resultEnd,
                results_per_page: formData.value.results_per_page,
                termGroups: response.data.term_groups,
                approximateGroups: response.data.approximate_groups || [],
                cutTerms: response.data.cut_terms || [],
                expansionCap: response.data.expansion_cap,
            });
        })
        .catch((error) => {
            debug({ $options: { name: "searchArguments" } }, error);
        });
}

function removeMetadata(metadata) {
    if (formData.value.q.length === 0 && currentReport.value !== "aggregation") {
        formData.value.report = "bibliography";
    }
    formData.value.start = "";
    formData.value.end = "";
    const localParams = copyObject(formData.value);
    localParams[metadata] = "";
    router.push(paramsToRoute(localParams));
}

function getQueryTerms(group, index) {
    groupIndexSelected.value = index;
    triggerButtonIndex.value = index;
    excludedCount.value = 0;
    $http
        .get(`${$dbUrl}/scripts/get_query_terms.py`, {
            params: {
                q: group,
                approximate: 0,
                ...paramsFilter(route.query),
            },
        })
        .then((response) => {
            words.value = response.data;
            showQueryTerms.value = true;
            nextTick(() => {
                if (closeButton.value) closeButton.value.focus();
            });
        })
        .catch((error) => {
            debug({ $options: { name: "searchArguments" } }, error);
        });
}

function closeTermsList() {
    showQueryTerms.value = false;
    nextTick(() => {
        const triggerButton = termGroupButtons.value[triggerButtonIndex.value];
        if (triggerButton) triggerButton.focus();
    });
}

function handleDialogKeydown(event) {
    if (event.key === "Escape") {
        closeTermsList();
        return;
    }
    if (event.key === "Tab") {
        const dialog = queryTermsDialog.value;
        if (!dialog) return;
        const focusable = dialog.querySelectorAll(
            'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
        );
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (event.shiftKey && document.activeElement === first) {
            event.preventDefault();
            last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
            event.preventDefault();
            first.focus();
        }
    }
}

// The remove buttons of the dialog's words, in order
const removeButtons = () => [...(termsList.value?.querySelectorAll(".term-chip-remove") || [])];

// Arrow keys, Home and End move between the words' remove buttons: Tab goes through them too, one by one
function moveBetweenTerms(event) {
    const buttons = removeButtons();
    const current = buttons.indexOf(document.activeElement);
    if (current === -1) return;
    const last = buttons.length - 1;
    const targets = {
        ArrowRight: current + 1, ArrowDown: current + 1, ArrowLeft: current - 1, ArrowUp: current - 1, Home: 0, End: last,
    };
    if (!(event.key in targets)) return;
    event.preventDefault();
    buttons[Math.max(0, Math.min(targets[event.key], last))].focus();
}

function removeFromTermsList(word, groupIndex) {
    const index = words.value.indexOf(word);
    const removedWithKeyboard = removeButtons().includes(document.activeElement);
    words.value.splice(index, 1);
    wordListChanged.value = true;
    excludedCount.value += 1;
    if (removedWithKeyboard) {
        // its button is gone: the next word's, or the last one's, or the dialog's rerun button
        nextTick(() => {
            const buttons = removeButtons();
            const next = buttons[Math.min(index, buttons.length - 1)] || rerunButton.value || closeButton.value;
            if (next) next.focus();
        });
    }
    if (termGroupsCopy.value.length === 0) {
        termGroupsCopy.value = copyObject(wordGroups.value);
    }
    if (termGroupsCopy.value[groupIndex].indexOf(" NOT ") !== -1) {
        // already a NOT in the clause: add an OR
        termGroupsCopy.value[groupIndex] += " | " + word.trim();
    } else {
        termGroupsCopy.value[groupIndex] += " NOT " + word.trim();
    }
    formData.value.q = termGroupsCopy.value.join(" ");
    formData.value.approximate = "no";
    formData.value.approximate_ratio = "";
}

function rerunQuery() {
    showQueryTerms.value = false;
    router.push(paramsToRoute({ ...formData.value, q: formData.value.q }));
}

function proximity() {
    return t("searchArgs.withinProximity", { n: formData.value.method_arg });
}

function removeTerm(index) {
    const queryTermGroup = copyObject(description.value.termGroups);
    queryTermGroup.splice(index, 1);
    formData.value.q = queryTermGroup.join(" ");
    if (queryTermGroup.length === 0 && currentReport.value !== "aggregation") {
        formData.value.report = "bibliography";
    }
    formData.value.start = 0;
    formData.value.end = 0;
    if (queryTermGroup.length === 1) {
        formData.value.method = "proxy";
        formData.value.method_arg = "";
        formData.value.arg_phrase = "";
    }
    store.updateDescription({ ...description.value, termGroups: queryTermGroup });
    router.push(paramsToRoute({ ...formData.value }));
}

watch(
    () => route.fullPath,
    (newPath, oldPath) => {
        const newQuery = router.resolve(newPath).query;
        const oldQuery = router.resolve(oldPath || "").query;
        if (["concordance", "kwic", "bibliography"].includes(formData.value.report)) {
            if (!isOnlyFacetChange(newQuery, oldQuery)) fetchSearchArgs();
        } else {
            fetchSearchArgs();
        }
    }
);

fetchSearchArgs();
</script>
<style scoped lang="scss">
@use "../assets/styles/theme.module.scss" as theme;

#search-arguments {
    line-height: 180%;
}

#query-terms {
    position: absolute;
    z-index: 100;
    width: min(46rem, 95vw);
    border: 1px solid theme.$card-header-color;
}

#query-terms .card-header {
    font-variant: small-caps;
    padding: 0.4rem 0.5rem 0.4rem 1rem;
}

.query-terms-summary {
    margin-bottom: 0.75rem;
}

#query-terms-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    list-style: none;
    margin: 0;
    padding: 2px;
    max-height: min(22rem, 50vh);
    overflow-y: auto;
}

.term-chip {
    display: inline-flex;
    align-items: center;
    border: 1px solid rgba(theme.$link-color, 0.45);
    border-radius: 50rem;
    background-color: rgba(theme.$link-color, 0.06);
    color: #212529;
    line-height: 1.5;
}

.term-chip-word {
    padding: 0.1rem 0.35rem 0.1rem 0.7rem;
}

#query-terms .term-chip-remove {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.75rem;
    height: 1.75rem;
    margin-right: 0.1rem;
    padding: 0;
    border: none;
    border-radius: 50%;
    background: transparent;
    color: theme.$link-color;
    cursor: pointer;
}

#query-terms .term-chip-remove .icon-x {
    flex-shrink: 0;
    background-color: theme.$link-color !important; /* the theme's .icon-x is !important too */
    width: 0.8em;
    height: 0.8em;
}

#query-terms .term-chip-remove:hover,
#query-terms .term-chip-remove:focus-visible {
    background-color: theme.$link-color;
}

#query-terms .term-chip-remove:hover .icon-x,
#query-terms .term-chip-remove:focus-visible .icon-x {
    background-color: #fff !important; /* 6.7:1 on the button's red */
}

#query-terms .term-chip-remove:focus-visible {
    outline-offset: 1px !important; /* the theme's ring, clear of the word */
}

#query-terms .card-footer {
    background-color: rgba(theme.$link-color, 0.06);
}

.term-groups-container {
    display: inline-flex;
    align-items: stretch;
    border: 1px solid theme.$link-color;
    border-radius: 50rem;
    margin: 5px 5px 5px 0px;
    background-color: #fff;
    overflow: hidden;
}

.term-group-word {
    display: block;
    padding: 0.1rem 0.5rem;
    text-decoration: none;
    background: none;
    color: theme.$link-color;
    border: none;
    border-right: solid 1px theme.$link-color;
    flex-grow: 1;
}

.term-word {
    display: block;
    padding: 0.1rem 0.5rem;
    border-right: solid 1px theme.$link-color;
    flex-grow: 1;
}

.close-pill {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 1.6rem;
    color: theme.$link-color;
    border: none;
    cursor: pointer;
}
</style>
