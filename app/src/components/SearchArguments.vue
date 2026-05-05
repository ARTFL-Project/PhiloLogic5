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
                            :aria-label="$t('searchArgs.expandTermGroup', { group: group })"
                            :ref="el => { if (el) termGroupButtons[index] = el }">
                            {{ group }}
                        </button>
                        <button type="button" class="close-pill" @click="removeTerm(index)"
                            :aria-label="$t('searchArgs.removeTerm', { term: group })">
                            <span class="icon-x"></span>
                        </button>
                    </div>
                    {{ queryArgs.proximity }}
                </span>
                <div class="card outline-secondary shadow" id="query-terms" v-if="showQueryTerms" role="dialog"
                    aria-modal="true" :aria-labelledby="'query-terms-title'" ref="queryTermsDialog"
                    @keydown="handleDialogKeydown">
                    <button type="button" class="btn btn-secondary btn-sm close" @click="closeTermsList()"
                        :aria-label="$t('common.close')" ref="closeButton">
                        <span class="icon-x"></span>
                    </button>
                    <span class="pe-4 h6" id="query-terms-title">
                        {{ $t("searchArgs.termsExpanded", { length: words.length }) }}:
                    </span>
                    <h4 class="h6" v-if="words.length > 100" id="query-terms-frequent">{{
                        $t("searchArgs.mostFrequentTerms") }}</h4>
                    <button type="button" class="btn btn-secondary btn-sm" style="margin: 10px 0px"
                        v-if="wordListChanged" @click="rerunQuery()">
                        {{ $t("searchArgs.rerunQuery") }}
                    </button>
                    <div class="row" id="query-terms-list">
                        <div class="col-3" v-for="word in words" :key="word">
                            <div class="term-groups-container">
                                <span class="term-word">{{ word.replace(/"/g, "") }}</span>
                                <button type="button" class="close-pill"
                                    @click="removeFromTermsList(word, groupIndexSelected)"
                                    :aria-label="$t('searchArgs.excludeTerm', { term: word })">
                                    <span class="icon-x"></span>
                                </button>
                            </div>
                        </div>
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
const { formData, currentReport, description } = storeToRefs(store);

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

    if ("q" in queryParams) {
        const method = queryParams.method || "proxy";
        if (queryParams.q.split(" ").length > 1) {
            if (method === "proxy") {
                queryArgs.proximity = (typeof queryParams.method_arg !== "undefined" || queryParams.method_arg)
                    ? t("searchArgs.withinProximity", { n: queryParams.method_arg })
                    : "";
            } else if (method === "exac_cooc") {
                queryArgs.proximity = (typeof queryParams.method_arg !== "undefined" || queryParams.arg_phrase)
                    ? t("searchArgs.withinExactlyProximity", { n: queryParams.arg_phrase })
                    : "";
            } else if (method === "sentence") {
                queryArgs.proximity = t("searchArgs.sameSentence");
            }
        } else {
            queryArgs.proximity = "";
        }
    }
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

function removeFromTermsList(word, groupIndex) {
    const index = words.value.indexOf(word);
    words.value.splice(index, 1);
    wordListChanged.value = true;
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
    padding: 10px 15px 0px 15px;
    box-shadow: 0px 0.2em 8px 0.01em rgba(0, 0, 0, 0.1);
}

#query-terms>button:first-child {
    position: absolute;
    right: 2px;
    top: 0;
}

#query-terms-list {
    margin: 10px -5px;
    max-height: 400px;
    max-width: 800px;
    overflow-y: scroll;
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
