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
                        <button type="button" class="term-group-word" @click="getQueryTerms(group, index)"
                            :aria-label="$t('searchArgs.expandTermGroup', { group: group })">
                            {{ group }}
                        </button>
                        <button type="button" class="close-pill" @click="removeTerm(index)"
                            :aria-label="$t('searchArgs.removeTerm', { term: group })">
                            <span class="icon-x"></span>
                        </button>
                    </div>
                    {{ queryArgs.proximity }}
                </span>
                <div class="card outline-secondary shadow" id="query-terms" v-show="showQueryTerms" role="dialog"
                    aria-modal="true" :aria-labelledby="'query-terms-title'">
                    <button type="button" class="btn btn-secondary btn-sm close" @click="closeTermsList()"
                        :aria-label="$t('common.close')">
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
<script>
import { mapStores, mapWritableState } from "pinia";
import { useMainStore } from "../stores/main";
import BibliographyCriteria from "./BibliographyCriteria";

export default {
    name: "searchArguments",
    components: {
        BibliographyCriteria,
    },
    props: ["resultStart", "resultEnd"],
    computed: {
        ...mapWritableState(useMainStore, [
            "formData",
            "currentReport",
            "resultsLength",
            "description"
        ]),
        ...mapStores(useMainStore),
        wordGroups() {
            return this.description.termGroups;
        },
        collocationFilter() {
            if (this.formData.colloc_filter_choice == 'attribute') {
                return {
                    attrib: this.formData.q_attribute,
                    value: this.formData.q_attribute_value,
                };
            } else {
                return false
            }
        },
        exactPhrase() {
            let querySplit = this.formData.q.split(" ");
            if (this.formData.q.split('"').length - 1 == 2 && querySplit.length > 1) {
                let lastWord = querySplit.pop();
                let firstWord = querySplit.shift();
                if (firstWord.startsWith('"') && lastWord.endsWith('"')) {
                    return true;
                }
            }
            return false
        },
    },
    inject: ["$http"],
    data() {
        return {
            currentWordQuery: typeof this.$route.query.q == "undefined" ? "" : this.$route.query.q,
            queryArgs: {},
            words: [],
            wordListChanged: false,
            restart: false,
            queryReport: this.$route.name,
            termGroupsCopy: [],
            showQueryTerms: false,
            groupIndexSelected: null,
        };
    },
    created() {
        this.fetchSearchArgs();
    },
    watch: {
        // call again the method if the route changes
        $route(newUrl, oldUrl) {
            let facetReports = ["concordance", "kwic", "bibliography"]
            if (facetReports.includes(this.formData.report)) {
                if (!this.isOnlyFacetChange(newUrl.query, oldUrl.query)) {
                    this.fetchSearchArgs();
                }
            } else {
                this.fetchSearchArgs();
            }
        }
    },
    methods: {
        fetchSearchArgs() {
            this.queryReport = this.$route.name;
            this.currentWordQuery = typeof this.$route.query.q == "undefined" ? "" : this.$route.query.q;
            let queryParams = { ...this.formData };
            if ("q" in queryParams) {
                this.queryArgs.queryTerm = queryParams.q;
            } else {
                this.queryArgs.queryTerm = "";
            }
            this.queryArgs.biblio = this.buildBiblioCriteria(this.$philoConfig, this.$route.query, this.formData)

            if ("q" in queryParams) {
                let method = queryParams.method;
                if (typeof method === "undefined") {
                    method = "proxy";
                }
                if (queryParams.q.split(" ").length > 1) {
                    if (method === "proxy") {
                        if (typeof queryParams.method_arg !== "undefined" || queryParams.method_arg) {
                            this.queryArgs.proximity = this.$t("searchArgs.withinProximity", {
                                n: queryParams.method_arg,
                            });
                        } else {
                            this.queryArgs.proximity = "";
                        }
                    } else if (method === "exac_cooc") {
                        if (typeof queryParams.method_arg !== "undefined" || queryParams.arg_phrase) {
                            this.queryArgs.proximity = this.$t("searchArgs.withinExactlyProximity", {
                                n: queryParams.arg_phrase,
                            });
                        } else {
                            this.queryArgs.proximity = "";
                        }
                    } else if (method === "sentence") {
                        this.queryArgs.proximity = this.$t("searchArgs.sameSentence");
                    }
                } else {
                    this.queryArgs.proximity = "";
                }
            }
            if (queryParams.approximate == "yes") {
                this.queryArgs.approximate = true;
            } else {
                this.queryArgs.approximate = false;
            }
            this.$http
                .get(`${this.$dbUrl}/scripts/get_term_groups.py`, {
                    params: this.paramsFilter({ report: this.formData.report, ...this.$route.query }),
                })
                .then((response) => {
                    this.mainStore.updateDescription({
                        ...this.description,
                        start: this.resultStart,
                        end: this.resultEnd,
                        results_per_page: this.formData.results_per_page,
                        termGroups: response.data.term_groups,
                    });
                    this.originalQuery = response.data.original_query;
                })
                .catch((error) => {
                    this.loading = false;
                    this.error = error.toString();
                    this.debug(this, error);
                });
        },
        proximity() {
            return this.$t("searchArgs.withinProximity", {
                n: this.formData.method_arg,
            });
        },
        removeMetadata(metadata) {
            if (this.formData.q.length == 0 && this.currentReport != "aggregation") {
                this.formData.report = "bibliography";
            }
            this.formData.start = "";
            this.formData.end = "";
            let localParams = this.copyObject(this.formData);
            localParams[metadata] = "";
            this.$router.push(this.paramsToRoute(localParams));
        },
        getQueryTerms(group, index) {
            this.groupIndexSelected = index;
            this.$http
                .get(`${this.$dbUrl}/scripts/get_query_terms.py`, {
                    params: {
                        q: group,
                        approximate: 0,
                        ...this.paramsFilter(this.$route.query),
                    },
                })
                .then((response) => {
                    this.words = response.data;
                    this.showQueryTerms = true;
                })
                .catch((error) => {
                    this.error = error.toString();
                    this.debug(this, error);
                });
        },
        closeTermsList() {
            this.showQueryTerms = false;
        },
        removeFromTermsList(word, groupIndex) {
            var index = this.words.indexOf(word);
            this.words.splice(index, 1);
            this.wordListChanged = true;
            if (this.termGroupsCopy.length == 0) {
                this.termGroupsCopy = this.copyObject(this.wordGroups);
            }
            if (this.termGroupsCopy[groupIndex].indexOf(" NOT ") !== -1) {
                // if there's already a NOT in the clause add an OR
                this.termGroupsCopy[groupIndex] += " | " + word.trim();
            } else {
                this.termGroupsCopy[groupIndex] += " NOT " + word.trim();
            }
            this.formData.q = this.termGroupsCopy.join(" ");
            this.formData.approximate = "no";
            this.formData.approximate_ratio = "";
        },
        rerunQuery() {
            this.$router.push(this.paramsToRoute({ ...this.formData, q: this.formData.q }));
        },
        removeTerm(index) {
            let queryTermGroup = this.copyObject(this.description.termGroups);
            queryTermGroup.splice(index, 1);
            this.formData.q = queryTermGroup.join(" ");
            if (queryTermGroup.length === 0 && this.currentReport != "aggregation") {
                this.formData.report = "bibliography";
            }
            this.formData.start = 0;
            this.formData.end = 0;
            if (queryTermGroup.length == 1) {
                this.formData.method = "proxy";
                this.formData.method_arg = "";
                this.formData.arg_phrase = "";
            }
            this.mainStore.updateDescription({
                ...this.description,
                termGroups: queryTermGroup,
            });
            this.$router.push(this.paramsToRoute({ ...this.formData }));
        },
    },
};
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
