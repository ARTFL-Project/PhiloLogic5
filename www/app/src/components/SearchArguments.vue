<template>
    <div id="search-arguments" class="pb-2">
        <div v-if="currentWordQuery !== ''">
            <div v-if="currentReport !== 'collocation'">
                {{ $t("searchArgs.searchingDbFor") }}
                <span v-if="exactPhrase">
                    {{ $t("searchArgs.exactPhrase") }}<button
                        class="btn rounded-pill btn-outline-secondary btn-sm ms-1">{{ q
                        }}</button>
                </span>
                <span v-else>
                    <span v-if="approximate == 'yes'">
                        {{ $t("searchArgs.termsSimilarTo") }} <b>{{ currentWordQuery }}</b>:
                    </span>
                    <span v-else>{{ $t("searchArgs.terms") }}&nbsp;</span>
                    <span v-if="approximate.length == 0 || approximate == 'no'"></span>

                    <div class="term-groups-container" v-for="(group, index) in wordGroups" :key="index">
                        <button type="button" class="term-group-word" @click="getQueryTerms(group, index)"
                            :aria-label="$t('searchArgs.expandTermGroup', { group: group })">
                            {{ group }}
                        </button>
                        <button type="button" class="close-pill" @click="removeTerm(index)"
                            :aria-label="$t('searchArgs.removeTerm', { term: group })">
                            X
                        </button>
                    </div>
                    {{ queryArgs.proximity }}
                </span>
                <div class="card outline-secondary shadow" id="query-terms" v-show="showQueryTerms" role="dialog"
                    aria-modal="true" :aria-labelledby="'query-terms-title'">
                    <button type="button" class="btn btn-secondary btn-sm close" @click="closeTermsList()"
                        :aria-label="$t('common.close')">
                        <span aria-hidden="true">&times;</span>
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
                                    X
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div v-else>
                {{ $t("searchArgs.searchingCollocates") }} <b>{{ currentWordQuery }}</b>&nbsp;
                <span v-if="colloc_within == 'sent'">
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
            :start_date="start_date" :end_date="end_date" :removeMetadata="removeMetadata"></bibliography-criteria>
        <div style="margin-top: 10px" v-if="queryReport === 'collocation'">
            {{ $t("searchArgs.collocOccurrences", { n: resultsLength }) }}
        </div>
    </div>
</template>
<script>
import { mapFields } from "vuex-map-fields";
import BibliographyCriteria from "./BibliographyCriteria";

export default {
    name: "searchArguments",
    components: {
        BibliographyCriteria,
    },
    props: ["resultStart", "resultEnd"],
    computed: {
        ...mapFields([
            "formData.report",
            "formData.q",
            "formData.method_arg",
            "formData.arg_phrase",
            "formData.method",
            "formData.start",
            "formData.end",
            "formData.approximate",
            "formData.approximate_ratio",
            "formData.metadataFields",
            "formData.start_date",
            "formData.end_date",
            "formData.colloc_within",
            "formData.colloc_filter_choice",
            "formData.q_attribute",
            "formData.q_attribute_value",
            "currentReport",
            "resultsLength",
            "description",
        ]),
        formData() {
            return this.$store.state.formData;
        },
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
            let querySplit = this.q.split(" ");
            if (this.q.split('"').length - 1 == 2 && querySplit.length > 1) {
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
        $route: "fetchSearchArgs",
    },
    methods: {
        fetchSearchArgs() {
            this.queryReport = this.$route.name;
            this.currentWordQuery = typeof this.$route.query.q == "undefined" ? "" : this.$route.query.q;
            let queryParams = { ...this.$store.state.formData };
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
                    params: this.paramsFilter({ report: this.report, ...this.$route.query }),
                })
                .then((response) => {
                    this.$store.commit("updateDescription", {
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
                n: this.method_arg,
            });
        },
        removeMetadata(metadata) {
            if (this.q.length == 0 && this.currentReport != "aggregation") {
                this.report = "bibliography";
            }
            this.start = "";
            this.end = "";
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
            this.q = this.termGroupsCopy.join(" ");
            this.approximate = "no";
            this.approximate_ratio = "";
        },
        rerunQuery() {
            this.$router.push(this.paramsToRoute({ ...this.$store.state.formData, q: this.q }));
        },
        removeTerm(index) {
            let queryTermGroup = this.copyObject(this.description.termGroups);
            queryTermGroup.splice(index, 1);
            this.q = queryTermGroup.join(" ");
            if (queryTermGroup.length === 0 && this.currentReport != "aggregation") {
                this.report = "bibliography";
            }
            this.start = 0;
            this.end = 0;
            if (queryTermGroup.length == 1) {
                this.method = "proxy";
                this.method_arg = "";
                this.arg_phrase = "";
            }
            this.$store.commit("updateDescription", {
                ...this.description,
                termGroups: queryTermGroup,
            });
            this.$router.push(this.paramsToRoute({ ...this.$store.state.formData }));
        },
    },
};
</script>
<style scoped lang="scss">
@import "../assets/styles/theme.module.scss";

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
    border: 1px solid $link-color;
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
    color: $link-color;
    border: none;
    border-right: solid 1px $link-color;
    flex-grow: 1;
}

.term-word {
    display: block;
    padding: 0.1rem 0.5rem;
    border-right: solid 1px $link-color;
    flex-grow: 1;
}

.close-pill {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 1.6rem;
    color: $link-color;
    border: none;
    cursor: pointer;
}
</style>
