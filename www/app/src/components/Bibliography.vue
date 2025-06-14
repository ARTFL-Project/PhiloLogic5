<template>
    <div class="container-fluid">
        <results-summary :description="results.description"></results-summary>
        <div style="position: relative" v-if="!showFacets && philoConfig.facets.length > 0">
            <button type="button" class="btn btn-sm btn-secondary"
                style="position: absolute; bottom: 0; right: 0.5rem; padding: 0.125rem 0.25rem" @click="toggleFacets()"
                :aria-label="$t('common.showFacetsLabel')">
                {{ $t("common.showFacets") }}
            </button>
        </div>
        <div class="row mt-4" style="padding-right: 0.5rem">
            <div class="col-12" :class="{ 'col-md-9': showFacets, 'col-xl-9': showFacets }"
                v-if="!philoConfig.dictionary_bibliography || results.result_type == 'doc'">
                <div role="region" :aria-label="$t('bibliography.resultsRegion')" aria-live="polite">
                    <transition-group tag="div" :css="false" v-on:before-enter="beforeEnter" v-on:enter="enter">
                        <article class="card philologic-occurrence mx-2 mb-4 shadow-sm"
                            v-for="(result, index) in results.results" :key="result.philo_id.join('-')" role="article"
                            :aria-labelledby="`citation-${results.description.start + index}`">
                            <div class="row citation-container">
                                <div class="col-12 col-sm-10 col-md-11">
                                    <div class="cite" :data-id="result.philo_id.join(' ')">
                                        <span class="number" aria-hidden="true">{{ results.description.start + index
                                        }}</span>
                                        <div class="form-check d-inline-block ms-3 me-2" style="vertical-align: middle"
                                            v-if="resultType == 'doc' && philoConfig.metadata.indexOf('title') !== -1">
                                            <input type="checkbox" class="form-check-input"
                                                :id="`biblio-checkbox-${results.description.start + index}`"
                                                @click="addToSearch(result.metadata_fields.title)"
                                                :aria-describedby="`citation-${results.description.start + index}`"
                                                :aria-label="$t('bibliography.selectForSearch')" />
                                        </div>
                                        <div :id="`citation-${results.description.start + index}`"
                                            class="d-inline-block">
                                            <citations :citation="result.citation"></citations>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </article>
                    </transition-group>
                </div>
            </div>
            <div class="col-12" :class="{ 'col-md-9': showFacets, 'col-xl-9': showFacets }"
                v-if="philoConfig.dictionary_bibliography && results.result_type != 'doc'">
                <div role="region" :aria-label="$t('bibliography.dictionaryResultsRegion')" aria-live="polite">
                    <div class="list-group" v-for="(group, groupKey) in results.results" :key="groupKey" role="group"
                        :aria-label="`${$t('bibliography.titleGroup')} ${groupKey + 1}`">
                        <article class="list-group-item p-0" v-for="(result, index) in group" :key="index"
                            style="border-width: 0" role="article"
                            :aria-labelledby="`dict-citation-${groupKey}-${index}`">
                            <div class="card philologic-occurrence mx-2 mb-4 shadow-sm">
                                <header class="citation-dico-container">
                                    <div class="cite" :data-id="result.philo_id.join(' ')"
                                        :id="`dict-citation-${groupKey}-${index}`">
                                        <span class="number" aria-hidden="true">{{ results.description.start + index
                                        }}</span>
                                        <citations :citation="result.citation"></citations>
                                    </div>
                                </header>
                                <div class="pt-3 px-3 text-content" select-word :position="result.position"
                                    role="region" :aria-label="$t('bibliography.contextRegion')">
                                    <div v-html="result.context"></div>
                                </div>
                            </div>
                        </article>
                    </div>
                </div>
            </div>
            <div class="col" md="5" xl="4" v-if="showFacets" role="complementary"
                :aria-label="$t('common.facetsRegion')">
                <facets></facets>
            </div>
        </div>
        <pages></pages>
    </div>
</template>
<script>
import gsap from "gsap";
import { computed } from "vue";
import { mapFields } from "vuex-map-fields";
import citations from "./Citations";
import facets from "./Facets";
import pages from "./Pages";
import ResultsSummary from "./ResultsSummary";

export default {
    name: "bibliography-report",
    components: {
        citations,
        ResultsSummary,
        facets,
        pages,
    },
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
            "description",
            "currentReport",
            "metadataUpdate",
            "urlUpdate",
            "showFacets",
        ]),
    },
    inject: ["$http"],
    provide() {
        return {
            results: computed(() => this.results.results),
        };
    },
    data() {
        return {
            philoConfig: this.$philoConfig,
            results: {},
            resultType: "doc",
            metadataAddition: [],
        };
    },
    created() {
        this.report = "bibliography";
        this.currentReport = "bibliography";
        this.fetchResults();
    },
    watch: {
        urlUpdate() {
            if (this.report == "bibliography") {
                this.fetchResults();
            }
        },
    },
    methods: {
        fetchResults() {
            this.results = {};
            this.searchParams = { ...this.$store.state.formData };
            this.$http
                .get(`${this.$dbUrl}/reports/bibliography.py`, {
                    params: this.paramsFilter(this.searchParams),
                })
                .then((response) => {
                    if (!this.philoConfig.dictionary_bibliography || response.data.doc_level) {
                        this.results = response.data;
                        this.resultType = this.results.result_type;
                    } else {
                        this.results = this.dictionaryBibliography(response.data);
                    }
                })
                .catch((error) => {
                    this.loading = false;
                    this.error = error.toString();
                    this.debug(this, error);
                });
        },
        dictionaryBibliography(data) {
            let groupedResults = [];
            let currentTitle = data.results[0].metadata_fields.title;
            let titleGroup = [];
            for (let i = 0; i < data.results.length; i += 1) {
                if (data.results[i].metadata_fields.title !== currentTitle) {
                    groupedResults.push(titleGroup);
                    titleGroup = [];
                    currentTitle = data.results[i].metadata_fields.title;
                }
                data.results[i].position = i + 1;
                titleGroup.push(data.results[i]);
                if (i + 1 == data.results.length) {
                    groupedResults.push(titleGroup);
                }
            }
            data.results = groupedResults;
            return data;
        },
        addToSearch(titleValue) {
            let title = '"' + titleValue + '"';
            let itemIndex = this.metadataAddition.indexOf(title);
            if (itemIndex === -1) {
                this.metadataAddition.push(title);
            } else {
                this.metadataAddition.splice(itemIndex, 1);
            }
            let newTitleValue = this.metadataAddition.join(" | ");
            this.$store.commit("updateFormDataField", {
                key: "title",
                value: newTitleValue,
            });
            this.metadataUpdate = { title: newTitleValue };
        },
        beforeEnter: function (el) {
            el.style.opacity = 0;
        },
        enter: function (el, done) {
            gsap.to(el, {
                opacity: 1,
                delay: el.dataset.index * 0.015,
                onComplete: done,
            });
        },
        toggleFacets() {
            if (this.showFacets) {
                this.showFacets = false;
            } else {
                this.showFacets = true;
            }
        },
    },
};
</script>
<style scoped>
.citation-container {
    border-width: 0 !important;
}

.citation-dico-container {
    border-bottom: solid 1px #eee !important;
}

.number {
    background-color: rgb(78, 93, 108);
    color: #fff;
    font-size: 1rem;
    line-height: 1.5;
    padding: 0.375rem 0.75rem;
    display: inline-block;
    margin-right: 5px;
    border-radius: 0.25rem;
}

.text-content {
    text-align: justify;
}
</style>
