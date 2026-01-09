<template>
    <div class="container-fluid">
        <results-summary :description="results.description"></results-summary>

        <div class="row bibliography-layout mt-3 pe-1">
            <!-- Facets sidebar - appears first in DOM for mobile accessibility -->
            <div role="region" class="col-12 col-md-3 col-xl-3 facets-column" :aria-label="$t('common.facetsRegion')"
                v-if="showFacets">
                <facets></facets>
            </div>

            <!-- Results column for regular bibliography -->
            <div role="region" class="col-12" :class="{ 'col-md-9': showFacets, 'col-xl-9': showFacets }"
                :aria-label="$t('bibliography.bibliographyResultsRegion')"
                v-if="!philoConfig.dictionary_bibliography || results.result_type == 'doc'">
                <transition-group tag="div" :css="false" v-on:before-enter="beforeEnter" v-on:enter="enter">
                    <article class="card philologic-occurrence mx-2 mb-4 shadow-sm"
                        v-for="(result, index) in results.results" :key="result.philo_id.join('-')" role="article">
                        <div class="row citation-container">
                            <div class="col-12">
                                <div class="cite d-flex align-items-center" :data-id="result.philo_id.join(' ')">
                                    <span class="number flex-shrink-0" aria-hidden="true">{{
                                        results.description.start + index
                                    }}</span>
                                    <div :id="`citation-${results.description.start + index}`" class="flex-grow-1">
                                        <citations :citation="result.citation"
                                            :result-number="results.description.start + index"></citations>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </article>
                </transition-group>
            </div>

            <!-- Results column for dictionary bibliography -->
            <div role="region" class="col-12" :class="{ 'col-md-9': showFacets, 'col-xl-9': showFacets }"
                :aria-label="$t('bibliography.dictionaryResultsRegion')"
                v-if="philoConfig.dictionary_bibliography && results.result_type != 'doc'">
                <div class="list-group" v-for="(group, groupKey) in results.results" :key="groupKey" role="group">
                    <article class="list-group-item p-0" v-for="(result, index) in group" :key="index"
                        style="border-width: 0">
                        <div class="card philologic-occurrence mx-2 mb-4 shadow-sm">
                            <header class="citation-dico-container">
                                <div class="cite" :data-id="result.philo_id.join(' ')"
                                    :id="`dict-citation-${groupKey}-${index}`">
                                    <span class="number" aria-hidden="true">{{ results.description.start + index
                                        }}</span>
                                    <citations :citation="result.citation"
                                        :result-number="results.description.start + index"></citations>
                                </div>
                            </header>
                            <div class="pt-3 px-3 text-content" select-word :position="result.position">
                                <div v-html="result.context"></div>
                            </div>
                        </div>
                    </article>
                </div>
            </div>

            <div class="pages-wrapper">
                <pages></pages>
            </div>
        </div>
    </div>
</template>
<script>
import gsap from "gsap";
import { mapStores, mapWritableState } from "pinia";
import { computed } from "vue";
import { useMainStore } from "../stores/main";
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
        ...mapWritableState(useMainStore, [
            "formData",
            "description",
            "currentReport",
            "metadataUpdate",
            "urlUpdate",
            "showFacets"
        ]),
        ...mapStores(useMainStore),
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
        };
    },
    created() {
        this.formData.report = "bibliography";
        this.currentReport = "bibliography";
        this.fetchResults();
    },
    watch: {
        urlUpdate(newUrl, oldUrl) {
            if (!this.isOnlyFacetChange(newUrl, oldUrl)) {
                this.fetchResults();
            }
        },
    },
    methods: {
        fetchResults() {
            this.results = {};
            this.searchParams = { ...this.formData };
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

/* Mobile layout: facets above results */
@media (max-width: 767px) {
    .bibliography-layout {
        display: flex;
        flex-direction: column;
    }

    .facets-column {
        order: 1;
        margin-bottom: 1rem;
    }

    .bibliography-layout>div[role="region"]:not(.facets-column) {
        order: 2;
    }

    .pages-wrapper {
        order: 3;
        width: 100%;
    }
}

/* Desktop layout: facets on right side */
@media (min-width: 768px) {
    .bibliography-layout {
        display: flex;
        flex-direction: row;
        flex-wrap: wrap;
    }

    .facets-column {
        order: 2;
        padding-left: 0 !important;
    }

    .bibliography-layout>div[role="region"]:not(.facets-column) {
        order: 1;
    }

    .pages-wrapper {
        order: 3;
        width: 100%;
    }
}
</style>
