<template>
    <div class="container-fluid">
        <results-summary :description="results.description"></results-summary>

        <!-- Facets toggle button with proper labeling -->
        <div style="position: relative" v-if="!showFacets && philoConfig.facets.length > 0">
            <button type="button" class="btn btn-secondary" style="position: absolute; bottom: 1rem; right: 0.5rem"
                @click="toggleFacets()" :aria-label="$t('common.showFacetsLabel')">
                {{ $t("common.showFacets") }}
            </button>
        </div>

        <div class="row" style="padding-right: 0.5rem">
            <main class="col-12" :class="{ 'col-md-9': showFacets, 'col-xl-9': showFacets }" role="main"
                :aria-label="$t('concordance.resultsRegion')">

                <transition-group tag="div" :css="false" v-on:before-enter="onBeforeEnter" v-on:enter="onEnter">
                    <article class="card philologic-occurrence text-view ms-2 me-2 mb-3 shadow-sm"
                        v-for="(result, index) in results.results" :key="result.philo_id.join('-')" :data-index="index"
                        :aria-labelledby="`result-${index}-heading`">

                        <!-- Citation header -->
                        <header class="row citation-container g-0">
                            <div class="col-12 col-sm-10 col-md-11">
                                <span class="cite" :id="`result-${index}-heading`">
                                    <span class="number"
                                        :aria-label="`${$t('concordance.resultNumber')} ${results.description.start + index}`">
                                        {{ results.description.start + index }}
                                    </span>
                                    <citations :citation="result.citation"></citations>
                                </span>
                            </div>
                            <div class="col-sm-2 col-md-1 d-none d-sm-inline-block">
                                <button type="button" class="btn btn-secondary more-context"
                                    @click="moreContext(index, $event)"
                                    :aria-label="`${$t('concordance.showMoreContext')} ${$t('concordance.forResult')} ${results.description.start + index}`">
                                    <span class="more d-none d-lg-inline-block">{{ $t("concordance.more") }}</span>
                                    <span class="visually-hidden">{{ $t("concordance.more") }}</span>
                                </button>
                            </div>
                        </header>

                        <!-- Concordance content -->
                        <div class="row">
                            <div class="col m-2 concordance-text" :position="results.description.start + index"
                                @keyup="dicoLookup($event, result.metadata_fields.year)"
                                :aria-label="`${$t('concordance.contextRegion')} ${results.description.start + index}`">
                                <div class="default-length" v-html="result.context"></div>
                                <div class="more-length"></div>
                            </div>
                        </div>
                    </article>
                </transition-group>
            </main>

            <!-- Facets sidebar -->
            <aside class="col col-md-3 col-xl-3 ps-0" v-if="showFacets" role="complementary"
                :aria-label="$t('common.facetsRegion')">
                <facets></facets>
            </aside>

            <pages></pages>
        </div>
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
    name: "concordance-report",
    components: {
        citations,
        ResultsSummary,
        facets,
        pages,
    },
    inject: ["$http"],
    provide() {
        return {
            results: computed(() => this.results.results),
        };
    },
    computed: {
        ...mapFields([
            "formData.report",
            "resultsLength",
            "searching",
            "currentReport",
            "description",
            "showFacets",
            "urlUpdate",
        ]),
    },
    data() {
        return {
            philoConfig: this.$philoConfig,
            results: { description: { end: 0 }, results: [] },
            searchParams: {},
            unbindUrlUpdate: null,
            start: 1,
        };
    },
    created() {
        this.report = "concordance";
        this.currentReport = "concordance";
        this.fetchResults();
    },
    watch: {
        urlUpdate() {
            if (this.report == "concordance") {
                this.fetchResults();
            }
        },
    },
    methods: {
        fetchResults() {
            this.results = { description: { end: 0 } };
            this.searchParams = { ...this.$store.state.formData };
            this.searching = true;
            this.$http
                .get(`${this.$dbUrl}/reports/concordance.py`, {
                    params: this.paramsFilter(this.searchParams),
                })
                .then((response) => {
                    this.results = response.data;
                    this.$store.commit("updateResultsLength", parseInt(response.data.results_length));
                    this.searching = false;
                })
                .catch((error) => {
                    this.searching = false;
                    this.error = error.toString();
                    this.debug(this, error);
                });
        },
        moreContext(index, event) {
            let button = event.srcElement;
            if (button.tagName == "BUTTON") {
                button = button.querySelector("span");
            }
            let defaultNode = document.getElementsByClassName("default-length")[index];
            let moreNode = document.getElementsByClassName("more-length")[index];
            let resultNumber = this.results.description.start + index - 1;
            let localParams = { hit_num: resultNumber, ...this.searchParams };
            if (button.innerHTML == this.$t("concordance.more")) {
                if (moreNode.innerHTML.length == 0) {
                    this.$http
                        .get(`${this.$dbUrl}/scripts/get_more_context.py`, {
                            params: this.paramsFilter(localParams),
                        })
                        .then((response) => {
                            let moreText = response.data;
                            moreNode.innerHTML = moreText;
                            defaultNode.style.display = "none";
                            moreNode.style.display = "block";
                            button.innerHTML = this.$t("concordance.less");
                        })
                        .catch((error) => {
                            this.loading = false;
                            this.error = error.toString();
                            this.debug(this, error);
                        });
                } else {
                    defaultNode.style.display = "none";
                    moreNode.style.display = "block";
                    button.innerHTML = this.$t("concordance.less");
                }
            } else {
                defaultNode.style.display = "block";
                moreNode.style.display = "none";
                button.innerHTML = this.$t("concordance.more");
            }
        },
        dicoLookup() { },
        toggleFacets() {
            if (this.showFacets) {
                this.showFacets = false;
            } else {
                this.showFacets = true;
            }
        },
        onBeforeEnter(el) {
            el.style.opacity = 0;
        },
        onEnter(el, done) {
            gsap.to(el, {
                opacity: 1,
                delay: el.dataset.index * 0.015,
                onComplete: done,
            });
        },
    },
};
</script>

<style>
.concordance-text {
    text-align: justify;
}

.philologic-occurrence {
    left: 0;
    position: relative;
}

.separator {
    padding: 5px;
    font-size: 60%;
    display: inline-block;
    vertical-align: middle;
}

.more-context {
    line-height: 1.8;
    position: absolute;
    right: 0;
}

.more_context,
.citation-container {
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
    height: 100%;
}

.hit_n {
    vertical-align: 5px;
    /*align numbers*/
}

.cite {
    height: 38px;
    display: inline-block;
}

.philologic-doc {
    font-variant: small-caps;
    font-weight: 700;
}

.citation-separator {
    margin-left: 8px;
    padding-left: 8px;
    border-left: double 3px darkgray;
}

.page-display {
    margin-left: 8px;
    padding-left: 8px;
    border-left: double 3px darkgray;
}

.citation-small-caps {
    font-variant: small-caps;
}

/* Concordance styling for theater */
.xml-speaker {
    font-weight: 700;
}

.xml-sp+.xml-l,
.xml-sp+.xml-ab {
    display: inline;
}

.xml-stage {
    font-style: italic;
}
</style>
