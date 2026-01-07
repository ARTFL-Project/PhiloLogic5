<template>
    <div class="container-fluid">
        <results-summary :description="results.description"></results-summary>

        <div class="row concordance-layout px-2">
            <!-- Facets sidebar - appears first in DOM for mobile accessibility -->
            <div role="region" class="col-12 col-md-3 col-xl-3 facets-column" :aria-label="$t('common.facetsRegion')"
                v-if="showFacets">
                <facets></facets>
            </div>

            <!-- Results column -->
            <div role="region" class="col-12" :class="{ 'col-md-9': showFacets, 'col-xl-9': showFacets }"
                :aria-label="$t('concordance.resultsRegion')">

                <transition-group tag="div" :css="false" v-on:before-enter="onBeforeEnter" v-on:enter="onEnter">
                    <article class="card philologic-occurrence text-view mb-3 shadow-sm"
                        v-for="(result, index) in results.results" :key="result.philo_id.join('-')" :data-index="index">

                        <!-- Citation header -->
                        <div class="row citation-container g-0">
                            <div class="col-10 col-md-11">
                                <span class="cite" :id="`result-${index}-heading`">
                                    <span class="number" :aria-describedby="`result-number-desc-${index}`">
                                        {{ results.description.start + index }}
                                        <span :id="`result-number-desc-${index}`" class="visually-hidden">
                                            {{ $t('concordance.resultNumber') }} {{ results.description.start + index }}
                                        </span>
                                    </span>
                                    <citations :citation="result.citation"
                                        :result-number="results.description.start + index"></citations>
                                </span>
                            </div>
                            <div class="col-2 col-md-1">
                                <button type="button" class="btn btn-secondary more-context"
                                    @click="moreContext(index, $event)"
                                    :aria-label="`${$t('concordance.showMoreContext')} ${$t('concordance.forResult')} ${results.description.start + index}`">
                                    <span class="more-text">{{ $t("concordance.more") }}</span>
                                    <i class="bi bi-plus-square more-icon" aria-hidden="true"></i>
                                </button>
                            </div>
                        </div>

                        <!-- Concordance content -->
                        <div class="row">
                            <div class="col m-2 concordance-text" :position="results.description.start + index"
                                @keyup="dicoLookup($event, result.metadata_fields.year)">
                                <div class="default-length" v-html="result.context"></div>
                                <div class="more-length"></div>
                            </div>
                        </div>
                    </article>
                </transition-group>
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
        ...mapWritableState(useMainStore, [
            "formData",
            "resultsLength",
            "searching",
            "currentReport",
            "description",
            "showFacets",
            "urlUpdate"
        ]),
        ...mapStores(useMainStore),
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
        this.formData.report = "concordance";
        this.currentReport = "concordance";
        this.fetchResults();
    },
    watch: {
        urlUpdate(newUrl, oldUrl) {
            console.log(newUrl.q, oldUrl.q);
            if (!this.isOnlyFacetChange(newUrl, oldUrl)) {
                this.fetchResults();
            }
        },
    },
    methods: {
        fetchResults() {
            this.results = { description: { end: 0 } };
            this.searchParams = { ...this.formData };
            this.searching = true;
            this.$http
                .get(`${this.$dbUrl}/reports/concordance.py`, {
                    params: this.paramsFilter(this.searchParams),
                })
                .then((response) => {
                    this.results = response.data;
                    this.mainStore.updateResultsLength(parseInt(response.data.results_length));
                    this.searching = false;
                })
                .catch((error) => {
                    this.searching = false;
                    this.error = error.toString();
                    this.debug(this, error);
                });
        },
        moreContext(index, event) {
            let button = event.target.closest("button");
            let textSpan = button.querySelector(".more-text");
            let icon = button.querySelector(".more-icon");
            let defaultNode = document.getElementsByClassName("default-length")[index];
            let moreNode = document.getElementsByClassName("more-length")[index];
            let resultNumber = this.results.description.start + index - 1;
            let localParams = { hit_num: resultNumber, ...this.searchParams };

            const isExpanded = textSpan.innerHTML == this.$t("concordance.less") ||
                icon.classList.contains("bi-dash-square");

            if (!isExpanded) {
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
                            textSpan.innerHTML = this.$t("concordance.less");
                            icon.classList.remove("bi-plus-square");
                            icon.classList.add("bi-dash-square");
                        })
                        .catch((error) => {
                            this.loading = false;
                            this.error = error.toString();
                            this.debug(this, error);
                        });
                } else {
                    defaultNode.style.display = "none";
                    moreNode.style.display = "block";
                    textSpan.innerHTML = this.$t("concordance.less");
                    icon.classList.remove("bi-plus-square");
                    icon.classList.add("bi-dash-square");
                }
            } else {
                defaultNode.style.display = "block";
                moreNode.style.display = "none";
                textSpan.innerHTML = this.$t("concordance.more");
                icon.classList.remove("bi-dash-square");
                icon.classList.add("bi-plus-square");
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

/* Show text by default on larger screens (> 991px / Bootstrap lg) */
.more-text {
    display: inline-block;
}

.more-icon {
    display: none;
    font-size: 1.125rem;
    vertical-align: middle;
}

/* At smaller viewports or zoom, switch to icon-only */
@media (max-width: 991px) {
    .more-text {
        display: none;
    }

    .more-icon {
        display: inline-block;
    }

    .more-context {
        padding: 0.25rem 0.5rem;
        min-width: auto;
        line-height: 1.2;
    }

    .number {
        padding: 0.25rem 0.5rem !important;
        line-height: 1.2 !important;
    }
}

.more_context,
.citation-container {
    border-bottom: solid 1px #eee !important;
}

.number {
    color: #fff;
    font-size: 1rem;
    line-height: 1.5;
    padding: 0.375rem 0.75rem;
    display: inline-block;
    margin-right: 5px;
    border-radius: 0.25rem;
}

.hit_n {
    vertical-align: 5px;
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

/* Mobile layout: facets above results */
@media (max-width: 767px) {
    .concordance-layout {
        display: flex;
        flex-direction: column;
    }

    .facets-column {
        order: 1;
        margin-bottom: 1rem;
    }

    .concordance-layout>div[role="region"]:not(.facets-column) {
        order: 2;
    }

    .pages-wrapper {
        order: 3;
        width: 100%;
    }
}

/* Desktop layout: facets on right side */
@media (min-width: 768px) {
    .concordance-layout {
        display: flex;
        flex-direction: row;
        flex-wrap: wrap;
    }

    .facets-column {
        order: 2;
        padding-left: 0 !important;
    }

    .concordance-layout>div[role="region"]:not(.facets-column) {
        order: 1;
    }

    .pages-wrapper {
        order: 3;
        width: 100%;
    }
}
</style>
