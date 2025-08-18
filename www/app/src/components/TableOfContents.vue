<template>
    <div class="container-fluid mt-4">
        <div class="row text-center pt-4" id="toc-report-title">
            <div class="col-8 offset-2">
                <h5>
                    <citations :citation="textNavigationCitation"></citations>
                </h5>
            </div>
        </div>
        <div class="text-center">
            <div class="row">
                <div class="col-12">
                    <div class="card mt-4 mb-4 py-4 px-2 d-inline-block shadow"
                        style="width: 100%; text-align: justify">
                        <button id="show-header" class="btn btn-outline-secondary mb-2" v-if="philoConfig.header_in_toc"
                            @click="toggleHeader()" :aria-expanded="showHeader" :aria-label="$t('toc.toggleHeader')"
                            :aria-controls="showHeader ? 'tei-header' : null">
                            {{ headerButton }}
                        </button>
                        <div class="card shadow-sm" id="tei-header" v-if="showHeader" role="region"
                            :aria-label="$t('toc.headerRegion')" v-html="teiHeader">
                        </div>

                        <nav id="toc-report" class="text-content-area" role="navigation"
                            :aria-label="$t('toc.navigationLabel')">
                            <div id="toc-content" v-scroll="handleScroll" ref="tocContent" tabindex="0" role="tree"
                                :aria-label="$t('toc.tocTree')">
                                <ol class="toc-list" role="list">
                                    <li v-for="(element, elIndex) in tocElements.slice(0, displayLimit)" :key="elIndex"
                                        :class="'toc-' + element.philo_type" role="listitem">
                                        <span :class="'bullet-point-' + element.philo_type" aria-hidden="true"></span>
                                        <router-link :to="element.href" class="toc-section" :aria-label="$t('toc.sectionLink', {
                                            type: element.philo_type,
                                            label: element.label
                                        })">
                                            {{ element.label }}
                                        </router-link>
                                        <citations :citation="element.citation"></citations>
                                    </li>
                                </ol>

                                <!-- Loading indicator for infinite scroll -->
                                <div v-if="isLoading" class="text-center mt-3" role="status"
                                    :aria-label="$t('common.loading')">
                                    <progress-spinner :sm="true" />
                                </div>

                                <!-- End of content indicator -->
                                <div v-if="displayLimit >= tocElements.length && tocElements.length > 0"
                                    class="text-center mt-3 text-muted" role="status" aria-live="polite">
                                    {{ $t('toc.endOfContent') }}
                                </div>
                            </div>
                        </nav>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>
<script>
import { mapFields } from "vuex-map-fields";
import citations from "./Citations";
import ProgressSpinner from "./ProgressSpinner";

export default {
    name: "tableOfContents",
    components: {
        citations,
        ProgressSpinner,
    },
    inject: ["$http"],
    computed: {
        ...mapFields({
            report: "formData.report",
            textNavigationCitation: "textNavigationCitation",
            searching: "searching",
        }),
    },
    data() {
        return {
            philoConfig: this.$philoConfig,
            displayLimit: 200,
            teiHeader: "",
            tocObject: {},
            tocElements: [],
            showHeader: false,
            headerButton: this.$t('toc.showHeader'),
            isLoading: false,
        };
    },
    created() {
        this.fetchToC();
    },
    methods: {
        fetchToC() {
            this.searching = true;
            this.$http
                .get(`${this.$dbUrl}/reports/table_of_contents.py`, {
                    params: { philo_id: this.$route.params.pathInfo },
                })
                .then((response) => {
                    this.searching = false;
                    this.tocObject = response.data;
                    this.tocElements = response.data.toc;
                    this.textNavigationCitation = response.data.citation;
                })
                .catch((error) => {
                    this.searching = false;
                    this.debug(this, error);
                });
        },
        toggleHeader() {
            if (!this.showHeader) {
                if (this.teiHeader.length == 0) {
                    this.$http
                        .get(`${this.$dbUrl}/scripts/get_header.py`, {
                            params: {
                                philo_id: this.$route.params.pathInfo,
                            },
                        })
                        .then((response) => {
                            this.teiHeader = response.data;
                            this.headerButton = this.$t('toc.hideHeader');
                            this.showHeader = true;
                        })
                        .catch((error) => {
                            this.debug(this, error);
                        });
                } else {
                    this.headerButton = this.$t('toc.hideHeader');
                    this.showHeader = true;
                }
            } else {
                this.headerButton = this.$t('toc.showHeader');
                this.showHeader = false;
            }
        },
        handleScroll() {
            const tocContent = this.$refs.tocContent;
            if (!tocContent) return;

            const scrollPosition = tocContent.getBoundingClientRect().bottom - 200;
            if (scrollPosition < window.innerHeight && !this.isLoading && this.displayLimit < this.tocElements.length) {
                this.isLoading = true;
                setTimeout(() => {
                    this.displayLimit += 200;
                    this.isLoading = false;
                }, 100);
            }
        },
    },
};
</script>
<style scoped lang="scss">
@use "../assets/styles/theme.module.scss" as theme;

.separator {
    padding: 5px;
    font-size: 60%;
    display: inline-block;
    vertical-align: middle;
}

.toc-section {
    font-size: 1.05rem;
    text-decoration: none;
    transition: all 0.15s ease-in-out;
}

.toc-section:hover,
.toc-section:focus {
    outline: 1px solid theme.$link-color;
    background-color: rgba(theme.$link-color, 0.05);
    border-radius: 4px;
}

.toc-list {
    list-style: none;
    padding-left: 0;
}

.toc-list li {
    margin-bottom: 0.5rem;
    border-radius: 4px;
    padding-top: 0.05rem;
    padding-bottom: 0.05rem;
}

.toc-list li:hover {
    background-color: #f8f9fa;
}

#tei-header {
    white-space: pre;
    font-family: monospace;
    font-size: 120%;
    overflow-x: auto;
    padding: 10px;
    background-color: rgb(253, 253, 253);
    margin: 10px;
    border: 1px solid #dee2e6;
}

#toc-content {
    padding: 1rem;
}

#toc-content:focus {
    outline: 2px solid theme.$link-color;
    outline-offset: -2px;
}

:deep(h5 .text-view) {
    font-size: inherit !important;
}
</style>
