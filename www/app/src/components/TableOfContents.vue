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
                                <ul class="toc-tree" role="tree">
                                    <li v-for="(element, elIndex) in processedTocElements.slice(0, displayLimit)"
                                        :key="elIndex" :class="'toc-item toc-' + element.philo_type" role="treeitem"
                                        :aria-level="element.level || 1">

                                        <!-- Section content -->
                                        <div class="toc-content-wrapper">
                                            <span v-if="element.philo_type === 'div1'" class="div1-marker"
                                                aria-hidden="true">§</span>
                                            <span v-else :class="'bullet-point-' + element.philo_type"
                                                aria-hidden="true"></span>
                                            <router-link :to="element.href" class="toc-section" :aria-label="$t('toc.sectionLink', {
                                                type: element.philo_type,
                                                label: element.label
                                            })">
                                                {{ element.label }}
                                            </router-link>
                                            <citations :citation="element.citation"></citations>
                                        </div>

                                        <!-- Child sections -->
                                        <ul v-if="element.children && element.children.length > 0" class="toc-children"
                                            role="group">
                                            <li v-for="(child, childIndex) in element.children" :key="childIndex"
                                                :class="'toc-item toc-child toc-' + child.philo_type" role="treeitem"
                                                :aria-level="child.level || 2">
                                                <div class="toc-content-wrapper">
                                                    <span :class="'bullet-point-' + child.philo_type"
                                                        aria-hidden="true"></span>
                                                    <router-link :to="child.href" class="toc-section" :aria-label="$t('toc.sectionLink', {
                                                        type: child.philo_type,
                                                        label: child.label
                                                    })">
                                                        {{ child.label }}
                                                    </router-link>
                                                    <citations :citation="child.citation"></citations>
                                                </div>

                                                <!-- Grandchildren (div3 level) -->
                                                <ul v-if="child.children && child.children.length > 0"
                                                    class="toc-children" role="group">
                                                    <li v-for="(grandchild, grandchildIndex) in child.children"
                                                        :key="grandchildIndex"
                                                        :class="'toc-item toc-child toc-' + grandchild.philo_type"
                                                        role="treeitem" :aria-level="grandchild.level || 3">
                                                        <div class="toc-content-wrapper">
                                                            <span :class="'bullet-point-' + grandchild.philo_type"
                                                                aria-hidden="true"></span>
                                                            <router-link :to="grandchild.href" class="toc-section"
                                                                :aria-label="$t('toc.sectionLink', {
                                                                    type: grandchild.philo_type,
                                                                    label: grandchild.label
                                                                })">
                                                                {{ grandchild.label }}
                                                            </router-link>
                                                            <citations :citation="grandchild.citation"></citations>
                                                        </div>
                                                    </li>
                                                </ul>
                                            </li>
                                        </ul>
                                    </li>
                                </ul>

                                <!-- Loading indicator for infinite scroll -->
                                <div v-if="isLoading" class="text-center mt-3" role="status"
                                    :aria-label="$t('common.loading')">
                                    <progress-spinner :sm="true" />
                                </div>

                                <!-- End of content indicator -->
                                <div v-if="displayLimit >= processedTocElements.length && processedTocElements.length > 0"
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
import { mapStores, mapWritableState } from "pinia";
import { useMainStore } from "../stores/main";
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
        ...mapWritableState(useMainStore, [
            "formData",
            "textNavigationCitation",
            "searching"
        ]),
        ...mapStores(useMainStore),
        processedTocElements() {
            return this.buildTocTree(this.tocElements);
        }
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
            if (scrollPosition < window.innerHeight && !this.isLoading && this.displayLimit < this.processedTocElements.length) {
                this.isLoading = true;
                setTimeout(() => {
                    this.displayLimit += 200;
                    this.isLoading = false;
                }, 100);
            }
        },
        buildTocTree(elements) {
            const tree = [];
            const typeHierarchy = {
                'doc': 1,
                'div1': 2,
                'div2': 3,
                'div3': 4,
                'para': 9
            };

            let stack = [];

            for (let element of elements) {
                const elementLevel = typeHierarchy[element.philo_type] || 1;
                element.level = elementLevel;
                element.children = [];

                // Find the appropriate parent in the stack
                while (stack.length > 0 && stack[stack.length - 1].level >= elementLevel) {
                    stack.pop();
                }

                if (stack.length === 0) {
                    // This is a top-level element
                    tree.push(element);
                } else {
                    // This is a child of the last element in the stack
                    stack[stack.length - 1].children.push(element);
                }

                stack.push(element);
            }

            return tree;
        },
    },
};
</script>
<style scoped lang="scss">
@use "../assets/styles/theme.module.scss" as theme;

#toc-content {
    padding: 1rem;
}

#toc-content:focus {
    outline: 2px solid theme.$link-color;
    outline-offset: -2px;
}

.toc-content-wrapper {
    display: inline-block;
    width: 100%;
    vertical-align: top;
}

.toc-tree {
    list-style: none;
    padding-left: 0;
    margin: 0;
}

.toc-item {
    margin-bottom: 0.25rem;
    border-radius: 4px;
    padding: 0.25rem 0;
    position: relative;
}

.toc-children {
    list-style: none;
    padding-left: 1.25rem;
    margin-bottom: 0;
    position: relative;
}

.toc-child {
    position: relative;
    padding-left: 0.75rem;
}

/* Vertical connecting lines */
.toc-children::before {
    content: '';
    position: absolute;
    left: 0.75rem;
    top: 0;
    bottom: 0;
    width: 1px;
    border-left: 1px dotted theme.$link-color;
    opacity: 0.4;
}

/* Horizontal connecting lines */
.toc-child::before {
    content: '';
    position: absolute;
    left: 0;
    top: 1.25rem;
    width: 0.75rem;
    height: 1px;
    border-top: 1px dotted theme.$link-color;
    opacity: 0.4;
}

/* Clean up vertical line for last child */
.toc-child:last-child::after {
    content: '';
    position: absolute;
    left: -0.25rem;
    top: 50%;
    bottom: -0.5rem;
    width: 1px;
    background-color: white;
}

.toc-section {
    font-size: 1.05rem;
    text-decoration: none;
    transition: all 0.15s ease-in-out;
    display: inline-block;
    padding: 0.25rem 0.25rem 0.25rem 0.25rem;
    border-radius: 4px;
}

.toc-section:hover,
.toc-section:focus {
    outline: 1px solid theme.$link-color;
}

.div1-marker {
    color: theme.$link-color;
    margin-right: 0.5rem;
    font-size: 1.2rem;
    margin-left: -0.5rem;
}

.toc-div1 .toc-section {
    font-size: 1.1rem;
}

.separator {
    padding: 5px;
    font-size: 60%;
    display: inline-block;
    vertical-align: middle;
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

:deep(h5 .text-view) {
    font-size: inherit !important;
}
</style>
