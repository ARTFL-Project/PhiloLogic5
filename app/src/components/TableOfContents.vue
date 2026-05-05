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
                            <div id="toc-content" v-scroll="handleScroll" ref="tocContent" tabindex="0">
                                <ul class="toc-tree">
                                    <li v-for="(element, elIndex) in processedTocElements.slice(0, displayLimit)"
                                        :key="elIndex" :class="'toc-item toc-' + element.philo_type">

                                        <!-- Section content -->
                                        <div class="toc-content-wrapper">
                                            <span v-if="element.philo_type === 'div1'" class="div1-marker"
                                                aria-hidden="true">※</span>
                                            <span v-else :class="'bullet-point-' + element.philo_type"
                                                aria-hidden="true"></span>
                                            <router-link :to="element.href" class="toc-section">
                                                {{ element.label }}
                                            </router-link>
                                        </div>

                                        <!-- Child sections -->
                                        <ul v-if="element.children && element.children.length > 0" class="toc-children">
                                            <li v-for="(child, childIndex) in element.children" :key="childIndex"
                                                :class="'toc-item toc-child toc-' + child.philo_type">
                                                <div class="toc-content-wrapper">
                                                    <span :class="'bullet-point-' + child.philo_type"
                                                        aria-hidden="true"></span>
                                                    <router-link :to="child.href" class="toc-section">
                                                        {{ child.label }}
                                                    </router-link>
                                                </div>

                                                <!-- Grandchildren (div3 level) -->
                                                <ul v-if="child.children && child.children.length > 0"
                                                    class="toc-children">
                                                    <li v-for="(grandchild, grandchildIndex) in child.children"
                                                        :key="grandchildIndex"
                                                        :class="'toc-item toc-child toc-' + grandchild.philo_type">
                                                        <div class="toc-content-wrapper">
                                                            <span :class="'bullet-point-' + grandchild.philo_type"
                                                                aria-hidden="true"></span>
                                                            <router-link :to="grandchild.href" class="toc-section">
                                                                {{ grandchild.label }}
                                                            </router-link>
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
<script setup>
import { computed, inject, ref, useTemplateRef } from "vue";
import { useRoute } from "vue-router";
import { storeToRefs } from "pinia";
import { useI18n } from "vue-i18n";
import { useMainStore } from "../stores/main";
import { buildTocTree, debug } from "../utils.js";
import Citations from "./Citations";
import ProgressSpinner from "./ProgressSpinner";

const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const philoConfig = inject("$philoConfig");
const route = useRoute();
const { t } = useI18n();
const store = useMainStore();
const { textNavigationCitation, searching } = storeToRefs(store);

const tocContent = useTemplateRef("tocContent");
const displayLimit = ref(200);
const teiHeader = ref("");
const tocObject = ref({});
const tocElements = ref([]);
const showHeader = ref(false);
const headerButton = ref(t("toc.showHeader"));
const isLoading = ref(false);

const processedTocElements = computed(() => buildTocTree(tocElements.value));

function fetchToC() {
    searching.value = true;
    $http
        .get(`${$dbUrl}/reports/table_of_contents.py`, {
            params: { philo_id: route.params.pathInfo },
        })
        .then((response) => {
            searching.value = false;
            tocObject.value = response.data;
            tocElements.value = response.data.toc;
            textNavigationCitation.value = response.data.citation;
        })
        .catch((error) => {
            searching.value = false;
            debug({ $options: { name: "tableOfContents" } }, error);
        });
}

function toggleHeader() {
    if (showHeader.value) {
        headerButton.value = t("toc.showHeader");
        showHeader.value = false;
        return;
    }
    if (teiHeader.value.length === 0) {
        $http
            .get(`${$dbUrl}/scripts/get_header.py`, {
                params: { philo_id: route.params.pathInfo },
            })
            .then((response) => {
                teiHeader.value = response.data;
                headerButton.value = t("toc.hideHeader");
                showHeader.value = true;
            })
            .catch((error) => {
                debug({ $options: { name: "tableOfContents" } }, error);
            });
    } else {
        headerButton.value = t("toc.hideHeader");
        showHeader.value = true;
    }
}

function handleScroll() {
    if (!tocContent.value) return;
    const scrollPosition = tocContent.value.getBoundingClientRect().bottom - 200;
    if (
        scrollPosition < window.innerHeight &&
        !isLoading.value &&
        displayLimit.value < processedTocElements.value.length
    ) {
        isLoading.value = true;
        setTimeout(() => {
            displayLimit.value += 200;
            isLoading.value = false;
        }, 100);
    }
}

fetchToC();
</script>
<style scoped lang="scss">
@use "../assets/styles/theme.module.scss" as theme;

#toc-content {
    padding: 0 2rem;
}

#toc-content:focus {
    outline: 2px solid theme.$link-color;
    outline-offset: -2px;
}

/* Component-specific styles only - tree structure now in App.vue */

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
