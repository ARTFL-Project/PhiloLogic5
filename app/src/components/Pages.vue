<template>
    <nav id="page-links" class="mt-4 pb-4 text-center" v-if="pages.length > 0"
        :aria-label="$t('pages.searchResultsPages')">
        <div class="row">
            <div class="col-12 col-sm-10 offset-sm-1 col-md-8 offset-md-2">
                <div class="btn-group shadow" role="group">
                    <button type="button" class="btn btn-outline-secondary" v-for="page in pages" :key="page.display"
                        :class="page.active" @click="goToPage(page.start, page.end)"
                        :aria-current="page.active === 'active' ? 'page' : null">
                        <span class="page-number">{{ page.label ? $t(page.label) : page.display }}</span>
                        <span class="page-range">{{ page.range }}</span>
                    </button>
                </div>
            </div>
        </div>
    </nav>
</template>
<script setup>
import { ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { storeToRefs } from "pinia";
import { useMainStore } from "../stores/main";
import { paramsToRoute } from "../utils.js";

const route = useRoute();
const router = useRouter();
const store = useMainStore();
const { formData, resultsLength, totalResultsDone, urlUpdate } = storeToRefs(store);

const pages = ref([]);

function buildPages() {
    const start = parseInt(route.query.start);
    const resultsPerPage = parseInt(formData.value.results_per_page) || 25;
    const total = resultsLength.value;
    if (!total) {
        pages.value = []; // no "First 1-0" button for no results
        return;
    }

    // Current page from start offset
    const currentPage = Math.floor(start / resultsPerPage) + 1 || 1;

    // Total page count (rounded up)
    let totalPages = Math.floor(total / resultsPerPage);
    if (total % resultsPerPage !== 0) totalPages += 1;
    totalPages = totalPages || 1;

    // Build the page-number list: up to 4 previous, current, up to 5 following
    const pageNumbers = [];
    for (let prev = currentPage - 4; prev < currentPage; prev += 1) {
        if (prev > 0) pageNumbers.push(prev);
    }
    pageNumbers.push(currentPage);
    for (let next = currentPage + 5; next > currentPage; next -= 1) {
        if (next < totalPages) pageNumbers.push(next);
    }
    if (pageNumbers[0] !== 1) pageNumbers.unshift(1);
    if (pageNumbers[pageNumbers.length - 1] !== totalPages) pageNumbers.push(totalPages);
    pageNumbers.sort((a, b) => a - b);

    // Build display objects
    const pageObjects = [];
    for (const page of new Set(pageNumbers)) {
        const pageStart = resultsPerPage * (page - 1) + 1;
        let pageEnd = pageStart + resultsPerPage - 1;
        if (pageEnd > total) pageEnd = total;
        const active = page === currentPage ? "active" : "";

        let label = "";
        if (page === 1 && pageNumbers.length > 1) label = "pages.first";
        if (page === totalPages) label = "pages.last";

        pageObjects.push({
            display: page,
            label,
            active,
            start: pageStart.toString(),
            end: pageEnd.toString(),
            range: `${pageStart}-${pageEnd}`,
        });
    }
    pages.value = pageObjects;
}

function goToPage(start, end) {
    return router.push(paramsToRoute({ ...formData.value, start, end }));
}

watch(totalResultsDone, (done) => {
    if (done) buildPages();
});
watch(urlUpdate, buildPages);
</script>
<style scoped lang="scss">
@use "../assets/styles/theme.module.scss" as theme;

.page {
    transition: width 0.4s ease !important;
}

.btn {
    line-height: initial !important;
}

.btn-group .btn {
    white-space: nowrap;
    min-width: auto;
}

@media (max-width: 767px) {
    .btn-group {
        flex-wrap: wrap; /* on phones, rather than wider than the page */
    }
}

.page-number {
    display: block;
    font-size: 110%;
    font-weight: 500;
}

.page-range {
    font-size: 80%;
    color: theme.$link-color;
}

.btn-outline-secondary {
    color: theme.$link-color !important;
    border-color: theme.$link-color !important;
    background-color: #fff !important;
}

.btn-outline-secondary:hover,
.btn-outline-secondary:focus {
    color: #fff !important;
    background-color: theme.$button-color !important;
    border-color: theme.$button-color !important;
}

.btn-outline-secondary:hover .page-range,
.btn-outline-secondary:focus .page-range {
    color: rgba(255, 255, 255, 0.9) !important;
    opacity: 1;
}

.btn-outline-secondary.active {
    color: #fff !important;
    background-color: theme.$button-color-active !important;
    border-color: theme.$button-color !important;
}

.btn-outline-secondary.active .page-range {
    color: rgba(255, 255, 255, 0.9) !important;
    opacity: 1;
}

.btn-outline-secondary:focus {
    box-shadow: 0 0 0 0.2rem rgba(theme.$button-color, 0.25) !important;
    outline: 2px solid theme.$button-color;
    outline-offset: 2px;
}

@media (prefers-contrast: high) {
    .page-range {
        opacity: 1;
        font-weight: 500;
    }

    .btn-outline-secondary {
        border-width: 2px;
    }
}
</style>
