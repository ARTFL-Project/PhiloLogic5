<template>
    <nav id="page-links" class="mt-4 pb-4 text-center" v-if="pages.length > 0"
        :aria-label="$t('pages.searchResultsPages')">
        <div class="row">
            <div class="col-12 col-sm-10 offset-sm-1 col-md-8 offset-md-2">
                <div class="btn-group shadow" role="group">
                    <button type="button" class="btn btn-outline-secondary" v-for="page in pages" :key="page.display"
                        :class="page.active" @click="goToPage(page.start, page.end)"
                        :aria-label="$t('pages.pageAriaLabel', { page: page.display, range: page.range })"
                        :aria-current="page.active === 'active' ? 'page' : null">
                        <span class="page-number">{{ page.display }}</span>
                        <span class="page-range">{{ page.range }}</span>
                    </button>
                </div>
            </div>
        </div>
    </nav>
</template>
<script>
import { mapFields } from "vuex-map-fields";

export default {
    name: "pages-component",
    computed: {
        ...mapFields(["formData.results_per_page", "resultsLength", "totalResultsDone", "urlUpdate"]),
    },
    data() {
        return { pages: [] };
    },
    watch: {
        totalResultsDone(done) {
            if (done) {
                this.buildPages();
            }
        },
        urlUpdate() {
            this.buildPages();
        },
    },
    methods: {
        buildPages() {
            let start = parseInt(this.$route.query.start);
            let resultsPerPage = parseInt(this.results_per_page) || 25;
            let resultsLength = this.resultsLength;

            // first find out what page we are on currently.
            let currentPage = Math.floor(start / resultsPerPage) + 1 || 1;

            // then how many total pages the query has
            let totalPages = Math.floor(resultsLength / resultsPerPage);
            let remainder = resultsLength % resultsPerPage;
            if (remainder !== 0) {
                totalPages += 1;
            }
            totalPages = totalPages || 1;

            // construct the list of page numbers we will output.
            let pages = [];
            // up to four previous pages
            let prev = currentPage - 4;
            while (prev < currentPage) {
                if (prev > 0) {
                    pages.push(prev);
                }
                prev += 1;
            }
            // the current page
            pages.push(currentPage);
            // up to five following pages
            let next = currentPage + 5;
            while (next > currentPage) {
                if (next < totalPages) {
                    pages.push(next);
                }
                next -= 1;
            }
            // first and last if not already there
            if (pages[0] !== 1) {
                pages.unshift(1);
            }
            if (pages[-1] !== totalPages) {
                pages.push(totalPages);
            }
            pages.sort(function (a, b) {
                return a - b;
            });

            // now we construct the actual links from the page numbers
            let pageObject = [];
            let lastPageName = "";
            let pageEnd, pageStart, active;
            for (let page of pages) {
                pageStart = page * resultsPerPage - resultsPerPage + 1;
                pageEnd = page * resultsPerPage;
                if (page === currentPage) {
                    active = "active";
                } else {
                    active = "";
                }
                pageStart = resultsPerPage * (page - 1) + 1;
                pageEnd = pageStart + resultsPerPage - 1;
                if (pageEnd > resultsLength) {
                    pageEnd = resultsLength;
                }
                if (page === 1 && pages.length > 1) {
                    page = "First";
                }
                if (page === totalPages) {
                    page = "Last";
                }
                if (page == lastPageName) {
                    continue;
                }
                lastPageName = page;
                pageObject.push({
                    display: page,
                    active: active,
                    start: pageStart.toString(),
                    end: pageEnd.toString(),
                    range: `${pageStart}-${pageEnd}`,
                });
            }
            this.pages = pageObject;
        },
        goToPage(start, end) {
            let route = this.paramsToRoute({
                ...this.$store.state.formData,
                start: start,
                end: end,
            });
            return this.$router.push(route);
        },
    },
};
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

.page-number {
    display: block;
    font-size: 110%;
    font-weight: 500;
}

.page-range {
    font-size: 80%;
    opacity: 0.85;
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
