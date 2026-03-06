<template>
    <header>
        <nav class="navbar navbar-expand-lg navbar-light bg-light shadow px-1"
            aria-label="Main navigation">
            <!-- Row 1: navigation links + brand -->
            <div class="header-main">
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse"
                    data-bs-target="#navbarContent" aria-controls="navbarContent" aria-expanded="false"
                    :aria-label="$t('header.toggleNavigation')">
                    <span class="navbar-toggler-icon"></span>
                </button>

                <div class="collapse navbar-collapse top-links" id="navbarContent">
                    <ul class="navbar-nav me-auto mb-lg-0">
                        <li class="nav-item">
                            <!-- Skip navigation for keyboard users -->
                            <a href="#main-content" class="visually-hidden-focusable"
                                @click.prevent="skipToMainContent">
                                {{ $t('common.skipToMain') }}
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" :href="philoConfig.link_to_home_page"
                                :aria-label="$t('header.goHome')"
                                v-if="philoConfig.link_to_home_page != ''">
                                {{ $t("header.goHome") }}
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="https://artfl-project.uchicago.edu"
                                aria-label="Visit ARTFL Project website">
                                ARTFL Project
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="https://textual-optics-lab.uchicago.edu"
                                aria-label="Visit Textual Optics Lab website">
                                Textual Optics Lab
                            </a>
                        </li>
                        <!-- Right side items - only shown in hamburger menu -->
                        <li class="nav-item d-lg-none">
                            <a class="nav-link" href="https://www.uchicago.edu"
                                aria-label="Visit University of Chicago website">
                                University of Chicago
                            </a>
                        </li>
                        <li class="nav-item d-lg-none">
                            <a class="nav-link" href="https://www.atilf.fr/"
                                aria-label="Visit ATILF-CNRS website">
                                ATILF-CNRS
                            </a>
                        </li>
                        <li class="nav-item d-lg-none">
                            <a class="nav-link"
                                href="https://artfl-project.uchicago.edu/content/contact-us"
                                title="Contact information for the ARTFL Project"
                                aria-label="Contact us">
                                {{ $t("header.contactUs") }}
                            </a>
                        </li>
                        <li class="nav-item d-lg-none">
                            <locale-changer />
                        </li>
                    </ul>
                </div>

                <router-link class="navbar-brand" to="/" :aria-label="stripHtmlTags(philoConfig.dbname)"
                    v-html="philoConfig.dbname">
                </router-link>

                <!-- Right side items - only shown on desktop -->
                <ul class="navbar-nav ml-auto top-links d-none d-lg-flex">
                    <li class="nav-item">
                        <a class="nav-link" href="https://www.uchicago.edu"
                            aria-label="Visit University of Chicago website">
                            University of Chicago
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="https://www.atilf.fr/"
                            aria-label="Visit ATILF-CNRS website">
                            ATILF-CNRS
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link"
                            href="https://artfl-project.uchicago.edu/content/contact-us"
                            title="Contact information for the ARTFL Project" aria-label="Contact us">
                            {{ $t("header.contactUs") }}
                        </a>
                    </li>
                    <li class="nav-item">
                        <locale-changer />
                    </li>
                </ul>
            </div>

            <!-- Row 2: action links -->
            <div class="header-actions">
                <button type="button" id="academic-citation-link" class="nav-link" data-bs-toggle="modal"
                    data-bs-target="#academic-citation" :aria-label="$t('header.citeUs')"
                    v-if="philoConfig.academic_citation.collection.length > 0">
                    {{ $t("header.citeUs") }}
                </button>
                <span v-else></span>

                <a id="report-error-link" class="nav-link" :href="philoConfig.report_error_link"
                    target="_blank" rel="noopener noreferrer"
                    :aria-label="$t('common.reportError') + ' (opens in new tab)'"
                    v-if="philoConfig.report_error_link.length > 0">
                    {{ $t("common.reportError") }}
                </a>
            </div>

            <!-- Modal -->
            <div class="modal fade" id="academic-citation" tabindex="-1" aria-labelledby="modal-title"
                aria-describedby="modal-body" aria-hidden="true" role="dialog" aria-modal="true">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <div class="modal-title" id="modal-title">
                                {{ $t('header.citationModalTitle', { dbname: philoConfig.dbname }) }}
                            </div>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"
                                :aria-label="$t('header.closeCitationDialog')" @click="$event.target.blur()">
                            </button>
                        </div>
                        <div class="modal-body" id="modal-body">
                            <span v-if="docCitation.citation.length > 0">
                                <citations :citation="docCitation.citation" separator=",&nbsp;" />,
                            </span>
                            <span v-html="philoConfig.academic_citation.collection"></span>:
                            <a :href="docCitation.link" :aria-label="docCitation.link">
                                {{ docCitation.link }}.&nbsp;
                            </a>
                            <span>Accessed on {{ date }}</span>
                        </div>
                    </div>
                </div>
            </div>
        </nav>
    </header>
</template>

<script>
import citations from "./Citations";
import LocaleChanger from "./LocaleChanger.vue";
export default {
    name: "Header-component",
    components: { citations, LocaleChanger },
    inject: ["$http"],
    data() {
        return {
            philoConfig: this.$philoConfig,
            date: this.getDate(),
            docCitation: { citation: [], link: this.$philoConfig.academic_citation.link },
        };
    },
    created() {
        this.getDocCitation();
    },
    watch: {
        $route() {
            this.getDocCitation();
        },
    },
    methods: {
        getDate() {
            let today = new Date();
            let months = [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ];
            let month = months[today.getMonth()];
            return `${today.getDate()} ${month}, ${today.getFullYear()}`;
        },
        getDocCitation() {
            let textObjectURL = this.$route.params;
            if ("pathInfo" in textObjectURL) {
                let philoID = textObjectURL.pathInfo.split("/").join(" ");
                this.$http
                    .get(`${this.$dbUrl}/scripts/get_academic_citation.py?philo_id=${philoID}`)
                    .then((response) => {
                        this.docCitation.citation = response.data.citation;
                        this.docCitation.link = response.data.link;
                    });
            } else {
                this.docCitation.citation = [];
            }
        },
        stripHtmlTags(html) {
            if (!html) return '';
            // Create a temporary element to extract text content
            const div = document.createElement('div');
            div.innerHTML = html;
            return div.textContent || div.innerText || '';
        },
        skipToMainContent() {
            // Find the main content element and scroll to it
            const mainContent = document.getElementById('main-content');
            if (mainContent) {
                mainContent.scrollIntoView({ behavior: 'smooth' });
                mainContent.focus({ preventScroll: true });
            }
        },
    },
};
</script>

<style lang="scss" scoped>
@use "../assets/styles/theme.module.scss" as theme;

// Override Bootstrap navbar-expand-lg nowrap so our two rows stack
nav.navbar {
    flex-wrap: wrap !important;
}

// Row 1: nav links + brand — grid ensures true centering
.header-main {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    align-items: center;
    width: 100%;
}

// Row 2: Cite Us (left) + Report Error (right)
.header-actions {
    display: flex;
    justify-content: space-between;
    width: 100%;
}

.top-links {
    margin-left: -0.25rem;
    margin-top: -2rem;
    font-variant: small-caps;
    white-space: nowrap;

    // Right group: align to end of its grid cell
    &.ml-auto {
        justify-self: end;
    }
}

// Proper spacing for hamburger menu items
@media (max-width: 991.98px) {
    .top-links {
        margin-top: 0;
        margin-left: 0;
    }

    .top-links .nav-item {
        margin-bottom: 0.25rem;
    }

    // Disable transform scale in hamburger menu
    nav.navbar a:hover,
    nav.navbar a:focus {
        transform: none !important;
    }

    .header-main {
        display: flex;
        flex-wrap: wrap;
    }

    .navbar-brand {
        order: 1;
        flex-grow: 1;
        text-align: center;
    }

    .navbar-toggler {
        order: 0;
    }
}

nav.navbar a:hover,
nav.navbar a:focus,
#academic-citation-link:focus,
#academic-citation-link:hover {
    transform: scale(1.1);
    border-radius: 0.25rem;
    transition: all 0.2s ease-in-out;
    text-decoration: underline dotted !important;
}

nav.navbar a,
#academic-citation-link {
    transition: all 0.2s ease-in-out;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
}

.navbar-brand {
    font-weight: 700;
    font-size: clamp(1.15rem, 2.5vw, 1.6rem) !important;
    font-variant: small-caps;
    text-align: center;
    flex: 0 1 auto;
    min-width: 0;
    white-space: normal;
    overflow-wrap: break-word;
    word-break: break-word;
    line-height: 1.2;
    margin: 0;
}

#report-error-link {
    font-variant: small-caps;
    font-weight: 700;
}

#academic-citation-link {
    background-color: inherit;
    font-variant: small-caps;
    font-weight: 700;
    border-width: 0;
}

.modal-dialog {
    max-width: fit-content;
}

.modal-title {
    font-weight: 700;
    font-size: 1.2rem;
    font-variant: small-caps;
}
</style>