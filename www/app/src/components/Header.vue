<template>
    <header role="banner">
        <nav class="navbar navbar-expand-lg navbar-light bg-light shadow" style="height: 53px"
            aria-label="Main navigation">
            <div class="collapse navbar-collapse top-links">
                <ul class="navbar-nav me-auto mb-2 mb-lg-0">
                    <li class="nav-item">
                        <a class="nav-link" :href="philoConfig.link_to_home_page"
                            :aria-label="`Go to ${philoConfig.dbname} home page`"
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
                </ul>
            </div>

            <button type="button" id="academic-citation-link" class="nav-link position-absolute" data-bs-toggle="modal"
                data-bs-target="#academic-citation" :aria-label="$t('header.citationHelpText')"
                v-if="philoConfig.academic_citation.collection.length > 0">
                {{ $t("header.citeUs") }}
            </button>

            <router-link class="navbar-brand" to="/" :aria-label="`Go to ${philoConfig.dbname} homepage`"
                v-html="philoConfig.dbname">
            </router-link>

            <ul class="navbar-nav ml-auto top-links">
                <li class="nav-item">
                    <a class="nav-link" href="https://www.uchicago.edu"
                        aria-label="Visit University of Chicago website">
                        University of Chicago
                    </a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="https://atilf.fr" aria-label="Visit ATILF-CNRS website">
                        ATILF-CNRS
                    </a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="https://artfl-project.uchicago.edu/content/contact-us"
                        title="Contact information for the ARTFL Project" aria-label="Contact ARTFL Project">
                        {{ $t("header.contactUs") }}
                    </a>
                </li>
                <li class="nav-item">
                    <locale-changer />
                </li>
            </ul>

            <a id="report-error-link" class="nav-link position-absolute" :href="philoConfig.report_error_link"
                target="_blank" rel="noopener noreferrer" aria-label="Report an error (opens in new tab)"
                v-if="philoConfig.report_error_link.length > 0">
                {{ $t("common.reportError") }}
            </a>

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
                            <a :href="docCitation.link" :aria-label="`Visit citation link: ${docCitation.link}`">
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
    },
};
</script>

<style lang="scss" scoped>
@import "../assets/styles/theme.module.scss";

.top-links {
    margin-left: -0.25rem;
    font-size: 80%;
    margin-top: -2rem;
    font-variant: small-caps;
}

#right-side-links {
    font-size: 80%;
    margin-top: -1rem;
    font-variant: small-caps;
}

.navbar-brand {
    font-weight: 700;
    font-size: 1.6rem !important;
    font-variant: small-caps;
    position: absolute;
    left: 50%;
    transform: translateX(-50%);
    line-height: 80%;
}

#report-error-link {
    right: 0.5rem;
    bottom: 0.25rem;
    font-variant: small-caps;
    font-weight: 700;
}

#academic-citation-link {
    background-color: inherit;
    left: 0.5rem;
    bottom: 0.25rem;
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