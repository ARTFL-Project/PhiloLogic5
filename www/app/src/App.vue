<template>
    <div v-if="!$philoConfig.valid_config" class="error" role="alert">
        <h1>{{ $t('common.invalidConfigTitle') }}</h1>
        <p>{{ $t('common.invalidConfigMessage') }}</p>
    </div>
    <Header v-if="$philoConfig.valid_config" />
    <main role="main" id="app" v-if="$philoConfig.valid_config">
        <!-- Skip navigation for keyboard users -->
        <a href="#main-content" class="visually-hidden-focusable">
            {{ $t('common.skipToMain') }}
        </a>

        <SearchForm v-if="accessAuthorized" />

        <!-- Main content landmark -->
        <nav id="main-content" role="navigation" tabindex="-1">
            <router-view v-if="accessAuthorized" />
        </nav>

        <access-control :client-ip="clientIp" :domain-name="domainName" v-if="!accessAuthorized" role="main"
            :aria-label="$t('common.accessControlLabel')" />
    </main>
    <!-- Footer -->
    <footer class="container-fluid" v-if="accessAuthorized" role="contentinfo">
        <div class="text-center mb-4">
            <hr class="mb-3" style="width: 20%; margin: auto" />
            <p>{{ $t('common.poweredBy') }}</p>
            <a href="https://artfl-project.uchicago.edu/" :title="$t('common.philoTitle')"
                :aria-label="$t('common.philoLinkLabel')">
                <img src="./assets/philo.png" :alt="$t('common.philoAlt')" height="40" width="110" />
            </a>
        </div>
    </footer>
</template>

<script>
import DOMPurify from "dompurify";
import { defineAsyncComponent } from "vue";
import { mapFields } from "vuex-map-fields";
import Header from "./components/Header.vue";
const SearchForm = defineAsyncComponent(() => import("./components/SearchForm.vue"));
const AccessControl = defineAsyncComponent(() => import("./components/AccessControl.vue"));

export default {
    name: "app",
    components: {
        Header,
        SearchForm,
        AccessControl,
    },
    inject: ["$http"],
    data() {
        return {
            initialLoad: true,
            clientIp: "",
            domainName: "",
            accessAuthorized: true,
        };
    },
    computed: {
        ...mapFields(["formData.report", "formData.q", "urlUpdate", "showFacets"]),
        defaultFieldValues() {
            let localFields = {
                report: "home",
                q: "",
                method: "proxy",
                cooc_order: "yes",
                method_arg: "",
                arg_phrase: "",
                results_per_page: 25,
                start: "",
                end: "",
                colloc_filter_choice: "",
                colloc_within: "sent",
                filter_frequency: 100,
                approximate: "no",
                approximate_ratio: 100,
                start_date: "",
                end_date: "",
                year_interval: this.$philoConfig.time_series_interval,
                sort_by: "rowid",
                first_kwic_sorting_option: "",
                second_kwic_sorting_option: "",
                third_kwic_sorting_option: "",
                start_byte: "",
                end_byte: "",
                group_by: this.$philoConfig.aggregation_config[0].field,
            };
            for (let field of this.$philoConfig.metadata) {
                localFields[field] = "";
            }
            return localFields;
        },
        reportValues() {
            let reportValues = {};
            let commonFields = ["q", "approximate", "approximate_ratio", ...this.$philoConfig.metadata];
            reportValues.concordance = new Set([
                ...commonFields,
                "method",
                "cooc_order",
                "method_arg",
                "results_per_page",
                "sort_by",
                "hit_num",
                "start",
                "end",
                "frequency_field",
                "word_property"
            ]);
            reportValues.kwic = new Set([
                ...commonFields,
                "method",
                "cooc_order",
                "method_arg",
                "results_per_page",
                "first_kwic_sorting_option",
                "second_kwic_sorting_option",
                "third_kwic_sorting_option",
                "start",
                "end",
                "frequency_field",
                "word_property"
            ]);
            reportValues.collocation = new Set([...commonFields, "start", "colloc_filter_choice", "filter_frequency", "colloc_within", "method_arg", "q_attribute", "q_attribute_value"]);
            reportValues.time_series = new Set([
                ...commonFields,
                "method",
                "cooc_order",
                "method_arg",
                "start_date",
                "end_date",
                "year_interval",
                "max_time",
            ]);
            reportValues.aggregation = new Set([...commonFields, "method", "cooc_order", "method_arg", "group_by"]);
            return reportValues;
        },
        pageTitle() {
            // Generate appropriate h1 text based on current route/report
            const routeName = this.$route.name;
            const searchQuery = this.q;
            const dbName = this.$philoConfig.dbname;

            switch (routeName) {
                case 'home':
                    return `${dbName} - ${this.$t('common.searchInterface')}`;
                case 'concordance':
                    return searchQuery ?
                        `${this.$t('concordance.pageTitle')}: "${searchQuery}" - ${dbName}` :
                        `${this.$t('concordance.pageTitle')} - ${dbName}`;
                case 'kwic':
                    return searchQuery ?
                        `${this.$t('kwic.pageTitle')}: "${searchQuery}" - ${dbName}` :
                        `${this.$t('kwic.pageTitle')} - ${dbName}`;
                case 'bibliography':
                    return searchQuery ?
                        `${this.$t('bibliography.pageTitle')}: "${searchQuery}" - ${dbName}` :
                        `${this.$t('bibliography.pageTitle')} - ${dbName}`;
                case 'collocation':
                    return searchQuery ?
                        `${this.$t('collocation.pageTitle')}: "${searchQuery}" - ${dbName}` :
                        `${this.$t('collocation.pageTitle')} - ${dbName}`;
                case 'time_series':
                    return searchQuery ?
                        `${this.$t('timeSeries.pageTitle')}: "${searchQuery}" - ${dbName}` :
                        `${this.$t('timeSeries.pageTitle')} - ${dbName}`;
                case 'aggregation':
                    return searchQuery ?
                        `${this.$t('aggregation.pageTitle')}: "${searchQuery}" - ${dbName}` :
                        `${this.$t('aggregation.pageTitle')} - ${dbName}`;
                case 'navigate':
                    return `${this.$t('navigation.pageTitle')} - ${dbName}`;
                case 'table-of-contents':
                    return `${this.$t('tableOfContents.pageTitle')} - ${dbName}`;
                default:
                    return `${dbName} - ${this.$t('common.searchInterface')}`;
            }
        },
    },
    created() {
        if (this.$philoConfig.valid_config) {
            document.title = DOMPurify.sanitize(this.$philoConfig.dbname);
            const html = document.documentElement;

            // Fix: Use dynamic locale instead of hardcoded 'sv'
            const currentLocale = this.$i18n.locale || localStorage.getItem("lang") || "en";
            html.setAttribute("lang", currentLocale);
            this.$i18n.locale = currentLocale;

            this.accessAuthorized = this.$philoConfig.access_control ? false : true;
            let baseUrl = this.getBaseUrl();
            if (this.$philoConfig.access_control) {
                this.$http
                    .get(`${baseUrl}/scripts/access_request.py`, {
                        headers: {
                            "Access-Control-Allow-Origin": "*",
                        },
                    })
                    .then((response) => {
                        this.accessAuthorized = response.data.access;
                        if (this.accessAuthorized) {
                            this.setupApp();
                        } else {
                            this.clientIp = response.data.incoming_address;
                            this.domainName = response.data.domain_name;
                        }
                    });
            } else {
                this.setupApp();
            }
        }
        if (this.$philoConfig.facets.length < 1) {
            this.showFacets = false;
        }
    },
    watch: {
        // call again the method if the route changes
        $route: "formDataUpdate",
        accessAuthorized(authorized) {
            if (authorized) {
                this.setupApp();
            }
        },
    },
    methods: {
        getBaseUrl() {
            let href = window.location.href;
            href = href.replace(/\/concordance.*/, "");
            href = href.replace(/\/kwic.*/, "");
            href = href.replace(/\/collocation.*/, "");
            href = href.replace(/\/aggregation.*/, "");
            href = href.replace(/\/table-of-contents.*/, "");
            href = href.replace(/\/navigate.*/, "");
            href = href.replace(/\/time_series.*/, "");
            href = href.replace(/\/bibliography.*/, "");
            return href;
        },

        setupApp() {
            this.$store.commit("setDefaultFields", this.defaultFieldValues);
            this.$store.commit("setReportValues", this.reportValues);
            this.formDataUpdate();
        },
        formDataUpdate() {
            let localParams = this.copyObject(this.defaultFieldValues);
            this.$store.commit("updateFormData", {
                ...localParams,
                ...this.$route.query,
            });
            if (!["textNavigation", "tableOfContents", "home"].includes(this.$route.name)) {
                this.evaluateRoute();
                this.urlUpdate = this.$route.fullPath;
            }
        },
        evaluateRoute() {
            if (this.$route.name == "bibliography") {
                this.report = "bibliography";
            }
            if (
                !["home", "textNavigation", "tableOfContents"].includes(this.$route.name) &&
                this.q.length > 0 &&
                this.$route.name == "bibliography"
            ) {
                this.$store.commit("updateFormDataField", {
                    key: "report",
                    value: "concordance",
                });
                this.debug(this, this.report);
                this.$router.push(this.paramsToRoute({ ...this.$store.state.formData }));
            } else {
                this.report = this.$route.name;
            }
        },
    },
};
</script>

<style lang="scss">
@import "./assets/styles/theme.module.scss";
@import "../node_modules/bootstrap/scss/bootstrap.scss";

a {
    text-decoration: none;
    transition: all 0.2s ease;
    border-radius: 2px;
}

/* Global link hover/focus effect */
a:hover,
a:focus {
    background-color: rgba($link-color, 0.08);
    box-shadow: 0 0 0 3px rgba($link-color, 0.1);
}

/* Special handling for button-style links */
.btn-link:hover,
.btn-link:focus {
    background-color: rgba($link-color, 0.08);
    box-shadow: 0 0 0 3px rgba($link-color, 0.1);
    border: 1px solid rgba($link-color, 0.2);
}

.dropdown-item:hover,
.dropdown-item:focus {
    transform: scale(1.02) !important;
    background-color: rgba($link-color, 0.15) !important;
    border: 1px solid rgba($link-color, 0.3) !important;
    box-shadow: 0 2px 8px rgba($link-color, 0.15) !important;
    z-index: 1 !important;
}

.dropdown-item:active {
    transform: scale(0.98) !important;
    background-color: rgba($link-color, 0.2) !important;
}

/* Ensure focus is always visible */
a:focus-visible {
    outline: 2px solid var(--bs-primary);
    outline-offset: 2px;
}

.btn:focus-visible {
    outline: 2px solid var(--bs-primary) !important;
    outline-offset: 2px !important;
}

.modal-backdrop {
    opacity: 0.7;
}

.passage-highlight {
    display: inline-block;
}

li {
    list-style-type: none;
}

body,
.btn,
select,
.custom-control-label,
.custom-control,
.input-group-text,
input {
    font-size: 14px !important;
}


.custom-control {
    min-height: auto;
}

// TOC styles with better accessibility
.toc-div1>a,
.toc-div2>a,
.toc-div3>a {
    padding: 5px 5px 5px 0px;
}

.toc-div1:focus,
.toc-div2:focus,
.toc-div3:focus {
    outline: 2px solid var(--bs-primary) !important;
    outline-offset: 2px !important;
}

.bullet-point-div1,
.bullet-point-div2,
.bullet-point-div3 {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 5px;
}

.bullet-point-div1 {
    background: #000;
}

.bullet-point-div2 {
    border: solid 2px;
}

.bullet-point-div3 {
    border: solid 1px;
}

.toc-div1,
.toc-div2,
.toc-div3 {
    text-indent: -0.9em;
    margin-bottom: 5px;
}

.toc-div1 {
    padding-left: 0.9em;
}

.toc-div2 {
    padding-left: 1.9em;
}

.toc-div3 {
    padding-left: 2.9em;
}

.toc-div1:hover,
.toc-div2:hover,
.toc-div3:hover {
    cursor: pointer;
}

br {
    content: " ";
    display: block;
}

.error {
    background-color: #f8d7da;
    color: #721c24;
    padding: 1rem;
    border: 1px solid #f5c6cb;
    border-radius: 0.25rem;
    margin: 1rem;
}

/*Text formatting*/
span.note {
    display: inline;
}

.xml-l {
    display: block;
}

.xml-l::before {
    content: "";
    font-family: "Droid Sans Mono", sans-serif;
    font-size: 0.7em;
    white-space: pre;
    width: 35px;
    display: inline-block;
}

.xml-l[id]::before {
    content: attr(id);
    color: #666666;
}

.xml-l[n]::before {
    content: attr(n);
    color: #666666;
}

.xml-l[type="split"]::before {
    content: "";
}

.xml-milestone::before {
    content: attr(n) "\00a0";
    color: #555555;
    font-family: "Droid Sans Mono", sans-serif;
    font-size: 0.6em;
    vertical-align: 0.3em;
}

.xml-milestone[unit="card"]::before {
    content: "";
}

.xml-lb[type="hyphenInWord"] {
    display: inline;
}

.xml-gap .xml-desc {
    display: none;
}

.xml-gap::after {
    content: "*";
}

.xml-w::before {
    margin: 0;
    padding: 0;
    float: left;
}

.xml-speaker {
    display: block;
}

/*Remove spacing before punctuation*/
.xml-w[pos=","],
.xml-w[pos="."],
.xml-w[pos=";"],
.xml-w[pos="?"],
.xml-w[pos="!"],
.xml-w[pos=":"],
.xml-w[pos="!"] {
    margin-left: -0.25em;
}
</style>