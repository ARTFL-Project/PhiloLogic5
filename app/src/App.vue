<template>
    <div v-if="!$philoConfig.valid_config" class="error" role="alert">
        <h1>{{ $t('common.invalidConfigTitle') }}</h1>
        <p>{{ $t('common.invalidConfigMessage') }}</p>
    </div>
    <Header v-if="$philoConfig.valid_config" />
    <main id="main" v-if="$philoConfig.valid_config">

        <!-- Access checking indicator -->
        <div v-if="checkingAccess" class="access-checking-container" role="status"
            :aria-label="$t('common.checkingAccess')">
            <div class="card shadow-sm">
                <div class="card-body text-center py-5">
                    <progress-spinner :lg="true" />
                    <h5 class="card-title mt-3">{{ $t('common.checkingAccess') }}</h5>
                    <p class="card-text text-muted">{{ $t('common.pleaseWait') }}</p>
                </div>
            </div>
        </div>

        <SearchForm v-if="accessAuthorized && !checkingAccess" />

        <!-- Main content landmark -->
        <nav id="main-content" tabindex="-1">
            <router-view v-if="accessAuthorized && !checkingAccess" />
        </nav>

        <access-control :client-ip="clientIp" :domain-name="domainName" v-if="!accessAuthorized && !checkingAccess"
            role="navigation" :aria-label="$t('common.accessControlLabel')" />
    </main>
    <!-- Footer -->
    <footer class="container-fluid" v-if="accessAuthorized">
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

<script setup>
import DOMPurify from "dompurify";
import { storeToRefs } from "pinia";
import { defineAsyncComponent, inject, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { useI18n } from "vue-i18n";
import Header from "./components/Header.vue";
import ProgressSpinner from "./components/ProgressSpinner.vue";
import { useMainStore } from "./stores/main";
import { copyObject, debug, paramsToRoute } from "./utils.js";
const SearchForm = defineAsyncComponent(() => import("./components/SearchForm.vue"));
const AccessControl = defineAsyncComponent(() => import("./components/AccessControl.vue"));

const $http = inject("$http");
const philoConfig = inject("$philoConfig");
const route = useRoute();
const router = useRouter();
const { locale: i18nLocale, t } = useI18n();
const store = useMainStore();
const { formData, urlUpdate, showFacets } = storeToRefs(store);

const clientIp = ref("");
const domainName = ref("");
const accessAuthorized = ref(true);
const checkingAccess = ref(false);

function toTitleCase(str) {
    return str.replace(/\w\S*/g, (txt) =>
        txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase()
    );
}

function updateDocumentTitle(routeName) {
    if (!philoConfig.valid_config) return;

    const titleCasedDbname = toTitleCase(DOMPurify.sanitize(philoConfig.dbname));
    const reportTypes = ["concordance", "kwic", "collocation", "aggregation", "time_series", "bibliography"];
    const pageTypes = ["textNavigation", "tableOfContents"];

    if (routeName && reportTypes.includes(routeName)) {
        const reportName = routeName === "bibliography"
            ? t("landingPage.bibliography")
            : t(`searchForm.${routeName}`);
        document.title = t("common.documentTitleWithReport", {
            dbname: titleCasedDbname,
            report: reportName,
        });
    } else if (routeName && pageTypes.includes(routeName)) {
        const pageName = t(`searchForm.${routeName}`);
        const baseTitle = t("common.documentTitle", { dbname: titleCasedDbname });
        document.title = `${baseTitle} - ${pageName}`;
    } else {
        document.title = `${t("common.documentTitle", { dbname: titleCasedDbname })} - ${t("landingPage.title")}`;
    }
}

function getBaseUrl() {
    let href = window.location.href;
    for (const segment of [
        "/landing", "/query", "/concordance", "/kwic", "/collocation",
        "/aggregation", "/table-of-contents", "/navigate", "/time_series", "/bibliography",
    ]) {
        href = href.replace(new RegExp(`${segment}.*`), "");
    }
    return href;
}

function evaluateRoute() {
    if (
        formData.value.q && formData.value.q.length > 0 &&
        route.name === "bibliography"
    ) {
        store.updateFormDataField({ key: "report", value: "concordance" });
        debug({ $options: { name: "app" } }, formData.value.report);
        router.push(paramsToRoute({ ...store.formData }));
    } else {
        formData.value.report = route.name;
    }
}

function formDataUpdate() {
    const localParams = copyObject(store.defaultFields);
    store.updateFormData({ ...localParams, ...route.query });
    if (!["textNavigation", "tableOfContents", "home"].includes(route.name)) {
        evaluateRoute();
        urlUpdate.value = route.query;
    }
}

function setupApp() {
    store.initFromConfig(philoConfig);
    formDataUpdate();
}

// ── Initial dispatch (replaces created()) ────────────────────────────────────
if (philoConfig.valid_config) {
    updateDocumentTitle(route.name);

    const currentLocale = i18nLocale.value || localStorage.getItem("lang") || "en";
    document.documentElement.setAttribute("lang", currentLocale);
    i18nLocale.value = currentLocale;

    accessAuthorized.value = !philoConfig.access_control;
    if (philoConfig.access_control) {
        checkingAccess.value = true;
        $http
            .get(`${getBaseUrl()}/scripts/access_request.py`, {
                headers: { "Access-Control-Allow-Origin": "*" },
            })
            .then((response) => {
                accessAuthorized.value = response.data.access;
                if (accessAuthorized.value) {
                    setupApp();
                } else {
                    clientIp.value = response.data.incoming_address;
                    domainName.value = response.data.domain_name;
                }
            })
            .finally(() => {
                checkingAccess.value = false;
            });
    } else {
        setupApp();
    }
}
if (philoConfig.facets.length < 1) {
    showFacets.value = false;
}

// ── Watchers ─────────────────────────────────────────────────────────────────
watch(
    () => route.fullPath,
    () => {
        formDataUpdate();
        updateDocumentTitle(route.name);
    }
);

watch(accessAuthorized, (authorized) => {
    if (authorized) setupApp();
});
</script>

<style lang="scss">
@use "../node_modules/bootstrap/scss/bootstrap.scss";
@use "./assets/styles/theme.module.scss" as theme;

a {
    text-decoration: none;
    transition: all 0.2s ease;
    border-radius: 2px;
}

/* Global link hover/focus effect */
a:hover,
a:focus {
    background-color: rgba(theme.$link-color, 0.08);
    box-shadow: 0 0 0 3px rgba(theme.$link-color, 0.1);
}

/* Special handling for button-style links */
.btn-link:hover,
.btn-link:focus {
    background-color: rgba(theme.$link-color, 0.08);
    box-shadow: 0 0 0 3px rgba(theme.$link-color, 0.1);
    border: 1px solid rgba(theme.$link-color, 0.2);
}

.dropdown-item:hover,
.dropdown-item:focus {
    transform: scale(1.02) !important;
    background-color: rgba(theme.$link-color, 0.15) !important;
    border: 1px solid rgba(theme.$link-color, 0.3) !important;
    box-shadow: 0 2px 8px rgba(theme.$link-color, 0.15) !important;
    z-index: 1 !important;
}

.dropdown-item:active {
    transform: scale(0.98) !important;
    background-color: rgba(theme.$link-color, 0.2) !important;
}

/* Ensure focus is always visible */
a:focus-visible {
    outline: 2px solid theme.$link-color;
    outline-offset: 2px;
}

.btn:focus-visible {
    outline: 2px solid theme.$link-color !important;
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
input,
.card,
.modal-content {
    font-size: 14px !important;
    color: #000;
}

input,
.form-select {
    border-color: #888 !important;
}


.custom-control {
    min-height: auto;
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

.access-checking-container {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 400px;
    padding: 2rem;
}

.access-checking-container .card {
    max-width: 500px;
    width: 100%;
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

.icon-x {
    display: inline-block;
    width: 1em;
    /* 1em will make the icon the same size as the surrounding text */
    height: 1em;

    -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='18' y1='6' x2='6' y2='18'%3E%3C/line%3E%3Cline x1='6' y1='6' x2='18' y2='18'%3E%3C/line%3E%3C/svg%3E");
    mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='18' y1='6' x2='6' y2='18'%3E%3C/line%3E%3Cline x1='6' y1='6' x2='18' y2='18'%3E%3C/line%3E%3C/svg%3E");
    -webkit-mask-size: cover;
    mask-size: cover;
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
    margin-top: 0.3rem;
    margin-bottom: 0;
    overflow: hidden;
}

.toc-child {
    position: relative;
    padding-left: 0.75rem;
    border-left: 1px dotted rgba(theme.$link-color, 0.4);
}

.toc-child:last-child {
    border-left-color: transparent;
}

.toc-content-wrapper {
    display: flex;
    align-items: baseline;
}

/* Horizontal branch — tied to content wrapper so it moves with text */
.toc-child > .toc-content-wrapper {
    position: relative;
}

.toc-child > .toc-content-wrapper::before {
    content: '';
    position: absolute;
    left: -0.75rem;
    top: calc(0.25rem + 0.85em);
    width: 0.75rem;
    height: 1px;
    border-top: 1px dotted rgba(theme.$link-color, 0.4);
}

/* Last child: partial vertical line from top down to the branch */
.toc-child:last-child > .toc-content-wrapper::after {
    content: '';
    position: absolute;
    left: -0.75rem;
    top: -100vh;
    height: calc(100vh + 0.25rem + 0.75em);
    width: 0px;
    border-left: 1px dotted rgba(theme.$link-color, 0.4);
}

/* Special markers */
.div1-marker {
    color: theme.$link-color;
    margin-right: 0.5rem;
    font-size: 1.1rem;
    margin-left: -1rem;
    font-weight: 700;
    flex-shrink: 0;
}

.toc-div1 .toc-section {
    font-size: 1.1rem;
}
</style>