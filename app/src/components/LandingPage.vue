<template>
    <div id="landing-page-container" class="mt-4">
        <div class="container-fluid">
            <div id="landing-page-logo" :class="{ dictionary: dictionary }" v-if="logo">
                <img style="max-height: 300px; width: auto" :src="logo"
                    :alt="$t('landingPage.logoAlt', { dbname: $philoConfig.dbname })" />
            </div>
            <div class="text-center mb-4">
                <h1 class="landing-page-title">{{ $t('landingPage.title') }}</h1>
            </div>
            <div class="d-flex justify-content-center position-relative">
                <div class="spinner-border text-secondary" role="status" v-if="loading"
                    style="width: 4rem; height: 4rem; position: absolute; z-index: 50; top: 10px" aria-hidden="true">
                </div>
                <!-- Persistent live region for screen readers -->
                <div aria-live="polite" aria-atomic="true" class="visually-hidden">
                    <span v-if="loading">{{ $t('common.loading') }}</span>
                </div>
            </div>
            <div id="default-landing-page" class="row justify-content-center" v-if="landingPageBrowsing === 'default'">
                <section class="col-12 col-sm-6 col-md-8 mb-4" v-for="browseType in defaultLandingPageBrowsing"
                    :key="browseType.label"
                    :aria-labelledby="`browse-${browseType.label.replace(/\s+/g, '-').toLowerCase()}`">
                    <div class="card shadow-sm">
                        <div class="card-header">
                            <h2 class="h6 mb-0" :id="`browse-${browseType.label.replace(/\s+/g, '-').toLowerCase()}`">
                                {{ browseType.label }}
                            </h2>
                        </div>
                        <div class="row g-0" role="group"
                            :aria-labelledby="`browse-${browseType.label.replace(/\s+/g, '-').toLowerCase()}`">
                            <div class="col" :class="{ 'col-2': browseType.queries.length > 6 }"
                                v-for="(range, rangeIndex) in browseType.queries" :key="rangeIndex">
                                <button class="btn btn-light landing-page-btn" :class="{
                                    first: rangeIndex === 0,
                                    last: rangeIndex === browseType.queries.length - 1,
                                }" style="border-radius: 0; width: 100%" @click="getContent(browseType, range)"
                                    :aria-label="$t('landingPage.browseRange', { type: browseType.label, range: range })">
                                    {{ range }}
                                </button>
                            </div>
                        </div>
                    </div>
                </section>
            </div>
            <section id="simple-landing-page" v-if="landingPageBrowsing === 'simple'"
                aria-labelledby="simple-bibliography-heading">
                <div class="row" id="landingGroup">
                    <div class="cols-12 col-sm-8 offset-sm-2 d-flex" style="justify-content: center">
                        <div class="card" style="width: fit-content">
                            <div class="card-header">
                                <h2 class="h6 mb-0" id="simple-bibliography-heading">
                                    {{ $t("landingPage.bibliography") }}
                                </h2>
                            </div>
                            <ul class="list-group" role="list">
                                <li class="list-group-item" v-for="(biblioObj, bibIndex) in bibliography.results"
                                    :key="bibIndex" role="listitem">
                                    <citations :citation="biblioObj.citation" :result-number="bibIndex + 1"></citations>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>
            </section>
            <section id="custom-landing-page" v-if="customLandingPage.length > 0" v-html="customLandingPage"
                :aria-label="$t('landingPage.customContent')">
            </section>
            <section id="dictionary-landing-page" v-if="landingPageBrowsing === 'dictionary'"
                :aria-label="$t('landingPage.dictionaryNavigation')">
                <div class="row">
                    <div class="col-6" :class="{ 'offset-3': !showDicoLetterRows }" id="dico-landing-volume">
                        <div class="card shadow-sm">
                            <div class="card-header">
                                <h2 class="h6 mb-0">{{ $t("landingPage.browseByVolume") }}</h2>
                            </div>
                            <nav class="list-group" flush v-if="volumeData.length"
                                :aria-label="$t('landingPage.browseByVolumeLabel')">
                                <router-link class="list-group-item list-group-item-action" v-for="volume in volumeData"
                                    :key="volume.philo_id" :to="`/navigate/${volume.philo_id}/table-of-contents`"
                                    :aria-label="$t('landingPage.volumeLink', { title: volume.title, start: volume.start_head, end: volume.end_head })">
                                    <i style="font-variant: small-caps">{{ volume.title }}</i>
                                    <span style="font-weight: 300; padding-left: 0.25rem" v-if="volume.start_head">({{
                                        volume.start_head }} - {{ volume.end_head }})</span>
                                </router-link>
                            </nav>
                        </div>
                    </div>
                    <div class="col" id="dico-landing-alpha" cols="6" style="border-width: 0px; box-shadow: 0 0 0"
                        v-if="showDicoLetterRows">
                        <div class="card">
                            <div class="card-header">
                                <h2 class="h6 mb-0">{{ $t("landingPage.browseByLetter") }}</h2>
                            </div>
                            <nav :aria-label="$t('landingPage.browseByLetterLabel')">
                                <table class="table table-borderless" style="margin-bottom: 0" role="presentation">
                                    <tr v-for="(row, rowIndex) in dicoLetterRows" :key="rowIndex">
                                        <td v-for="letter in row" :key="letter.letter">
                                            <button class="btn btn-link letter" @click="goToLetter(letter.letter)"
                                                :aria-label="$t('landingPage.browseByLetterAction', { letter: letter.letter })">
                                                {{ letter.letter }}
                                            </button>
                                        </td>
                                    </tr>
                                </table>
                            </nav>
                        </div>
                    </div>
                </div>
            </section>
            <section id="landing-page-content" class="mt-4" :aria-label="$t('landingPage.browseResults')">
                <div class="row">
                    <div class="col-12 col-sm-9 offset-sm-1 col-md-8 offset-md-2 text-content-area">
                        <article class="card mb-4 shadow-sm" v-for="(group, groupIndex) in resultGroups"
                            :key="group.prefix" :id="`landing-group-${groupIndex}`"
                            :aria-labelledby="`group-heading-${groupIndex}`">
                            <div class="card-header">
                                <h3 class="h6 mb-0" :id="`group-heading-${groupIndex}`">{{ group.prefix.toString() }}
                                </h3>
                            </div>
                            <ul class="list-group list-group-flush" role="list">
                                <li class="pt-1"
                                    v-for="(result, resultIndex) in group.results.slice(0, groupDisplay[groupIndex])"
                                    :key="resultIndex" role="listitem">
                                    <citations :citation="buildCitationObject(result.metadata, citationList)"
                                        :result-number="resultIndex + 1"></citations>
                                    <span v-if="displayCount == 'true'">&nbsp;({{ result.count }})</span>
                                </li>
                            </ul>
                            <div class="card-footer" v-if="group.results.length > 100">
                                <button type="button" class="btn btn-outline-secondary" @click="seeAll(groupIndex)"
                                    :aria-label="$t('landingPage.seeAllResults', { n: group.results.length, prefix: group.prefix })">
                                    {{ $t("landingPage.seeResults", { n: group.results.length }) }}
                                </button>
                            </div>
                        </article>
                    </div>
                </div>
            </section>
        </div>
    </div>
</template>
<script setup>
import { inject, reactive, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { storeToRefs } from "pinia";
import { useI18n } from "vue-i18n";
import { useMainStore } from "../stores/main";
import { debug, paramsToRoute } from "../utils.js";
import Citations from "./Citations";  // eslint-disable-line no-unused-vars

const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const philoConfig = inject("$philoConfig");
const route = useRoute();
const router = useRouter();
const { t } = useI18n();
const store = useMainStore();
const { formData } = storeToRefs(store);

const dictionary = philoConfig.dictionary;
const logo = philoConfig.logo;
const landingPageBrowsing = philoConfig.landing_page_browsing;
const defaultLandingPageBrowsing = philoConfig.default_landing_page_browsing;

const customLandingPage = ref("");
const displayCount = ref(true);
const resultGroups = ref([]);
const contentType = ref("");
const selectedField = ref("");
const loading = ref(false);
const showDicoLetterRows = ref(true);
const volumeData = ref([]);
const dicoLetterRows = ref([]);
const citationList = ref([]);
const groupDisplay = reactive({});
const bibliography = ref([]);

function setupDictView() {
    $http.get(`${$dbUrl}/scripts/get_bibliography.py?object_level=doc`).then((response) => {
        volumeData.value.push(...response.data);
    });

    const dicoLetterRange = philoConfig.dico_letter_range;
    let row = [];
    let position = 0;
    for (let i = 0; i < dicoLetterRange.length; i++) {
        position++;
        row.push({
            letter: dicoLetterRange[i],
            url: "bibliography&head=^" + dicoLetterRange[i] + ".*",
        });
        if (position === 4) {
            dicoLetterRows.value.push(row);
            row = [];
            position = 0;
        }
    }
    if (row.length) dicoLetterRows.value.push(row);
    if (dicoLetterRows.value.length === 0) showDicoLetterRows.value = false;
}

function setupCustomPage() {
    $http
        .get(`${$dbUrl}/scripts/get_custom_landing_page.py`)
        .then((response) => (customLandingPage.value = response.data));
}

function getContent(browseType, range) {
    // Navigate to URL with query params instead of loading content directly
    router.push({
        path: "/landing",
        query: {
            browse: browseType.group_by_field,
            range,
            display_count: browseType.display_count,
            is_range: browseType.is_range,
        },
    });
}

function loadContentFromUrl(browseType, range) {
    selectedField.value = browseType.group_by_field;
    loading.value = true;
    $http
        .get(`${$dbUrl}/scripts/get_landing_page_content.py`, {
            params: {
                group_by_field: browseType.group_by_field,
                display_count: browseType.display_count,
                is_range: browseType.is_range,
                query: range,
            },
        })
        .then((response) => {
            for (const i in response.data.content) {
                groupDisplay[i] = 100;
            }
            resultGroups.value = Object.freeze(response.data.content);
            citationList.value = response.data.citations;
            displayCount.value = response.data.display_count;
            contentType.value = response.data.content_type;
            loading.value = false;
        })
        .catch((error) => {
            debug({ $options: { name: "landingPage" } }, error);
            loading.value = false;
        });
}

function handleUrlParameters() {
    const { browse, range, display_count, is_range } = route.query;
    if (browse && range) {
        loadContentFromUrl(
            {
                group_by_field: browse,
                display_count: display_count || "true",
                is_range: is_range || "true",
            },
            range
        );
    }
}

function getSimpleLandingPageData() {
    $http
        .get(`${$dbUrl}/reports/bibliography.py`, { params: { simple_bibliography: "all" } })
        .then((response) => {
            bibliography.value = response.data;
        })
        .catch((error) => {
            debug({ $options: { name: "landingPage" } }, error);
            loading.value = false;
        });
}

function buildCitationObject(metadataFields, citations) {
    const out = [];
    for (const citation of citations) {
        let label = metadataFields[citation.field] || "";
        if (!citation.link) {
            out.push({ ...citation, href: "", label });
            continue;
        }
        if (citation.field === "title") {
            const docId = metadataFields.philo_id.split(" ")[0];
            const link = philoConfig.skip_table_of_contents
                ? `/navigate/${docId}`
                : `/navigate/${docId}/table-of-contents`;
            out.push({ ...citation, href: link, label: metadataFields.title });
            continue;
        }
        const queryParams = { ...formData.value, start: "0", end: "25" };
        if (label === "") {
            queryParams[citation.field] = ""; // Should be NULL but that's broken in the philo lib
            label = t("common.na");
        } else {
            queryParams[citation.field] = `"${label}"`;
        }
        // workaround for broken NULL searches
        const href = queryParams[citation.field].length
            ? paramsToRoute({ ...queryParams, report: "concordance" })
            : "";
        out.push({ ...citation, href, label });
    }
    return out;
}

function goToLetter(letter) {
    router.push(`/bibliography?head=^${letter}.*`);
}

function seeAll(groupIndex) {
    groupDisplay[groupIndex] = resultGroups.value[groupIndex].length;
}

watch(() => route.fullPath, handleUrlParameters);

if (landingPageBrowsing === "toc") {
    router.push("/navigate/1/table-of-contents");
}
if (!["simple", "default", "dictionary"].includes(landingPageBrowsing)) {
    setupCustomPage();
} else if (dictionary) {
    setupDictView();
} else if (landingPageBrowsing === "simple") {
    getSimpleLandingPageData();
}
handleUrlParameters();
</script>
<style lang="scss">
@use "../assets/styles/theme.module.scss" as theme;

#landing-page-container {
    .btn-light {
        background-color: #fff;
        border-width: 0px 1px 0px 0px;
        border-color: rgba(0, 0, 0, 0.125);
    }

    .first {
        border-bottom-left-radius: 0.25rem !important;
    }

    .last {
        border-bottom-right-radius: 0.25rem !important;
        border-right-width: 0px;
    }

    .btn-light:hover {
        background-color: #f8f8f8;
    }

    .card-header {
        text-align: center;
        font-variant: small-caps;
    }

    .letter {
        text-align: center;
        cursor: pointer;
        color: theme.$link-color;
        text-decoration: none;
        border: none;
        padding: 0.5rem;
        width: 100%;
    }

    .letter:focus {
        background-color: #f8f8f8;
        color: #fff;
        text-decoration: none;
    }

    .letter:hover {
        color: #fff !important;
        text-decoration: none;
    }

    td {
        padding: 0;
    }

    tr:nth-child(odd) td {
        background-color: #f8f8f8;
    }

    tr:nth-child(odd) td:nth-child(2n + 1) {
        background-color: #fff;
    }

    tr:nth-child(even) td {
        background-color: #fff;
    }

    tr:nth-child(even) td:nth-child(2n + 1) {
        background-color: #f8f8f8;
    }

    .landing-page-btn:focus {
        border-width: 3px;
    }

    #dico-landing-volume .list-group-item {
        padding: 0.5rem 1rem;
    }

    #dico-landing-volume a {
        display: inline-block;
        padding: 0.5rem 0;
    }

    #landing-page-logo {
        text-align: center;
    }

    .landing-page-title {
        margin-top: 2rem;
        font-size: 1.5rem;
        font-weight: 600;
        font-variant: small-caps;
        letter-spacing: 0.05em;
        margin-bottom: 0;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
}
</style>
