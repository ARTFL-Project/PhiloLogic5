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
                    style="width: 4rem; height: 4rem; position: absolute; z-index: 50; top: 10px">
                    <span class="visually-hidden">{{ $t('common.loading') }}</span>
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
                                    <citations :citation="biblioObj.citation"></citations>
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
                            :key="group.prefix" :id="`landing-${group.prefix}`"
                            :aria-labelledby="`group-heading-${groupIndex}`">
                            <div class="card-header">
                                <h3 class="h6 mb-0" :id="`group-heading-${groupIndex}`">{{ group.prefix.toString() }}
                                </h3>
                            </div>
                            <ul class="list-group list-group-flush" role="list">
                                <li class="list-group-item contentClass p-2"
                                    v-for="(result, resultIndex) in group.results.slice(0, groupDisplay[groupIndex])"
                                    :key="resultIndex" role="listitem">
                                    <citations :citation="buildCitationObject(result.metadata, citations)"></citations>
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
<script>
import { mapFields } from "vuex-map-fields";
import citations from "./Citations";

export default {
    name: "landingPage",
    components: {
        citations,
    },
    computed: { ...mapFields(["accessAuthorized"]) },
    inject: ["$http"],
    data() {
        return {
            dictionary: this.$philoConfig.dictionary,
            logo: this.$philoConfig.logo,
            landingPageBrowsing: this.$philoConfig.landing_page_browsing,
            defaultLandingPageBrowsing: this.$philoConfig.default_landing_page_browsing,
            customLandingPage: "",
            displayCount: true,
            resultGroups: [],
            contentType: "",
            selectedField: "",
            loading: false,
            showDicoLetterRows: true,
            volumeData: [],
            dicoLetterRows: [],
            citations: [],
            groupDisplay: {},
            bibliography: [],
        };
    },
    created() {
        if (this.landingPageBrowsing == "toc") {
            this.$router.push("/navigate/1/table-of-contents");
        }
        if (!["simple", "default", "dictionary"].includes(this.landingPageBrowsing)) {
            this.setupCustomPage();
        } else if (this.dictionary) {
            this.setupDictView();
        } else if (this.landingPageBrowsing == "simple") {
            this.getSimpleLandingPageData();
        }
    },
    methods: {
        setupDictView() {
            this.$http.get(`${this.$dbUrl}/scripts/get_bibliography.py?object_level=doc`).then((response) => {
                for (let i = 0; i < response.data.length; i++) {
                    this.volumeData.push(response.data[i]);
                }
            });

            let dicoLetterRange = this.$philoConfig.dico_letter_range;
            let row = [];
            let position = 0;
            for (var i = 0; i < dicoLetterRange.length; i++) {
                position++;
                row.push({
                    letter: dicoLetterRange[i],
                    url: "bibliography&head=^" + dicoLetterRange[i] + ".*",
                });
                if (position === 4) {
                    this.dicoLetterRows.push(row);
                    row = [];
                    position = 0;
                }
            }
            if (row.length) {
                this.dicoLetterRows.push(row);
            }
            if (this.dicoLetterRows.length == 0) {
                this.showDicoLetterRows = false;
            }
        },
        setupCustomPage() {
            this.$http
                .get(`${this.$dbUrl}/${this.landingPageBrowsing}`, {
                    withCredentials: false,
                    headers: { "Access-Control-Allow-Origin": "*", "Content-Type": "text/html" },
                })
                .then((response) => (this.customLandingPage = response.data));
        },
        getContent(browseType, range) {
            this.selectedField = browseType.group_by_field;
            this.loading = true;
            this.$http
                .get(`${this.$dbUrl}/scripts/get_landing_page_content.py`, {
                    params: {
                        group_by_field: browseType.group_by_field,
                        display_count: browseType.display_count,
                        is_range: browseType.is_range,
                        query: range,
                    },
                })
                .then((response) => {
                    for (let i in response.data.content) {
                        this.groupDisplay[i] = 100;
                    }
                    this.resultGroups = Object.freeze(response.data.content);
                    this.citations = response.data.citations;
                    this.displayCount = response.data.display_count;
                    this.contentType = response.data.content_type;
                    this.loading = false;
                })
                .catch((error) => {
                    this.debug(this, error);
                    this.loading = false;
                });
        },
        getSimpleLandingPageData() {
            this.$http
                .get(`${this.$dbUrl}/reports/bibliography.py`, { params: { simple_bibliography: "all" } })
                .then((response) => {
                    this.bibliography = response.data;
                })
                .catch((error) => {
                    this.debug(this, error);
                    this.loading = false;
                });
        },
        buildCitationObject(metadataFields, citations) {
            // Used because too many results are returned from server
            let citationObject = [];
            for (let citation of citations) {
                let label = metadataFields[citation.field] || "";
                if (citation.link) {
                    let link = "";
                    if (citation.field == "title") {
                        if (this.$philoConfig.skip_table_of_contents) {
                            link = `/navigate/${metadataFields.philo_id.split(" ")[0]}`;
                        } else {
                            link = `/navigate/${metadataFields.philo_id.split(" ")[0]}/table-of-contents`;
                        }
                        citationObject.push({ ...citation, href: link, label: metadataFields.title });
                    } else {
                        let queryParams = {
                            ...this.$store.state.formData,
                            start: "0",
                            end: "25",
                        };
                        if (label == "") {
                            queryParams[citation.field] = ""; // Should be NULL, but that's broken in the philo lib
                            label = this.$t("common.na");
                        } else {
                            queryParams[citation.field] = `"${label}"`;
                        }

                        // workaround for broken NULL searches
                        if (queryParams[citation.field].length) {
                            link = this.paramsToRoute({
                                ...queryParams,
                                report: "concordance",
                            });
                            citationObject.push({ ...citation, href: link, label: label });
                        } else {
                            citationObject.push({ ...citation, href: "", label: label });
                        }
                    }
                } else {
                    citationObject.push({ ...citation, href: "", label: label });
                }
            }
            return citationObject;
        },
        goToLetter(letter) {
            this.$router.push(`/bibliography?head=^${letter}.*`);
        },
        seeAll(groupIndex) {
            this.groupDisplay[groupIndex] = this.resultGroups[groupIndex].length;
        },
    },
};
</script>
<style lang="scss" scoped>
@use "../assets/styles/theme.module.scss" as theme;

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
    background: transparent;
    padding: 0.5rem;
    width: 100%;
}

.letter:hover,
.letter:focus {
    background-color: #e8e8e8;
    color: theme.$link-color;
    text-decoration: none;
}

tr:nth-child(odd) {
    background-color: #f8f8f8;
}

tr:nth-child(odd) td.letter:nth-child(2n + 1) {
    background-color: #fff;
}

tr:nth-child(even) {
    background-color: #fff;
}

tr:nth-child(even) td.letter:nth-child(2n + 1) {
    background-color: #f8f8f8;
}

.landing-page-btn:focus {
    border-width: 3px;
}

#dico-landing-volume .list-group-item {
    padding: 0 1rem;
}

#dico-landing-volume a {
    display: inline-block;
    padding: 0.5rem 0;
}

.letter {
    text-align: center;
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
</style>
