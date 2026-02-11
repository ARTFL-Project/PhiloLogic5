<template>
    <div class="modal-dialog modal-xl modal-dialog-scrollable" style="margin-top: 12rem;">
        <div class="modal-content">
            <div class="card-header text-center position-relative">
                <h2 class="modal-title mb-0 h6" id="biblio-modal-title"
                    style="font-variant: small-caps; font-size: 1.2rem; font-weight: 700;">{{
                        $t("resultsBiblio.heading") }}</h2>
                <button type="button" class="btn btn-secondary btn-sm close-box" data-bs-dismiss="modal"
                    :aria-label="$t('common.closeModal')" @click="$event.target.blur()"
                    style="position: absolute; top: 1px; right: 0;">
                    <span class="icon-x"></span>
                </button>
            </div>
            <div class="modal-body">
                <ul id="results-bibliography">
                    <li v-for="(result, resultIndex) in uniquedResults" :key="resultIndex">
                        <router-link :to="`/${formData.report}?${buildLink(result.metadata_fields.title)}`"
                            class="result-card-link" role="listitem" :aria-label="$t('resultsBiblio.viewOccurrences', {
                                count: result.count,
                                title: result.metadata_fields.title || 'Unknown title'
                            })">
                            <article class="result-card">
                                <div class="result-content">
                                    <div class="result-number">
                                        {{ resultIndex + 1 }}
                                    </div>
                                    <div class="citation-content">
                                        <citations :citation="result.citation"></citations>
                                    </div>
                                    <div class="occurrence-badge">
                                        <span class="occurrence-count">{{ result.count }} {{
                                            $t("resultsBiblio.occurrences")
                                            }}</span>
                                    </div>
                                </div>
                            </article>
                        </router-link>
                    </li>
                </ul>
            </div>
        </div>
    </div>
</template>

<script>
import { mapWritableState } from "pinia";
import variables from "../assets/styles/theme.module.scss";
import { useMainStore } from "../stores/main";
import citations from "./Citations";

export default {
    name: "ResultsBibliography",
    components: { citations },
    inject: ["results"],
    mounted() {
        // Fix accessibility issue: remove aria-hidden when modal is shown
        const modal = document.getElementById("results-bibliography");
        if (modal && modal.closest('.modal')) {
            modal.closest('.modal').removeAttribute('aria-hidden');
        }
    },
    computed: {
        ...mapWritableState(useMainStore, ["formData"]),
        themeColors() {
            return {
                linkColor: variables.linkColor,
                buttonColor: variables.buttonColor,
                cardHeaderColor: variables.cardHeaderColor,
                headerColor: variables.headerColor,
                color: variables.color
            };
        },
        uniquedResults() {
            //TODO: We should provide the object level of hits. This is a HACK.
            if (
                typeof this.results != "undefined" &&
                typeof this.results[0] != "undefined" &&
                typeof this.results[0].citation != "undefined"
            ) {
                // time series sends a results object which is incompatible
                let objectLevel = this.results[0].citation[0].object_type;
                let uniqueResults = [];
                let previousPhiloId = "";
                for (let result of this.results) {
                    if (result.metadata_fields[`philo_${objectLevel}_id`] == previousPhiloId) {
                        uniqueResults[uniqueResults.length - 1].count++;
                        continue;
                    }
                    result = this.copyObject(result);
                    let citation = [];
                    for (let i = 0; i < result.citation.length; i++) {
                        if (result.citation[i].object_type == objectLevel) {
                            citation.push(result.citation[i]);
                        }
                    }
                    result.citation = citation;
                    result.count = 1;
                    uniqueResults.push(result);
                    previousPhiloId = result.metadata_fields[`philo_${objectLevel}_id`];
                }
                return uniqueResults;
            } else {
                return [];
            }
        },
    },
    methods: {
        buildLink(title) {
            return this.paramsToUrlString({
                ...this.formData,
                title: `"${title}"`,
                start: 1,
                end: this.formData.results_per_page,
            });
        },
    },
};
</script>

<style scoped lang="scss">
@use "../assets/styles/theme.module.scss" as theme;

#results-bibliography {
    padding: 0;
    margin: 0;
}

.result-card-link {
    display: block;
    text-decoration: none;
    color: inherit;
    margin-bottom: 0.75rem;

    &:hover {
        text-decoration: none;
        color: inherit;
    }

    &:focus {
        outline: 2px solid theme.$button-color;
        outline-offset: 2px;
        border-radius: 8px;
    }
}

.result-card {
    background: #fff;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 0.75rem;
    transition: all 0.2s ease;
    position: relative;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.result-card-link:hover .result-card {
    background-color: rgba(theme.$link-color, 0.08) !important;
    border-color: theme.$link-color;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.result-content {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
}

.result-number {
    background: theme.$card-header-color;
    color: white;
    border-radius: 50%;
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 0.8rem;
    flex-shrink: 0;
    margin-top: 0.125rem;
}

.citation-content {
    flex: 1;
    line-height: 1.5;
    font-size: 0.9rem;
}

.occurrence-badge {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    margin-top: 0.125rem;
}

.occurrence-count {
    background: theme.$link-color;
    color: white;
    border-radius: 12px;
    padding: 0.25rem 0.5rem;
    font-weight: bold;
    font-size: 0.75rem;
    white-space: nowrap;
    display: inline-block;
}

.result-card-link:hover .occurrence-count {
    background: theme.$button-color;
    transform: scale(1.05);
}

/* Responsive design */
@media (max-width: 768px) {
    .result-content {
        flex-direction: column;
        gap: 0.5rem;
    }

    .occurrence-badge {
        align-self: flex-start;
        margin-top: 0;
    }

    .result-number {
        width: 24px;
        height: 24px;
        font-size: 0.75rem;
    }

    .result-card {
        padding: 0.5rem;
    }
}

/* Focus indicator for close button */
button.close-box:focus,
button.btn.close-box:focus-visible {
    outline: 2px solid #fff !important;
    outline-offset: -2px !important;
    background-color: rgba(255, 255, 255, 0.2) !important;
}
</style>