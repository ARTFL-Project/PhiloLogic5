<template>
    <div class="modal-dialog modal-xl modal-dialog-scrollable" role="dialog" aria-labelledby="biblio-modal-title">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title" id="biblio-modal-title">{{ $t("resultsBiblio.heading") }}</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal" :aria-label="$t('common.closeModal')">
                </button>
            </div>
            <div class="modal-body">
                <ul id="results-bibliography" role="list">
                    <li class="result" v-for="(result, resultIndex) in uniquedResults" :key="resultIndex"
                        role="listitem">
                        <citations :citation="result.citation"></citations>
                        <router-link :to="`/${report}?${buildLink(result.metadata_fields.title)}`"
                            class="btn rounded-pill btn-outline-secondary btn-sm ms-3" :aria-label="$t('resultsBiblio.viewOccurrences', {
                                count: result.count,
                                title: result.metadata_fields.title || 'Unknown title'
                            })">
                            {{ result.count }} {{ $t("resultsBiblio.occurrences") }}
                        </router-link>
                    </li>
                </ul>
            </div>
        </div>
    </div>
</template>
<script>
import { mapFields } from "vuex-map-fields";
import citations from "./Citations";

export default {
    name: "ResultsBibliography",
    components: { citations },
    inject: ["results"],
    computed: {
        ...mapFields(["formData.report"]),
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
                ...this.$store.state.formData,
                title: `"${title}"`,
                start: 1,
                end: this.$store.state.formData,
            });
        },
    },
};
</script>
<style scoped>
#results-bibliography {
    padding-inline-start: 2rem;
}

.result {
    list-style-type: circle;
    line-height: 2.5;
}
</style>
