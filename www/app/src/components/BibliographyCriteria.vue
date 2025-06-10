<template>
    <div role="region" :aria-label="$t('biblioCriteria.searchCriteriaRegion')">
        <div>
            <span v-if="!hideCriteriaString">{{ $t("searchArgs.biblioCriteria") }}:&nbsp;</span>

            <!-- Criteria list with proper semantics -->
            <div role="list" aria-live="polite" :aria-label="$t('biblioCriteria.activeFiltersLabel')">
                <div class="metadata-args rounded-pill" v-for="metadata in biblio" :key="metadata.key" role="listitem"
                    :aria-label="`${$t('biblioCriteria.filterLabel')}: ${metadata.alias} ${metadata.value.replace('<=>', '—')}`">

                    <span class="metadata-label" :id="`label-${metadata.key}`">{{ metadata.alias }}</span>
                    <span class="metadata-value" :id="`value-${metadata.key}`">{{metadata.value.replace("<=>",
                        "&#8212;")}}</span>

                    <!-- Accessible remove button -->
                    <button type="button" class="remove-metadata" v-if="removable" @click="removeMetadata(metadata.key)"
                        :aria-label="`${$t('biblioCriteria.removeFilter')} ${metadata.alias}`"
                        :aria-describedby="`label-${metadata.key} value-${metadata.key}`">
                        <span aria-hidden="true">×</span>
                    </button>
                </div>
            </div>

            <strong v-if="biblio.length === 0" role="status">{{ $t("common.none") }}</strong>
        </div>

        <!-- Time series section -->
        <div v-if="queryReport === 'time_series'" role="group" :aria-label="$t('biblioCriteria.dateRangeGroup')">
            {{ $t("searchArgs.occurrencesBetween", { n: resultsLength }) }}

            <!-- Start date -->
            <div class="biblio-criteria d-inline-block">
                <div class="metadata-args rounded-pill" role="listitem"
                    :aria-label="`${$t('biblioCriteria.startDate')}: ${start_date}`">
                    <span class="metadata-value" :id="`start - date - value`">{{ start_date }}</span>
                    <button type="button" class="remove-metadata" v-if="removable" @click="removeMetadata('start_date')"
                        :aria-label="`${$t('biblioCriteria.removeStartDate')}`" aria-describedby="start-date-value">
                        <span aria-hidden="true">×</span>
                    </button>
                </div>
            </div>

            &nbsp;{{ $t("common.and") }}&nbsp;

            <!-- End date -->
            <div class="biblio-criteria d-inline-block">
                <div class="metadata-args rounded-pill" role="listitem"
                    :aria-label="`${$t('biblioCriteria.endDate')}: ${end_date}`">
                    <span class="metadata-value" :id="`end - date - value`">{{ end_date }}</span>
                    <button type="button" class="remove-metadata" v-if="removable" @click="removeMetadata('end_date')"
                        :aria-label="`${$t('biblioCriteria.removeEndDate')}`" aria-describedby="end-date-value">
                        <span aria-hidden="true">×</span>
                    </button>
                </div>
            </div>
        </div>
    </div>
</template>
<script>
export default {
    name: "BibliographyCriteria",
    props: {
        biblio: Object,
        queryReport: String,
        resultsLength: Number,
        start_date: String || null,
        end_date: String | null,
        removeMetadata: Function || null,
        hideCriteriaString: Boolean || null,
    },
    computed: {
        removable() {
            if (typeof this.removeMetadata === "function") {
                return true
            }
            return false
        },
    }
}
</script>
<style scoped lang="scss">
@import "../assets/styles/theme.module.scss";

.metadata-args {
    border: 1px solid #ddd;
    display: inline-flex !important;
    margin-right: 5px;
    border-radius: 50rem;
    width: fit-content;
    line-height: 2;
    margin-bottom: 0.5rem;
    align-items: center;
}

.metadata-label {
    background-color: #e9ecef;
    border: solid #ddd;
    border-width: 0 1px 0 0;
    border-top-left-radius: 50rem;
    border-bottom-left-radius: 50rem;
    padding: 0 0.5rem;
    display: inline-flex;
    align-items: center;
}

.metadata-value {
    -webkit-box-decoration-break: clone;
    box-decoration-break: clone;
    padding: 0 0.5rem;
    display: inline-flex;
    align-items: center;
}

.remove-metadata {
    border-left: #ddd solid 1px;
    border-top-right-radius: 50rem;
    border-bottom-right-radius: 50rem;
    padding: 0 0.5rem;
    border-right: none;
    border-top: none;
    border-bottom: none;
    background: transparent;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    height: 100%;
}

.remove-metadata:hover,
.remove-metadata:focus {
    background-color: #e9ecef;
    cursor: pointer;
    outline: 2px solid $link-color;
    outline-offset: -2px;
}

.rounded-pill a {
    margin-right: 0.5rem;
    text-decoration: none;
}

/* Ensure the criteria list stays inline */
div[role="list"] {
    display: inline;
}

/* Ensure list items stay inline */
div[role="listitem"] {
    display: inline-block;
    vertical-align: middle;
}
</style>