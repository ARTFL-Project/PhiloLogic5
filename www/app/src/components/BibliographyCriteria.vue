<template>
    <div role="region" :aria-label="$t('biblioCriteria.searchCriteriaRegion')">
        <div>
            <span>{{ $t("searchArgs.biblioCriteria") }}:&nbsp;</span>

            <ul aria-live="polite" :aria-label="$t('biblioCriteria.activeFiltersLabel')" v-if="biblio.length !== 0">
                <li class="metadata-args rounded-pill" v-for="metadata in biblio" :key="metadata.key"
                    :aria-label="`${$t('biblioCriteria.filterLabel')}: ${metadata.alias} ${metadata.value.replace('<=>', '—')}`">

                    <span class="metadata-label" :id="`label-${metadata.key.replace(/\s+/g, '-')}`">{{ metadata.alias
                    }}</span>
                    <span class="metadata-value"
                        :id="`value-${metadata.key.replace(/\s+/g, '-')}`">{{metadata.value.replace("<=>",
                            "&#8212;")}}</span>

                    <!-- Accessible remove button -->
                    <button type="button" class="remove-metadata" v-if="removable" @click="removeMetadata(metadata.key)"
                        :aria-label="`${$t('biblioCriteria.removeFilter')} ${metadata.alias}`"
                        :aria-describedby="`label-${metadata.key.replace(/\s+/g, '-')} value-${metadata.key.replace(/\s+/g, '-')}`">
                        <span class="icon-x" aria-hidden="true"></span>
                    </button>
                </li>
            </ul>

            <strong v-if="biblio.length === 0" role="status">{{ $t("common.none") }}</strong>
        </div>

        <!-- Time series section -->
        <div v-if="queryReport === 'time_series'" role="group" :aria-label="$t('biblioCriteria.dateRangeGroup')">
            {{ $t("searchArgs.occurrencesBetween", { n: resultsLength }) }}

            <!-- Start date -->
            <div class="biblio-criteria d-inline-block">
                <div class="metadata-args rounded-pill"
                    :aria-label="`${$t('biblioCriteria.startDate')}: ${start_date}`">
                    <span class="metadata-value" id="start-date-value">{{ start_date }}</span>
                    <button type="button" class="remove-metadata" v-if="removable" @click="removeMetadata('start_date')"
                        :aria-label="`${$t('biblioCriteria.removeStartDate')}`" aria-describedby="start-date-value">
                        <span class="icon-x" aria-hidden="true"></span>
                    </button>
                </div>
            </div>

            &nbsp;{{ $t("common.and") }}&nbsp;

            <!-- End date -->
            <div class="biblio-criteria d-inline-block">
                <div class="metadata-args rounded-pill" :aria-label="`${$t('biblioCriteria.endDate')}: ${end_date}`">
                    <span class="metadata-value" id="end-date-value">{{ end_date }}</span>
                    <button type="button" class="remove-metadata" v-if="removable" @click="removeMetadata('end_date')"
                        :aria-label="`${$t('biblioCriteria.removeEndDate')}`" aria-describedby="end-date-value">
                        <span class="icon-x" aria-hidden="true"></span>
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
        start_date: [String, Number, null],
        end_date: [String, Number, null],
        removeMetadata: Function || null,
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
@use "../assets/styles/theme.module.scss" as theme;

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
    padding: 0.4rem 0.5rem;
    border-right: none;
    border-top: none;
    border-bottom: none;
    background: transparent;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    min-width: 2rem;
    line-height: 1;
}

.remove-metadata:hover,
.remove-metadata:focus {
    background-color: theme.$button-color !important;
    color: #fff !important;
    cursor: pointer;
    outline: 2px solid theme.$link-color;
    outline-offset: -2px;
}

.remove-metadata:hover .icon-x,
.remove-metadata:focus .icon-x {
    background-color: #fff !important;
}

.rounded-pill a {
    margin-right: 0.5rem;
    text-decoration: none;
}

/* Ensure the criteria list stays inline */
ul {
    display: inline;
    list-style: none;
    margin: 0;
    padding: 0;
}

/* Ensure list items stay inline */
li {
    display: inline-block;
    vertical-align: middle;
}
</style>