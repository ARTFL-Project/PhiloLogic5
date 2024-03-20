<template>
    <div>
        <div>
            {{ $t("searchArgs.biblioCriteria") }}:
            <span class="metadata-args rounded-pill" v-for="metadata in biblio" :key="metadata.key">
                <span class="metadata-label">{{ metadata.alias }}</span>
                <span class="metadata-value">{{ metadata.value.replace("<=>", "&#8212;") }}</span>
                <span class="remove-metadata" v-if="removable" @click="removeMetadata(metadata.key)">X</span>
            </span>
            <b v-if="biblio.length === 0">{{ $t("common.none") }}</b>
        </div>
        <div v-if="queryReport === 'time_series'">
            {{ $t("searchArgs.occurrencesBetween", { n: resultsLength }) }}
            <span class="biblio-criteria">
                <span class="metadata-args rounded-pill">
                    <span class="metadata-value">{{ start_date }}</span>
                    <span class="remove-metadata" v-if="removable" @click="removeMetadata('start_date')">X</span>
                </span> </span>&nbsp; {{ $t("common.and") }}&nbsp;
            <span class="biblio-criteria">
                <span class="metadata-args rounded-pill">
                    <span class="metadata-value">{{ end_date }}</span>
                    <span class="remove-metadata" v-if="removable" @click="removeMetadata('end_date')">X</span>
                </span>
            </span>
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
<style scoped>
.metadata-args {
    border: 1px solid #ddd;
    display: inline-flex !important;
    margin-right: 5px;
    border-radius: 50rem;
    width: fit-content;
    line-height: 2;
    margin-bottom: 0.5rem;
}

.metadata-label {
    background-color: #e9ecef;
    border: solid #ddd;
    border-width: 0 1px 0 0;
    border-top-left-radius: 50rem;
    border-bottom-left-radius: 50rem;
    padding: 0 0.5rem;
}

.metadata-value {
    -webkit-box-decoration-break: clone;
    box-decoration-break: clone;
    padding: 0 0.5rem;
}

.remove-metadata {
    padding-right: 5px;
    padding-left: 5px;
    border-left: #ddd solid 1px;
    border-top-right-radius: 50rem;
    border-bottom-right-radius: 50rem;
    padding: 0 0.5rem;
}

.remove-metadata:hover,
.close-pill:hover {
    background-color: #e9ecef;
    cursor: pointer;
}

.rounded-pill a {
    margin-right: 0.5rem;
    text-decoration: none;
}

.metadata-label,
.metadata-value,
.remove-metadata {
    display: flex;
    align-items: center;
}

.remove-metadata:hover {
    background-color: #e9ecef;
    cursor: pointer;
}
</style>