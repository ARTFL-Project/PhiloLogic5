<template>
    <div>
        <ul class="nav nav-tabs mb-3">
            <li class="nav-item" v-for="name in kinds" :key="name">
                <button class="nav-link" :class="{ active: kind === name }" type="button" @click="kind = name">{{ $t(`previews.${name}`) }}</button>
            </li>
        </ul>
        <div class="d-flex gap-2 align-items-center mb-3">
            <select v-if="kind === 'tokens' && fileNames.length" class="form-select form-select-sm w-auto" v-model="tokensFile" :aria-label="$t('previews.file')">
                <option v-for="name in fileNames" :key="name" :value="name">{{ name }}</option>
            </select>
            <label v-else class="small">{{ $t("previews.sample") }} <input type="number" min="1" max="50" class="form-control form-control-sm d-inline w-auto" v-model.number="count" /></label>
            <button class="btn btn-sm btn-secondary" type="button" :disabled="loading" @click="run">
                <span v-if="loading" class="spinner-border spinner-border-sm me-1"></span>{{ $t("previews.run") }}
            </button>
            <span class="small text-body-secondary">{{ $t(`previews.${kind}Help`) }}</span>
        </div>
        <div v-if="error" class="alert alert-danger py-2">{{ error }}</div>

        <template v-if="kind === 'header' && results.header">
            <p class="small">{{ $t("previews.sortedAs") }} <span class="mono">{{ results.header.sorted.join(" → ") }}</span></p>
            <p class="small text-body-secondary" v-if="notFound.length">{{ $t("previews.notFound", { fields: notFound.join(", ") }) }}</p>
            <div class="table-responsive">
                <table class="table table-sm table-bordered small align-top">
                    <thead>
                        <tr>
                            <th>{{ $t("previews.file") }}</th>
                            <th v-for="field in headerFields" :key="field" class="mono">{{ field }} <span v-if="field !== 'year'" class="badge text-bg-light">{{ results.header.found[field] || 0 }}</span></th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr v-for="row in results.header.rows" :key="row.file">
                            <td class="mono">
                                {{ row.file }}
                                <button v-if="row.header" class="btn btn-link btn-sm p-0 d-block" type="button" @click="shownHeader = shownHeader === row.file ? null : row.file">{{ $t("previews.showHeader") }}</button>
                            </td>
                            <td v-if="row.error" :colspan="headerFields.length" class="text-danger">{{ row.error }}</td>
                            <template v-else>
                                <td v-for="field in headerFields" :key="field" :class="{ 'table-warning': !row.metadata[field] }" :title="row.xpaths[field] || row.metadata[field] || ''">{{ shorten(row.metadata[field]) }}</td>
                            </template>
                        </tr>
                    </tbody>
                </table>
            </div>
            <pre v-if="shownHeaderText" class="bg-light p-2 rounded header-text mono">{{ shownHeaderText }}</pre>
        </template>

        <template v-if="kind === 'tags' && results.tags">
            <table class="table table-sm small">
                <thead>
                    <tr>
                        <th>{{ $t("previews.tag") }}</th>
                        <th class="text-end">{{ $t("previews.count") }}</th>
                        <th>{{ $t("previews.treatedAs") }}</th>
                        <th>{{ $t("previews.attributes") }}</th>
                    </tr>
                </thead>
                <tbody>
                    <tr v-for="tag in results.tags.tags" :key="tag.tag">
                        <td class="mono" :title="tag.example">&lt;{{ tag.tag }}&gt;</td>
                        <td class="text-end">{{ tag.count }}</td>
                        <td>
                            <span v-if="tag.mapped_to" class="badge text-bg-secondary me-1">{{ tag.mapped_to }}</span>
                            <span v-if="tag.suppressed" class="badge text-bg-secondary me-1">{{ $t("previews.suppressed") }}</span>
                            <span v-if="tag.exception" class="badge text-bg-info me-1">{{ $t("previews.exception") }}</span>
                        </td>
                        <td class="mono">{{ Object.keys(tag.attributes).join(", ") }}</td>
                    </tr>
                </tbody>
            </table>
        </template>

        <template v-if="kind === 'tokens' && results.tokens">
            <p class="small">{{ $t("previews.tokensSummary", { words: results.tokens.words, sentences: results.tokens.sentence_count, punctuation: results.tokens.punctuation }) }}</p>
            <p v-for="sentence in results.tokens.sentences" :key="sentence.id" class="mb-2">
                <span v-for="(token, index) in sentence.tokens" :key="index" class="token" :class="token.kind">{{ token.text }}</span>
            </p>
        </template>
    </div>
</template>

<script setup>
import { computed, reactive, ref, watch } from "vue";
import { api } from "../api";

// Previews of a load on a sample of its files, with the options being edited
const props = defineProps({
    files: { type: Object, required: true },
    options: { type: Object, required: true },
    fileNames: { type: Array, default: () => [] },
});
const kinds = ["header", "tags", "tokens"];
const kind = ref("header");
const count = ref(10);
const loading = ref(false);
const error = ref(null);
const results = reactive({ header: null, tags: null, tokens: null });
const shownHeader = ref(null);
const tokensFile = ref(props.fileNames[0] || "");
watch(
    () => props.fileNames,
    (names) => {
        if (!names.includes(tokensFile.value)) {
            tokensFile.value = names[0] || "";
        }
    },
);

// Fields found in at least one file, the year first (it sets the order of the documents)
const headerFields = computed(() => {
    if (!results.header) {
        return [];
    }
    return ["year", ...results.header.fields.filter((field) => field !== "year" && results.header.found[field])];
});
const notFound = computed(() => (results.header ? results.header.fields.filter((field) => !results.header.found[field]) : []));
const shorten = (value) => (value && value.length > 120 ? `${value.slice(0, 120)}…` : value);
const shownHeaderText = computed(() => {
    const row = results.header && results.header.rows.find((item) => item.file === shownHeader.value);
    return row ? row.header : null;
});

async function run() {
    loading.value = true;
    error.value = null;
    try {
        results[kind.value] = await api.post(`previews/${kind.value}`, {
            files: props.files,
            options: props.options,
            count: count.value,
            file: tokensFile.value,
        });
    } catch (requestError) {
        error.value = requestError.message;
    } finally {
        loading.value = false;
    }
}
</script>
