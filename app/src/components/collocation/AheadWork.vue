<template>
    <section class="card shadow-sm mb-3 work-card" :class="{ 'perhaps-card': perhaps }">
        <div class="card-body py-2">
            <h4 class="h6 mb-1"><span class="work-year">{{ work.year }} &middot;</span> {{ workName(work) }}</h4>
            <p class="mb-1 small">
                {{ $t("aheadOfTime.closer", {
                    share: `${Math.round(work.share * 100)}%`, after: work.after, before: work.before, sentences: work.sentences,
                }) }}
            </p>
            <p v-if="work.words.length" class="mb-1">
                <span class="small text-muted me-1">{{ $t("aheadOfTime.words") }}</span>
                <span v-for="w in work.words" :key="`${w.group}-${w.word}`" class="work-chip">{{ wordText(w) }}</span>
            </p>
            <p v-if="work.followers.length" class="mb-1 small">
                <span class="text-muted me-1">{{ $t("aheadOfTime.followers") }}</span>
                {{ work.followers.map((f) => `${workName(f)} (${f.year})`).join("; ") }}
            </p>
            <button v-if="work.sources.length" type="button" class="btn btn-sm btn-link p-0" @click="emit('read', work)">
                {{ $t("aheadOfTime.readSentences") }} &rarr;</button>
        </div>
    </section>
</template>

<script setup>
import { useI18n } from "vue-i18n";

const props = defineProps({
    work: { type: Object, required: true },
    queryWord: { type: String, default: "" },
    perhaps: { type: Boolean, default: false },
});
const emit = defineEmits(["read"]);
const { t } = useI18n();

function wordText(w) {
    if (w.phrase) return w.phrase;
    if (w.group === 1) return `${w.word} ${props.queryWord}`;
    if (w.group === 2) return `${props.queryWord} ${w.word}`;
    return w.word;
}

function workName(w) {
    return [w.author, w.title].filter(Boolean).join(", ") || t("aheadOfTime.untitled");
}
</script>

<style scoped>
.work-year {
    font-weight: 600;
}

.work-chip {
    display: inline-block;
    padding: 0.1rem 0.45rem;
    margin: 0.15rem 0.25rem 0.15rem 0;
    border: 1px solid #ced4da;
    border-radius: 0.5rem;
    font-size: 0.85rem;
}

.perhaps-card {
    border-style: dashed;
}
</style>
