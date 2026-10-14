<template>
    <!-- A sense's periods from left to right: the words new in each against those before, the fading ones, and its most
         representative works; the first period has what set it apart from those after -->
    <ol class="periods" :aria-label="$t('sensePeriods.periodsOf', { sense })">
        <li v-for="(p, i) in periods" :key="p.start" class="period">
            <div class="period-years">{{ years(p) }}</div>
            <div class="period-size">{{ $t("sensePeriods.size", { sentences: p.sentences, authors: p.authors }) }}</div>

            <template v-if="i === 0 && p.characteristic">
                <div class="period-label">{{ $t("sensePeriods.atStart") }}</div>
                <div v-if="!p.characteristic.length" class="fading">{{ $t("sensePeriods.nothingApart") }}</div>
                <div class="chips">
                    <button v-for="w in p.characteristic" :key="`${w.group}-${w.word}`" type="button"
                        class="chip chip-start" :title="startTitle(w)" :style="{ borderColor: color(0.8) }"
                        @click="emit('word', p, w)">{{ wordText(w) }}</button>
                </div>
            </template>
            <template v-else-if="p.new">
                <div class="period-label">{{ $t("sensePeriods.new") }}</div>
                <div class="chips">
                    <button v-for="w in p.new" :key="`${w.group}-${w.word}`" type="button" class="chip"
                        :title="newTitle(w)" :style="{ backgroundColor: color(0.18) }"
                        @click="emit('word', p, w)">{{ wordText(w) }}</button>
                </div>
                <div v-if="p.fading.length" class="fading">
                    <span class="period-label">{{ $t("sensePeriods.fading") }}</span>
                    {{ p.fading.map((w) => wordText(w)).join(", ") }}
                </div>
            </template>

            <div v-if="p.representative?.length" class="works">
                <div class="period-label">{{ $t("sensePeriods.representative") }}</div>
                <button v-for="w in p.representative" :key="w.hits[0]" type="button"
                    class="btn btn-link btn-sm p-0 work-link" @click="emit('work', p, w)">
                    {{ shortAuthor(w.author) }}<template v-if="w.author">, </template><i>{{ shortTitle(w.title) }}</i>
                    {{ " " }}<span class="text-muted">({{ w.year }})</span>
                </button>
            </div>
        </li>
    </ol>
</template>

<script setup>
import { useI18n } from "vue-i18n";
import { senseColor } from "./senseColors.js";

const props = defineProps({
    periods: { type: Array, required: true },
    sense: { type: String, required: true },
    shades: { type: Object, required: true },
    queryWord: { type: String, default: "" },
});
const emit = defineEmits(["word", "work"]);
const { t } = useI18n();

function color(alpha) {
    return senseColor(props.sense, alpha, props.shades);
}

function years(p) {
    return periodYears(p);
}

function wordText(w) {
    return periodWord(w, props.queryWord);
}

function times(r) {
    return r >= 10 ? Math.round(r) : r.toFixed(1);
}

function newTitle(w) {
    const rate = w.ratio === null ? t("sensePeriods.notBefore") : t("sensePeriods.newTitle", { ratio: times(w.ratio) });
    const corpus = w.corpus === null ? "" : ` ${t("sensePeriods.corpusTitle", { corpus: times(w.corpus) })}`;
    return `${rate}${corpus}`;
}

function startTitle(w) {
    return w.ratio === null ? t("sensePeriods.notAfter") : t("sensePeriods.startTitle", { ratio: times(w.ratio) });
}

// The catalogue's author and title, short enough for a column: the name without its dates, the title's first words
function shortAuthor(author) {
    const parts = (author || "").split(",").map((p) => p.trim());
    return parts.slice(0, 2).filter((p, i) => i === 0 || !/\d/.test(p)).join(", ");
}

function shortTitle(title) {
    const clean = (title || "").replace(/\s+/g, " ").trim();
    return clean.length > 48 ? `${clean.slice(0, 46).replace(/[\s,.;:]+\S*$/, "")}…` : clean;
}
</script>

<script>
// Shared with the tabs that open a period's passages
export function periodYears(p) {
    return p.start === p.end ? `${p.start}` : `${p.start}–${p.end}`;
}

export function periodWord(w, queryWord = "") {
    if (w.phrase) return w.phrase;
    if (w.group === 1) return `${w.word} ${queryWord}`;
    if (w.group === 2) return `${queryWord} ${w.word}`;
    return w.word;
}
</script>

<style scoped>
/* Left to right, wrapping onto the next line rather than scrolling: on a narrow screen, one period under another */
.periods {
    display: flex;
    flex-wrap: wrap;
    row-gap: 0.75rem;
    list-style: none;
    padding: 0 0 0.5rem 0;
    margin: 0;
}

.period {
    flex: 1 0 10.5rem;
    max-width: 16rem;
    padding: 0.25rem 0.6rem 0.25rem 0.6rem;
    border-left: 1px solid #dee2e6;
}

.period-years {
    font-weight: 600;
}

.period-size {
    font-size: 0.75rem;
    color: #6c757d;
    margin-bottom: 0.3rem;
}

.period-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    color: #6c757d;
    margin-top: 0.3rem;
}

.chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
    margin-top: 0.15rem;
}

.chip {
    border: 1px solid transparent;
    border-radius: 0.5rem;
    padding: 0.05rem 0.4rem;
    font-size: 0.85rem;
    line-height: 1.3;
    color: #212529;
    background: #f1f3f5;
}

.chip:hover,
.chip:focus-visible {
    border-color: #495057;
}

.chip-start {
    background: #fff;
    border-style: dashed;
}

.fading {
    font-size: 0.8rem;
    color: #6c757d;
    margin-top: 0.3rem;
}

.works {
    margin-top: 0.3rem;
}

.work-link {
    display: block;
    text-align: left;
    font-size: 0.78rem;
    line-height: 1.25;
    margin-top: 0.15rem;
}
</style>
