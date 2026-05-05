<template>
    <div ref="rootRef" class="word-cloud-container" role="region" :aria-label="wordCloudAriaLabel"
        aria-describedby="word-cloud-instructions">
        <!-- Instructions for screen readers -->
        <div :id="`word-cloud-instructions-${label}`" class="visually-hidden">
            {{ $t("wordCloud.instructions") }}
        </div>

        <!-- Word cloud items -->
        <button v-for="(word, index) in collocCloudWords" :key="word.collocate" class="cloud-word text-view"
            :style="getWordCloudStyle(word)" @click="clickHandler(word)" @keydown="handleKeydown($event, word, index)"
            :aria-label="getWordAriaLabel(word)" :title="getWordTitle(word)" type="button">
            {{ word.collocate }}
        </button>

        <!-- Status for screen readers -->
        <div class="visually-hidden" aria-live="polite" aria-atomic="true" id="word-cloud-status">
            {{ statusMessage }}
        </div>
    </div>
</template>

<script setup>
import { computed, nextTick, onMounted, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import variables from "../assets/styles/theme.module.scss";

const props = defineProps({
    wordWeights: { type: Array, required: true, default: () => [] },
    clickHandler: { type: Function, required: true },
    label: { type: String, required: false, default: "default" },
});

const { t } = useI18n();

const rootRef = ref(null);
const collocCloudWords = ref([]);
const cloudColor = ref(variables.color || "#8e3232");
const selectedWordIndex = ref(0);

const wordCloudAriaLabel = computed(() =>
    t("wordCloud.ariaLabel", { count: collocCloudWords.value.length })
);

const statusMessage = computed(() => {
    if (collocCloudWords.value.length === 0) return t("wordCloud.noWords");
    return t("wordCloud.wordsLoaded", { count: collocCloudWords.value.length });
});

const colorCodes = computed(() => {
    let r, g, b;
    const color = cloudColor.value || variables.color || "#8e3232";
    const rgbMatch = color.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);

    if (rgbMatch) {
        r = parseInt(rgbMatch[1]);
        g = parseInt(rgbMatch[2]);
        b = parseInt(rgbMatch[3]);
    } else if (color.length === 4) {
        r = parseInt("0x" + color[1] + color[1]);
        g = parseInt("0x" + color[2] + color[2]);
        b = parseInt("0x" + color[3] + color[3]);
    } else if (color.length === 7) {
        r = parseInt("0x" + color[1] + color[2]);
        g = parseInt("0x" + color[3] + color[4]);
        b = parseInt("0x" + color[5] + color[6]);
    }

    const codes = {};
    const step = 0.03;
    for (let i = 0; i < 21; i += 1) {
        const rLocal = Math.max(0, Math.round(r - r * step * i));
        const gLocal = Math.max(0, Math.round(g - g * step * i));
        const bLocal = Math.max(0, Math.round(b - b * step * i));
        // Start at 0.82 to meet WCAG AA contrast requirement (4.5:1)
        codes[i] = `rgba(${rLocal}, ${gLocal}, ${bLocal}, ${0.82 + i * 0.03})`;
    }
    return codes;
});

function buildWordCloud() {
    if (!props.wordWeights || props.wordWeights.length === 0) return;

    const lowestValue = props.wordWeights[props.wordWeights.length - 1].count;
    const highestValue = props.wordWeights[0].count;
    const coeff = (highestValue - lowestValue) / 20;

    const adjustWeight = (count) => parseInt(Math.round((count - lowestValue) / coeff));

    const weightedWordList = props.wordWeights.map((wordObject) => {
        const adjustedWeight = adjustWeight(wordObject.count);
        return {
            collocate: wordObject.collocate,
            surfaceForm: wordObject.surfaceForm,
            weight: 1.1 + adjustedWeight / 10,
            color: colorCodes.value[adjustedWeight],
            originalCount: wordObject.count,
            adjustedWeight,
        };
    });
    weightedWordList.sort((a, b) => a.collocate.localeCompare(b.collocate));
    collocCloudWords.value = weightedWordList;
    selectedWordIndex.value = 0;
}

function getWordCloudStyle(word) {
    return `font-size: ${word.weight}rem; color: ${word.color}`;
}

function getOriginalCount(word) {
    return word.originalCount || 0;
}

function getSignificanceLevel(word) {
    const weight = word.adjustedWeight || 0;
    if (weight >= 15) return t("wordCloud.veryHigh");
    if (weight >= 10) return t("wordCloud.high");
    if (weight >= 5) return t("wordCloud.medium");
    return t("wordCloud.low");
}

function getWordAriaLabel(word) {
    return t("wordCloud.wordAriaLabel", {
        word: word.collocate,
        frequency: getOriginalCount(word),
        significance: getSignificanceLevel(word),
    });
}

function getWordTitle(word) {
    return t("wordCloud.wordTooltip", {
        word: word.collocate,
        frequency: getOriginalCount(word),
    });
}

function announceToScreenReader(message) {
    const announcer = document.createElement("div");
    announcer.setAttribute("aria-live", "assertive");
    announcer.setAttribute("aria-atomic", "true");
    announcer.className = "visually-hidden";
    announcer.textContent = message;
    document.body.appendChild(announcer);

    setTimeout(() => {
        if (document.body.contains(announcer)) {
            document.body.removeChild(announcer);
        }
    }, 1000);
}

function announceWordSelection(word) {
    announceToScreenReader(getWordAriaLabel(word));
}

function handleKeydown(event, word, index) {
    const words = collocCloudWords.value;
    let newIndex = index;

    switch (event.key) {
        case "ArrowRight":
            newIndex = (index + 1) % words.length;
            event.preventDefault();
            break;
        case "ArrowLeft":
            newIndex = (index - 1 + words.length) % words.length;
            event.preventDefault();
            break;
        case "ArrowDown":
            // Move to next "row" (approximate)
            newIndex = Math.min(index + 5, words.length - 1);
            event.preventDefault();
            break;
        case "ArrowUp":
            // Move to previous "row" (approximate)
            newIndex = Math.max(index - 5, 0);
            event.preventDefault();
            break;
        case "Home":
            newIndex = 0;
            event.preventDefault();
            break;
        case "End":
            newIndex = words.length - 1;
            event.preventDefault();
            break;
        case "Enter":
        case " ":
            props.clickHandler(word);
            event.preventDefault();
            return;
    }

    if (newIndex !== index) {
        selectedWordIndex.value = newIndex;
        nextTick(() => {
            const wordButtons = rootRef.value?.querySelectorAll(".cloud-word");
            if (wordButtons && wordButtons[newIndex]) {
                wordButtons[newIndex].focus();
                announceWordSelection(words[newIndex]);
            }
        });
    }
}

watch(() => props.wordWeights, buildWordCloud);
onMounted(buildWordCloud);
</script>

<style lang="scss" scoped>
@use "../assets/styles/theme.module.scss" as theme;

.word-cloud-container {
    position: relative;
    line-height: 1.4;
}

.cloud-word {
    display: inline-block;
    min-width: 24px;
    padding: 5px;
    cursor: pointer;
    line-height: initial;
    border: none;
    background: transparent;
    border-radius: 5px;
    transition: all 0.2s ease;
}

.cloud-word:hover {
    background-color: rgba(theme.$link-color, 0.08);
    transform: scale(1.10);
}

.cloud-word:focus {
    outline: 2px solid theme.$link-color;
    outline-offset: 2px;
    background-color: rgba(theme.$link-color, 0.05);
}

.cloud-word:active {
    transform: scale(0.95);
}

/* Ensure proper spacing */
.cloud-word+.cloud-word {
    margin-left: 1px;
}

.visually-hidden {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
}
</style>