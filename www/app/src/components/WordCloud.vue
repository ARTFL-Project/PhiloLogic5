<template>
    <div class="word-cloud-container" role="region" :aria-label="wordCloudAriaLabel"
        aria-describedby="word-cloud-instructions">
        <!-- Instructions for screen readers -->
        <div id="word-cloud-instructions" class="visually-hidden">
            {{ $t("wordCloud.instructions") }}
        </div>

        <!-- Word cloud items -->
        <button v-for="(word, index) in collocCloudWords" :key="word.collocate" class="cloud-word text-view"
            :style="getWordCloudStyle(word)" @click="clickHandler(word)" @keydown="handleKeydown($event, word, index)"
            :aria-label="getWordAriaLabel(word)" :title="getWordTitle(word)" type="button">
            {{ word.collocate }}
        </button>

        <!-- Hidden data table for screen readers -->
        <div class="visually-hidden">
            <table role="table" aria-label="Word cloud data table">
                <caption>{{ $t("wordCloud.tableCaption") }}</caption>
                <thead>
                    <tr>
                        <th scope="col">{{ $t("wordCloud.word") }}</th>
                        <th scope="col">{{ $t("wordCloud.frequency") }}</th>
                        <th scope="col">{{ $t("wordCloud.significance") }}</th>
                    </tr>
                </thead>
                <tbody>
                    <tr v-for="word in sortedWordsForTable" :key="word.collocate">
                        <td>{{ word.collocate }}</td>
                        <td>{{ getOriginalCount(word) }}</td>
                        <td>{{ getSignificanceLevel(word) }}</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- Status for screen readers -->
        <div class="visually-hidden" aria-live="polite" aria-atomic="true" id="word-cloud-status">
            {{ statusMessage }}
        </div>
    </div>
</template>

<script>
import variables from "../assets/styles/theme.module.scss";

export default {
    name: 'WordCloud',
    props: {
        wordWeights: {
            type: Array,
            required: true,
            default: () => []
        },
        clickHandler: {
            type: Function,
            required: true
        }
    },
    computed: {
        wordCloudAriaLabel() {
            const wordCount = this.collocCloudWords.length;
            return this.$t("wordCloud.ariaLabel", { count: wordCount });
        },

        sortedWordsForTable() {
            return [...this.collocCloudWords].sort((a, b) => {
                // Sort by original frequency (highest first)
                const aCount = this.getOriginalCount(a);
                const bCount = this.getOriginalCount(b);
                return bCount - aCount;
            });
        },

        statusMessage() {
            if (this.collocCloudWords.length === 0) {
                return this.$t("wordCloud.noWords");
            }
            return this.$t("wordCloud.wordsLoaded", { count: this.collocCloudWords.length });
        },

        colorCodes() {
            let r, g, b;

            // Use theme color with fallback
            const color = this.cloudColor || variables.color || '#8e3232';

            // Check if the color is in RGB format
            const rgbMatch = color.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);

            if (rgbMatch) {
                r = parseInt(rgbMatch[1]);
                g = parseInt(rgbMatch[2]);
                b = parseInt(rgbMatch[3]);
            } else {
                // Parse as hex
                if (color.length == 4) {
                    r = parseInt("0x" + color[1] + color[1]);
                    g = parseInt("0x" + color[2] + color[2]);
                    b = parseInt("0x" + color[3] + color[3]);
                } else if (color.length == 7) {
                    r = parseInt("0x" + color[1] + color[2]);
                    g = parseInt("0x" + color[3] + color[4]);
                    b = parseInt("0x" + color[5] + color[6]);
                }
            }

            let colorCodes = {};
            var step = 0.03;
            for (let i = 0; i < 21; i += 1) {
                let rLocal = Math.max(0, Math.round(r - r * step * i));
                let gLocal = Math.max(0, Math.round(g - g * step * i));
                let bLocal = Math.max(0, Math.round(b - b * step * i));
                let opacityStep = i * 0.03;
                colorCodes[i] = `rgba(${rLocal}, ${gLocal}, ${bLocal}, ${0.8 + opacityStep})`;
            }
            return colorCodes;
        }
    },
    data() {
        return {
            collocCloudWords: [],
            cloudColor: variables.color || '#8e3232', // Use theme color with fallback
            selectedWordIndex: 0
        }
    },
    mounted() {
        this.buildWordCloud();
    },
    watch: {
        wordWeights: function () {
            this.buildWordCloud();
        }
    },
    methods: {
        buildWordCloud() {
            if (!this.wordWeights || this.wordWeights.length === 0) {
                return;
            }
            let lowestValue = this.wordWeights[this.wordWeights.length - 1].count;
            let higestValue = this.wordWeights[0].count;
            let diff = higestValue - lowestValue;
            let coeff = diff / 20;

            var adjustWeight = function (count) {
                let adjustedCount = count - lowestValue;
                let adjustedWeight = Math.round(adjustedCount / coeff);
                adjustedWeight = parseInt(adjustedWeight);
                return adjustedWeight;
            };

            let weightedWordList = [];
            for (let wordObject of this.wordWeights) {
                let adjustedWeight = adjustWeight(wordObject.count);
                weightedWordList.push({
                    collocate: wordObject.collocate,
                    surfaceForm: wordObject.surfaceForm,
                    weight: 1 + adjustedWeight / 10,
                    color: this.colorCodes[adjustedWeight],
                    originalCount: wordObject.count,
                    adjustedWeight: adjustedWeight
                });
            }
            weightedWordList.sort(function (a, b) {
                return a.collocate.localeCompare(b.collocate);
            });
            this.collocCloudWords = weightedWordList;
            this.selectedWordIndex = 0;
        },

        getWordCloudStyle(word) {
            return `font-size: ${word.weight}rem; color: ${word.color}`;
        },

        getWordAriaLabel(word) {
            const frequency = this.getOriginalCount(word);
            const significance = this.getSignificanceLevel(word);
            return this.$t("wordCloud.wordAriaLabel", {
                word: word.collocate,
                frequency: frequency,
                significance: significance
            });
        },

        getWordTitle(word) {
            const frequency = this.getOriginalCount(word);
            return this.$t("wordCloud.wordTooltip", {
                word: word.collocate,
                frequency: frequency
            });
        },

        getOriginalCount(word) {
            return word.originalCount || 0;
        },

        getSignificanceLevel(word) {
            const weight = word.adjustedWeight || 0;
            if (weight >= 15) return this.$t("wordCloud.veryHigh");
            if (weight >= 10) return this.$t("wordCloud.high");
            if (weight >= 5) return this.$t("wordCloud.medium");
            return this.$t("wordCloud.low");
        },

        handleKeydown(event, word, index) {
            const words = this.collocCloudWords;
            let newIndex = index;

            switch (event.key) {
                case 'ArrowRight':
                    newIndex = (index + 1) % words.length;
                    event.preventDefault();
                    break;
                case 'ArrowLeft':
                    newIndex = (index - 1 + words.length) % words.length;
                    event.preventDefault();
                    break;
                case 'ArrowDown':
                    // Move to next "row" (approximate)
                    newIndex = Math.min(index + 5, words.length - 1);
                    event.preventDefault();
                    break;
                case 'ArrowUp':
                    // Move to previous "row" (approximate)
                    newIndex = Math.max(index - 5, 0);
                    event.preventDefault();
                    break;
                case 'Home':
                    newIndex = 0;
                    event.preventDefault();
                    break;
                case 'End':
                    newIndex = words.length - 1;
                    event.preventDefault();
                    break;
                case 'Enter':
                case ' ':
                    this.clickHandler(word);
                    event.preventDefault();
                    return;
            }

            // Focus the new word
            if (newIndex !== index) {
                this.selectedWordIndex = newIndex;
                this.$nextTick(() => {
                    const wordButtons = this.$el.querySelectorAll('.cloud-word');
                    if (wordButtons[newIndex]) {
                        wordButtons[newIndex].focus();
                        this.announceWordSelection(words[newIndex]);
                    }
                });
            }
        },

        announceWordSelection(word) {
            const announcement = this.getWordAriaLabel(word);
            this.announceToScreenReader(announcement);
        },

        announceToScreenReader(message) {
            const announcer = document.createElement('div');
            announcer.setAttribute('aria-live', 'assertive');
            announcer.setAttribute('aria-atomic', 'true');
            announcer.className = 'visually-hidden';
            announcer.textContent = message;
            document.body.appendChild(announcer);

            setTimeout(() => {
                if (document.body.contains(announcer)) {
                    document.body.removeChild(announcer);
                }
            }, 1000);
        }
    }
}
</script>

<style lang="scss" scoped>
@use "../assets/styles/theme.module.scss" as theme;

.word-cloud-container {
    position: relative;
    line-height: 1.4;
}

.cloud-word {
    display: inline-block;
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