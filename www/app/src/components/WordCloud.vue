<template>
    <div>
        <span class="cloud-word text-view" v-for="word in collocCloudWords" :key="word.word"
            :style="getWordCloudStyle(word)" @click="clickHandler(word)">{{ word.collocate }}</span>
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
            default: () => []  // Ensure default empty array
        },
        clickHandler: {
            type: Function,
            required: true
        }
    },
    computed: {
        colorCodes: function () {
            let r, g, b;

            // Check if the color is in RGB format
            const rgbMatch = this.cloudColor.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);

            if (rgbMatch) {
                // Parse RGB format
                r = parseInt(rgbMatch[1]);
                g = parseInt(rgbMatch[2]);
                b = parseInt(rgbMatch[3]);
            } else {
                // Parse as hex
                // 3 digits
                if (this.cloudColor.length == 4) {
                    r = parseInt("0x" + this.cloudColor[1] + this.cloudColor[1]);
                    g = parseInt("0x" + this.cloudColor[2] + this.cloudColor[2]);
                    b = parseInt("0x" + this.cloudColor[3] + this.cloudColor[3]);
                    // 6 digits
                } else if (this.cloudColor.length == 7) {
                    r = parseInt("0x" + this.cloudColor[1] + this.cloudColor[2]);
                    g = parseInt("0x" + this.cloudColor[3] + this.cloudColor[4]);
                    b = parseInt("0x" + this.cloudColor[5] + this.cloudColor[6]);
                }
            }

            let colorCodes = {};
            var step = 0.03;
            for (let i = 0; i < 21; i += 1) {
                // Use Math.max to prevent negative values
                let rLocal = Math.max(0, Math.round(r - r * step * i));
                let gLocal = Math.max(0, Math.round(g - g * step * i));
                let bLocal = Math.max(0, Math.round(b - b * step * i));
                let opacityStep = i * 0.03;
                colorCodes[i] = `rgba(${rLocal}, ${gLocal}, ${bLocal}, ${0.4 + opacityStep})`;
            }
            return colorCodes;
        }
    },
    data() {
        return {
            collocCloudWords: [],
            cloudColor: variables.color
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
                return;  // Don't try to build if no data
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
                });
            }
            weightedWordList.sort(function (a, b) {
                return a.collocate.localeCompare(b.collocate);
            });
            this.collocCloudWords = weightedWordList;
        },
        getWordCloudStyle(word) {
            return `font-size: ${word.weight}rem; color: ${word.color}`;
        },
    }
}
</script>
<style lang="scss" scoped>
@import "../assets/styles/theme.module.scss";

.cloud-word {
    display: inline-block;
    padding: 5px;
    cursor: pointer;
    line-height: initial;
}
</style>