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
            required: true
        },
        clickHandler: {
            type: Function,
            required: true
        }
    },
    computed: {
        colorCodes: function () {
            let r = 0,
                g = 0,
                b = 0;
            // 3 digits
            if (this.cloudColor.length == 4) {
                r = "0x" + this.cloudColor[1] + this.cloudColor[1];
                g = "0x" + this.cloudColor[2] + this.cloudColor[2];
                b = "0x" + this.cloudColor[3] + this.cloudColor[3];

                // 6 digits
            } else if (this.cloudColor.length == 7) {
                r = "0x" + this.cloudColor[1] + this.cloudColor[2];
                g = "0x" + this.cloudColor[3] + this.cloudColor[4];
                b = "0x" + this.cloudColor[5] + this.cloudColor[6];
            }
            let colorCodes = {};
            var step = 0.03;
            for (let i = 0; i < 21; i += 1) {
                let rLocal = r - r * step * i;
                let gLocal = g - g * step * i;
                let bLocal = b - b * step * i;
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