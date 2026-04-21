import { describe, it, expect, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { nextTick } from "vue";
import { createI18n } from "vue-i18n";
import WordCloud from "../src/components/WordCloud.vue";

const i18n = createI18n({
    legacy: false,
    locale: "en",
    messages: {
        en: {
            wordCloud: {
                ariaLabel: "Word cloud containing {count} words",
                instructions: "Use arrow keys",
                tableCaption: "Word cloud data",
                word: "Word", frequency: "Frequency", significance: "Significance",
                wordAriaLabel: "{word}, appears {frequency} times, {significance} significance",
                wordTooltip: "{word} (appears {frequency} times)",
                noWords: "No words available",
                wordsLoaded: "{count} words loaded",
                veryHigh: "very high", high: "high", medium: "medium", low: "low",
            },
        },
    },
});

const sampleWords = [
    { collocate: "power", surfaceForm: "power", count: 100 },
    { collocate: "freedom", surfaceForm: "freedom", count: 50 },
    { collocate: "people", surfaceForm: "people", count: 25 },
];

async function mountWordCloud(props = {}) {
    const wrapper = mount(WordCloud, {
        props: { wordWeights: [], clickHandler: vi.fn(), label: "test", ...props },
        global: { plugins: [i18n] },
    });
    await nextTick();
    return wrapper;
}

describe("WordCloud", () => {
    // --- Rendering ---
    it("renders no word buttons when empty", async () => {
        const wrapper = await mountWordCloud();
        expect(wrapper.findAll(".cloud-word").length).toBe(0);
    });

    it("renders word buttons from wordWeights", async () => {
        const wrapper = await mountWordCloud({ wordWeights: sampleWords });
        expect(wrapper.findAll(".cloud-word").length).toBe(3);
    });

    it("has accessible button type", async () => {
        const wrapper = await mountWordCloud({ wordWeights: sampleWords });
        expect(wrapper.find(".cloud-word").attributes("type")).toBe("button");
    });

    it("sets aria-label on word buttons", async () => {
        const wrapper = await mountWordCloud({ wordWeights: sampleWords });
        expect(wrapper.find(".cloud-word").attributes("aria-label")).toBeTruthy();
    });

    // --- @click="clickHandler(word)" ---
    it("calls click handler with word object on click", async () => {
        const handler = vi.fn();
        const wrapper = await mountWordCloud({ wordWeights: sampleWords, clickHandler: handler });
        await wrapper.find(".cloud-word").trigger("click");
        expect(handler).toHaveBeenCalledWith(expect.objectContaining({ collocate: expect.any(String) }));
    });

    // --- @keydown="handleKeydown($event, word, index)" ---
    it("navigates to next word on ArrowRight", async () => {
        const wrapper = await mountWordCloud({ wordWeights: sampleWords });
        const firstWord = wrapper.findAll(".cloud-word")[0];
        await firstWord.trigger("keydown", { key: "ArrowRight" });
        await nextTick();
        expect(wrapper.vm.selectedWordIndex).toBe(1);
    });

    it("navigates to previous word on ArrowLeft", async () => {
        const wrapper = await mountWordCloud({ wordWeights: sampleWords });
        wrapper.vm.selectedWordIndex = 2;
        const lastWord = wrapper.findAll(".cloud-word")[2];
        await lastWord.trigger("keydown", { key: "ArrowLeft" });
        await nextTick();
        expect(wrapper.vm.selectedWordIndex).toBe(1);
    });

    it("navigates to first word on Home", async () => {
        const wrapper = await mountWordCloud({ wordWeights: sampleWords });
        wrapper.vm.selectedWordIndex = 2;
        const word = wrapper.findAll(".cloud-word")[2];
        await word.trigger("keydown", { key: "Home" });
        await nextTick();
        expect(wrapper.vm.selectedWordIndex).toBe(0);
    });

    it("navigates to last word on End", async () => {
        const wrapper = await mountWordCloud({ wordWeights: sampleWords });
        const word = wrapper.findAll(".cloud-word")[0];
        await word.trigger("keydown", { key: "End" });
        await nextTick();
        expect(wrapper.vm.selectedWordIndex).toBe(2);
    });

    it("calls click handler on Enter key", async () => {
        const handler = vi.fn();
        const wrapper = await mountWordCloud({ wordWeights: sampleWords, clickHandler: handler });
        await wrapper.find(".cloud-word").trigger("keydown", { key: "Enter" });
        expect(handler).toHaveBeenCalled();
    });

    it("calls click handler on Space key", async () => {
        const handler = vi.fn();
        const wrapper = await mountWordCloud({ wordWeights: sampleWords, clickHandler: handler });
        await wrapper.find(".cloud-word").trigger("keydown", { key: " " });
        expect(handler).toHaveBeenCalled();
    });

    // --- Word order: sorted alphabetically ---
    it("sorts words alphabetically", async () => {
        const wrapper = await mountWordCloud({ wordWeights: sampleWords });
        const words = wrapper.findAll(".cloud-word");
        const labels = words.map(w => w.text().trim());
        const sorted = [...labels].sort();
        expect(labels).toEqual(sorted);
    });
});
