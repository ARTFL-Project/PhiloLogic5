import { describe, it, expect } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { createGlobalConfig, createMockHttp } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import RisingWords from "../src/components/collocation/RisingWords.vue";
import DistinctivePassagesModal from "../src/components/DistinctivePassagesModal.vue";

const flat = (n, v) => new Array(n).fill(v);
const word = (w, fields = {}) => ({
    word: w, group: 0, onset: 1790, before: 0.01, after: 0.2, new_word: false, corpus_before: 3, corpus_after: 9, ...fields,
});

// A use of two words that rises in 1790, and a word coined then that rises alone
const report = {
    n_hits: 120, n_sentences: 100, n_works: 12, filter_list: [], candidates: 40, rising: 3,
    uses: [
        {
            words: [word("peuple"), word("française", { group: 2, phrase: "révolution française" })], n_words: 2,
            onset: 1790, rises: true, before: 0.01, after: 0.25, sentences: 30, works: 6, sentences_before: 2,
            sources: [{ hit: 21, year: 1770 }, { hit: 22, year: 1788 }],
        },
        {
            words: [word("révolutionnaire", { new_word: true, corpus_before: 0 })], n_words: 1, onset: 1792, rises: true,
            before: 0, after: 0.1, sentences: 12, works: 4, sentences_before: 0, sources: [{ hit: 31, year: 1792 }],
        },
    ],
    time: { start: 1700, end: 1800, share: { value: [flat(101, 0.1), flat(101, 0.05)] } },
};

async function loaded() {
    const http = createMockHttp({
        "get_rising_words.py": report,
        "get_sense_passages.py": { passages: [] },
    });
    const global = createGlobalConfig({
        http,
        route: { name: "collocation", path: "/collocation", query: { q: "lemma:révolution", report: "collocation" } },
        stubs: {
            ProgressSpinner: { template: "<div class='spinner-stub' />" },
            DistinctivePassagesModal: { template: "<div class='modal-stub' />", props: ["groupName", "signature", "passages", "loading", "hasMore", "orderingNote"] },
        },
    });
    const store = useMainStore();
    store.formData = { ...store.formData, q: "lemma:révolution", report: "collocation" };
    const wrapper = mount(RisingWords, { global });
    await flushPromises();
    return { wrapper, http };
}

describe("RisingWords", () => {
    it("shows the uses of several words with their rise, and the words that rise alone", async () => {
        const { wrapper } = await loaded();
        expect(wrapper.text()).toContain("3 of 40 words that go with it rise: 1 emerging uses of several words, and 1 words that rise alone.");
        const cards = wrapper.findAll(".use-card");
        expect(cards).toHaveLength(1);
        expect(cards[0].text()).toContain("Rises in 1790: from 1.0% of the sentences before to 25.0% after");
        expect(cards[0].text()).toContain("révolution française");
        expect(cards[0].text()).toContain("The earliest sentences, from 1770, 1788");
        const alone = wrapper.find(".alone-chip");
        expect(alone.text()).toContain("révolutionnaire");
        expect(alone.text()).toContain("new in the corpus");
    });

    it("reads a use's earliest sentences", async () => {
        const { wrapper, http } = await loaded();
        await wrapper.find(".use-card .btn-link").trigger("click");
        await flushPromises();
        const call = http.get.mock.calls.find(([url]) => url.includes("get_sense_passages.py"));
        expect(call[1].params.hits).toBe("21,22");
        expect(wrapper.findComponent(DistinctivePassagesModal).props("groupName")).toContain("peuple");
    });
});
