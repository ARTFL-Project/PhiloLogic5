import { describe, it, expect } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { createGlobalConfig, createMockHttp } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import SensePeriods from "../src/components/collocation/SensePeriods.vue";
import DistinctivePassagesModal from "../src/components/DistinctivePassagesModal.vue";

const node = (id, parent, children, share, fields = {}) => ({
    id, parent, children, share, sense: null, rank: null, words: [], before: [], after: [], ...fields,
});
const word = (w, fields = {}) => ({ word: w, group: 0, g2: 30, ratio: 4, corpus: 1.5, hits: [7, 8], ...fields });

// Sense 1 in three periods; sense 2 in one
const report = {
    n_hits: 120, n_sentences: 100, n_works: 30, filter_list: [], detail: 1,
    nodes: [
        node(0, null, [1, 2], 1),
        node(1, 0, [], 0.6, { sense: "1", rank: 1, words: [{ word: "peuple" }, { word: "nation" }] }),
        node(2, 0, [], 0.4, { sense: "2", rank: 2, words: [{ word: "astre" }, { word: "globe" }] }),
    ],
    senses: [
        {
            id: 1, sense: "1", tried: [[2, true], [3, true], [4, false]],
            periods: [
                { start: 1600, end: 1780, sentences: 40, works: 10, authors: 9, characteristic: [word("étrange", { ratio: 9 })],
                    representative: [{ author: "Volney, C.-F. (Constantin-François), 1757-1820.", title: "Les ruines", year: 1771, sentences: 3, hits: [1, 2] }] },
                { start: 1781, end: 1830, sentences: 30, works: 10, authors: 8, new: [word("assemblée"), word("française", { group: 2, phrase: "révolution française", ratio: null })],
                    fading: [word("étrange", { ratio: 0.1 })], representative: [] },
                { start: 1831, end: 1900, sentences: 30, works: 10, authors: 9, new: [word("prolétariat")], fading: [], representative: [] },
            ],
        },
        { id: 2, sense: "2", tried: [[2, false]], periods: [{ start: 1600, end: 1900, sentences: 40, works: 30, authors: 20 }] },
    ],
};

async function loaded() {
    const http = createMockHttp({ "get_sense_periods.py": report, "get_sense_passages.py": { passages: [] } });
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
    const wrapper = mount(SensePeriods, { global });
    await flushPromises();
    return { wrapper, http };
}

describe("SensePeriods", () => {
    it("shows each sense's periods from left to right, with what is new and what fades", async () => {
        const { wrapper } = await loaded();
        const cards = wrapper.findAll(".period-card");
        expect(cards).toHaveLength(2);
        const text = (el) => el.text().replace(/\s+/g, " ");
        expect(text(cards[0])).toContain("Sense 1: peuple, nation");
        expect(cards[0].text()).toContain("3 periods");
        const periods = cards[0].findAll(".period");
        expect(periods.map((p) => p.find(".period-years").text())).toEqual(["1600–1780", "1781–1830", "1831–1900"]);
        expect(periods[0].text()).toContain("At the start");
        expect(periods[0].text()).toContain("étrange");
        expect(periods[1].findAll(".chip").map((c) => c.text())).toEqual(["assemblée", "révolution française"]);
        expect(periods[1].find(".chip").attributes("title")).toBe("×4.0 its rate in the periods before. In the whole corpus, over the same years: ×1.5.");
        expect(periods[1].findAll(".chip")[1].attributes("title")).toContain("Not in the periods before.");
        expect(periods[1].find(".fading").text()).toContain("étrange");
        expect(text(periods[0].find(".work-link"))).toContain("Volney, C.-F. (Constantin-François), Les ruines (1771)");
        expect(cards[1].text()).toContain("one period: no change shows");
    });

    it("reads a word's sentences in a period, and a work's most typical", async () => {
        const { wrapper, http } = await loaded();
        await wrapper.findAll(".period-card")[0].findAll(".period")[1].find(".chip").trigger("click");
        await flushPromises();
        let call = http.get.mock.calls.filter(([url]) => url.includes("get_sense_passages.py")).at(-1);
        expect(call[1].params.hits).toBe("7,8");
        expect(wrapper.findComponent(DistinctivePassagesModal).props("groupName")).toContain("1781–1830 · “assemblée”");
        await wrapper.find(".work-link").trigger("click");
        await flushPromises();
        call = http.get.mock.calls.filter(([url]) => url.includes("get_sense_passages.py")).at(-1);
        expect(call[1].params.hits).toBe("1,2");
    });
});
