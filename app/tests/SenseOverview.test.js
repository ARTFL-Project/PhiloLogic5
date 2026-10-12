import { describe, it, expect } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { createGlobalConfig, createMockHttp } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import SenseOverview from "../src/components/collocation/SenseOverview.vue";

const word = (w) => ({ word: w, n: 10, ratio: 3, lift: 6, hits: [3, 4] });
const node = (id, parent, children, share, fields = {}) => ({
    id, parent, children, share, sentences: Math.round(share * 100), words: [], before: [], after: [], hidden: [],
    rarer: [], split: null, reason: null, works: 10, top_work_share: 0.2, examples: [], sense: null, rank: null, ...fields,
});
const flat = (n, v) => new Array(n).fill(v);

const senses = {
    n_hits: 120, n_sentences: 100, n_works: 30, detail: 1, finest: 0.1, filter_list: [],
    nodes: [
        node(0, null, [1, 2], 1, { split: { relative: 1, z: 3, importance: 1, seed: { word: "loi", group: 0 } } }),
        node(1, 0, [], 0.6, { sense: "1", rank: 1, words: [word("peuple"), word("nation")] }),
        node(2, 0, [], 0.4, { sense: "2", rank: 2, words: [word("astre"), word("globe")] }),
    ],
    time: {
        start: 1700, end: 1900, sentences: flat(201, 1), works: flat(201, 1),
        share: { smoothness: 1, value: [flat(201, 1), flat(201, 0.6), flat(201, 0.4)], low: [], high: [] }, rate: null,
    },
};

const periods = {
    n_hits: 120, n_sentences: 100, n_works: 30, nodes: senses.nodes,
    senses: [
        {
            id: 1, sense: "1", tried: [[2, true], [3, false]],
            periods: [
                { start: 1700, end: 1790, sentences: 50, works: 15, authors: 12, characteristic: [{ word: "étrange", group: 0, ratio: 5, hits: [1] }], representative: [] },
                { start: 1791, end: 1900, sentences: 50, works: 15, authors: 13, new: [{ word: "assemblée", group: 0, ratio: 4, corpus: 1.2, hits: [5] }], fading: [], representative: [] },
            ],
        },
    ],
};

async function loaded() {
    const http = createMockHttp({ "get_word_senses.py": senses, "get_sense_periods.py": periods, "get_sense_passages.py": { passages: [] } });
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
    const wrapper = mount(SenseOverview, { global, attachTo: document.body });
    await flushPromises();
    await new Promise((r) => setTimeout(r, 0));
    await flushPromises();
    return { wrapper, http };
}

describe("SenseOverview", () => {
    it("draws the tree with each sense's course over the years, cut into its periods", async () => {
        const { wrapper, http } = await loaded();
        const rows = wrapper.findAll(".sense-row");
        expect(rows).toHaveLength(2);
        expect(rows[0].text()).toContain("Sense 1");
        expect(rows[0].findAll("path")).toHaveLength(1); // its band
        const call = http.get.mock.calls.find(([url]) => url.includes("get_sense_periods.py"));
        expect(call[1].params.nodes).toBe("1,2");
        expect(rows[0].findAll(".period-tick")).toHaveLength(1);
        expect(rows[0].findAll(".period-word").map((w) => w.text())).toEqual(["étrange", "assemblée"]);
        wrapper.unmount();
    });

    it("opens a sense's words and periods, and keeps the explanation in a popover", async () => {
        const { wrapper } = await loaded();
        expect(wrapper.find(".info-pop").exists()).toBe(false);
        await wrapper.find(".info-button").trigger("click");
        expect(wrapper.find(".info-pop").text()).toContain("On the left, the tree of the senses");
        await wrapper.findAll(".sense-row")[0].trigger("click");
        const panel = wrapper.find(".selected-sense");
        expect(panel.text()).toContain("peuple");
        expect(panel.findAll(".period")).toHaveLength(2);
        expect(panel.text()).toContain("2 periods");
        await wrapper.findAll(".sense-row")[1].trigger("click");
        expect(wrapper.find(".selected-sense").text()).toContain("Too few dated works");
        wrapper.unmount();
    });
});
