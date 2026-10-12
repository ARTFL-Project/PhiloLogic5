import { afterAll, afterEach, describe, it, expect } from "vitest";
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

// jsdom lays nothing out: the box's width is given, the rest measures as zero
let boxWidth = 1000;
const clientWidth = Object.getOwnPropertyDescriptor(Element.prototype, "clientWidth");
Object.defineProperty(Element.prototype, "clientWidth", { configurable: true, get: () => boxWidth });
afterEach(() => {
    boxWidth = 1000;
});
afterAll(() => {
    if (clientWidth) Object.defineProperty(Element.prototype, "clientWidth", clientWidth);
});

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
        expect(wrapper.find(".overview").classes()).toContain("wide");
        const rows = wrapper.findAll(".sense-rows .sense-row");
        expect(rows).toHaveLength(2);
        expect(rows[0].find("button.sense-name").text()).toContain("Sense 1");
        expect(rows[0].findAll(".band .band-area")).toHaveLength(1);
        expect(wrapper.findAll(".tree .branch")).toHaveLength(2);
        const call = http.get.mock.calls.find(([url]) => url.includes("get_sense_periods.py"));
        expect(call[1].params.nodes).toBe("1,2");
        expect(rows[0].findAll(".period-tick")).toHaveLength(1);
        expect(rows[0].findAll(".period-word").map((w) => w.text())).toEqual(["étrange", "assemblée"]);
        wrapper.unmount();
    });

    it("says in words what the band and the periods draw", async () => {
        const { wrapper } = await loaded();
        const name = wrapper.findAll("button.sense-name")[0];
        const about = wrapper.find(`#${name.attributes("aria-describedby")}`);
        expect(about.classes()).toContain("visually-hidden");
        expect(about.text()).toContain("highest around 1700 (60%)");
        expect(about.text()).toContain("1700–1790 (étrange); 1791–1900 (assemblée)");
        expect(wrapper.findAll(".band, .tree, .period-words, .years-row").every((e) => e.attributes("aria-hidden") === "true")).toBe(true);
        wrapper.unmount();
    });

    it("stacks the rows without the tree on a narrow screen", async () => {
        boxWidth = 300;
        const { wrapper } = await loaded();
        expect(wrapper.find(".overview").classes()).not.toContain("wide");
        expect(wrapper.find(".tree").exists()).toBe(false);
        expect(wrapper.find(".tree-cell").exists()).toBe(false);
        expect(wrapper.findAll(".sense-rows .sense-row")).toHaveLength(2);
        expect(wrapper.findAll(".band")).toHaveLength(2);
        wrapper.unmount();
    });

    it("opens a sense's words and periods, and keeps the explanation in a popover", async () => {
        const { wrapper } = await loaded();
        expect(wrapper.find(".info-pop").exists()).toBe(false);
        await wrapper.find(".info-button").trigger("click");
        expect(wrapper.find(".info-pop").text()).toContain("On the left, the tree of the senses");
        await wrapper.find(".info-pop").trigger("keydown", { key: "Escape" });
        expect(wrapper.find(".info-pop").exists()).toBe(false);
        await wrapper.find(".info-button").trigger("click");
        expect(wrapper.find(".info-pop").exists()).toBe(true);
        const names = () => wrapper.findAll("button.sense-name");
        expect(names()[0].attributes("aria-expanded")).toBe("false");
        expect(names()[0].attributes("aria-controls")).toBeUndefined();
        await names()[0].trigger("click");
        expect(names()[0].attributes("aria-expanded")).toBe("true");
        expect(names()[0].attributes("aria-controls")).toBe("sense-overview-panel");
        const panel = wrapper.find("#sense-overview-panel");
        expect(panel.text()).toContain("peuple");
        expect(panel.findAll(".period")).toHaveLength(2);
        expect(panel.text()).toContain("2 periods");
        await names()[1].trigger("click");
        expect(names()[0].attributes("aria-expanded")).toBe("false");
        expect(wrapper.find(".selected-sense").text()).toContain("Too few dated works");
        expect(wrapper.find(".info-pop").exists()).toBe(false); // the click elsewhere closed it
        wrapper.unmount();
    });
});
