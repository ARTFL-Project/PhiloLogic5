import { describe, it, expect } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { createGlobalConfig, createMockHttp } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import SenseTree from "../src/components/collocation/SenseTree.vue";

const word = (w, ratio = 3, n = 10) => ({ word: w, n, ratio, lift: ratio * 2 });
const split = (relative, z = 3) => ({ relative, z, importance: relative, seed: { word: "loi", group: 0 } });
const node = (id, parent, children, share, fields = {}) => ({
    id, parent, children, share, sentences: Math.round(share * 100), words: [], before: [], after: [], hidden: [],
    rarer: [], split: null, reason: null, works: 10, top_work_share: 0.2, examples: [], sense: null, rank: null, ...fields,
});

// The main distinction (0), one 40% as important (2) and one 15% (3); the first view down to 25%: senses 1, 2 and 2.1.
// Then a rest: 7, out of which 2.1's split carves nothing (8 is the rest of the sentences)
const report = {
    n_hits: 120, n_sentences: 100, n_works: 12, folds: 5, evidence: 2, detail: 0.25, finest: 0.1, filter_list: [],
    nodes: [
        node(0, null, [1, 2], 1, { words: [word("loi")], split: split(1) }),
        node(1, 0, [], 0.5, { sense: "1", rank: 1, words: [word("dieu"), word("foi")] }),
        node(2, 0, [3, 6], 0.5, { sense: "2", rank: 2, words: [word("ciel")], split: split(0.4) }),
        node(3, 2, [4, 5], 0.3, { sense: "2.1", rank: 3, words: [word("mer")], split: split(0.15) }),
        node(4, 3, [], 0.2, { sense: "2.1", rank: 3, words: [word("port"), word("navire")] }),
        node(5, 3, [], 0.1, { sense: "2.1.1", rank: 4, words: [word("vague")] }),
        node(6, 2, [], 0.2, { sense: "2", rank: 2, before: [{ ...word("son"), phrase: "sa raison" }] }),
    ],
    time: null,
};

function mountTree(response = report) {
    const http = createMockHttp({ "get_word_senses.py": response });
    const global = createGlobalConfig({
        http,
        route: { name: "collocation", path: "/collocation", query: { q: "lemma:raison", report: "collocation" } },
        stubs: { ProgressSpinner: { template: "<div class='spinner-stub' />" } },
    });
    const store = useMainStore();
    store.formData = { ...store.formData, q: "lemma:raison", report: "collocation" };
    return { wrapper: mount(SenseTree, { global }), http };
}

async function loaded(response = report) {
    const mounted = mountTree(response);
    mounted.wrapper.vm.runDetection();
    await flushPromises();
    return mounted;
}

const atCut = (wrapper) => wrapper.findAll(".at-cut").map((b) => b.text().replace(/\s+/g, " ").trim());
const asBlocks = (wrapper) => wrapper.findAll(".btn-group button")[1].trigger("click");

describe("SenseTree", () => {
    it("draws the senses at the cut as branches, as thick as their shares, forking where they split", async () => {
        const { wrapper } = await loaded();
        const rows = () => wrapper.findAll("[id^='sense-row-']").map((t) => t.text().replace(/\s+/g, " ").trim());
        expect(rows()).toEqual([
            "Sense 1 · 50% · dieu, foi", "Sense 2 · 20% · sa raison", "Sense 2.1 · 30% · mer",
        ]);
        const width = (id) => Number(wrapper.find(`#sense-branch-${id}`).attributes("stroke-width"));
        expect(width(1)).toBeGreaterThan(width(3));
        expect(wrapper.find("#sense-branch-4").exists()).toBe(false); // finer than the cut: not drawn
        await wrapper.find("#sense-tree-cut").setValue(3);
        expect(rows().map((r) => r.split(" ·")[0])).toEqual(["Sense 1", "Sense 2", "Sense 2.1", "Sense 2.1.1"]);
        await wrapper.find("#sense-row-6").trigger("click");
        expect(wrapper.find(".selected-sense").text()).toContain("sa raison");
    });

    it("draws the senses carved out of a sense in shades of its color", async () => {
        const { wrapper } = await loaded();
        const hsl = (id) => wrapper.find(`#sense-branch-${id}`).attributes("stroke").match(/hsla\((\d+), (\d+)%, (\d+)%/).slice(1).map(Number);
        const [h1] = hsl(1);
        const [h2, , l2] = hsl(6); // sense 2
        const [h21, , l21] = hsl(3); // sense 2.1
        expect(h21).toBe(h2);
        expect(l21).not.toBe(l2);
        expect(h1).not.toBe(h2);
        expect(wrapper.find("line.cut").exists()).toBe(false);
    });

    it("draws a block for each sense, as wide as its share", async () => {
        const { wrapper } = await loaded();
        await asBlocks(wrapper);
        expect(wrapper.findAll("rect[id^='sense-block-']")).toHaveLength(report.nodes.length - 1);
        const width = (id) => Number(wrapper.find(`#sense-block-${id}`).attributes("width"));
        expect(Math.abs(width(1) - width(2))).toBeLessThan(1); // half and half (each a pixel short, for the gap)
        expect(Math.abs(width(3) + width(6) - width(2))).toBeLessThanOrEqual(1);
    });

    it("cuts the tree at the first view, and lists the senses the cut crosses", async () => {
        const { wrapper } = await loaded();
        await asBlocks(wrapper);
        expect(wrapper.find("#sense-tree-cut").element.value).toBe("2");
        expect(atCut(wrapper)).toEqual([
            "Sense 1: dieu, foi · 50% of the sentences",
            "Sense 2: sa raison · 20% of the sentences",
            "Sense 2.1: mer · 30% of the sentences",
        ]);
        expect(wrapper.text()).toContain("3 senses: the distinctions at least 40% as important as the main one.");
        expect(wrapper.find("#sense-block-3").classes()).toContain("shown");
        expect(wrapper.find("#sense-block-4").classes()).not.toContain("shown");
    });

    it("moves the cut with the slider, and back to the first view", async () => {
        const { wrapper } = await loaded();
        await wrapper.find("#sense-tree-cut").setValue(3);
        expect(atCut(wrapper).map((s) => s.split(":")[0])).toEqual(["Sense 1", "Sense 2", "Sense 2.1", "Sense 2.1.1"]);
        await wrapper.find("#sense-tree-cut").setValue(0);
        expect(atCut(wrapper)).toEqual([]);
        expect(wrapper.text()).toContain("1 sense: no distinction shown.");
        await wrapper.find("#sense-tree-cut").element.parentElement.querySelector("button.btn-link").click();
        await flushPromises();
        expect(wrapper.find("#sense-tree-cut").element.value).toBe("2");
    });

    it("shows the words of a sense picked in the list", async () => {
        const { wrapper } = await loaded();
        await wrapper.findAll(".at-cut")[1].trigger("click");
        const picked = wrapper.find(".selected-sense");
        expect(picked.text()).toContain("Sense 2");
        expect(picked.text()).toContain("sa raison");
    });

    it("shows the rest of the sentences grey, as no sense", async () => {
        const withRest = {
            ...report,
            detail: 1,
            nodes: [
                node(0, null, [1, 2], 1, { words: [word("loi")], split: split(1) }),
                node(1, 0, [], 0.6),
                node(2, 0, [], 0.4, { sense: "1", rank: 1, words: [word("ciel")] }),
            ],
        };
        const { wrapper } = await loaded(withRest);
        await asBlocks(wrapper);
        expect(atCut(wrapper)).toEqual(["Sense 1: ciel · 40% of the sentences", "the rest · 60% of the sentences"]);
        expect(wrapper.find("#sense-block-1").attributes("fill")).toContain("hsla(0, 0%, 62%");
    });
});
