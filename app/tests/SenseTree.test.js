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

// The main distinction (0), one 40% as important (2) and one 15% (3); the first view down to 25%: senses 1, 2 and 2.1
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

async function loaded(response = report) {
    const http = createMockHttp({ "get_word_senses.py": response });
    const global = createGlobalConfig({
        http,
        route: { name: "collocation", path: "/collocation", query: { q: "lemma:raison", report: "collocation" } },
        stubs: { ProgressSpinner: { template: "<div class='spinner-stub' />" } },
    });
    const store = useMainStore();
    store.formData = { ...store.formData, q: "lemma:raison", report: "collocation" };
    const wrapper = mount(SenseTree, { global });
    await flushPromises();
    return wrapper;
}

const rows = (wrapper) => wrapper.findAll("[id^='sense-row-']").map((t) => t.text().replace(/\s+/g, " ").trim());

describe("SenseTree", () => {
    it("draws the senses shown as branches, as thick as their shares, at the first view", async () => {
        const wrapper = await loaded();
        expect(wrapper.find("#sense-tree-detail").element.value).toBe("2");
        expect(rows(wrapper)).toEqual(["Sense 1 · 50% · dieu, foi", "Sense 2 · 20% · sa raison", "Sense 2.1 · 30% · mer"]);
        expect(wrapper.text()).toContain("3 senses: the distinctions at least 40% as important as the main one.");
        const width = (id) => Number(wrapper.find(`#sense-branch-${id}`).attributes("stroke-width"));
        expect(width(1)).toBeGreaterThan(width(3));
        expect(wrapper.find("#sense-branch-4").exists()).toBe(false); // finer than the detail: not drawn
        expect(wrapper.find("line.cut").exists()).toBe(false);
        expect(wrapper.find("rect").exists()).toBe(false); // no blocks
    });

    it("grows and shrinks with the detail, and goes back to the first view", async () => {
        const wrapper = await loaded();
        await wrapper.find("#sense-tree-detail").setValue(3);
        expect(rows(wrapper).map((r) => r.split(" ·")[0])).toEqual(["Sense 1", "Sense 2", "Sense 2.1", "Sense 2.1.1"]);
        await wrapper.find("#sense-tree-detail").setValue(0);
        expect(rows(wrapper)).toEqual(["the rest · 100%"]);
        expect(wrapper.text()).toContain("1 sense: no distinction shown.");
        await wrapper.find("button.btn-link").trigger("click");
        expect(wrapper.find("#sense-tree-detail").element.value).toBe("2");
    });

    it("shows the words of a sense picked by click or by keyboard", async () => {
        const wrapper = await loaded();
        await wrapper.find("#sense-row-6").element.parentElement.dispatchEvent(new Event("click"));
        await flushPromises();
        expect(wrapper.find(".selected-sense").text()).toContain("sa raison");
        const row = wrapper.findAll(".row-label")[0];
        expect(row.attributes("tabindex")).toBe("0");
        await row.trigger("keydown", { key: "Enter" });
        expect(wrapper.find(".selected-sense").text()).toContain("dieu");
    });

    it("finds a word among the senses, and marks the rows of those shown", async () => {
        const wrapper = await loaded();
        await wrapper.find("#sense-tree-find").setValue("mer");
        expect(wrapper.text()).toContain("“mer” marks Sense 2.1 (×3.0).");
        expect(wrapper.findAll(".row-label.found")).toHaveLength(1);
        expect(wrapper.find(".row-label.found").text()).toContain("Sense 2.1");
        await wrapper.find("#sense-tree-find").setValue("sa"); // the form next to the query, for the lemma son
        expect(wrapper.text()).toContain("“sa” marks Sense 2 (×3.0).");
    });

    it("draws the senses carved out of a sense in shades of its color", async () => {
        const wrapper = await loaded();
        const hsl = (id) => wrapper.find(`#sense-branch-${id}`).attributes("stroke").match(/hsla\((\d+), (\d+)%, ([\d.]+)%/).slice(1).map(Number);
        const [h1] = hsl(1);
        const [h2, , l2] = hsl(6); // sense 2
        const [h21, , l21] = hsl(3); // sense 2.1
        expect(h21).toBe(h2);
        expect(l21).not.toBe(l2);
        expect(h1).not.toBe(h2);
    });

    it("draws the rest of the sentences grey, as no sense", async () => {
        const withRest = {
            ...report,
            detail: 1,
            nodes: [
                node(0, null, [1, 2], 1, { words: [word("loi")], split: split(1) }),
                node(1, 0, [], 0.6),
                node(2, 0, [], 0.4, { sense: "1", rank: 1, words: [word("ciel")] }),
            ],
        };
        const wrapper = await loaded(withRest);
        expect(rows(wrapper)).toEqual(["Sense 1 · 40% · ciel", "the rest · 60%"]);
        expect(wrapper.find("#sense-branch-1").attributes("stroke")).toContain("hsla(0, 0%, 62%");
    });
});
