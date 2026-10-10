import { describe, it, expect } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { createGlobalConfig, createMockHttp } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import SenseMap from "../src/components/collocation/SenseMap.vue";

const word = (w) => ({ word: w, n: 10, ratio: 3, lift: 6 });
const split = (relative) => ({ relative, z: 3, importance: relative, seed: { word: "loi", group: 0 } });
const node = (id, parent, children, share, fields = {}) => ({
    id, parent, children, share, sentences: Math.round(share * 100), words: [], before: [], after: [], hidden: [],
    rarer: [], split: null, reason: null, works: 10, top_work_share: 0.2, examples: [], sense: null, rank: null, ...fields,
});

// Sense 1 splits into 1 and 1.1, close together; 2 is far from both
const report = {
    n_hits: 120, n_sentences: 100, n_works: 12, detail: 0.5, finest: 0.1, filter_list: [],
    nodes: [
        node(0, null, [1, 2], 1, { split: split(1) }),
        node(1, 0, [3, 4], 0.6, { sense: "1", rank: 1, words: [word("loi")], split: split(0.5) }),
        node(2, 0, [], 0.4, { sense: "2", rank: 3, words: [word("ciel")] }),
        node(3, 1, [], 0.4, { sense: "1", rank: 1, words: [word("droit")] }),
        node(4, 1, [], 0.2, { sense: "1.1", rank: 2, words: [word("code")] }),
    ],
    distances: [
        [0, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0, 0.3, 0.05, 0.05],
        [0.1, 0.3, 0, 0.3, 0.32],
        [0.1, 0.05, 0.3, 0, 0.1],
        [0.1, 0.05, 0.32, 0.1, 0],
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
    const wrapper = mount(SenseMap, { global });
    await flushPromises();
    return wrapper;
}

const at = (wrapper, id) => {
    const c = wrapper.find(`#sense-point-${id}`);
    return [Number(c.attributes("cx")), Number(c.attributes("cy"))];
};
const gap = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]);

describe("SenseMap", () => {
    it("places the senses shown closer together when they use more alike words", async () => {
        const wrapper = await loaded();
        expect(wrapper.findAll("circle[id^='sense-point-']")).toHaveLength(3); // 1, 1.1 and 2 at the first view
        expect(gap(at(wrapper, 3), at(wrapper, 4))).toBeLessThan(gap(at(wrapper, 3), at(wrapper, 2)));
        expect(gap(at(wrapper, 3), at(wrapper, 4))).toBeLessThan(gap(at(wrapper, 4), at(wrapper, 2)));
        expect(wrapper.text()).toContain("The map keeps");
    });

    it("gives the distances in a table, and from a sense picked to the others", async () => {
        const wrapper = await loaded();
        const cells = wrapper.findAll(".distance-table tbody tr")[0].findAll("td").map((td) => td.text());
        expect(cells).toEqual(["", "0.10", "0.30"]); // sense 1 (node 3) to 1.1 and 2
        await wrapper.find("#sense-point-2").trigger("click");
        expect(wrapper.findAll("line.link")).toHaveLength(2);
        expect(wrapper.findAll("text.distance").map((t) => t.text())).toEqual(["0.30", "0.32"]);
    });

    it("shows the senses at the detail asked for", async () => {
        const wrapper = await loaded();
        await wrapper.find("#sense-map-detail").setValue(1);
        expect(wrapper.findAll("circle[id^='sense-point-']")).toHaveLength(2);
        expect(wrapper.text()).toContain("2 senses at this cut");
    });
});
