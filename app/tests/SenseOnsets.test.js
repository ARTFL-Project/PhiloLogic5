import { describe, it, expect } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { createGlobalConfig, createMockHttp } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import SenseOnsets from "../src/components/collocation/SenseOnsets.vue";
import DistinctivePassagesModal from "../src/components/DistinctivePassagesModal.vue";

const word = (w) => ({ word: w, n: 10, ratio: 3, lift: 6 });
const split = (relative) => ({ relative, z: 3, importance: relative, seed: { word: "loi", group: 0 } });
const node = (id, parent, children, share, fields = {}) => ({
    id, parent, children, share, sentences: Math.round(share * 100), words: [], before: [], after: [], hidden: [],
    rarer: [], split: null, reason: null, works: 10, top_work_share: 0.2, examples: [], sense: null, rank: null,
    onset: null, ...fields,
});
const flat = (n, v) => new Array(n).fill(v);

// Sense 2 rises in 1789, with two earlier sentences; sense 1 does not rise
const report = {
    n_hits: 120, n_sentences: 100, n_works: 12, detail: 1, finest: 0.1, filter_list: [],
    nodes: [
        node(0, null, [1, 2], 1, { split: split(1) }),
        node(1, 0, [], 0.6, { sense: "1", rank: 1, words: [word("globe"), word("planète")] }),
        node(2, 0, [], 0.4, {
            sense: "2", rank: 2, words: [word("peuple"), word("ennemi")],
            onset: {
                year: 1789, before: 0.02, after: 0.3, g2: 40, sentences_before: 5, works_before: 4,
                precursors: [{ hit: 11, year: 1688, p: 0.9 }, { hit: 12, year: 1750, p: 0.8 }],
            },
        }),
    ],
    time: {
        start: 1680, end: 1800, sentences: flat(121, 1), works: flat(121, 1),
        share: { smoothness: 1, value: [flat(121, 1), flat(121, 0.6), flat(121, 0.4)], low: [], high: [] }, rate: null,
    },
};

async function loaded() {
    const http = createMockHttp({
        "get_word_senses.py": report,
        "get_sense_passages.py": { passages: [{ hit: 11, philo_id: [1, 0, 0, 0, 0, 0, 0], citation: [], context: "<p>x</p>" }] },
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
    const wrapper = mount(SenseOnsets, { global });
    await flushPromises();
    return { wrapper, http };
}

describe("SenseOnsets", () => {
    it("lists the senses that take off first, with their rise and what came before", async () => {
        const { wrapper } = await loaded();
        const cards = wrapper.findAll(".onset-card");
        expect(cards.map((c) => c.attributes("id"))).toEqual(["sense-onset-2", "sense-onset-1"]);
        expect(cards[0].text()).toContain("Takes off in 1789: from 2.0% of the works' sentences before to 30.0% after.");
        expect(cards[0].text()).toContain("Before 1789: 5 sentences in 4 works. The earliest, from 1688, 1750");
        expect(cards[0].findAll("circle.precursor")).toHaveLength(2);
        expect(cards[1].text()).toContain("No clear rise.");
        expect(wrapper.text()).toContain("2 senses, 1 of which take off.");
    });

    it("reads the earliest sentences of a sense", async () => {
        const { wrapper, http } = await loaded();
        await wrapper.find("#sense-onset-2 .btn-link").trigger("click");
        await flushPromises();
        const call = http.get.mock.calls.find(([url]) => url.includes("get_sense_passages.py"));
        expect(call[1].params.hits).toBe("11,12");
        const modal = wrapper.findComponent(DistinctivePassagesModal);
        expect(modal.props("orderingNote")).toContain("before its rise in 1789");
    });
});
