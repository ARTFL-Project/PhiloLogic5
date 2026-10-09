import { describe, it, expect } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { createGlobalConfig, createMockHttp } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import WordSenses from "../src/components/collocation/WordSenses.vue";
import DistinctivePassagesModal from "../src/components/DistinctivePassagesModal.vue";

const word = (w, ratio = 3, n = 10) => ({ word: w, n, ratio, lift: ratio * 2 });
const split = (relative, z, seed = { word: "loi", group: 0 }) => ({
    tested: true, gain: 120 * relative, bits_per_sentence: 0.4 * relative, z, sentences_won: 60, sentences: 100,
    importance: 0.4 * relative, seed, relative,
});
const node = (id, parent, children, share, fields = {}) => ({
    id, parent, children, share, sentences: Math.round(share * 100), words: [], before: [], after: [],
    split: null, reason: null, works: 10, top_work_share: 0.2, examples: [], sense: null, rank: null, ...fields,
});
const flat = (n, v) => new Array(n).fill(v);
const rows = (values) => values.map((v) => flat(5, v));

// Three distinctions: the main one (0), one 40% as important (2), and one 15% as important (3), below the 25% default.
// The numbers are those the report gives: sense 6, which nothing marks, goes on as 2 once 2.1 (3) is carved out of it
const senses = {
    n_hits: 120, n_sentences: 100, n_works: 12, duplicates: 0, folds: 5, evidence: 2, resamples: 200,
    band: [2.5, 97.5], detail: 0.25, finest: 0.1, filter_list: ["de", "la"],
    nodes: [
        node(0, null, [1, 2], 1, { words: [word("loi"), word("droit")], before: [word("la")], split: split(1, 3.4) }),
        node(1, 0, [], 0.5, {
            sense: "1", rank: 1,
            words: [word("dieu", 8.3), word("foi", 12.4)], split: split(0, -0.7), reason: "no_gain",
            examples: [{ hit: 7, p: 0.97, evidence: [{ word: "dieu", group: 0 }, { word: "avoir", group: 1 }] }],
        }),
        node(2, 0, [3, 6], 0.5, {
            sense: "2", rank: 2, words: [word("ciel")], split: split(0.4, 2.6, { word: "la", group: 1 }),
        }),
        node(3, 2, [4, 5], 0.3, {
            sense: "2.1", rank: 3, words: [word("mer")], split: split(0.15, 2.2, { word: "vent", group: 2 }),
        }),
        node(4, 3, [], 0.2, {
            sense: "2.1", rank: 3, words: [word("port"), word("navire"), word("quai"), word("vent")], reason: "inseparable",
            groups: [{ hub: 0, members: [1, 2] }],
        }),
        node(5, 3, [], 0.1, { sense: "2.1.1", rank: 4, words: [word("vague")], split: split(0, 0.5), reason: "no_gain" }),
        node(6, 2, [], 0.2, {
            sense: "2", rank: 2, split: split(0.05, 2.5), reason: "faint", works: 1, top_work_share: 1,
            rarer: [{ word: "ciel", group: 0, n: 1, ratio: 0.05 }, { word: "la", group: 1, n: 2, ratio: 0.25 }],
        }),
    ],
    time: {
        start: 1700, end: 1704, sentences: flat(5, 20), works: [2, 3, 2, 3, 2],
        share: {
            smoothness: 10,
            value: rows([1, 0.5, 0.5, 0.3, 0.2, 0.1, 0.2]),
            low: rows([1, 0.4, 0.4, 0.2, 0.1, 0.05, 0.1]),
            high: rows([1, 0.6, 0.6, 0.4, 0.3, 0.15, 0.3]),
        },
        rate: null,
    },
};

const oneSense = {
    ...senses,
    nodes: [node(0, null, [], 1, { words: [word("loi")], before: [word("la")], split: split(0, 0.8), reason: "no_gain" })],
    time: null,
};

function mountSenses(response, extra = {}) {
    const http = createMockHttp({
        "get_word_senses.py": response,
        "get_sense_passages.py": { passages: [{ hit: 7, philo_id: [1, 0, 0, 0, 0, 0, 0], citation: [], context: "<p>x</p>" }] },
        ...extra,
    });
    const global = createGlobalConfig({
        http,
        route: { name: "collocation", path: "/collocation", query: { q: "lemma:raison", report: "collocation" } },
        stubs: {
            ProgressSpinner: { template: "<div class='spinner-stub' />" },
            DistinctivePassagesModal: {
                template: "<div class='modal-stub' />",
                props: ["groupName", "signature", "passages", "loading", "hasMore", "orderingNote"],
            },
        },
    });
    const store = useMainStore();
    store.formData = { ...store.formData, q: "lemma:raison", report: "collocation" };
    return { wrapper: mount(WordSenses, { global }), http };
}

async function loaded(response = senses) {
    const mounted = mountSenses(response);
    mounted.wrapper.vm.runDetection();
    await flushPromises();
    return mounted;
}

const cards = (wrapper) => wrapper.findAll(".sense-card").map((c) => c.attributes("id"));

describe("WordSenses", () => {
    it("shows the distinctions at least as important as the default detail", async () => {
        const { wrapper } = await loaded();
        expect(wrapper.text()).toContain("3 senses in 100 sentences from 12 works");
        expect(cards(wrapper)).toEqual(["sense-card-1", "sense-card-6", "sense-card-3"]);
        expect(wrapper.find("#word-senses-detail").element.value).toBe("2");
        expect(wrapper.text()).toContain("3 senses: the distinctions at least 40% as important as the main one.");
    });

    it("tells what each distinction started from, how important it is and how sure", async () => {
        const { wrapper } = await loaded();
        const splits = wrapper.findAll(".split-row").map((r) => r.text());
        expect(splits).toHaveLength(2);
        expect(splits[0]).toContain("Senses 1, 2, 2.1 (100% of the sentences)");
        expect(splits[0]).toContain("split starting from “loi”; 100% as important as the main distinction; gain on unseen works 3.4 times its standard error");
        expect(splits[1]).toContain("Senses 2, 2.1 (50% of the sentences)");
        expect(splits[1]).toContain("starting from “la raison”; 40% as important");
    });

    it("shows finer distinctions as the slider moves, and fewer back", async () => {
        const { wrapper } = await loaded();
        await wrapper.find("#word-senses-detail").setValue(3);
        expect(cards(wrapper)).toEqual(["sense-card-1", "sense-card-6", "sense-card-4", "sense-card-5"]);
        expect(wrapper.text()).toContain("4 senses: the distinctions at least 15% as important as the main one.");
        expect(wrapper.text()).toContain("starting from “raison vent”");

        await wrapper.find("#word-senses-detail").setValue(1);
        expect(cards(wrapper)).toEqual(["sense-card-1", "sense-card-2"]);
        expect(wrapper.text()).toContain("2 senses: the main distinction only.");

        await wrapper.find("#word-senses-detail").setValue(0);
        expect(cards(wrapper)).toEqual([]);
        expect(wrapper.text()).toContain("100 sentences from 12 works, shown as one sense: move the slider");
    });

    it("numbers each sense by the split that made it, the same at any detail", async () => {
        const { wrapper } = await loaded();
        const titles = () => wrapper.findAll(".sense-card h3").map((h3) => h3.text().split(" ·")[0].trim());
        expect(titles()).toEqual(["Sense 1", "Sense 2", "Sense 2.1"]);
        await wrapper.find("#word-senses-detail").setValue(3);
        expect(titles()).toEqual(["Sense 1", "Sense 2", "Sense 2.1", "Sense 2.1.1"]);
        await wrapper.find("#word-senses-detail").setValue(1);
        expect(titles()).toEqual(["Sense 1", "Sense 2"]);
    });

    it("shows the rest of the sentences, which no sense takes, as a line of its own, not a sense", async () => {
        const withRest = {
            ...senses,
            detail: 1,
            nodes: [
                node(0, null, [1, 2], 1, { words: [word("loi")], split: split(1, 3.4) }),
                node(1, 0, [], 0.6, {
                    rarer: [{ word: "ciel", group: 0, n: 1, ratio: 0.05 }], split: split(0, 1.2), reason: "no_gain",
                }),
                node(2, 0, [], 0.4, { sense: "1", rank: 1, words: [word("ciel")], reason: "no_gain" }),
            ],
            time: { ...senses.time, share: { smoothness: 10, value: rows([1, 0.6, 0.4]), low: rows([1, 0.5, 0.3]), high: rows([1, 0.7, 0.5]) } },
        };
        const { wrapper } = await loaded(withRest);
        expect(cards(wrapper)).toEqual(["sense-card-2"]);
        expect(wrapper.find("#sense-rest-1").text()).toContain("The rest of the sentences (60%): no clear sense emerges in them");
        expect(wrapper.find("#sense-rest-1").text()).toContain("best split's gain on unseen works was 1.2 times its standard error");
        expect(wrapper.text()).toContain("1 sense in 100 sentences from 12 works · 60% of them in no clear sense");
        expect(wrapper.find(".split-row").text()).toContain("Senses 1, the rest (100% of the sentences)");
        expect(wrapper.findAll(".legend-chip").map((c) => c.text())).toEqual(["Sense 1: ciel"]);
        const fills = wrapper.findAll("svg.overview-chart path").map((p) => p.attributes("fill"));
        expect(fills).toHaveLength(2);
        expect(fills[1]).toContain("hsla(0, 0%, 62%");
    });

    it("says why a sense was not split further", async () => {
        const { wrapper } = await loaded();
        expect(wrapper.find("#sense-card-1").text()).toContain("best split's gain on unseen works was -0.7 times its standard error");
        expect(wrapper.find("#sense-card-3").text()).toContain("Splits further at finer detail (15% as important as the main distinction");
        expect(wrapper.find("#sense-card-6").text()).toContain("its distinctions were less than 10% as important as the main one");
        await wrapper.find("#word-senses-detail").setValue(3);
        expect(wrapper.find("#sense-card-4").text()).toContain("its sentences could not be told apart");
    });

    it("groups a sense's words that go together around their hub word, the others after", async () => {
        const { wrapper } = await loaded();
        await wrapper.find("#word-senses-detail").setValue(3);
        const card = wrapper.find("#sense-card-4");
        const group = card.find(".word-group");
        expect(group.find(".hub-chip").text()).toContain("port");
        expect(group.text()).toContain("navire");
        expect(group.text()).toContain("quai");
        expect(card.text()).toContain("Other words:");
        expect(group.text()).not.toContain("vent");
    });

    it("lists each sense's words with how many times more frequent they are", async () => {
        const { wrapper } = await loaded();
        const first = wrapper.find("#sense-card-1");
        expect(first.text()).toContain("dieu");
        expect(first.text()).toContain(`${String.fromCharCode(215)}8.3`);
        expect(first.text()).toContain(`${String.fromCharCode(215)}12`);
    });

    it("tells how many works a sense is from, and when one work makes most of it", async () => {
        const { wrapper } = await loaded();
        expect(wrapper.find("#sense-card-1").text()).toContain("from 10 works");
        expect(wrapper.find("#sense-card-1 .one-work").exists()).toBe(false);
        expect(wrapper.find("#sense-card-6 .one-work").text()).toContain("100% from a single work");
    });

    it("tells what a sense lacks, and names a sense by it when nothing marks it", async () => {
        const { wrapper } = await loaded();
        const card = wrapper.find("#sense-card-6");
        expect(card.text()).toContain("The rest of the sense it was split from");
        expect(card.text()).not.toContain("No word is markedly more frequent");
        expect(card.text()).toContain("Rarer here:");
        expect(card.text()).toContain(`ciel ${String.fromCharCode(247)}20`);
        expect(card.text()).toContain(`la raison ${String.fromCharCode(247)}4.0`);
        const chips = wrapper.findAll(".legend-chip").map((c) => c.text());
        expect(chips[1]).toContain("Sense 2: without ciel, la raison");
    });

    it("shows the common words that alone mark a sense, greyed, and names the sense by them", async () => {
        const marked = {
            ...senses,
            nodes: senses.nodes.map((n) => (n.id === 6 ? { ...n, rarer: [], hidden: [word("de", 4), word("que", 2.5)] } : n)),
        };
        const { wrapper } = await loaded(marked);
        const card = wrapper.find("#sense-card-6");
        expect(card.text()).toContain("Marked only by common words, which the collocation filter otherwise hides:");
        expect(card.findAll(".hidden-chip").map((c) => c.text())).toEqual([
            `de ${String.fromCharCode(215)}4.0`, `que ${String.fromCharCode(215)}2.5`,
        ]);
        expect(card.text()).not.toContain("No word is markedly more frequent");
        const chips = wrapper.findAll(".legend-chip").map((c) => c.text());
        expect(chips[1]).toContain("Sense 2: de, que");
    });

    it("tells what was split off from a sense nothing marks, and names the sense by it", async () => {
        const unmarked = { ...senses, nodes: senses.nodes.map((n) => (n.id === 6 ? { ...n, rarer: [] } : n)) };
        const { wrapper } = await loaded(unmarked);
        expect(wrapper.find("#sense-card-6").text()).toContain(
            "The rest of the sense it was split from, once the sentences marked by mer are split off",
        );
        const chips = wrapper.findAll(".legend-chip").map((c) => c.text());
        expect(chips[1]).toContain("Sense 2: without mer");
    });

    it("words next to the query read with the query word", async () => {
        const { wrapper } = await loaded();
        expect(wrapper.text()).toContain("la raison");
    });

    it("draws a chart for each sense shown over time", async () => {
        const { wrapper } = await loaded();
        expect(wrapper.findAll("svg.sense-chart")).toHaveLength(3);
        expect(wrapper.find("svg.overview-chart").exists()).toBe(true);
        expect(wrapper.findAll("svg.overview-chart path")).toHaveLength(3);
    });

    it("tells when no distinct senses are found", async () => {
        const { wrapper } = await loaded(oneSense);
        expect(wrapper.text()).toContain("No distinct senses in 100 sentences from 12 works");
        expect(wrapper.findAll(".sense-card")).toHaveLength(0);
        expect(wrapper.find("#word-senses-detail").exists()).toBe(false);
        expect(wrapper.text()).toContain("No dates");
    });

    it("passes up the words the collocation filter left out", async () => {
        const { wrapper } = await loaded();
        expect(wrapper.emitted("filterList")[0][0]).toEqual(["de", "la"]);
        expect(wrapper.emitted("stopwordsMissing")[0][0]).toBe(false);
    });

    it("fetches a sense's examples by their hits and notes what decided them", async () => {
        const { wrapper, http } = await loaded();
        await wrapper.find("#sense-card-1 .btn-link").trigger("click");
        await flushPromises();
        const call = http.get.mock.calls.find(([url]) => url.includes("get_sense_passages.py"));
        expect(call[1].params.hits).toBe("7");
        const modal = wrapper.findComponent(DistinctivePassagesModal);
        const note = modal.props("passages")[0].note;
        expect(note).toContain("97% in this sense");
        expect(note).toContain("dieu, avoir raison");
    });
});
