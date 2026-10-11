import { describe, it, expect } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { createGlobalConfig, createMockHttp } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import AheadOfTime from "../src/components/collocation/AheadOfTime.vue";
import DistinctivePassagesModal from "../src/components/DistinctivePassagesModal.vue";

const work = (doc, year, fields = {}) => ({
    doc, year, author: "Volney", title: `Work ${doc}`, z: 5, share: 0.74, before: 40, after: 60, sentences: 3,
    words: [{ word: "peuple", group: 0 }, { word: "française", group: 2, phrase: "révolution française" }],
    sources: [{ hit: doc * 10, year }, { hit: doc * 10 + 1, year }],
    followers: [{ doc: 9, year: year + 3, author: "Marat", title: "Pamphlets" }],
    ...fields,
});

const report = (clear, perhaps) => ({
    n_hits: 120, n_sentences: 100, n_works: 30, filter_list: [], tested: 28, span: 10, threshold: 3.9, evidence: 2,
    by_chance: 0.6, clear, perhaps, n_clear: clear.length, n_perhaps: perhaps.length,
    periods: [
        { start: 1770, end: 1779, tested: 10, clear: 0, perhaps: perhaps.length },
        { start: 1780, end: 1789, tested: 18, clear: clear.length, perhaps: 0 },
    ],
});

async function loaded(data) {
    const http = createMockHttp({ "get_ahead_of_time.py": data, "get_sense_passages.py": { passages: [] } });
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
    const wrapper = mount(AheadOfTime, { global });
    await flushPromises();
    return { wrapper, http };
}

describe("AheadOfTime", () => {
    it("shows the works clearly ahead, and keeps those perhaps ahead folded", async () => {
        const { wrapper } = await loaded(report([work(1, 1788)], [work(2, 1775, { z: 2.5, share: 0.6 })]));
        expect(wrapper.text()).toContain("28 works compared with those of the 10 years before and after them: 1 clearly ahead");
        const cards = wrapper.findAll(".work-card");
        expect(cards).toHaveLength(1);
        expect(cards[0].text()).toContain("1788 · Volney, Work 1");
        expect(cards[0].text()).toContain("closer to it than those before in 74% of comparisons (60 works after, 40 before)");
        expect(cards[0].text()).toContain("révolution française");
        expect(cards[0].text()).toContain("Closest later works: Marat, Pamphlets (1791)");
        expect(wrapper.findAll("rect.bar-clear")).toHaveLength(2);
        const toggle = wrapper.find("[aria-controls='ahead-perhaps']");
        expect(toggle.attributes("aria-expanded")).toBe("false");
        await toggle.trigger("click");
        expect(wrapper.findAll(".perhaps-card")).toHaveLength(1);
        expect(wrapper.text()).toContain("about 1 would pass it by chance alone");
    });

    it("opens the works perhaps ahead when none is clearly ahead", async () => {
        const { wrapper } = await loaded(report([], [work(2, 1775, { z: 2.5, share: 0.6 })]));
        expect(wrapper.text()).toContain("No work is clearly ahead of its time");
        expect(wrapper.findAll(".perhaps-card")).toHaveLength(1);
    });

    it("reads a work's sentences most like the later works", async () => {
        const { wrapper, http } = await loaded(report([work(1, 1788)], []));
        await wrapper.find(".work-card .btn-link").trigger("click");
        await flushPromises();
        const call = http.get.mock.calls.find(([url]) => url.includes("get_sense_passages.py"));
        expect(call[1].params.hits).toBe("10,11");
        expect(wrapper.findComponent(DistinctivePassagesModal).props("groupName")).toContain("1788 · Volney, Work 1");
    });
});
