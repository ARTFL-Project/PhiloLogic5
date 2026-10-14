import { describe, it, expect } from "vitest";
import { mount } from "@vue/test-utils";
import { nextTick } from "vue";
import { createGlobalConfig } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import CollocationOptions from "../src/components/CollocationOptions.vue";

const WORD_ATTRIBUTES = { pos: ["ADJ", "NOUN", "VERB"], tag: ["NC", "VINF"] };

function mountOptions(philoConfig = {}) {
    const global = createGlobalConfig({
        philoConfig: { word_attributes: WORD_ATTRIBUTES, collocation_lemmas: true, stopwords: "stop.txt", ...philoConfig },
    });
    const store = useMainStore();
    store.initFromConfig(global.provide.$philoConfig);
    store.formData = { ...store.defaultFields, report: "collocation", colloc_filter_choice: "frequency" };
    const wrapper = mount(CollocationOptions, { global, attachTo: document.body });
    return { wrapper, store };
}

describe("CollocationOptions", () => {
    it("groups each choice in a fieldset with its legend", () => {
        const { wrapper } = mountOptions();
        const legends = wrapper.findAll("fieldset > legend").map((l) => l.text());
        expect(legends).toEqual(expect.arrayContaining([
            "Count collocates within", "Count collocates as", "Leave out common words",
            "Keep only collocates with these word properties",
        ]));
        wrapper.unmount();
    });

    it("shows the lemma choice, the stopwords and the properties only when the database has them", () => {
        const { wrapper } = mountOptions({ collocation_lemmas: false, stopwords: "", word_attributes: {} });
        expect(wrapper.find("#collocLemmas").exists()).toBe(false);
        expect(wrapper.find("#collocStopwords").exists()).toBe(false);
        expect(wrapper.find(".colloc-properties").exists()).toBe(false);
        wrapper.unmount();
    });

    it("counts collocates as lemmas when picked", async () => {
        const { wrapper, store } = mountOptions();
        expect(wrapper.find("#collocForms").element.checked).toBe(true);
        await wrapper.find("#collocLemmas").setValue(true);
        expect(store.formData.colloc_lemmas).toBe("yes");
        wrapper.unmount();
    });

    it("leaves out either list of common words, both, or none", async () => {
        const { wrapper, store } = mountOptions();
        await wrapper.find("#collocStopwords").setValue(true);
        expect(store.formData.colloc_filter_choice).toBe("frequency | stopwords");
        await wrapper.find("#collocFrequent").setValue(false);
        expect(store.formData.colloc_filter_choice).toBe("stopwords");
        await wrapper.find("#collocStopwords").setValue(false);
        expect(store.formData.colloc_filter_choice).toBe("nofilter");
        await wrapper.find("#filter-df-pct").setValue("2");
        expect(store.formData.colloc_filter_choice).toBe("frequency");
        wrapper.unmount();
    });

    it("names the controls next to their inputs by what they hold", () => {
        const { wrapper } = mountOptions();
        for (const control of ["#collocWithinN", "#collocFrequent"]) {
            const ids = wrapper.find(control).attributes("aria-labelledby").split(" ");
            for (const id of ids) expect(document.getElementById(id)).not.toBeNull();
        }
        wrapper.unmount();
    });

    it("selects within n words when a number is typed, not when tabbing through", async () => {
        const { wrapper, store } = mountOptions();
        await wrapper.find("#collocNWords").trigger("focus");
        expect(store.formData.colloc_within).toBe("sent");
        await wrapper.find("#collocNWords").setValue("5");
        expect(store.formData.colloc_within).toBe("n");
        wrapper.unmount();
    });

    it("keeps the values ticked of each property, its button saying which", async () => {
        const { wrapper, store } = mountOptions();
        const toggle = wrapper.find("#colloc-property-pos-toggle");
        const panel = wrapper.find("#colloc-property-pos");
        expect(toggle.attributes("aria-expanded")).toBe("false");
        expect(toggle.attributes("aria-controls")).toBe("colloc-property-pos");
        expect(panel.attributes("hidden")).toBeDefined();
        expect(toggle.text()).toContain("any");
        await toggle.trigger("click");
        expect(toggle.attributes("aria-expanded")).toBe("true");
        await wrapper.find("#colloc-property-pos-2").setValue(true);  // VERB
        await wrapper.find("#colloc-property-pos-0").setValue(true);  // ADJ
        expect(store.formData.colloc_attr_pos).toBe("ADJ | VERB");
        expect(toggle.text()).toContain("ADJ, VERB");
        const [selectAll, clear] = wrapper.findAll("#colloc-property-pos .colloc-value-actions button");
        await selectAll.trigger("click");
        expect(store.formData.colloc_attr_pos).toBe("ADJ | NOUN | VERB");
        await clear.trigger("click");
        expect(store.formData.colloc_attr_pos).toBe("");
        await nextTick();
        expect(toggle.text()).toContain("any");
        wrapper.unmount();
    });
});
