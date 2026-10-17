import { afterEach, describe, it, expect } from "vitest";
import { mount } from "@vue/test-utils";
import { nextTick, reactive } from "vue";
import { createGlobalConfig } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import CollocationOptions from "../src/components/CollocationOptions.vue";

const WORD_ATTRIBUTES = { pos: ["ADJ", "NOUN", "VERB"], tag: ["NC", "VINF"] };
const mounted = [];

// Each form is unmounted, even after a failure: one left in the page would answer for the next one's ids
afterEach(() => {
    while (mounted.length) mounted.pop().unmount();
});

function mountOptions(philoConfig = {}, formData = {}) {
    const global = createGlobalConfig({
        philoConfig: { word_attributes: WORD_ATTRIBUTES, collocation_lemmas: true, stopwords: "stop.txt", ...philoConfig },
    });
    const store = useMainStore();
    store.initFromConfig(global.provide.$philoConfig);
    // the search form's search, which the options edit
    const form = reactive({ ...store.defaultFields, report: "collocation", colloc_filter_choice: "frequency", ...formData });
    const wrapper = mount(CollocationOptions, { global, props: { form }, attachTo: document.body });
    mounted.push(wrapper);
    return { wrapper, form, store };
}

describe("CollocationOptions", () => {
    it("labels each choice, as the other rows of the form", () => {
        const { wrapper } = mountOptions();
        for (const [id, text] of [
            ["colloc-within", "Count collocates"],
            ["colloc-common", "Exclude"],
        ]) {
            expect(wrapper.find(`label[for="${id}"]`).text()).toBe(text);
            expect(wrapper.find(`select#${id}`).exists()).toBe(true);
        }
        // the filters, under a heading of their own
        const filters = wrapper.find("#collocate-filters");
        expect(document.getElementById(filters.attributes("aria-labelledby")).textContent.trim()).toBe("Collocate Filters:");
        expect(filters.find("#colloc-common").exists() && filters.find(".colloc-properties").exists()).toBe(true);
        const group = wrapper.find(".colloc-properties");
        expect(group.attributes("role")).toBe("group");
        expect(document.getElementById(group.attributes("aria-labelledby")).textContent.trim()).toBe(
            "Keep only collocates with"
        );
    });

    it("shows the lemma choice, the stopwords and the properties only when the database has them", () => {
        const { wrapper } = mountOptions({ collocation_lemmas: false, stopwords: "", word_attributes: {} });
        expect(wrapper.find("#collocLemmas").exists()).toBe(false);
        expect(wrapper.find('#colloc-common option[value="stopwords"]').exists()).toBe(false);
        expect(wrapper.find(".colloc-properties").exists()).toBe(false);
    });

    it("edits the form's search, not the search shown", async () => {
        const { wrapper, form, store } = mountOptions();
        await wrapper.find("#colloc-common").setValue("nofilter");
        await wrapper.find("#colloc-within").setValue("n");
        expect(form).toMatchObject({ colloc_filter_choice: "nofilter", colloc_within: "n", method_arg: "5" });
        expect(store.formData).toMatchObject({ colloc_filter_choice: "", colloc_within: "sent", method_arg: "" });
    });

    it("counts collocates as lemmas where the database has them, as word forms when picked", async () => {
        const { wrapper, form } = mountOptions();
        const group = wrapper.find(".colloc-choice");
        expect(group.attributes("role")).toBe("radiogroup");
        expect(document.getElementById(group.attributes("aria-labelledby")).textContent.trim()).toBe(
            "Count collocates as"
        );
        expect(form.colloc_lemmas).toBe("yes");
        expect(wrapper.find("#collocLemmas").element.checked).toBe(true);
        await wrapper.find("#collocForms").setValue(true);
        expect(form.colloc_lemmas).toBe("no");
    });

    it("counts within n words, the number shown only then, 5 until another is typed", async () => {
        const { wrapper, form } = mountOptions();
        expect(wrapper.find("#collocNWords").exists()).toBe(false);
        await wrapper.find("#colloc-within").setValue("n");
        expect(wrapper.find("#collocNWords").element.value).toBe("5");
        await wrapper.find("#collocNWords").setValue("8");
        expect(form.colloc_within).toBe("n");
        expect(String(form.method_arg)).toBe("8");
        await wrapper.find("#colloc-within").setValue("sent");
        await wrapper.find("#colloc-within").setValue("n");
        expect(String(form.method_arg)).toBe("8");
    });

    it("leaves out either list of common words, both, or none, the threshold shown with the most frequent", async () => {
        const { wrapper, form } = mountOptions();
        expect(wrapper.find("#filter-df-pct").exists()).toBe(true);
        await wrapper.find("#colloc-common").setValue("frequency | stopwords");
        expect(form.colloc_filter_choice).toBe("frequency | stopwords");
        await wrapper.find("#colloc-common").setValue("stopwords");
        expect(wrapper.find("#filter-df-pct").exists()).toBe(false);
        await wrapper.find("#colloc-common").setValue("nofilter");
        expect(form.colloc_filter_choice).toBe("nofilter");
    });

    it("shows a new search's common words, and those of links in any order", () => {
        const fresh = mountOptions({}, { colloc_filter_choice: "" });
        expect(fresh.wrapper.find("#colloc-common").element.value).toBe("stopwords");
        const old = mountOptions({}, { colloc_filter_choice: "stopwords | frequency" });
        expect(old.wrapper.find("#colloc-common").element.value).toBe("frequency | stopwords");
    });

    it("keeps the values ticked of each property, its button saying which", async () => {
        const { wrapper, form } = mountOptions();
        const toggle = wrapper.find("#colloc-property-pos-toggle");
        const popup = wrapper.find("#colloc-property-pos");
        expect(toggle.attributes("aria-expanded")).toBe("false");
        expect(toggle.attributes("aria-controls")).toBe("colloc-property-pos");
        expect(popup.attributes("hidden")).toBeDefined();
        expect(toggle.text()).toContain("any");
        await toggle.trigger("click");
        expect(toggle.attributes("aria-expanded")).toBe("true");
        expect(popup.attributes("hidden")).toBeUndefined();
        await wrapper.find("#colloc-property-pos-2").setValue(true); // VERB
        await wrapper.find("#colloc-property-pos-0").setValue(true); // ADJ
        expect(form.colloc_attr_pos).toBe("ADJ | VERB");
        expect(toggle.text()).toContain("ADJ, VERB");
        const [selectAll, clear] = wrapper.findAll("#colloc-property-pos .colloc-value-actions button");
        await selectAll.trigger("click");
        expect(form.colloc_attr_pos).toBe("ADJ | NOUN | VERB");
        await clear.trigger("click");
        expect(form.colloc_attr_pos).toBe("");
        await nextTick();
        expect(toggle.text()).toContain("any");
    });

    it("shows one property's values at a time, until Escape (back on its button) or the focus going elsewhere", async () => {
        const { wrapper } = mountOptions();
        const pos = wrapper.find("#colloc-property-pos-toggle");
        const tag = wrapper.find("#colloc-property-tag-toggle");
        await pos.trigger("click");
        await tag.trigger("click");
        expect(pos.attributes("aria-expanded")).toBe("false");
        expect(tag.attributes("aria-expanded")).toBe("true");
        const value = wrapper.find("#colloc-property-tag-0");
        value.element.focus();
        await value.trigger("keydown", { key: "Escape" });
        await nextTick();
        expect(tag.attributes("aria-expanded")).toBe("false");
        expect(document.activeElement).toBe(tag.element);
        await tag.trigger("click");
        await tag.trigger("focusout", { relatedTarget: wrapper.find("#colloc-common").element });
        expect(tag.attributes("aria-expanded")).toBe("false");
    });
});
