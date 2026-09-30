import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import { createI18n } from "vue-i18n";
import ListEditor from "../src/components/ListEditor.vue";
import OptionField from "../src/components/OptionField.vue";
import OrderedChoice from "../src/components/OrderedChoice.vue";
import en from "../src/locales/en.json";
import fr from "../src/locales/fr.json";
import { deepEqual, formatDuration, formatSize } from "../src/utils";

const i18n = createI18n({ legacy: false, locale: "en", messages: { en } });
const options = { global: { plugins: [i18n] } };

function keys(object, prefix = "") {
    return Object.entries(object).flatMap(([key, value]) => (typeof value === "object" ? keys(value, `${prefix}${key}.`) : [`${prefix}${key}`]));
}

describe("locales", () => {
    it("have the same messages", () => {
        expect(keys(fr).sort()).toEqual(keys(en).sort());
    });
});

describe("utils", () => {
    it("format", () => {
        expect(formatSize(512)).toBe("512 B");
        expect(formatSize(1536)).toBe("1.5 KB");
        expect(formatDuration(3725)).toBe("1 h 02 min");
        expect(formatDuration(65)).toBe("1 min 05 s");
        expect(deepEqual({ a: [1] }, { a: [1] })).toBe(true);
    });
});

describe("ListEditor", () => {
    it("emits the non-empty lines", async () => {
        const wrapper = mount(ListEditor, { ...options, props: { modelValue: ["a"] } });
        await wrapper.find("textarea").setValue(" x \n\n y\n");
        expect(wrapper.emitted("update:modelValue").at(-1)).toEqual([["x", "y"]]);
    });
});

describe("OrderedChoice", () => {
    it("adds and moves items", async () => {
        const wrapper = mount(OrderedChoice, { ...options, props: { modelValue: ["author", "title"], choices: ["author", "title", "year"] } });
        await wrapper.find("select").setValue("year");
        await wrapper.findAll("button").at(-1).trigger("click");
        expect(wrapper.emitted("update:modelValue")[0]).toEqual([["author", "title", "year"]]);
        await wrapper.find('button[aria-label="Down"]').trigger("click");
        expect(wrapper.emitted("update:modelValue")[1]).toEqual([["title", "author"]]);
    });
});

describe("OptionField", () => {
    const option = { key: "break_apost", kind: "bool", default: true, help: "Break words on apostrophes", choices: [], value_choices: [] };

    it("marks changes and resets to the default", async () => {
        const wrapper = mount(OptionField, { ...options, props: { option, modelValue: false } });
        expect(wrapper.classes()).toContain("changed");
        await wrapper.find("button").trigger("click");
        expect(wrapper.emitted("update:modelValue")[0]).toEqual([true]);
    });

    it("keeps multiple choices in their order", async () => {
        const multichoice = { key: "navigable_objects", kind: "multichoice", default: ["doc", "div1"], help: "", choices: ["doc", "div1", "div2"], value_choices: [] };
        const wrapper = mount(OptionField, { ...options, props: { option: multichoice, modelValue: ["div1"] } });
        await wrapper.find('input[id="option-navigable_objects-doc"]').setValue(true);
        expect(wrapper.emitted("update:modelValue")[0]).toEqual([["doc", "div1"]]);
    });
});
