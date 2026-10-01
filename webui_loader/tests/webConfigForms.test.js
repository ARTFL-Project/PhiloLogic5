import { EditorView } from "@codemirror/view";
import { flushPromises, mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import { createI18n } from "vue-i18n";
import CitationListEditor from "../src/components/CitationListEditor.vue";
import RecordEditor from "../src/components/RecordEditor.vue";
import ReplacementsEditor from "../src/components/ReplacementsEditor.vue";
import en from "../src/locales/en.json";
import JsonEditor from "../src/components/JsonEditor.vue";
import { highlightJsonLines } from "../src/jsonHighlight";
import { lineDiff } from "../src/utils";
import { FORMS, characterNote, citationName, followCitations } from "../src/webConfigForms";

const i18n = createI18n({ legacy: false, locale: "en", messages: { en } });
const options = { global: { plugins: [i18n] } };

const author = { field: "author", object_level: "doc", prefix: "", suffix: "", link: true, style: { "font-variant": "small-caps" } };
const title = { field: "title", object_level: "doc", prefix: "", suffix: "", link: true, style: {} };

describe("highlightJsonLines", () => {
    it("highlights keys, strings, numbers and literals, line by line, escaping everything", () => {
        const lines = highlightJsonLines('{\n  "a": "<b>",\n  "n": -1.5,\n  "ok": true,\n  "no": null\n}');
        expect(lines).toHaveLength(6);
        expect(lines[1]).toContain('<span class="tok-propertyName">&quot;a&quot;</span>');
        expect(lines[1]).toContain('<span class="tok-string">&quot;&lt;b&gt;&quot;</span>');
        expect(lines[2]).toContain('<span class="tok-number">-1.5</span>');
        expect(lines[3]).toContain('<span class="tok-bool">true</span>');
        expect(lines[4]).toContain('<span class="tok-keyword">null</span>');
        expect(lines.join("\n")).not.toContain("<b>");
    });
    it("copes with text being typed", () => {
        expect(highlightJsonLines('{"a": "unfinished').join("")).toContain("unfinished");
    });
});

describe("JsonEditor", () => {
    it("passes on valid JSON, and reports invalid JSON", async () => {
        const wrapper = mount(JsonEditor, { ...options, props: { modelValue: { a: 1 } }, attachTo: document.body });
        for (let attempt = 0; attempt < 50 && !wrapper.find(".cm-editor").exists(); attempt += 1) {
            await flushPromises();
            await new Promise((resolve) => setTimeout(resolve, 20));
        }
        const view = EditorView.findFromDOM(wrapper.find(".cm-editor").element);
        expect(view.state.doc.toString()).toBe('{\n  "a": 1\n}');
        view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: '{"a": 2}' } });
        expect(wrapper.emitted("update:modelValue").at(-1)).toEqual([{ a: 2 }]);
        view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: '{"a": ' } });
        await wrapper.vm.$nextTick();
        expect(wrapper.find(".invalid-feedback").exists()).toBe(true);
        expect(wrapper.emitted("update:modelValue")).toHaveLength(1);
        // A value set from outside replaces the text
        await wrapper.setProps({ modelValue: { b: true } });
        expect(view.state.doc.toString()).toBe('{\n  "b": true\n}');
        wrapper.unmount();
    });
});

describe("citations", () => {
    it("are named by the entry of citations they are", () => {
        expect(citationName({ ...author }, { author, title })).toBe("author");
        expect(citationName({ ...author, prefix: "by " }, { author, title })).toBe(null);
    });

    it("lists follow the changes of the named citations they use", () => {
        const values = {
            citations: { author, title },
            concordance_citation: [author, title],
            aggregation_config: [{ field: "author", field_citation: [author], break_up_field: null, break_up_field_citation: null }],
            default_landing_page_browsing: [{ label: "Author", citation: [author, title] }],
            navigation_citation: [author],
        };
        const changed = { ...author, suffix: ", " };
        followCitations(values, { author, title }, { author: changed, title }, (key) => key !== "navigation_citation");
        expect(values.concordance_citation).toEqual([changed, title]);
        expect(values.aggregation_config[0].field_citation).toEqual([changed]);
        expect(values.aggregation_config[0].break_up_field_citation).toBe(null);
        expect(values.default_landing_page_browsing[0].citation).toEqual([changed, title]);
        // Not editable (set by code): left as it is
        expect(values.navigation_citation).toEqual([author]);
    });

    it("the list editor shows them by name, others as citations of their own", () => {
        const wrapper = mount(CitationListEditor, { ...options, props: { modelValue: [author, { ...title, prefix: "in " }], citations: { author, title } } });
        const items = wrapper.findAll("li");
        expect(items[0].text()).toContain("author");
        expect(items[1].text()).toContain(en.citations.custom);
    });
});

describe("replacements", () => {
    it("say in words what can't be seen in them", () => {
        const i18n = createI18n({ legacy: false, locale: "en", messages: { en } });
        const t = i18n.global.t;
        const ideographicSpace = String.fromCodePoint(0x3000);
        expect(characterNote(" ", {}, t)).toBe("a space");
        expect(characterNote("  ", {}, t)).toBe("2 spaces");
        expect(characterNote(" OR ", {}, t)).toBe("a space before and after");
        expect(characterNote("and ", {}, t)).toBe("a space after");
        expect(characterNote(ideographicSpace, { [ideographicSpace]: "ideographic space" }, t)).toBe("ideographic space");
        expect(characterNote(String.fromCodePoint(0xff5c), {}, t)).toBe("U+FF5C");
        expect(characterNote("-", {}, t)).toBe("");
    });

    it("are edited as rows, in order", async () => {
        const wrapper = mount(ReplacementsEditor, { ...options, props: { modelValue: [["-", " "], [" OR ", " | "]], query: true } });
        const inputs = wrapper.findAll("tbody input");
        expect(inputs.map((input) => input.element.value)).toEqual(["-", " ", " OR ", " | "]);
        await inputs[3].setValue(" || ");
        await inputs[3].trigger("change");
        expect(wrapper.emitted("update:modelValue").at(-1)).toEqual([[["-", " "], [" OR ", " || "]]]);
    });
});

describe("RecordEditor", () => {
    it("keeps the keys its form doesn't know, and clears the citation of a removed breakdown", async () => {
        const entry = { field: "author", object_level: "doc", field_citation: [], break_up_field: "title", break_up_field_citation: [title], extra: 1 };
        const wrapper = mount(RecordEditor, { ...options, props: { modelValue: entry, fields: FORMS.aggregation_config, citations: { author, title } } });
        const breakUp = wrapper.findAll("input").find((input) => input.element.value === "title");
        await breakUp.setValue("");
        await breakUp.trigger("change");
        expect(wrapper.emitted("update:modelValue").at(-1)[0]).toEqual({ ...entry, break_up_field: null, break_up_field_citation: null });
    });
});

describe("lineDiff", () => {
    it("shows the changed lines with their context", () => {
        const before = ["a", "b", "c", "d", "e", "f", "g", "h"].join("\n");
        const after = ["a", "b", "c", "d", "E", "f", "g", "h"].join("\n");
        expect(lineDiff(before, after, 1).map(({ type, text }) => ({ type, text }))).toEqual([
            { type: "gap", text: "\u22ef" },
            { type: "same", text: "d" },
            { type: "removed", text: "e" },
            { type: "added", text: "E" },
            { type: "same", text: "f" },
            { type: "gap", text: "\u22ef" },
        ]);
    });
    it("matches the lines which stay, between changes", () => {
        const rows = lineDiff("x\nkeep\ny", "X\nkeep\nY", 0).filter((row) => row.type !== "gap");
        expect(rows.map((row) => `${row.type} ${row.text}`)).toEqual(["removed x", "added X", "removed y", "added Y"]);
        expect(lineDiff("1\n2\n3", "1\n2\n3", 1)).toEqual([{ type: "gap", text: "\u22ef" }]);
        // The index of each line in its text, for its highlighting
        expect(lineDiff("a\nb", "a\nB", 1).filter((row) => row.type !== "gap")).toEqual([
            { type: "same", text: "a", line: 0 },
            { type: "removed", text: "b", line: 1 },
            { type: "added", text: "B", line: 1 },
        ]);
    });
});
