import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import { createI18n } from "vue-i18n";
import CitationListEditor from "../src/components/CitationListEditor.vue";
import RecordEditor from "../src/components/RecordEditor.vue";
import ReplacementsEditor from "../src/components/ReplacementsEditor.vue";
import en from "../src/locales/en.json";
import { highlightJson, lineDiff } from "../src/utils";
import { FORMS, citationName, followCitations, visibleCharacters } from "../src/webConfigForms";

const i18n = createI18n({ legacy: false, locale: "en", messages: { en } });
const options = { global: { plugins: [i18n] } };

const author = { field: "author", object_level: "doc", prefix: "", suffix: "", link: true, style: { "font-variant": "small-caps" } };
const title = { field: "title", object_level: "doc", prefix: "", suffix: "", link: true, style: {} };

describe("highlightJson", () => {
    it("highlights keys, strings, numbers and literals, escaping everything", () => {
        const html = highlightJson('{"a": "<b>", "n": -1.5, "ok": true}');
        expect(html).toContain('<span class="json-key">&quot;a&quot;</span>');
        expect(html).toContain('<span class="json-string">&quot;&lt;b&gt;&quot;</span>');
        expect(html).toContain('<span class="json-number">-1.5</span>');
        expect(html).toContain('<span class="json-literal">true</span>');
        expect(html).not.toContain("<b>");
    });
    it("copes with text being typed", () => {
        expect(highlightJson('{"a": "unfinished')).toContain("unfinished");
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
    it("show their spaces and invisible characters", () => {
        expect(visibleCharacters(" OR ")).toBe("\u2423OR\u2423");
        expect(visibleCharacters("\u3000")).toBe("[U+3000]");
        expect(visibleCharacters("-")).toBe(null);
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
        expect(lineDiff(before, after, 1)).toEqual([
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
    });
});
