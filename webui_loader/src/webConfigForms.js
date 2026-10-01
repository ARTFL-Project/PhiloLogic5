import { clone, deepEqual } from "./utils";

// The kinds of options edited with a form of their own, which can also be edited as JSON
export const STRUCTURED = ["string_map", "string_lists", "replacements", "citations", "citation_list", "sort_orders", "choice_values", "record", "records"];

// The object levels of citations and metadata
export const LEVELS = ["doc", "div1", "div2", "div3", "para", "page", "line"];

// The forms of the web config options which are objects, or lists of objects (kinds record and records): their
// fields, in order. Types: text, year (a number, or empty), bool, field (a metadata field), optional_field (none if
// empty; clears: the field emptied with it, when: shown only if that field is set), level, strings (one per line),
// string_map, citations (a list of citations). The fields a form doesn't know are kept as they are.
export const FORMS = {
    academic_citation: [
        { key: "collection", type: "text" },
        { key: "citation", type: "citations" },
    ],
    time_series_start_end_date: [
        { key: "start_date", type: "year" },
        { key: "end_date", type: "year" },
    ],
    dictionary_lookup: [
        { key: "url_root", type: "text" },
        { key: "keywords", type: "bool" },
    ],
    dictionary_lookup_keywords: [
        { key: "selected_keyword", type: "text" },
        { key: "immutable_key_values", type: "string_map" },
        { key: "variable_key_values", type: "string_map" },
    ],
    results_summary: [
        { key: "field", type: "field" },
        { key: "object_level", type: "level" },
    ],
    aggregation_config: [
        { key: "field", type: "field" },
        { key: "object_level", type: "level" },
        { key: "field_citation", type: "citations" },
        { key: "break_up_field", type: "optional_field", clears: "break_up_field_citation" },
        { key: "break_up_field_citation", type: "citations", when: "break_up_field" },
    ],
    default_landing_page_browsing: [
        { key: "label", type: "text" },
        { key: "group_by_field", type: "field" },
        { key: "is_range", type: "bool" },
        { key: "display_count", type: "bool" },
        { key: "queries", type: "strings" },
        { key: "citation", type: "citations" },
    ],
};

// What "Add" puts in the lists of objects
export const NEW_ITEMS = {
    results_summary: { field: "", object_level: "doc" },
    aggregation_config: { field: "", object_level: "doc", field_citation: [], break_up_field: null, break_up_field_citation: null },
    default_landing_page_browsing: { label: "", group_by_field: "", display_count: true, queries: [], is_range: true, citation: [] },
};

// The field shown as the title of each object of a list
export const ITEM_TITLES = { aggregation_config: "field", default_landing_page_browsing: "label" };

export const BLANK_CITATION = { field: "", object_level: "doc", prefix: "", suffix: "", link: false, style: {} };

// The name of a citation in citations, if it is one of them (the web config then refers to it as citations["name"])
export function citationName(citation, citations) {
    return Object.keys(citations || {}).find((name) => deepEqual(citations[name], citation)) || null;
}

const CITATION_LISTS = ["concordance_citation", "bibliography_citation", "table_of_contents_citation", "navigation_citation", "simple_landing_citation"];

// When an entry of citations changes, the lists which use it change with it: else they would keep the old one, which
// would then be saved as a citation of its own rather than as citations["name"]. editable(key) says which options can
// be changed.
export function followCitations(values, before, after, editable) {
    for (const [name, old] of Object.entries(before || {})) {
        const updated = (after || {})[name];
        if (updated === undefined || deepEqual(old, updated)) {
            continue;
        }
        const swap = (list) => (Array.isArray(list) ? list.map((item) => (deepEqual(item, old) ? clone(updated) : item)) : list);
        const update = (key, change) => {
            if (editable(key) && values[key] !== undefined && values[key] !== null) {
                const changed = change(values[key]);
                if (!deepEqual(changed, values[key])) {
                    values[key] = changed;
                }
            }
        };
        for (const key of CITATION_LISTS) {
            update(key, swap);
        }
        update("academic_citation", (value) => ({ ...value, citation: swap(value.citation) }));
        update("aggregation_config", (value) =>
            value.map((entry) => ({ ...entry, field_citation: swap(entry.field_citation), break_up_field_citation: swap(entry.break_up_field_citation) })),
        );
        update("default_landing_page_browsing", (value) => value.map((entry) => ({ ...entry, citation: swap(entry.citation) })));
    }
}

// A string with its spaces and invisible characters shown, or null if it has none
export function visibleCharacters(text) {
    if (!/[\s\u200b-\u200f\u2028-\u202f\u2060\ufeff]/.test(text || "")) {
        return null;
    }
    return Array.from(text)
        .map((character) => {
            if (character === " ") {
                return "\u2423"; // open box
            }
            if (character === "\t") {
                return "\u21e5"; // rightwards arrow to bar
            }
            if (/[\s\u200b-\u200f\u2028-\u202f\u2060\ufeff]/.test(character)) {
                return `[U+${character.codePointAt(0).toString(16).toUpperCase().padStart(4, "0")}]`;
            }
            return character;
        })
        .join("");
}
