import { describe, it, expect } from "vitest";
import { reactive } from "vue";
import { useAutocomplete } from "../src/composables/useAutocomplete.js";

function autocomplete() {
    const metadataValues = reactive({ title: "" });
    const { autoCompleteResults, arrowCounters, setMetadataResult, comboboxAttributes } = useAutocomplete({
        http: {}, dbUrl: "", philoConfig: { metadata: ["title"], autocomplete: ["title"] }, metadataValues, route: { query: {} },
    });
    return { metadataValues, autoCompleteResults, arrowCounters, setMetadataResult, comboboxAttributes };
}

describe("setMetadataResult", () => {
    it("quotes the value chosen, its quotes doubled", () => {
        const { metadataValues, setMetadataResult } = autocomplete();
        setMetadataResult('Les Révoltés de la <span class="highlight">"Bounty"</span>', "title");
        expect(metadataValues.title).toBe('"Les Révoltés de la ""Bounty"""');
    });

    it("keeps the rest of the value before it", () => {
        const { metadataValues, setMetadataResult } = autocomplete();
        setMetadataResult('"Lady Audley\'s Secret" | <last/>Folle-Farine', "title");
        expect(metadataValues.title).toBe('"Lady Audley\'s Secret" | "Folle-Farine"');
    });
});

describe("comboboxAttributes", () => {
    it("names the list collapsed while it has no suggestions, with no active descendant", () => {
        const { comboboxAttributes } = autocomplete();
        expect(comboboxAttributes("title", "autocomplete-title")).toEqual({
            role: "combobox",
            "aria-autocomplete": "list",
            "aria-controls": "autocomplete-title",
            "aria-expanded": false,
            "aria-activedescendant": undefined,
        });
    });

    it("names the suggestion highlighted as the active descendant", () => {
        const { autoCompleteResults, arrowCounters, comboboxAttributes } = autocomplete();
        autoCompleteResults.title = ["Candide", "Zadig"];
        expect(comboboxAttributes("title", "autocomplete-title")["aria-activedescendant"]).toBeUndefined();
        arrowCounters.title = 1;
        const attributes = comboboxAttributes("title", "autocomplete-title");
        expect(attributes["aria-expanded"]).toBe(true);
        expect(attributes["aria-activedescendant"]).toBe("autocomplete-title-1");
    });

    it("names none once the suggestions are fewer than the one highlighted", () => {
        const { autoCompleteResults, arrowCounters, comboboxAttributes } = autocomplete();
        autoCompleteResults.title = ["Candide"];
        arrowCounters.title = 3;
        expect(comboboxAttributes("title", "autocomplete-title")["aria-activedescendant"]).toBeUndefined();
    });
});
