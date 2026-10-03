import { describe, it, expect } from "vitest";
import { reactive } from "vue";
import { useAutocomplete } from "../src/composables/useAutocomplete.js";

function autocomplete() {
    const metadataValues = reactive({ title: "" });
    const { setMetadataResult } = useAutocomplete({
        http: {}, dbUrl: "", philoConfig: { metadata: ["title"], autocomplete: ["title"] }, metadataValues, route: { query: {} },
    });
    return { metadataValues, setMetadataResult };
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
