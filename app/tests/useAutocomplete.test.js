import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { reactive, ref } from "vue";
import { useAutocomplete } from "../src/composables/useAutocomplete.js";

let input;

beforeEach(() => {
    vi.useFakeTimers();
    // the suggestions are only shown in the field still focused when they come
    input = document.createElement("input");
    document.body.appendChild(input);
    input.focus();
});

afterEach(() => {
    input.remove();
    vi.useRealTimers();
});

function autocomplete(response, options = {}) {
    const metadataValues = reactive({ title: "", year: "" });
    const queryTerm = ref("");
    const http = { get: vi.fn(() => Promise.resolve({ data: response })) };
    const onSelect = vi.fn();
    const ac = useAutocomplete({
        http, dbUrl: "", metadataValues, queryTerm, onSelect, route: { query: {} },
        philoConfig: { metadata: ["title", "year"], autocomplete: ["q", "title"] }, ...options,
    });
    return { ...ac, metadataValues, queryTerm, http, onSelect };
}

/** Type value in field, and wait for its suggestions. */
async function type(ac, field, value) {
    if (field === "q") ac.queryTerm.value = value;
    else ac.metadataValues[field] = value;
    ac.onChange(field);
    await vi.advanceTimersByTimeAsync(200);
}

function key(name) {
    return { key: name, isComposing: false, preventDefault: vi.fn() };
}

describe("metadata suggestions", () => {
    it("shows the value without what comes before it", async () => {
        const ac = autocomplete(['"Lady Audley\'s Secret" | CUTHERE <span class="highlight">Fol</span>le-Farine']);
        await type(ac, "title", '"Lady Audley\'s Secret" | fol');
        expect(ac.autoCompleteResults.title).toEqual([
            { html: '<span class="highlight">Fol</span>le-Farine', value: '"Folle-Farine"', selected: false },
        ]);
    });

    it("quotes the value chosen, its quotes doubled", async () => {
        const ac = autocomplete(['Les Révoltés de la <span class="highlight">"Bounty"</span>']);
        await type(ac, "title", "bounty");
        ac.toggle("title", 0);
        expect(ac.metadataValues.title).toBe('"Les Révoltés de la ""Bounty"""');
        expect(ac.onSelect).toHaveBeenCalledWith("title", '"Les Révoltés de la ""Bounty"""');
    });

    it("joins the values chosen by OR, after the rest of the value", async () => {
        const ac = autocomplete([
            '"Lady Audley\'s Secret" | CUTHERE <span class="highlight">Fol</span>le-Farine',
            '"Lady Audley\'s Secret" | CUTHERE <span class="highlight">Fol</span>ly',
        ]);
        await type(ac, "title", '"Lady Audley\'s Secret" | fol');
        ac.toggle("title", 0);
        ac.toggle("title", 1);
        expect(ac.metadataValues.title).toBe('"Lady Audley\'s Secret" | "Folle-Farine" | "Folly"');
        expect(ac.autoCompleteResults.title.length).toBe(2); // the list stays open
    });

    it("puts back what was typed when none is chosen any more", async () => {
        const ac = autocomplete(['<span class="highlight">Fol</span>le-Farine']);
        await type(ac, "title", "fol");
        ac.toggle("title", 0);
        ac.toggle("title", 0);
        expect(ac.metadataValues.title).toBe("fol");
    });

    it("asks for the values of the field", async () => {
        const ac = autocomplete([]);
        await type(ac, "title", "fol");
        expect(ac.http.get).toHaveBeenCalledWith("/scripts/autocomplete_metadata.py", {
            params: { term: "fol", field: "title" },
        });
    });

    it("has none for a field without autocomplete", async () => {
        const ac = autocomplete(["1850"]);
        await type(ac, "year", "18");
        expect(ac.http.get).not.toHaveBeenCalled();
    });
});

describe("search term suggestions", () => {
    const libert = ['amour "la" "haine" <span class="highlight">libert</span>é', 'amour "la" "haine" <span class="highlight">libert</span>és'];

    it("keeps the rest of the terms as typed, phrases with it", async () => {
        const ac = autocomplete(libert);
        await type(ac, "q", 'amour "la haine" libert');
        expect(ac.autoCompleteResults.q[0].html).toBe('<span class="highlight">libert</span>é');
        ac.toggle("q", 0);
        expect(ac.queryTerm.value).toBe('amour "la haine" "liberté"');
        ac.toggle("q", 1);
        expect(ac.queryTerm.value).toBe('amour "la haine" "liberté" | "libertés"');
    });

    it("completes the term after an OR", async () => {
        const ac = autocomplete(['amour | <span class="highlight">libert</span>é']);
        await type(ac, "q", "amour|libert");
        ac.toggle("q", 0);
        expect(ac.queryTerm.value).toBe('amour|"liberté"');
    });

    it("completes a quoted term", async () => {
        const ac = autocomplete(['"<span class="highlight">libert</span>é"']);
        await type(ac, "q", '"libert');
        expect(ac.autoCompleteResults.q[0].html).toBe('<span class="highlight">libert</span>é');
        ac.toggle("q", 0);
        expect(ac.queryTerm.value).toBe('"liberté"');
    });

    it("keeps the server's rest of the terms for a term in an unclosed phrase", async () => {
        const ac = autocomplete(['"la" "<span class="highlight">lib</span>re"']);
        await type(ac, "q", '"la lib');
        ac.toggle("q", 0);
        expect(ac.queryTerm.value).toBe('"la" "libre"');
    });

    it("does not quote word properties", async () => {
        const ac = autocomplete(['<span class="highlight">lemma:lib</span>erté', '<span class="highlight">lemma:lib</span>re']);
        await type(ac, "q", "lemma:lib");
        ac.toggle("q", 0);
        ac.toggle("q", 1);
        expect(ac.queryTerm.value).toBe("lemma:liberté | lemma:libre");
    });

    it("asks for nothing under two letters, or for the terms searched", async () => {
        const ac = autocomplete([], { route: { query: { q: "liberté" } } });
        await type(ac, "q", '"l');
        await type(ac, "q", "liberté");
        expect(ac.http.get).not.toHaveBeenCalled();
    });

    it("drops suggestions that come once the field is left", async () => {
        const ac = autocomplete(libert);
        ac.queryTerm.value = "libert";
        ac.onChange("q");
        input.blur();
        await vi.advanceTimersByTimeAsync(200);
        expect(ac.autoCompleteResults.q).toEqual([]);
    });
});

describe("keyboard", () => {
    const words = ['<span class="highlight">lib</span>re', '<span class="highlight">lib</span>erté'];

    it("moves through the suggestions with the arrows, up from the first back to the field", async () => {
        const ac = autocomplete(words);
        await type(ac, "q", "lib");
        ac.onKeydown("q", key("ArrowDown"));
        ac.onKeydown("q", key("ArrowDown"));
        ac.onKeydown("q", key("ArrowDown"));
        expect(ac.arrowCounters.q).toBe(1);
        ac.onKeydown("q", key("ArrowUp"));
        ac.onKeydown("q", key("ArrowUp"));
        expect(ac.arrowCounters.q).toBe(-1);
    });

    it("adds or removes a suggestion with Space, which is typed when none is highlighted", async () => {
        const ac = autocomplete(words);
        await type(ac, "q", "lib");
        const typed = key(" ");
        ac.onKeydown("q", typed);
        expect(typed.preventDefault).not.toHaveBeenCalled();
        ac.onKeydown("q", key("ArrowDown"));
        ac.onKeydown("q", key(" "));
        expect(ac.queryTerm.value).toBe('"libre"');
        ac.onKeydown("q", key(" "));
        expect(ac.queryTerm.value).toBe("lib");
    });

    it("adds a suggestion with Enter, never removes it, and closes the list", async () => {
        const ac = autocomplete(words);
        await type(ac, "q", "lib");
        ac.onKeydown("q", key("ArrowDown"));
        ac.onKeydown("q", key(" "));
        ac.onKeydown("q", key("Enter"));
        expect(ac.queryTerm.value).toBe('"libre"');
        expect(ac.autoCompleteResults.q).toEqual([]);
    });

    it("submits the form with Enter when no suggestion is highlighted", async () => {
        const ac = autocomplete(words);
        await type(ac, "q", "lib");
        const enter = key("Enter");
        ac.onKeydown("q", enter);
        expect(enter.preventDefault).not.toHaveBeenCalled();
        expect(ac.autoCompleteResults.q.length).toBe(2);
    });

    it("closes the list with Escape, keeping the suggestions added", async () => {
        const ac = autocomplete(words);
        await type(ac, "q", "lib");
        ac.toggle("q", 1);
        ac.onKeydown("q", key("Escape"));
        expect(ac.autoCompleteResults.q).toEqual([]);
        expect(ac.queryTerm.value).toBe('"liberté"');
    });

    it("leaves the keys of an input method composing alone", async () => {
        const ac = autocomplete(words);
        await type(ac, "q", "lib");
        ac.onKeydown("q", { ...key("ArrowDown"), isComposing: true });
        expect(ac.arrowCounters.q).toBe(-1);
    });
});

describe("comboboxAttrs", () => {
    it("makes a field with autocomplete a combobox, its list controlled once open", async () => {
        const ac = autocomplete(['<span class="highlight">lib</span>re'], { idPrefix: "compare-" });
        expect(ac.comboboxAttrs("title")).toMatchObject({
            role: "combobox", "aria-expanded": "false", "aria-controls": undefined,
            "aria-describedby": "compare-autocomplete-instructions",
        });
        await type(ac, "title", "lib");
        expect(ac.comboboxAttrs("title")).toMatchObject({
            "aria-expanded": "true", "aria-controls": "compare-autocomplete-title", "aria-activedescendant": undefined,
        });
        ac.onKeydown("title", key("ArrowDown"));
        expect(ac.comboboxAttrs("title")["aria-activedescendant"]).toBe("compare-autocomplete-title-option-0");
    });

    it("leaves a field without autocomplete as it is", () => {
        expect(autocomplete([]).comboboxAttrs("year")).toEqual({});
    });

    it("tells how many suggestions the open list has, and for what", async () => {
        const ac = autocomplete(['<span class="highlight">lib</span>re']);
        expect(ac.openList.value).toBe(null);
        await type(ac, "title", "lib");
        expect(ac.openList.value).toEqual({ count: 1, term: "lib" });
    });
});
