import { computed, getCurrentScope, onScopeDispose, reactive } from "vue";
import { quoteMetadataValue } from "../utils.js";

const OR = " | ";

// How long typing must pause before suggestions are asked for: keys typed fast, or a paste, make one request. A request
// takes the server about a millisecond: the 200 ms waited when each one started a CGI process made the list lag.
export const AUTOCOMPLETE_DELAY = 50;

function stripTags(html) {
    return html.replace(/<[^>]+>/g, "");
}

/**
 * The words suggested for the last term of term: { html, value, selected } each, value as it goes in the query, and
 * what is kept before them. That is the rest of term as typed (the server's copy of it splits phrases), unless the
 * last term is inside an unclosed phrase, where the server's copy, which closes it, is kept.
 */
function termSuggestions(results, term) {
    let serverPrefix = "";
    const suggestions = results.map((html) => {
        const text = stripTags(html);
        const cut = text.search(/\S*$/); // the server's copy of the rest has no tags: as long in html as in text
        serverPrefix = text.slice(0, cut);
        const word = text.slice(cut).replace(/^"|"$/g, "");
        // word properties (lemma:, pos:) as they are, words quoted
        return { html: html.slice(cut).replace(/^"|"$/g, ""), value: word.includes(":") ? word : `"${word}"` };
    });
    const typedPrefix = term.trimEnd().replace(/[^\s|]*$/, "");
    const unclosed = (typedPrefix.match(/"/g) || []).length % 2 === 1;
    return { prefix: unclosed ? serverPrefix : typedPrefix, suggestions };
}

/**
 * The values suggested for the term being typed at the end of a metadata value, quoted, and what is kept before
 * them: the server's copy of the rest, up to CUTHERE, quoted already.
 */
function metadataSuggestions(results) {
    let prefix = "";
    const suggestions = results.map((result) => {
        const cut = result.lastIndexOf("CUTHERE");
        prefix = cut < 0 ? "" : result.slice(0, cut);
        const html = (cut < 0 ? result : result.slice(cut + "CUTHERE".length)).trim();
        return { html, value: quoteMetadataValue(stripTags(html).trim()) };
    });
    return { prefix, suggestions };
}

/**
 * Composable for search term and metadata field autocomplete: a list of suggestions under a field, each of which can
 * be added to or removed from it, the ones chosen joined by OR. The field is an ARIA combobox (comboboxAttrs), with
 * the keys of onKeydown: arrows to move through the list, Space to add or remove a suggestion, Enter to add it and
 * close the list, Escape to close it.
 *
 * @param {Object} options
 * @param {Object} options.http - Axios instance (injected $http)
 * @param {string} options.dbUrl - Database URL (injected $dbUrl)
 * @param {Object} options.philoConfig - PhiloLogic config ($philoConfig)
 * @param {Object} options.metadataValues - Reactive object of field → current value
 * @param {Object} options.route - Vue Router route (for checking current field values)
 * @param {Function} [options.onSelect] - Optional callback after a metadata value is set: (field, value) => void
 * @param {Object} [options.queryTerm] - Ref of the search terms, to complete them as field "q"
 * @param {string} [options.idPrefix] - Prefix of the ids of the lists, for more than one form in a page
 */
export function useAutocomplete({ http, dbUrl, philoConfig, metadataValues, route, onSelect, queryTerm, idPrefix = "" }) {
    const fields = queryTerm ? ["q", ...philoConfig.metadata] : philoConfig.metadata;
    const autoCompleteResults = reactive(Object.fromEntries(fields.map((field) => [field, []])));
    const arrowCounters = reactive(Object.fromEntries(fields.map((field) => [field, -1])));
    // field → the value its suggestions are for, and what is kept before the ones chosen
    const bases = reactive({});

    let timeout = null;
    if (getCurrentScope()) {
        onScopeDispose(() => clearTimeout(timeout));
    }

    function getValue(field) {
        return field === "q" ? queryTerm.value : metadataValues[field];
    }

    function setValue(field, value) {
        if (field === "q") {
            queryTerm.value = value;
            return;
        }
        metadataValues[field] = value;
        if (onSelect) {
            onSelect(field, value);
        }
    }

    function onChange(field) {
        if (!philoConfig.autocomplete.includes(field)) return;
        arrowCounters[field] = -1;
        const input = document.activeElement;
        if (timeout) clearTimeout(timeout);
        timeout = setTimeout(() => {
            const term = getValue(field) || "";
            if (term.replace(/"/g, "").trim().length < 2 || term === route.query[field]) {
                closeAutocomplete(field);
                return;
            }
            const [script, params] =
                field === "q" ? ["autocomplete_term.py", { term }] : ["autocomplete_metadata.py", { term, field }];
            http.get(`${dbUrl}/scripts/${script}`, { params })
                .then((response) => {
                    // not once the field is left or changed: a late list would cover what is clicked next
                    if (getValue(field) !== term || document.activeElement !== input) return;
                    const { prefix, suggestions } =
                        field === "q" ? termSuggestions(response.data, term) : metadataSuggestions(response.data);
                    bases[field] = { term, prefix };
                    autoCompleteResults[field] = suggestions.map((s) => ({ ...s, selected: false }));
                    arrowCounters[field] = -1;
                })
                .catch(() => {});
        }, AUTOCOMPLETE_DELAY);
    }

    function toggle(field, index) {
        const result = autoCompleteResults[field][index];
        if (!result) return;
        result.selected = !result.selected;
        const chosen = autoCompleteResults[field].filter((r) => r.selected).map((r) => r.value);
        const { term, prefix } = bases[field];
        setValue(field, chosen.length ? prefix + chosen.join(OR) : term);
    }

    function onKeydown(field, event) {
        const results = autoCompleteResults[field];
        if (!results?.length || event.isComposing) return;
        const active = arrowCounters[field];
        switch (event.key) {
            case "ArrowDown":
                event.preventDefault();
                arrowCounters[field] = Math.min(active + 1, results.length - 1);
                break;
            case "ArrowUp":
                // up from the first suggestion is back to the field: Enter searches again, Space types a space
                event.preventDefault();
                arrowCounters[field] = Math.max(active - 1, -1);
                break;
            case " ":
                if (active < 0) return;
                event.preventDefault();
                toggle(field, active);
                break;
            case "Enter":
                if (active < 0) return; // the form is submitted
                event.preventDefault();
                if (!results[active].selected) toggle(field, active);
                closeAutocomplete(field);
                break;
            case "Escape":
                event.preventDefault();
                closeAutocomplete(field);
                break;
        }
    }

    function closeAutocomplete(field) {
        autoCompleteResults[field] = [];
        arrowCounters[field] = -1;
        delete bases[field];
    }

    function clearAutoCompletePopup() {
        for (let field in autoCompleteResults) {
            closeAutocomplete(field);
        }
    }

    function listId(field) {
        return `${idPrefix}autocomplete-${field}`;
    }

    const instructionsId = `${idPrefix}autocomplete-instructions`;

    /** The ARIA attributes of a field with autocomplete, to bind on its input. */
    function comboboxAttrs(field) {
        if (!philoConfig.autocomplete.includes(field)) return {};
        const open = autoCompleteResults[field].length > 0;
        const active = arrowCounters[field];
        return {
            role: "combobox",
            "aria-autocomplete": "list",
            "aria-expanded": open ? "true" : "false",
            "aria-controls": open ? listId(field) : undefined,
            "aria-activedescendant": open && active >= 0 ? `${listId(field)}-option-${active}` : undefined,
            "aria-describedby": instructionsId,
        };
    }

    /** The open list, for a status message: { count, term }, or null. */
    const openList = computed(() => {
        const field = fields.find((f) => autoCompleteResults[f].length > 0);
        return field && bases[field] ? { count: autoCompleteResults[field].length, term: bases[field].term } : null;
    });

    function autoCompletePosition(field) {
        let parent = document.getElementById(`${idPrefix}${field}-group`);
        if (parent) {
            let input = parent.querySelector("input");
            if (input.offsetWidth < 240) {
                // under a narrow field (a small screen): as wide as the field with its label, to be read (WCAG 1.4.10)
                return `left: 0; margin-left: 0; width: ${parent.offsetWidth}px`;
            }
            let childOffset = input.offsetLeft;
            return `left: ${childOffset}px; width: ${input.offsetWidth}px`;
        }
    }

    return {
        autoCompleteResults,
        arrowCounters,
        onChange,
        onKeydown,
        toggle,
        closeAutocomplete,
        clearAutoCompletePopup,
        comboboxAttrs,
        listId,
        instructionsId,
        openList,
        autoCompletePosition,
    };
}
