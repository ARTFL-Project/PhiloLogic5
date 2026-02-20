import { reactive } from "vue";

/**
 * Composable for metadata field autocomplete.
 *
 * @param {Object} options
 * @param {Object} options.http - Axios instance (injected $http)
 * @param {string} options.dbUrl - Database URL (injected $dbUrl)
 * @param {Object} options.philoConfig - PhiloLogic config ($philoConfig)
 * @param {Object} options.metadataValues - Reactive object of field → current value
 * @param {Object} options.route - Vue Router route (for checking current field values)
 * @param {Function} [options.onSelect] - Optional callback after a result is selected: (field, value) => void
 */
export function useAutocomplete({ http, dbUrl, philoConfig, metadataValues, route, onSelect }) {
    const autoCompleteResults = reactive(
        Object.fromEntries(philoConfig.metadata.map((field) => [field, []]))
    );
    const arrowCounters = reactive(
        Object.fromEntries(philoConfig.metadata.map((field) => [field, -1]))
    );

    let timeout = null;

    function onChange(field) {
        if (!philoConfig.autocomplete.includes(field)) return;
        if (timeout) clearTimeout(timeout);
        timeout = setTimeout(() => {
            let currentFieldValue = route.query[field];
            if (metadataValues[field] && metadataValues[field].length > 1 && metadataValues[field] !== currentFieldValue) {
                http
                    .get(`${dbUrl}/scripts/autocomplete_metadata.py`, {
                        params: { term: metadataValues[field], field: field },
                    })
                    .then((response) => {
                        autoCompleteResults[field] = response.data.map((result) =>
                            result.replace(/CUTHERE/, "<last/>")
                        );
                    })
                    .catch(() => {});
            }
        }, 200);
    }

    function onArrowDown(field) {
        if (arrowCounters[field] < autoCompleteResults[field].length) {
            arrowCounters[field] = arrowCounters[field] + 1;
        }
        if (arrowCounters[field] > 5) {
            let container = document.getElementById(`autocomplete-${field}`);
            if (container) {
                container.scrollTop = container.scrollTop + 36;
            }
        }
    }

    function onArrowUp(field) {
        if (arrowCounters[field] > 0) {
            arrowCounters[field] = arrowCounters[field] - 1;
        }
        let container = document.getElementById(`autocomplete-${field}`);
        if (container) {
            container.scrollTop = container.scrollTop - 36;
        }
    }

    function onEnter(field) {
        let result = autoCompleteResults[field][arrowCounters[field]];
        setMetadataResult(result, field);
    }

    function setMetadataResult(inputString, field) {
        if (typeof inputString !== "undefined") {
            let prefix = "";
            let lastInput;
            if (inputString.match(/<last\/>/)) {
                let inputGroup = inputString.split(/<last\/>/);
                lastInput = inputGroup.pop();
                lastInput = lastInput.trim().replace(/<[^>]+>/g, "");
                prefix = inputGroup.join("");
            } else {
                lastInput = inputString.replace(/<[^>]+>/g, "").trim();
            }
            if (lastInput.match(/"/)) {
                if (lastInput.startsWith('"')) {
                    lastInput = lastInput.slice(1);
                }
                if (lastInput.endsWith('"')) {
                    lastInput = lastInput.slice(0, lastInput.length - 1);
                }
            }
            let finalInput = `${prefix}"${lastInput}"`;
            metadataValues[field] = finalInput;
            if (onSelect) {
                onSelect(field, finalInput);
            }
        }
        autoCompleteResults[field] = [];
        arrowCounters[field] = -1;
    }

    function clearAutoCompletePopup() {
        for (let field in autoCompleteResults) {
            autoCompleteResults[field] = [];
        }
    }

    function autoCompletePosition(field) {
        let parent = document.getElementById(`${field}-group`);
        if (parent) {
            let input = parent.querySelector("input");
            let childOffset = input.offsetLeft;
            return `left: ${childOffset}px; width: ${input.offsetWidth}px`;
        }
    }

    return {
        autoCompleteResults,
        arrowCounters,
        onChange,
        onArrowDown,
        onArrowUp,
        onEnter,
        setMetadataResult,
        clearAutoCompletePopup,
        autoCompletePosition,
    };
}
