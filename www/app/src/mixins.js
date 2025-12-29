import { useMainStore } from "./stores/main";

export function paramsFilter(formValues) {
    let localFormData = {};
    let validFields = [];
    const store = useMainStore();

    if ("report" in formValues && formValues.report in store.reportValues) {
        validFields = store.reportValues[formValues.report];
    } else {
        validFields = new Set(Object.keys(formValues));
    }
    if (formValues.approximate == "no") {
        formValues.approximate_ratio = "";
    }
    for (const field in formValues) {
        let value = formValues[field];
        if (field === "report") {
            continue;
        }
        // Check if this field should skip validFields validation
        const facetSupportedReports = ["concordance", "kwic", "bibliography"];
        const isFacetParam =
            field === "facet" ||
            field === "relative_frequency" ||
            field === "word_property";
        const isCollocationParam =
            field === "collocation_method" ||
            field === "similarity_by" ||
            field === "time_series_interval";
        const isCompareParam =
            field.startsWith("compare_") && formValues.report === "collocation";

        const shouldSkipValidation =
            (isFacetParam &&
                facetSupportedReports.includes(formValues.report)) ||
            (isCollocationParam && formValues.report === "collocation") ||
            isCompareParam;

        if (!shouldSkipValidation && !validFields.has(field)) {
            continue;
        }
        if (
            (value !== undefined && value !== null && value.length > 0) ||
            field === "results_per_page"
        ) {
            if (
                (field === "method" && value === "proxy") ||
                (field === "approximate" && value == "no") ||
                (field === "sort_by" && value === "rowid")
            ) {
                continue;
            } else if (
                (field == "start" && value == 0) ||
                (field == "end" && value == 0)
            ) {
                continue;
            } else {
                localFormData[field] = value;
            }
        } else if (
            field == "hit_num" ||
            field == "start_date" ||
            field == "end_date" ||
            field == "method_arg"
        ) {
            localFormData[field] = value;
        }
    }

    return localFormData;
}
export function paramsToRoute(formValues) {
    let report;
    if (
        (!formValues.q || formValues.q.length == 0) &&
        !["bibliography", "aggregation", "time_series"].includes(
            formValues.report
        )
    ) {
        report = "bibliography";
    } else {
        report = formValues.report;
    }
    let localFormData = this.paramsFilter(formValues);
    let routeObject = {
        path: `/${report}`,
        query: localFormData,
    };
    return routeObject;
}
export function paramsToUrlString(params) {
    let filteredParams = this.paramsFilter(params);
    let queryParams = [];
    for (let param in filteredParams) {
        queryParams.push(
            `${param}=${encodeURIComponent(filteredParams[param])}`
        );
    }
    return queryParams.join("&");
}
export function copyObject(objectToCopy) {
    return JSON.parse(JSON.stringify(objectToCopy));
}
export function saveToLocalStorage(urlString, results) {
    try {
        sessionStorage[urlString] = JSON.stringify(results);
        console.log("saved results to localStorage");
    } catch (e) {
        sessionStorage.clear();
        console.log("Clearing sessionStorage for space...");
        try {
            sessionStorage[urlString] = JSON.stringify(results);
            console.log("saved results to localStorage");
        } catch (e) {
            sessionStorage.clear();
            console.log("Quota exceeded error: the JSON object is too big...");
        }
    }
}
export function mergeResults(fullResults, newData, sortKey) {
    if (
        typeof fullResults === "undefined" ||
        Object.keys(fullResults).length === 0
    ) {
        fullResults = newData;
    } else {
        for (let key in newData) {
            let value = newData[key];
            if (typeof value.count !== "undefined") {
                if (key in fullResults) {
                    fullResults[key].count += value.count;
                } else {
                    fullResults[key] = value;
                }
            }
        }
    }
    let sortedList = this.sortResults(fullResults, sortKey);
    return {
        sorted: sortedList,
        unsorted: fullResults,
    };
}
export function sortResults(fullResults, sortKey) {
    let sortedList = [];
    for (let key in fullResults) {
        sortedList.push({
            label: key,
            count: parseFloat(fullResults[key].count),
            metadata: fullResults[key].metadata,
        });
    }
    if (sortKey === "label") {
        sortedList.sort(function (a, b) {
            return a.label - b.label;
        });
    } else {
        sortedList.sort(function (a, b) {
            return b.count - a.count;
        });
    }
    return sortedList;
}
export function deepEqual(x, y) {
    const ok = Object.keys,
        tx = typeof x,
        ty = typeof y;
    return x && y && tx === "object" && tx === ty
        ? ok(x).length === ok(y).length &&
              ok(x).every((key) => deepEqual(x[key], y[key]))
        : x === y;
}
export function dictionaryLookup(event, year) {
    if (event.key === "d") {
        var selection = window.getSelection().toString();
        var century = parseInt(year.slice(0, year.length - 2));
        var range = century.toString() + "00-" + String(century + 1) + "00";
        if (range == "NaN00-NaN00") {
            range = "";
        }
        var link =
            this.$philoConfig.dictionary_lookup +
            "?docyear=" +
            range +
            "&strippedhw=" +
            selection;
        window.open(link);
    }
}

export function dateRangeHandler(
    metadataInputStyle,
    dateRange,
    dateType,
    metadataValues
) {
    for (let metadata in metadataInputStyle) {
        if (
            ["date", "int"].includes(metadataInputStyle[metadata]) &&
            dateType[metadata] != "exact"
        ) {
            let separator = "-";
            if (metadataInputStyle[metadata] == "date") {
                separator = "<=>";
            }
            if (
                dateRange[metadata].start.length > 0 &&
                dateRange[metadata].end.length > 0
            ) {
                metadataValues[
                    metadata
                ] = `${dateRange[metadata].start}${separator}${dateRange[metadata].end}`;
            } else if (
                dateRange[metadata].start.length > 0 &&
                dateRange[metadata].end.length == 0
            ) {
                metadataValues[
                    metadata
                ] = `${dateRange[metadata].start}${separator}`;
            } else if (
                dateRange[metadata].start.length == 0 &&
                dateRange[metadata].end.length > 0
            ) {
                metadataValues[
                    metadata
                ] = `${separator}${dateRange[metadata].end}`;
            }
        }
    }
    return metadataValues;
}

export function buildBiblioCriteria(philoConfig, query, formData) {
    let queryArgs = {};
    for (let field of philoConfig.metadata) {
        if (field in query && formData[field].length > 0) {
            queryArgs[field] = formData[field];
        }
    }
    let biblio = [];
    if (queryArgs.report === "time_series") {
        delete queryArgs[philoConfig.time_series_year_field];
    }
    let config = philoConfig;
    let facets = [];
    for (let i = 0; i < config.facets.length; i++) {
        let alias = Object.keys(config.facets[i])[0];
        let facet = config.facets[i][alias];
        if (typeof facet == "string") {
            facets.push(facet);
        } else {
            for (let value of facets) {
                if (facets.indexOf(value) < 0) {
                    facets.push(value);
                }
            }
        }
    }
    for (let k in queryArgs) {
        if (config.available_metadata.indexOf(k) >= 0) {
            if (this.report == "time_series" && k == "year") {
                continue;
            }
            let v = queryArgs[k];
            let alias = k;
            if (v) {
                if (k in config.metadata_aliases) {
                    alias = config.metadata_aliases[k];
                }
                biblio.push({ key: k, alias: alias, value: v });
            }
        }
    }
    return biblio;
}

export function extractSurfaceFromCollocate(words) {
    let newWords = [];
    for (let wordObj of words) {
        let collocate = `${wordObj[0]}`.replace(/lemma:/, "");
        if (collocate.search(/\w+:.*/) != -1) {
            collocate = collocate.replace(/(\p{L}+):.*/u, "$1");
        }
        let surfaceForm = wordObj[0];
        newWords.push({
            collocate: collocate,
            surfaceForm: surfaceForm,
            count: wordObj[1],
        });
    }
    return newWords;
}

export function debug(component, message) {
    console.log(`MESSAGE FROM ${component.$options.name}:`, message);
}

export function isOnlyFacetChange(newUrl, oldUrl) {
    const differences = [];
    const ok = Object.keys;

    // Get all unique keys from both objects
    const allKeys = new Set([...ok(newUrl || {}), ...ok(oldUrl || {})]);

    for (const key of allKeys) {
        if (newUrl?.[key] != oldUrl?.[key]) {
            if (
                key != "facet" &&
                key != "relative_frequency" &&
                key != "collocation_method" &&
                key != "similarity_by" &&
                key != "word_property"
            ) {
                return false;
            }
            differences.push(key);
        }
    }

    return differences.length > 0;
}

export function buildTocTree(elements) {
    const tree = [];
    const typeHierarchy = {
        div1: 1,
        div2: 2,
        div3: 3,
    };

    let stack = [];

    for (let element of elements) {
        const elementLevel = typeHierarchy[element.philo_type] || 1;
        element.level = elementLevel;
        element.children = [];

        // Find the appropriate parent in the stack
        while (
            stack.length > 0 &&
            stack[stack.length - 1].level >= elementLevel
        ) {
            stack.pop();
        }

        if (stack.length === 0) {
            // This is a top-level element
            tree.push(element);
        } else {
            // This is a child of the last element in the stack
            stack[stack.length - 1].children.push(element);
        }

        stack.push(element);
    }

    return tree;
}

export default {
    paramsFilter,
    paramsToRoute,
    paramsToUrlString,
    copyObject,
    saveToLocalStorage,
    mergeResults,
    sortResults,
    deepEqual,
    dictionaryLookup,
    dateRangeHandler,
    buildBiblioCriteria,
    extractSurfaceFromCollocate,
    debug,
    isOnlyFacetChange,
    buildTocTree,
};
