import { useMainStore } from "./stores/main";

export function dbUrlOnThisHost(dbUrl, location = window.location) {
    // The database's URL on the host this page came from: behind a proxy such as EZproxy, which rewrites host names
    // in pages but not in this code, requests must go through the proxy too, as it is what access control lets in,
    // and it holds the cookies.
    return location.origin + new URL(dbUrl, location.href).pathname;
}

export function databaseUrl(doc = document, devUrl = import.meta.env.VITE_DB_URL) {
    // The database's URL: the <base href> the server gives its pages, the database's own path. The client is built
    // for no path in particular, so a database can be served under any prefix, or copied to another machine,
    // without rebuilding it. With `npm run dev`, VITE_DB_URL is the database to work with (in .env.development.local).
    return devUrl || doc.baseURI;
}

export function refusedSearchMessage(error) {
    // Why the server refused a search (a 400 for a report or the total of results), or "" for any other error
    const url = error?.config?.url || "";
    if (error?.response?.status !== 400 || !/\/(reports\/\w+|scripts\/get_total_results)\.py/.test(url)) {
        return "";
    }
    return error.response.data?.description || "";
}

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
            formValues.report,
        )
    ) {
        report = "bibliography";
    } else {
        report = formValues.report;
    }
    // the fields of the report the route goes to: with no search terms, a concordance's became a bibliography URL
    // with cooc_order and method_arg
    let localFormData = paramsFilter({ ...formValues, report });
    let routeObject = {
        path: `/${report}`,
        query: localFormData,
    };
    return routeObject;
}

// A metadata value quoted, which matches it exactly: its quotes doubled ("Les Révoltés de la ""Bounty""")
export function quoteMetadataValue(value) {
    return `"${String(value).replace(/"/g, '""')}"`;
}

export function paramsToUrlString(params) {
    let filteredParams = paramsFilter(params);
    let queryParams = [];
    for (let param in filteredParams) {
        queryParams.push(
            `${param}=${encodeURIComponent(filteredParams[param])}`,
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
    let sortedList = sortResults(fullResults, sortKey);
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

export function dictionaryLookup(event, year, philoConfig) {
    if (event.key === "d") {
        var selection = window.getSelection().toString();
        var century = parseInt(year.slice(0, year.length - 2));
        var range = century.toString() + "00-" + String(century + 1) + "00";
        if (range == "NaN00-NaN00") {
            range = "";
        }
        var link =
            philoConfig.dictionary_lookup +
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
    metadataValues,
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
                metadataValues[metadata] =
                    `${dateRange[metadata].start}${separator}${dateRange[metadata].end}`;
            } else if (
                dateRange[metadata].start.length > 0 &&
                dateRange[metadata].end.length == 0
            ) {
                metadataValues[metadata] =
                    `${dateRange[metadata].start}${separator}`;
            } else if (
                dateRange[metadata].start.length == 0 &&
                dateRange[metadata].end.length > 0
            ) {
                metadataValues[metadata] =
                    `${separator}${dateRange[metadata].end}`;
            } else {
                metadataValues[metadata] = ""; // both boxes emptied: no range left
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

// The request parameter of the values kept of a word property, for collocates (colloc_attr_pos=NOUN | ADJ)
export const PROPERTY_PARAM = "colloc_attr_";

// The word properties collocates are filtered by, of formData: [name, values] pairs, by name, the values in order
export function collocateProperties(formData, wordAttributes = {}) {
    const properties = [];
    for (const name of Object.keys(wordAttributes).sort()) {
        const values = splitValues(formData[`${PROPERTY_PARAM}${name}`]);
        if (values.length > 0) properties.push([name, values.sort()]);
    }
    return properties;
}

// The values of a property parameter: those " | " separates
export function splitValues(value) {
    return [...new Set(String(value || "").split("|").map((v) => v.trim()).filter((v) => v))];
}

// The query of a concordance of a collocate: the hits it was counted in. With word properties, the collocate with
// one of the kept values of each (as the report's collocate_keys)
export function collocateCleanup(collocate, q, formData = {}, wordAttributes = {}) {
    const properties = collocateProperties(formData, wordAttributes);
    if (properties.length > 0) {
        let keys = [collocate.surfaceForm];
        for (const [name, values] of properties) {
            keys = keys.flatMap((key) => values.map((value) => `${key}:${name}:${value}`));
        }
        return `${q} ${keys.join(" | ")}`;
    }
    if (collocate.surfaceForm.startsWith("lemma:") || collocate.surfaceForm.search(/\w+:.*/) !== -1) {
        return `${q} ${collocate.surfaceForm}`;
    }
    return `${q} "${collocate.surfaceForm}"`;
}

// The common words a new collocation search leaves out: the database's stopwords, else the most frequent words
export function defaultCommonWords(philoConfig) {
    return philoConfig.stopwords?.length > 0 ? "stopwords" : "frequency";
}

// The collocation parameters of links made before they could be combined, as the form has them now: collocates
// counted as lemmas when the query has one (without colloc_lemmas), and colloc_filter_choice=attribute as the one
// value q_attribute_value of q_attribute, without any other filter. Changes formData; the counts are the same.
export function upgradeCollocationParams(formData, query) {
    if (!formData.colloc_filter_choice) {  // the most frequent words, as the report has it
        formData.colloc_filter_choice = "frequency";
    }
    if (!("colloc_lemmas" in query)) {
        formData.colloc_lemmas = String(formData.q || "").includes("lemma:") ? "yes" : "no";
    }
    if (formData.colloc_filter_choice === "attribute") {
        if (formData.q_attribute) {
            formData[`${PROPERTY_PARAM}${formData.q_attribute}`] = formData.q_attribute_value || "";
        }
        formData.colloc_filter_choice = "nofilter";
    }
    formData.q_attribute = "";
    formData.q_attribute_value = "";
    return formData;
}

// The parameters of a collocation report of the form's query and filters, with metadata instead of the form's when
// given (comparing with other metadata). Not the comparison's, which the form keeps from the URL.
export function collocationParams(formData, philoConfig, metadata = null) {
    const params = paramsFilter({ ...formData, report: "collocation" });
    for (const key of Object.keys(params)) {
        if (key.startsWith("compare_")) delete params[key];
    }
    if (metadata !== null) {
        for (const field of philoConfig.metadata) delete params[field];
        Object.assign(params, paramsFilter({ ...metadata }));
    }
    return params;
}

export function concordanceMethod(collocWithin) {
    return collocWithin === "n" ? "proxy" : "sentence";
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
    const name = component?.$options?.name ?? "unknown";
    console.log(`MESSAGE FROM ${name}:`, message);
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

// The position to go back to, once the page is long enough to scroll there (or after 3 s): the results load after
// the route changes, so the page was too short, and coming back from a text went back to the top
export function whenTallEnough(position, timeout = 3000) {
    const start = Date.now();
    return new Promise((resolve) => {
        const check = () => {
            const tallEnough = document.documentElement.scrollHeight >= position.top + window.innerHeight;
            if (tallEnough || Date.now() - start >= timeout) {
                resolve(position);
            } else {
                setTimeout(check, 100);
            }
        };
        check();
    });
}
