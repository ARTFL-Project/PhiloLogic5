import { onActivated, onDeactivated } from "vue";
import { isViewOnlyParam } from "../../utils.js";

/**
 * A senses tab kept, with its result, while another tab is shown. The page asks for its result each time it is shown
 * (and a tab loaded on demand fetches on its own as it mounts): wanted() says whether to fetch it, not when the result
 * of the same parameters, but those of the view, is there or coming. The words it left out (filter_list) and whether the stopwords were
 * missing go to the results summary above the tabs when its result comes while it is shown, and each time it is shown.
 *
 * @param {Function} emit - The tab's emit, with the filterList and stopwordsMissing events
 * @param {Object} result - Ref of the tab's result
 * @param {Object} loading - Ref of whether it is coming
 * @returns {{ wanted: Function, giveFilters: Function }} wanted(params), giveFilters(result)
 */
export function useKeptTab(emit, result, loading) {
    let fetchedFor = null; // the parameters of the result there, or coming
    let shown = true; // false while the tab is kept and another shown

    function wanted(params) {
        // as the URL has them: strings (the form's defaults are numbers until a URL gives them)
        const search = JSON.stringify(
            Object.entries(params)
                .filter(([key]) => !isViewOnlyParam(key))
                .map(([key, value]) => [key, String(value)])
                .sort()
        );
        if (search === fetchedFor && (result.value || loading.value)) return false;
        fetchedFor = search;
        return true;
    }

    function giveFilters(data) {
        if (!shown) return;
        emit("filterList", data?.filter_list || []);
        emit("stopwordsMissing", Boolean(data?.stopwords_missing));
    }

    onActivated(() => {
        shown = true;
        if (result.value) giveFilters(result.value);
    });
    onDeactivated(() => {
        shown = false;
    });
    return { wanted, giveFilters };
}
