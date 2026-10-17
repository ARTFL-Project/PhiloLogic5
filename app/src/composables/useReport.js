import { storeToRefs } from "pinia";
import { onScopeDispose, watch } from "vue";
import { useMainStore } from "../stores/main";
import { debug, isOnlyFacetChange } from "../utils.js";
import { useLatestRequest } from "./useLatestRequest";

/**
 * The search of a report's page, one at a time. The report is the one the form and the store's currentReport say;
 * run() fetches its results with fetch(signal), which makes its requests with the AbortSignal it gets, while the
 * store's searching says so, an error logged. A new search aborts the one in flight, whose response, coming last,
 * would replace the new one's (paging fast), and so does leaving the page.
 *
 * With watchUrl, the search runs again when the URL changes it: not when only the facets change, which are fetched on
 * their own, nor for another report's URL, which the page is leaving for. The page runs its first search itself.
 *
 * @param {string} report - The report's name, as the route has it
 * @param {Function} fetch - (signal) => Promise, fetching the results and showing them
 * @param {Object} [options]
 * @param {boolean} [options.watchUrl=true] - Whether a change of the URL's search runs it again
 * @returns {{ run: Function }}
 */
export function useReport(report, fetch, { watchUrl = true } = {}) {
    const store = useMainStore();
    const { formData, currentReport, searching, urlUpdate } = storeToRefs(store);
    const search = useLatestRequest();

    async function run() {
        const signal = search.start();
        searching.value = true;
        try {
            await fetch(signal);
        } catch (error) {
            if (!signal.aborted) debug({ $options: { name: `${report}-report` } }, error);
        } finally {
            // Not when aborted: a newer search is in flight, or the page is gone
            if (search.finish(signal)) searching.value = false;
        }
    }

    if (watchUrl) {
        watch(urlUpdate, (newUrl, oldUrl) => {
            if (formData.value.report === report && !isOnlyFacetChange(newUrl, oldUrl)) run();
        });
    }

    // Leaving the page, its search in flight ends: the next page's starts after
    onScopeDispose(() => {
        if (search.abort()) searching.value = false;
    });

    formData.value.report = report;
    currentReport.value = report;
    return { run };
}
