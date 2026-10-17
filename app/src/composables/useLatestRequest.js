import { deepEqual } from "../utils.js";

/**
 * Requests of which only the latest's answers are wanted: start() aborts the one in flight and gives the AbortSignal
 * for the next; requests started with the same key (one search's) share theirs. finish(signal) says whether a request
 * that ended was the latest, abort() ends the one in flight (leaving the page) and says whether there was one.
 *
 * @returns {{ start: Function, finish: Function, abort: Function }}
 */
export function useLatestRequest() {
    let controller = null;
    let current; // the key of the requests in flight

    function start(key) {
        if (key === undefined || !controller || !deepEqual(key, current)) {
            controller?.abort();
            controller = new AbortController();
            current = key;
        }
        return controller.signal;
    }

    function finish(signal) {
        if (controller?.signal !== signal) return false;
        controller = null;
        return true;
    }

    function abort() {
        if (!controller) return false;
        controller.abort();
        controller = null;
        return true;
    }

    return { start, finish, abort };
}
