import { nextTick, onBeforeUnmount, watch } from "vue";

/**
 * More of a long list as the page scrolls to its end: more() is called when the element after the list comes within
 * 200 px of the window's bottom, and again once it has shown more rows while that element still is (they may not
 * reach that far, as in a tall window). more() returns false when it has nothing more to show, else when the rows
 * are there (a promise, or at once).
 *
 * @param {Object} end - Template ref of the element after the list
 * @param {Function} more - Shows more of the list
 */
export function useLoadOnScroll(end, more) {
    if (typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver(
        async ([entry]) => {
            if (!entry.isIntersecting) return;
            if ((await more()) === false) return;
            await nextTick();
            // seen again: an element observed anew is reported where it is now
            if (end.value) {
                observer.unobserve(end.value);
                observer.observe(end.value);
            }
        },
        { rootMargin: "0px 0px 200px 0px" }
    );
    watch(
        end,
        (element, old) => {
            if (old) observer.unobserve(old);
            if (element) observer.observe(element);
        },
        { immediate: true, flush: "post" }
    );
    onBeforeUnmount(() => observer.disconnect());
}
