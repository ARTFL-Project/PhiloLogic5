import gsap from "gsap";

/**
 * Stagger-fade transition handlers for <transition-group> result lists.
 * Used by Bibliography, Concordance, and Kwic to fade in result items
 * with a small per-item delay driven by the element's data-index attribute.
 *
 * Returns Vue transition hook handlers `beforeEnter` and `enter`. Consumers
 * can destructure-rename if their template uses different names:
 *   const { beforeEnter: onBeforeEnter, enter: onEnter } = useFadeTransition();
 */
export function useFadeTransition(staggerDelay = 0.015) {
    return {
        beforeEnter(el) {
            el.style.opacity = 0;
        },
        enter(el, done) {
            gsap.to(el, {
                opacity: 1,
                delay: el.dataset.index * staggerDelay,
                onComplete: done,
            });
        },
    };
}
