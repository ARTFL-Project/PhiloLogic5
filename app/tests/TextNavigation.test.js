import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { nextTick } from "vue";
import { createTestPinia, createTestI18n, createTestConfig, createTestRouter, createMockHttp } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import {
    paramsFilter, paramsToRoute, paramsToUrlString, copyObject, saveToLocalStorage,
    mergeResults, sortResults, deepEqual, dictionaryLookup, dateRangeHandler,
    buildBiblioCriteria, extractSurfaceFromCollocate, debug, isOnlyFacetChange, buildTocTree,
} from "../src/mixins.js";
import TextNavigation from "../src/components/TextNavigation.vue";

const mixinMethods = {
    paramsFilter, paramsToRoute, paramsToUrlString, copyObject, saveToLocalStorage,
    mergeResults, sortResults, deepEqual, dictionaryLookup, dateRangeHandler,
    buildBiblioCriteria, extractSurfaceFromCollocate, debug, isOnlyFacetChange, buildTocTree,
};

const navFixture = {
    text: "<p>Sample text with liberty.</p>",
    current_obj_img: [],
    citation: [{ label: "Brown", href: "", style: {}, prefix: "", suffix: "" }],
    prev: "1 4 0 0 0 0 0",
    next: "1 6 0 0 0 0 0",
    toc: [],
    page_images: [],
    graphics: [],
    imgs: {},
    metadata_fields: { year: "1798" },
};

const tocFixture = {
    toc: [
        { philo_type: "div1", philo_id: "1 1 0 0 0 0 0", head: "Chapter 1", n: "1", byte: "0" },
        { philo_type: "div1", philo_id: "1 2 0 0 0 0 0", head: "Chapter 2", n: "2", byte: "1000" },
    ],
    current_obj_position: 50,
};

let consoleWarnSpy;
let consoleLogSpy;

async function mountTextNavigation(overrides = {}) {
    const http = overrides.http || createMockHttp({
        "navigation.py": navFixture,
        "get_table_of_contents.py": tocFixture,
    });
    const pinia = createTestPinia();
    const i18n = createTestI18n();
    const config = createTestConfig();
    const router = createTestRouter({
        name: "textNavigation",
        path: "/navigate/1/5",
        query: {},
    });
    await router.isReady();

    const store = useMainStore();
    store.formData = { ...store.formData, report: "textNavigation" };

    // Suppress expected DOM warnings/errors
    consoleWarnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    consoleLogSpy = vi.spyOn(console, "log").mockImplementation(() => {});

    // Create stub DOM elements that mounted() expects
    const stubIds = ["show-toc", "toc-top-bar", "toc-wrapper", "nav-buttons"];
    for (const id of stubIds) {
        if (!document.getElementById(id)) {
            const el = document.createElement("div");
            el.id = id;
            el.getBoundingClientRect = () => ({ top: 100, left: 0, width: 100, height: 30 });
            document.body.appendChild(el);
        }
    }

    const wrapper = mount(TextNavigation, {
        global: {
            plugins: [pinia, i18n, router],
            provide: { $http: http, $dbUrl: "/testdb", $philoConfig: config },
            mixins: [{ methods: mixinMethods }],
            stubs: {
                Citations: { template: "<span class='citations-stub' />" },
                ProgressSpinner: { template: "<div class='spinner-stub' />" },
            },
            mocks: { $philoConfig: config, $dbUrl: "/testdb", $scrollTo: vi.fn() },
            directives: { scroll: { mounted() {}, unmounted() {} } },
        },
    });

    // Let async operations settle (ignore DOM errors)
    try { await flushPromises(); } catch {}
    await nextTick();

    return { wrapper, http, router };
}

afterEach(() => {
    if (consoleWarnSpy) consoleWarnSpy.mockRestore();
    if (consoleLogSpy) consoleLogSpy.mockRestore();
});

describe("TextNavigation", () => {
    // --- HTTP requests ---
    it("fetches text and TOC on mount", async () => {
        const http = createMockHttp({
            "navigation.py": navFixture,
            "get_table_of_contents.py": tocFixture,
        });
        await mountTextNavigation({ http });
        const navCalls = http.get.mock.calls.filter(c => c[0].includes("navigation.py"));
        const tocCalls = http.get.mock.calls.filter(c => c[0].includes("get_table_of_contents"));
        expect(navCalls.length).toBe(1);
        expect(tocCalls.length).toBe(1);
    });

    it("passes philo_id in navigation request", async () => {
        const http = createMockHttp({
            "navigation.py": navFixture,
            "get_table_of_contents.py": tocFixture,
        });
        await mountTextNavigation({ http });
        const navCall = http.get.mock.calls.find(c => c[0].includes("navigation.py"));
        expect(navCall[0]).toMatch(/philo_id=1(\+|%20| )5/);
    });

    // --- toggleTableOfContents() ---
    it("toggles tocOpen state", async () => {
        const { wrapper } = await mountTextNavigation();
        expect(wrapper.vm.tocOpen).toBe(false);
        wrapper.vm.toggleTableOfContents();
        expect(wrapper.vm.tocOpen).toBe(true);
        wrapper.vm.toggleTableOfContents();
        expect(wrapper.vm.tocOpen).toBe(false);
    });

    // --- backToTop() ---
    it("calls window.scrollTo on backToTop", async () => {
        const scrollSpy = vi.spyOn(window, "scrollTo").mockImplementation(() => {});
        const { wrapper } = await mountTextNavigation();
        wrapper.vm.backToTop();
        expect(scrollSpy).toHaveBeenCalledWith({ top: 0, behavior: "smooth" });
        scrollSpy.mockRestore();
    });

    // --- goToTextObject(philoID) ---
    it("navigates to text object by formatting philo_id", async () => {
        const { wrapper, router } = await mountTextNavigation();
        const pushSpy = vi.spyOn(router, "push");
        wrapper.vm.goToTextObject("1 6 0 0 0 0 0");
        expect(pushSpy).toHaveBeenCalledWith({ path: "/navigate/1/6/0/0/0/0/0" });
    });

    it("closes TOC when navigating to text object", async () => {
        const { wrapper } = await mountTextNavigation();
        wrapper.vm.tocOpen = true;
        wrapper.vm.goToTextObject("1 6 0 0 0 0 0");
        expect(wrapper.vm.tocOpen).toBe(false);
    });

    it("handles hyphenated philo_id format", async () => {
        const { wrapper, router } = await mountTextNavigation();
        const pushSpy = vi.spyOn(router, "push");
        wrapper.vm.goToTextObject("1-6-0-0-0-0-0");
        expect(pushSpy).toHaveBeenCalledWith({ path: "/navigate/1/6/0/0/0/0/0" });
    });

    // --- textObjectSelection() ---
    it("adjusts TOC bounds and navigates on selection", async () => {
        const { wrapper, router } = await mountTextNavigation();
        wrapper.vm.tocElements = {
            docId: "1",
            elements: tocFixture.toc,
            start: 50,
            end: 150,
        };
        const pushSpy = vi.spyOn(router, "push");
        const mockEvent = { preventDefault: vi.fn() };
        wrapper.vm.textObjectSelection("1 2 0 0 0 0 0", 10, mockEvent);
        expect(mockEvent.preventDefault).toHaveBeenCalled();
        expect(pushSpy).toHaveBeenCalled();
    });

    // --- loadBefore() / loadAfter() ---
    it("decreases start index on loadBefore", async () => {
        const { wrapper } = await mountTextNavigation();
        wrapper.vm.start = 300;
        wrapper.vm.loadBefore();
        expect(wrapper.vm.start).toBe(100);
    });

    it("clamps start to 0 on loadBefore", async () => {
        const { wrapper } = await mountTextNavigation();
        wrapper.vm.start = 50;
        wrapper.vm.loadBefore();
        expect(wrapper.vm.start).toBe(0);
    });

    it("increases end index on loadAfter", async () => {
        const { wrapper } = await mountTextNavigation();
        wrapper.vm.end = 100;
        wrapper.vm.loadAfter();
        expect(wrapper.vm.end).toBe(300);
    });

    // --- dicoLookup(event) ---
    it("opens dictionary lookup on 'd' key", async () => {
        const openSpy = vi.spyOn(window, "open").mockImplementation(() => {});
        const { wrapper } = await mountTextNavigation();
        // Mock window.getSelection
        vi.spyOn(window, "getSelection").mockReturnValue({ toString: () => "liberty" });
        wrapper.vm.dicoLookup({ key: "d" });
        expect(openSpy).toHaveBeenCalled();
        expect(openSpy.mock.calls[0][0]).toContain("liberty");
        openSpy.mockRestore();
    });

    it("does not open dictionary on other keys", async () => {
        const openSpy = vi.spyOn(window, "open").mockImplementation(() => {});
        const { wrapper } = await mountTextNavigation();
        wrapper.vm.dicoLookup({ key: "a" });
        expect(openSpy).not.toHaveBeenCalled();
        openSpy.mockRestore();
    });

    // --- State after fetch ---
    it("sets textObject after navigation fetch", async () => {
        const { wrapper } = await mountTextNavigation();
        expect(wrapper.vm.textObject).toBeTruthy();
        expect(wrapper.vm.textObject.prev).toBe("1 4 0 0 0 0 0");
        expect(wrapper.vm.textObject.next).toBe("1 6 0 0 0 0 0");
    });

    it("stores TOC elements after fetch", async () => {
        const { wrapper } = await mountTextNavigation();
        expect(wrapper.vm.tocElements.docId).toBe("1");
        expect(wrapper.vm.tocElements.elements.length).toBe(2);
    });
});
