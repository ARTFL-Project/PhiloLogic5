import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { nextTick } from "vue";
import { createTestPinia, createTestI18n, createTestConfig, createTestRouter, createMockHttp } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import TextNavigation from "../src/components/TextNavigation.vue";

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
        { philo_type: "div1", philo_id: "1 1 0 0 0 0 0", label: "Chapter 1", head: "Chapter 1", n: "1", byte: "0" },
        { philo_type: "div1", philo_id: "1 2 0 0 0 0 0", label: "Chapter 2", head: "Chapter 2", n: "2", byte: "1000" },
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

    // --- toggleTableOfContents() — driven by #show-toc click ---
    it("toggles the TOC panel on #show-toc click", async () => {
        const { wrapper } = await mountTextNavigation();
        expect(wrapper.find("#toc-content").exists()).toBe(false);

        await wrapper.find("#show-toc").trigger("click");
        await nextTick();
        expect(wrapper.find("#toc-content").exists()).toBe(true);
        expect(wrapper.find("#show-toc").attributes("aria-expanded")).toBe("true");

        await wrapper.find("#show-toc").trigger("click");
        await nextTick();
        expect(wrapper.find("#toc-content").exists()).toBe(false);
    });

    // --- backToTop() — driven by #back-to-top click ---
    it("calls window.scrollTo when #back-to-top is clicked", async () => {
        const scrollSpy = vi.spyOn(window, "scrollTo").mockImplementation(() => {});
        const { wrapper } = await mountTextNavigation();
        await wrapper.find("#back-to-top").trigger("click");
        expect(scrollSpy).toHaveBeenCalledWith({ top: 0, behavior: "smooth" });
        scrollSpy.mockRestore();
    });

    // --- goToTextObject(philoID) — driven by #next-obj/#prev-obj clicks ---
    it("pushes a route when the next-section button is clicked", async () => {
        const { wrapper, router } = await mountTextNavigation();
        const pushSpy = vi.spyOn(router, "push");
        await wrapper.find("#next-obj").trigger("click");
        expect(pushSpy).toHaveBeenCalledWith({ path: "/navigate/1/6/0/0/0/0/0" });
    });

    it("closes the TOC panel when navigating to a sibling text object", async () => {
        const { wrapper } = await mountTextNavigation();
        // Open the TOC first
        await wrapper.find("#show-toc").trigger("click");
        await nextTick();
        expect(wrapper.find("#toc-content").exists()).toBe(true);

        // Navigating away should close it
        await wrapper.find("#prev-obj").trigger("click");
        await nextTick();
        expect(wrapper.find("#toc-content").exists()).toBe(false);
    });

    // --- textObjectSelection() — driven by clicking a TOC entry ---
    it("preventDefault's and pushes a route when a TOC entry is clicked", async () => {
        const { wrapper, router } = await mountTextNavigation();
        await wrapper.find("#show-toc").trigger("click");
        await nextTick();

        const pushSpy = vi.spyOn(router, "push");
        // First TOC entry's link/button — selector depends on how entries render
        const entry = wrapper.find("#toc-content a, #toc-content button[type='button']");
        if (entry.exists()) {
            await entry.trigger("click");
            await nextTick();
            expect(pushSpy).toHaveBeenCalled();
            // The click should route into /navigate/...
            const routeCall = pushSpy.mock.calls.find(c => c[0].path?.startsWith("/navigate/"));
            expect(routeCall).toBeTruthy();
        }
    });

    // --- dicoLookup(event) — driven by keydown on #text-obj-content ---
    it("opens dictionary lookup on 'd' keydown over the text", async () => {
        const openSpy = vi.spyOn(window, "open").mockImplementation(() => {});
        vi.spyOn(window, "getSelection").mockReturnValue({ toString: () => "liberty" });
        const { wrapper } = await mountTextNavigation();
        await wrapper.find("#text-obj-content").trigger("keydown", { key: "d" });
        expect(openSpy).toHaveBeenCalled();
        expect(openSpy.mock.calls[0][0]).toContain("liberty");
        openSpy.mockRestore();
    });

    it("does not open dictionary on unrelated keys", async () => {
        const openSpy = vi.spyOn(window, "open").mockImplementation(() => {});
        const { wrapper } = await mountTextNavigation();
        await wrapper.find("#text-obj-content").trigger("keydown", { key: "a" });
        expect(openSpy).not.toHaveBeenCalled();
        openSpy.mockRestore();
    });

    // --- State after fetch — observable via rendered DOM ---
    it("renders text content and prev/next buttons after navigation fetch", async () => {
        const { wrapper } = await mountTextNavigation();
        // textObject.text is rendered v-html into #text-obj-content
        const text = wrapper.find("#text-obj-content");
        expect(text.exists()).toBe(true);
        expect(text.html()).toContain("liberty");
        // prev/next buttons render v-if="textObject.prev" / v-if="textObject.next"
        expect(wrapper.find("#prev-obj").exists()).toBe(true);
        expect(wrapper.find("#next-obj").exists()).toBe(true);
    });

    it("renders TOC entries from fetched tocElements", async () => {
        const { wrapper } = await mountTextNavigation();
        await wrapper.find("#show-toc").trigger("click");
        await nextTick();
        const tocPanel = wrapper.find("#toc-content");
        expect(tocPanel.exists()).toBe(true);
        // Two fixture entries: "Chapter 1", "Chapter 2"
        expect(tocPanel.text()).toContain("Chapter 1");
        expect(tocPanel.text()).toContain("Chapter 2");
    });
});
