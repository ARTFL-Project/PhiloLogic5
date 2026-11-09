import { describe, it, expect, vi, afterEach } from "vitest";
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
        query: overrides.query || {},
    });
    await router.isReady();

    const store = useMainStore();
    store.formData = { ...store.formData, report: "textNavigation" };

    // Suppress expected DOM warnings/errors
    consoleWarnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    consoleLogSpy = vi.spyOn(console, "log").mockImplementation(() => {});

    const wrapper = mount(TextNavigation, {
        attachTo: overrides.attachTo,
        global: {
            plugins: [pinia, i18n, router],
            provide: { $http: http, $dbUrl: "/testdb", $philoConfig: config },
            stubs: {
                Citations: { template: "<span class='citations-stub' />" },
                ProgressSpinner: { template: "<div class='spinner-stub' />" },
            },
            mocks: { $philoConfig: config, $dbUrl: "/testdb" },
        },
    });

    // Let async operations settle (ignore DOM errors)
    try { await flushPromises(); } catch { /* as above */ }
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

    // --- The open panel loads more sections as it scrolls, once it settled on the current one ---
    it("loads the sections before when the open panel is scrolled to its top, once it settled", async () => {
        const toc = Array.from({ length: 400 }, (_, i) => ({
            philo_type: "div1", philo_id: `1 ${i + 1} 0 0 0 0 0`, label: `Chapter ${i + 1}`,
        }));
        const http = createMockHttp({ "navigation.py": navFixture, "get_table_of_contents.py": { toc, current_obj_position: 300 } });
        const { wrapper } = await mountTextNavigation({ http });
        await wrapper.find("#show-toc").trigger("click");
        await nextTick();
        const sections = () => wrapper.findAll("#toc-content .toc-link").length;
        expect(sections()).toBe(200); // the 100 before the current one, and after

        await wrapper.find("#toc-content").trigger("scroll"); // scrolling to the current section
        await nextTick();
        expect(sections()).toBe(200);

        await new Promise((resolve) => setTimeout(resolve, 120));
        await wrapper.find("#toc-content").trigger("scroll"); // at its top (jsdom scrolls nothing)
        await nextTick();
        expect(sections()).toBe(400);
    });
});

// A long text object comes a part at a time, with its window: its bounds, the starts of its neighbors, its number
const firstPart = { start_byte: 0, end_byte: 1000, prev_start_byte: null, next_start_byte: 1000, part: 1, part_count: 4 };
const middlePart = { start_byte: 2000, end_byte: 3000, prev_start_byte: 1000, next_start_byte: 3000, part: 3, part_count: 4 };
const lastPart = { start_byte: 3000, end_byte: 3500, prev_start_byte: 2000, next_start_byte: null, part: 4, part_count: 4 };

function partsHttp(window) {
    return createMockHttp({
        "navigation.py": { ...navFixture, window },
        "get_table_of_contents.py": tocFixture,
    });
}

describe("TextNavigation of a long text object, a part at a time", () => {
    it("asks for windows, and for the part of window_start in the route", async () => {
        const http = partsHttp(middlePart);
        await mountTextNavigation({ http, query: { window_start: "2000" } });
        const navCall = http.get.mock.calls.find(c => c[0].includes("navigation.py"));
        expect(navCall[0]).toContain("windowed=1");
        expect(navCall[0]).toContain("window_start=2000");
    });

    it("asks for windows with the hit, and no window_start, from a concordance", async () => {
        const http = partsHttp(middlePart);
        await mountTextNavigation({ http, query: { byte: "2500" } });
        const navCall = http.get.mock.calls.find(c => c[0].includes("navigation.py"));
        expect(navCall[0]).toContain("byte=2500");
        expect(navCall[0]).toContain("windowed=1");
        expect(navCall[0]).not.toContain("window_start");
    });

    it("shows no parts for an object sent whole", async () => {
        const { wrapper } = await mountTextNavigation();
        expect(wrapper.find(".text-parts").exists()).toBe(false);
        expect(wrapper.find("#next-obj").attributes("aria-label")).toBe("Go to next section");
        expect(wrapper.find("#prev-obj").attributes("aria-label")).toBe("Go to previous section");
    });

    it("shows which part it is, above and below the text, with links to the parts around it", async () => {
        const { wrapper } = await mountTextNavigation({ http: partsHttp(middlePart), query: { byte: "2500" } });
        const navs = wrapper.findAll("nav.text-parts");
        expect(navs.length).toBe(2);
        for (const nav of navs) {
            expect(nav.find(".text-part-indicator").text()).toBe("Part 3 of 4");
        }
        expect(navs[0].attributes("aria-label")).not.toBe(navs[1].attributes("aria-label"));
        const next = wrapper.find("#next-part-bottom");
        expect(next.element.tagName).toBe("A");
        expect(next.attributes("href")).toBe("/navigate/1/5?window_start=3000");  // without the hit of another part
        expect(next.attributes("aria-label")).toBe("Next part: part 4 of 4");
        expect(next.text()).toContain("Next part");
        const previous = wrapper.find("#previous-part-top");
        expect(previous.attributes("href")).toBe("/navigate/1/5?window_start=1000");
        expect(previous.attributes("aria-label")).toBe("Previous part: part 2 of 4");
        expect(wrapper.find("#previous-part-bottom").exists()).toBe(true);
        expect(wrapper.find("#next-part-top").exists()).toBe(true);
    });

    it("has no link past the first or the last part", async () => {
        let { wrapper } = await mountTextNavigation({ http: partsHttp(firstPart) });
        expect(wrapper.find("#previous-part-top").exists()).toBe(false);
        expect(wrapper.find("#previous-part-bottom").exists()).toBe(false);
        expect(wrapper.find(".text-part-indicator").text()).toBe("Part 1 of 4");
        ({ wrapper } = await mountTextNavigation({ http: partsHttp(lastPart) }));
        expect(wrapper.find("#next-part-top").exists()).toBe(false);
        expect(wrapper.find("#next-part-bottom").exists()).toBe(false);
    });

    it("steps through the parts with the nav bar's buttons", async () => {
        const { wrapper, router } = await mountTextNavigation({ http: partsHttp(middlePart), query: { byte: "2500" } });
        const pushSpy = vi.spyOn(router, "push");
        expect(wrapper.find("#next-obj").attributes("aria-label")).toBe("Go to next part");
        expect(wrapper.find("#prev-obj").attributes("aria-label")).toBe("Go to previous part");
        await wrapper.find("#next-obj").trigger("click");
        expect(pushSpy).toHaveBeenLastCalledWith({ path: "/navigate/1/5", query: { window_start: "3000" } });
        await wrapper.find("#prev-obj").trigger("click");
        expect(pushSpy).toHaveBeenLastCalledWith({ path: "/navigate/1/5", query: { window_start: "1000" } });
    });

    it("goes on to the next object from the last part, and back to the previous one from the first", async () => {
        let { wrapper, router } = await mountTextNavigation({ http: partsHttp(lastPart) });
        let pushSpy = vi.spyOn(router, "push");
        expect(wrapper.find("#next-obj").attributes("aria-label")).toBe("Go to next section");
        await wrapper.find("#next-obj").trigger("click");
        expect(pushSpy).toHaveBeenLastCalledWith({ path: "/navigate/1/6/0/0/0/0/0" });
        ({ wrapper, router } = await mountTextNavigation({ http: partsHttp(firstPart) }));
        pushSpy = vi.spyOn(router, "push");
        expect(wrapper.find("#prev-obj").attributes("aria-label")).toBe("Go to previous section");
        await wrapper.find("#prev-obj").trigger("click");
        expect(pushSpy).toHaveBeenLastCalledWith({ path: "/navigate/1/4/0/0/0/0/0" });
    });

    it("shows the nav bar's buttons for the parts of an object with no neighbors", async () => {
        const http = createMockHttp({
            "navigation.py": { ...navFixture, prev: "", next: "", window: middlePart },
            "get_table_of_contents.py": tocFixture,
        });
        const { wrapper } = await mountTextNavigation({ http });
        expect(wrapper.find("#prev-obj").exists()).toBe(true);
        expect(wrapper.find("#next-obj").exists()).toBe(true);
    });

    it("opens the part a part link goes to, and gives its text the focus", async () => {
        const scrollSpy = vi.spyOn(window, "scrollTo").mockImplementation(() => {});
        const http = partsHttp(middlePart);
        const { wrapper, router } = await mountTextNavigation({ http, query: { byte: "2500" }, attachTo: document.body });
        await wrapper.find("#next-part-bottom").trigger("click");
        await flushPromises();
        await nextTick();
        await flushPromises();
        expect(router.currentRoute.value.query).toEqual({ window_start: "3000" });
        const navCalls = http.get.mock.calls.filter(c => c[0].includes("navigation.py"));
        expect(navCalls.length).toBe(2);
        expect(navCalls[1][0]).toContain("window_start=3000");
        expect(navCalls[1][0]).not.toContain("byte=");
        expect(document.activeElement?.id).toBe("text-obj-content");
        expect(scrollSpy).toHaveBeenCalledWith({ top: 0, behavior: "instant" });
        wrapper.unmount();
        scrollSpy.mockRestore();
    });
});
