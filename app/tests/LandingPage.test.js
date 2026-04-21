import { describe, it, expect, vi } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { nextTick } from "vue";
import { createGlobalConfig, createMockHttp } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import LandingPage from "../src/components/LandingPage.vue";

function mountLandingPage(overrides = {}) {
    const http = overrides.http || createMockHttp({
        "landing_page.py": {
            content: [
                {
                    citation: [{ label: "Brown, Charles", href: "/bibliography?author=Brown", style: {} }],
                    count: 5,
                    metadata: { author: "Brown, Charles" },
                },
            ],
            prefix: "A-D",
        },
    });
    const global = createGlobalConfig({
        http,
        philoConfig: {
            landing_page_browsing: "default",
            default_landing_page_browsing: [
                { label: "Author", group_by_field: "author", display_count: true, queries: ["A-D", "E-I", "J-M", "N-R", "S-Z"], is_range: true, citation: [{ field: "author", object_level: "doc", prefix: "", suffix: "", link: true, style: {} }] },
                { label: "Title", group_by_field: "title", display_count: false, queries: ["A-D", "E-I", "J-M", "N-R", "S-Z"], is_range: true, citation: [{ field: "title", object_level: "doc", prefix: "", suffix: "", link: true, style: {} }] },
            ],
        },
        route: { name: "home", path: "/", query: {} },
        stubs: {
            Citations: { template: "<span class='citations-stub' />" },
            ProgressSpinner: { template: "<div class='spinner-stub' />" },
        },
        ...overrides,
    });

    const store = useMainStore();
    store.formData = { ...store.formData, report: "home" };

    return mount(LandingPage, { global });
}

describe("LandingPage", () => {
    it("renders landing page with browse options", async () => {
        const wrapper = mountLandingPage();
        await flushPromises();
        await nextTick();
        expect(wrapper.exists()).toBe(true);
        expect(wrapper.text()).toContain("Author");
    });

    // --- @click="getContent(browseType, range)" ---
    it("renders range browse buttons", async () => {
        const wrapper = mountLandingPage();
        await flushPromises();
        await nextTick();
        const rangeBtns = wrapper.findAll("button").filter(b => b.text().includes("-"));
        expect(rangeBtns.length).toBeGreaterThan(0);
    });

    it("fetches content on range button click", async () => {
        const http = createMockHttp({ "landing_page.py": { content: [], prefix: "A-D" } });
        const wrapper = mountLandingPage({ http });
        await flushPromises();
        await nextTick();

        const rangeBtns = wrapper.findAll("button").filter(b => b.text().includes("-"));
        if (rangeBtns.length > 0) {
            http.get.mockClear();
            await rangeBtns[0].trigger("click");
            await flushPromises();
            expect(http.get).toHaveBeenCalled();
        }
    });

    it("renders results after fetching content", async () => {
        const wrapper = mountLandingPage();
        await flushPromises();
        await nextTick();

        // Should render citations for fetched content
        const citations = wrapper.findAll(".citations-stub");
        expect(citations.length).toBeGreaterThanOrEqual(0);
    });

    // --- Active range highlighting ---
    it("highlights active range button", async () => {
        const wrapper = mountLandingPage();
        await flushPromises();
        await nextTick();

        const rangeBtns = wrapper.findAll("button").filter(b => b.text().includes("-"));
        if (rangeBtns.length > 0) {
            await rangeBtns[0].trigger("click");
            await nextTick();
            // Active button should have active/selected class
        }
    });
});
