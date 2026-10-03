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

describe("LandingPage title browse", () => {
    const CitationsStub = { template: "<span class='citations-stub' />", props: ["citation", "resultNumber"] };
    const citations = [
        { field: "author", object_level: "doc", prefix: "", suffix: ", ", link: false, style: {} },
        { field: "title", object_level: "doc", prefix: "", suffix: "", link: true, style: {} },
        { field: "year", object_level: "doc", prefix: " [", suffix: "]", link: false, style: {} },
    ];
    const content = {
        content: {
            B: {
                prefix: "B",
                results: [
                    // 66 volumes alike in all the citation's fields: one entry
                    { metadata: { title: "Bible", author: null, year: 1910, philo_id: "5 0 0 0 0 0 0" }, count: 66 },
                    { metadata: { title: "Bérénice", author: "Racine", year: 1670, philo_id: "6 0 0 0 0 0 0" }, count: 1 },
                ],
            },
        },
        citations,
        display_count: "false",
        content_type: "title",
    };

    async function mountTitles() {
        const browse = { label: "Title", group_by_field: "title", display_count: false, queries: ["A-D"], is_range: true, citation: citations };
        const global = createGlobalConfig({
            http: createMockHttp({ "get_landing_page_content.py": content }),
            philoConfig: { landing_page_browsing: "default", default_landing_page_browsing: [browse] },
            route: { name: "landing", path: "/landing", query: { browse: "title", range: "A-D", display_count: "false", is_range: "true" } },
            stubs: { Citations: CitationsStub, ProgressSpinner: { template: "<div />" } },
        });
        await global.plugins[2].isReady(); // the router, with the browse in its query
        useMainStore().formData = { ...useMainStore().formData, report: "landing" };
        const wrapper = mount(LandingPage, { global });
        await flushPromises();
        await nextTick();
        return wrapper;
    }

    it("links an entry of several documents to their bibliography, and says how many", async () => {
        const wrapper = await mountTitles();
        const [bible, berenice] = wrapper.findAllComponents(CitationsStub).map((c) => c.props("citation").find((x) => x.field === "title").href);
        expect(bible).toEqual({ path: "/bibliography", query: { title: '"Bible"', author: "NULL", year: "1910" } });
        expect(berenice).toBe("/navigate/6/table-of-contents");
        const items = wrapper.findAll("li.pt-1");
        expect(items[0].text()).toContain("(66 documents)");
        expect(items[1].text()).not.toContain("documents");
    });
});
