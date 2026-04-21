import { describe, it, expect } from "vitest";
import { mount } from "@vue/test-utils";
import { createI18n } from "vue-i18n";
import Citations from "../src/components/Citations.vue";

const i18n = createI18n({
    legacy: false,
    locale: "en",
    messages: { en: { citations: { citationGroup: "Citation", viewText: "View text {resultNumber}" } } },
});

function mountCitations(props = {}) {
    return mount(Citations, {
        props: {
            citation: [],
            resultNumber: "1",
            ...props,
        },
        global: {
            plugins: [i18n],
            stubs: { "router-link": { template: '<a class="router-link-stub"><slot /></a>', props: ["to"] } },
        },
    });
}

describe("Citations", () => {
    it("renders citation text from plain object", () => {
        const wrapper = mountCitations({
            citation: [{ label: "Voltaire", href: "" }],
        });
        expect(wrapper.text()).toContain("Voltaire");
    });

    it("renders multiple citations with separators", () => {
        const wrapper = mountCitations({
            citation: [
                { label: "Author", href: "" },
                { label: "Title", href: "" },
            ],
            separator: " | ",
        });
        expect(wrapper.text()).toContain("Author");
        expect(wrapper.text()).toContain("Title");
    });

    it("renders linked citations as router-links", () => {
        const wrapper = mountCitations({
            citation: [{ label: "Chapter 1", href: "/navigate/1/2/0" }],
        });
        expect(wrapper.find(".router-link-stub").exists()).toBe(true);
    });

    it("renders span for citations without href", () => {
        const wrapper = mountCitations({
            citation: [{ label: "No link", href: "" }],
        });
        expect(wrapper.find(".router-link-stub").exists()).toBe(false);
        expect(wrapper.text()).toContain("No link");
    });

    it("renders empty state for empty citation array", () => {
        const wrapper = mountCitations({ citation: [] });
        const citeEl = wrapper.find("cite");
        expect(citeEl.findAll(".citation").length).toBe(0);
    });
});
