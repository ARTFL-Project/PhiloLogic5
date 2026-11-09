import { describe, it, expect } from "vitest";
import { mount } from "@vue/test-utils";
import { createGlobalConfig } from "./helpers.js";
import Footer from "../src/components/Footer.vue";

function mountFooter(philoConfig = {}) {
    return mount(Footer, { global: createGlobalConfig({ philoConfig }) });
}

function accessibilityLink(wrapper) {
    return wrapper.findAll("a").find((a) => a.text() === "Accessibility");
}

describe("Footer", () => {
    it("links to the accessibility site of the config", () => {
        const link = accessibilityLink(mountFooter({ accessibility_link: "https://example.org/accessibility" }));
        expect(link.attributes("href")).toBe("https://example.org/accessibility");
        // its name is the text it shows, and it opens where the page was
        expect(link.attributes("aria-label")).toBeUndefined();
        expect(link.attributes("target")).toBeUndefined();
        expect(link.find("i").attributes("aria-hidden")).toBe("true");
    });

    it("has no accessibility link when the config empties it", () => {
        expect(accessibilityLink(mountFooter({ accessibility_link: "" }))).toBeUndefined();
    });

    it("has no accessibility link with a broken config, which sends none", () => {
        expect(accessibilityLink(mountFooter({ accessibility_link: undefined }))).toBeUndefined();
    });

    it("still says it is powered by PhiloLogic", () => {
        const wrapper = mountFooter({ accessibility_link: "" });
        expect(wrapper.text()).toContain("Powered by");
        expect(wrapper.find('a[href="https://artfl-project.uchicago.edu/"] img').exists()).toBe(true);
    });
});
