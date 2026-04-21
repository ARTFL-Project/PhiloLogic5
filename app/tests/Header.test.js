import { describe, it, expect, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createGlobalConfig } from "./helpers.js";
import Header from "../src/components/Header.vue";

function mountHeader(overrides = {}) {
    const global = createGlobalConfig({
        philoConfig: {
            dbname: "My Test DB",
            academic_citation: { link: "https://example.com", collection: "" },
            header_in_footer: false,
            ...overrides.philoConfig,
        },
        ...overrides,
    });
    return mount(Header, { global });
}

describe("Header", () => {
    it("renders database name in header", () => {
        const wrapper = mountHeader();
        expect(wrapper.text()).toContain("My Test DB");
    });

    it("renders navigation element", () => {
        const wrapper = mountHeader();
        expect(wrapper.find("nav").exists()).toBe(true);
    });

    it("renders cite button", () => {
        const wrapper = mountHeader();
        const text = wrapper.text().toLowerCase();
        // i18n key is header.citeUs — check for the key or translated text
        expect(text).toContain("cit");
    });
});
