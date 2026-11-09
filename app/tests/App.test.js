import { describe, it, expect, vi } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { createGlobalConfig, createMockHttp } from "./helpers.js";
import App from "../src/App.vue";

// what is around the footer, loaded asynchronously, and not under test
vi.mock("../src/components/Header.vue", () => ({ default: { template: "<header />" } }));
vi.mock("../src/components/SearchForm.vue", () => ({ __esModule: true, default: { template: "<div />" } }));
vi.mock("../src/components/AccessControl.vue", () => ({ __esModule: true, default: { template: "<div />" } }));

function mountApp({ philoConfig = {}, access } = {}) {
    const http = createMockHttp(access ? { access_request: access } : {});
    http.interceptors = { response: { use: vi.fn() } };
    return mount(App, {
        global: createGlobalConfig({
            philoConfig,
            http,
        }),
    });
}

describe("App footer", () => {
    it("is under the pages of a database without access control", async () => {
        const wrapper = mountApp();
        await flushPromises();
        expect(wrapper.find("footer").text()).toContain("Accessibility");
    });

    it("is under the login screen of a client refused access", async () => {
        const wrapper = mountApp({
            philoConfig: { access_control: true },
            access: { access: false, incoming_address: "192.0.2.1", domain_name: "" },
        });
        await flushPromises();
        expect(wrapper.find("footer").text()).toContain("Accessibility");
    });

    it("waits for the access check", () => {
        const wrapper = mountApp({ philoConfig: { access_control: true }, access: { access: true } });
        expect(wrapper.find("footer").exists()).toBe(false);
    });
});
