import { describe, it, expect, vi, afterEach } from "vitest";

// main.js asks the database for its config before anything else: a request that never answers, so nothing is mounted
const { get } = vi.hoisted(() => ({ get: vi.fn(() => new Promise(() => {})) }));
vi.mock("axios", () => ({ default: { get } }));
vi.mock("../src/App.vue", () => ({ default: {} }));
vi.mock("../src/router", () => ({ default: {} }));
vi.mock("../src/i18n", () => ({ default: {} }));

afterEach(() => {
    vi.unstubAllEnvs();
    vi.resetModules();
    vi.restoreAllMocks();
    get.mockClear();
    document.head.querySelectorAll("base").forEach((base) => base.remove());
});

describe("main", () => {
    it("asks the database of VITE_DB_URL for its config with npm run host, not the dev server", async () => {
        vi.stubEnv("VITE_DB_URL", "http://philologic.example/philologic5/mydb/");
        await import("../src/main.js");
        expect(get).toHaveBeenCalledWith(
            expect.stringMatching(/^http:\/\/philologic\.example\/philologic5\/mydb\/+scripts\/get_web_config\.py$/),
            {}
        );
    });

    it("asks the database of the page's <base href> for its config when the server serves it", async () => {
        vi.stubEnv("DEV", false);
        const error = vi.spyOn(console, "error");
        const base = document.createElement("base");
        base.href = "/philologic5/mydb/";
        document.head.appendChild(base);
        await import("../src/main.js");
        expect(get).toHaveBeenCalledWith(
            expect.stringMatching(new RegExp(`^${window.location.origin}/philologic5/mydb/+scripts/get_web_config\\.py$`)),
            {}
        );
        expect(error).not.toHaveBeenCalled();
    });

    it("says what to set when npm run host runs without VITE_DB_URL", async () => {
        const error = vi.spyOn(console, "error").mockImplementation(() => {});
        await import("../src/main.js");
        expect(error).toHaveBeenCalledWith(expect.stringContaining("VITE_DB_URL"));
    });
});
