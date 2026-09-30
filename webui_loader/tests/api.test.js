import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, headers, request, setCsrfToken, takeTokenFromUrl } from "../src/api";

function memoryStorage() {
    const values = {};
    return {
        getItem: (key) => (key in values ? values[key] : null),
        setItem: (key, value) => (values[key] = value),
        removeItem: (key) => delete values[key],
    };
}

describe("token of the URL", () => {
    it("is stored and taken out of the address", () => {
        const storage = memoryStorage();
        const history = { replaceState: vi.fn() };
        const location = { pathname: "/", search: "", hash: "#/load?token=abc&x=1" };
        expect(takeTokenFromUrl(location, history, storage)).toBe("abc");
        expect(storage.getItem("philologic5-webui-loader-token")).toBe("abc");
        expect(history.replaceState).toHaveBeenCalledWith(null, "", "/#/load?x=1");
    });

    it("leaves an address without token alone", () => {
        const history = { replaceState: vi.fn() };
        expect(takeTokenFromUrl({ pathname: "/", search: "", hash: "#/" }, history, memoryStorage())).toBeNull();
        expect(history.replaceState).not.toHaveBeenCalled();
    });
});

describe("headers", () => {
    it("send the token, and the CSRF token only with changes", () => {
        const storage = memoryStorage();
        storage.setItem("philologic5-webui-loader-token", "abc");
        setCsrfToken("csrf1");
        expect(headers("GET", storage)).toEqual({ Accept: "application/json", Authorization: "Bearer abc" });
        expect(headers("POST", storage)["X-CSRF-Token"]).toBe("csrf1");
        setCsrfToken(null);
    });
});

describe("request", () => {
    afterEach(() => vi.unstubAllGlobals());

    it("gives the description of errors", async () => {
        vi.stubGlobal(
            "fetch",
            vi.fn(async () => ({ ok: false, status: 409, text: async () => JSON.stringify({ title: "409 Conflict", description: "already loading" }) })),
        );
        await expect(request("POST", "jobs", { body: {} })).rejects.toMatchObject({ status: 409, message: "already loading" });
        await expect(request("POST", "jobs", { body: {} })).rejects.toBeInstanceOf(ApiError);
    });

    it("builds the query and sends JSON", async () => {
        const fetch = vi.fn(async () => ({ ok: true, status: 200, text: async () => '{"ok": true}' }));
        vi.stubGlobal("fetch", fetch);
        expect(await request("PUT", "x", { params: { a: 1, b: "", c: null }, body: { d: 2 } })).toEqual({ ok: true });
        const [url, options] = fetch.mock.calls[0];
        expect(url).toBe("api/x?a=1");
        expect(options.body).toBe('{"d":2}');
        expect(options.headers["Content-Type"]).toBe("application/json");
    });
});
