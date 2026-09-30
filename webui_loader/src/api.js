// Requests to the API of philologic5-webui-loader. On your own machine, the token of the URL printed by the command is
// kept in localStorage (which only this origin, port included, can read) and sent as an Authorization header. In the
// service, the session is a cookie. Requests which change something carry the CSRF token of the session.

const TOKEN_KEY = "philologic5-webui-loader-token";

let csrfToken = null;

export class ApiError extends Error {
    constructor(status, description, data) {
        super(description || `Error ${status}`);
        this.status = status;
        this.data = data;
    }
}

// The token of the URL (#/?token=...), stored and removed from the address bar
export function takeTokenFromUrl(location = window.location, history = window.history, storage = window.localStorage) {
    const hash = location.hash || "";
    const query = hash.includes("?") ? hash.slice(hash.indexOf("?") + 1) : "";
    const params = new URLSearchParams(query);
    const token = params.get("token");
    if (token) {
        storage.setItem(TOKEN_KEY, token);
        params.delete("token");
        const rest = params.toString();
        const route = hash.slice(0, hash.includes("?") ? hash.indexOf("?") : hash.length) || "#/";
        history.replaceState(null, "", `${location.pathname}${location.search}${route}${rest ? "?" + rest : ""}`);
    }
    return token;
}

export function forgetToken(storage = window.localStorage) {
    storage.removeItem(TOKEN_KEY);
}

export function setCsrfToken(token) {
    csrfToken = token;
}

export function headers(method, storage = window.localStorage) {
    const result = { Accept: "application/json" };
    const token = storage.getItem(TOKEN_KEY);
    if (token) {
        result.Authorization = `Bearer ${token}`;
    }
    if (method !== "GET" && csrfToken) {
        result["X-CSRF-Token"] = csrfToken;
    }
    return result;
}

export async function request(method, path, { body, params, raw, contentType } = {}) {
    let url = `api/${path}`;
    if (params) {
        const query = new URLSearchParams();
        for (const [key, value] of Object.entries(params)) {
            if (value !== undefined && value !== null && value !== "") {
                query.set(key, value);
            }
        }
        const queryString = query.toString();
        if (queryString) {
            url += `?${queryString}`;
        }
    }
    const options = { method, headers: headers(method), credentials: "same-origin" };
    if (raw !== undefined) {
        options.body = raw;
        options.headers["Content-Type"] = contentType || "application/octet-stream";
    } else if (body !== undefined) {
        options.body = JSON.stringify(body);
        options.headers["Content-Type"] = "application/json";
    }
    const response = await fetch(url, options);
    let data = null;
    const text = await response.text();
    if (text) {
        try {
            data = JSON.parse(text);
        } catch {
            data = { description: text };
        }
    }
    if (!response.ok) {
        throw new ApiError(response.status, data && (data.description || data.title), data);
    }
    return data;
}

export const api = {
    get: (path, params) => request("GET", path, { params }),
    post: (path, body) => request("POST", path, { body }),
    put: (path, body) => request("PUT", path, { body }),
    patch: (path, body) => request("PATCH", path, { body }),
    delete: (path) => request("DELETE", path),
};
