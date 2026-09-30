// The session: who is logged in, and the settings of the server
import { reactive } from "vue";
import { api, ApiError, setCsrfToken } from "./api";

export const session = reactive({
    loaded: false,
    mode: null,
    authenticated: false,
    stage: null,
    user: null,
    role: null,
    mustChangePassword: false,
    server: {},
    error: null,
});

export async function loadSession() {
    try {
        const data = await api.get("session");
        session.mode = data.mode;
        session.authenticated = data.authenticated;
        session.stage = data.stage || null;
        session.user = data.user || null;
        session.role = data.role || null;
        session.mustChangePassword = !!data.must_change_password;
        session.server = data;
        session.error = null;
        setCsrfToken(data.csrf || null);
    } catch (error) {
        session.error = error instanceof ApiError ? error.message : String(error);
    }
    session.loaded = true;
    return session;
}

export const isAdmin = () => session.role === "admin";
export const isService = () => session.mode === "service";
