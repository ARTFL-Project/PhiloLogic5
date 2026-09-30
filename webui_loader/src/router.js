import { createRouter, createWebHashHistory } from "vue-router";
import { session } from "./session";

import DatabasesView from "./views/DatabasesView.vue";
import JobsView from "./views/JobsView.vue";
import JobView from "./views/JobView.vue";
import LoadView from "./views/LoadView.vue";
import LoginView from "./views/LoginView.vue";
import PasswordView from "./views/PasswordView.vue";
import UsersView from "./views/UsersView.vue";
import WebConfigView from "./views/WebConfigView.vue";

const routes = [
    { path: "/", name: "databases", component: DatabasesView },
    { path: "/load", name: "load", component: LoadView },
    { path: "/jobs", name: "jobs", component: JobsView },
    { path: "/jobs/:id", name: "job", component: JobView, props: true },
    { path: "/databases/:name/web_config", name: "web_config", component: WebConfigView, props: true },
    { path: "/login", name: "login", component: LoginView, meta: { public: true } },
    { path: "/password", name: "password", component: PasswordView },
    { path: "/users", name: "users", component: UsersView, meta: { admin: true } },
];

const router = createRouter({ history: createWebHashHistory(), routes });

router.beforeEach((to) => {
    if (session.mode === "service") {
        if (!session.authenticated && !to.meta.public) {
            return { name: "login" };
        }
        if (session.authenticated && session.mustChangePassword && to.name !== "password") {
            return { name: "password" };
        }
        if (to.meta.admin && session.role !== "admin") {
            return { name: "databases" };
        }
    }
    return true;
});

export default router;
