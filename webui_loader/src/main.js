import "bootstrap-icons/font/bootstrap-icons.css";
import "./styles.scss";

import { createApp } from "vue";
import { takeTokenFromUrl } from "./api";
import App from "./App.vue";
import i18n from "./i18n";
import router from "./router";
import { loadSession } from "./session";

takeTokenFromUrl();
loadSession().then(() => {
    createApp(App).use(i18n).use(router).mount("#app");
});
