import axios from "axios";
import "bootstrap";
import "bootstrap-icons/font/bootstrap-icons.css";
import { createPinia } from "pinia";
import { createApp } from "vue";
import App from "./App.vue";
import router from "./router";

import i18n from "./i18n";
import { databaseUrl, dbUrlOnThisHost } from "./utils";

const dbUrl = dbUrlOnThisHost(databaseUrl());

axios
    .get(`${dbUrl}/scripts/get_web_config.py`, {})
    .then((response) => {
        const app = createApp(App).use(i18n);
        const pinia = createPinia();

        app.config.globalProperties.$philoConfig = response.data;
        app.config.globalProperties.$dbUrl = dbUrl;
        app.provide("$http", axios);
        app.provide("$dbUrl", dbUrl);
        app.provide("$philoConfig", response.data);
        app.use(router);
        app.use(pinia);

        router.isReady().then(() => app.mount("#app"));
    })
    .catch((error) => {
        // this.loading = false
        console.log(error.toString());
    });
