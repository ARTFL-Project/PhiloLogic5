<template>
    <nav class="navbar navbar-expand-md navbar-dark mb-3">
        <div class="container-fluid">
            <router-link class="navbar-brand" to="/"><i class="bi bi-database-gear me-2"></i>{{ $t("app.title") }}</router-link>
            <button class="navbar-toggler" type="button" @click="menuOpen = !menuOpen" :aria-label="$t('app.menu')">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" :class="{ show: menuOpen }" v-if="session.authenticated">
                <ul class="navbar-nav me-auto">
                    <li class="nav-item"><router-link class="nav-link" to="/">{{ $t("nav.databases") }}</router-link></li>
                    <li class="nav-item"><router-link class="nav-link" to="/load">{{ $t("nav.newLoad") }}</router-link></li>
                    <li class="nav-item"><router-link class="nav-link" to="/jobs">{{ $t("nav.loads") }}</router-link></li>
                    <li class="nav-item" v-if="session.mode === 'service' && session.role === 'admin'">
                        <router-link class="nav-link" to="/users">{{ $t("nav.users") }}</router-link>
                    </li>
                </ul>
                <span class="navbar-text me-3 small">
                    <i class="bi bi-hdd-network me-1"></i>{{ session.server.hostname }}
                    <span v-if="session.user"> · <i class="bi bi-person me-1"></i>{{ session.user }}</span>
                </span>
                <div class="d-flex gap-2">
                    <button class="btn btn-sm btn-outline-light" @click="switchLanguage">{{ otherLanguage.toUpperCase() }}</button>
                    <template v-if="session.mode === 'service'">
                        <router-link class="btn btn-sm btn-outline-light" to="/password">{{ $t("nav.password") }}</router-link>
                        <button class="btn btn-sm btn-outline-light" @click="logout">{{ $t("nav.logout") }}</button>
                    </template>
                </div>
            </div>
            <button v-else class="btn btn-sm btn-outline-light ms-auto" @click="switchLanguage">{{ otherLanguage.toUpperCase() }}</button>
        </div>
    </nav>
    <main class="container-fluid pb-5">
        <div v-if="session.error" class="alert alert-danger">
            <i class="bi bi-exclamation-triangle me-2"></i>{{ $t("app.unreachable") }}: {{ session.error }}
        </div>
        <div v-else-if="session.mode === 'personal' && !session.authenticated" class="card mx-auto mt-5 narrow">
            <div class="card-body">
                <h1 class="h4">{{ $t("app.needToken") }}</h1>
                <p>{{ $t("app.needTokenHelp") }}</p>
                <pre class="bg-light p-2 rounded">philologic5-webui-loader url</pre>
            </div>
        </div>
        <router-view v-else :key="$route.fullPath" />
    </main>
</template>

<script setup>
import { computed, ref } from "vue";
import { useI18n } from "vue-i18n";
import { useRouter } from "vue-router";
import { api } from "./api";
import { loadSession, session } from "./session";

const { locale } = useI18n();
const router = useRouter();
const menuOpen = ref(false);
const otherLanguage = computed(() => (locale.value === "fr" ? "en" : "fr"));

function switchLanguage() {
    locale.value = otherLanguage.value;
    localStorage.setItem("philologic5-webui-loader-lang", locale.value);
    document.documentElement.lang = locale.value;
}

async function logout() {
    await api.post("logout");
    await loadSession();
    router.push({ name: "login" });
}
</script>
