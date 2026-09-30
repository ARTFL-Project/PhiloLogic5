<template>
    <div class="card mx-auto mt-4 narrow">
        <div class="card-body">
            <h1 class="h4 mb-3">{{ $t("password.title") }}</h1>
            <div v-if="session.mustChangePassword" class="alert alert-info py-2">{{ $t("password.mustChange") }}</div>
            <div v-if="error" class="alert alert-danger py-2">{{ error }}</div>
            <div v-if="done" class="alert alert-success py-2">{{ $t("password.changed") }}</div>
            <form @submit.prevent="submit">
                <div class="mb-3">
                    <label class="form-label" for="current">{{ $t("password.current") }}</label>
                    <input id="current" type="password" class="form-control" v-model="current" autocomplete="current-password" required />
                </div>
                <div class="mb-3">
                    <label class="form-label" for="new">{{ $t("password.new") }}</label>
                    <input id="new" type="password" class="form-control" v-model="newPassword" autocomplete="new-password" minlength="12" required />
                    <div class="form-text">{{ $t("password.rules") }}</div>
                </div>
                <div class="mb-3">
                    <label class="form-label" for="confirm">{{ $t("password.confirm") }}</label>
                    <input id="confirm" type="password" class="form-control" :class="{ 'is-invalid': confirm && confirm !== newPassword }" v-model="confirm" autocomplete="new-password" required />
                </div>
                <button class="btn btn-secondary" :disabled="busy || newPassword !== confirm">{{ $t("password.change") }}</button>
            </form>
            <template v-if="!session.mustChangePassword && session.server.trusted_browser_days > 0">
                <hr />
                <h2 class="h5">{{ $t("password.trustedBrowsers") }}</h2>
                <p class="small">{{ session.server.browser_trusted ? $t("password.thisBrowserTrusted") : $t("password.thisBrowserNotTrusted") }}</p>
                <div v-if="forgotten" class="alert alert-success py-2">{{ $t("password.forgotten") }}</div>
                <button class="btn btn-outline-danger btn-sm" type="button" @click="forgetBrowsers">{{ $t("password.forget") }}</button>
            </template>
        </div>
    </div>
</template>

<script setup>
import { ref } from "vue";
import { useRouter } from "vue-router";
import { api } from "../api";
import { loadSession, session } from "../session";

const router = useRouter();
const current = ref("");
const newPassword = ref("");
const confirm = ref("");
const error = ref(null);
const busy = ref(false);
const done = ref(false);
const forgotten = ref(false);

async function forgetBrowsers() {
    error.value = null;
    try {
        await api.delete("account/trusted_browsers");
        await loadSession();
        forgotten.value = true;
    } catch (requestError) {
        error.value = requestError.message;
    }
}

async function submit() {
    busy.value = true;
    error.value = null;
    try {
        await api.post("account/password", { current: current.value, new: newPassword.value });
        const mustChange = session.mustChangePassword;
        await loadSession();
        current.value = newPassword.value = confirm.value = "";
        done.value = true;
        if (mustChange) {
            router.push("/");
        }
    } catch (requestError) {
        error.value = requestError.message;
    } finally {
        busy.value = false;
    }
}
</script>
