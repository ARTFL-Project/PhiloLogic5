<template>
    <div class="card mx-auto mt-4 narrow">
        <div class="card-body">
            <h1 class="h4 mb-3"><i class="bi bi-shield-lock me-2"></i>{{ $t("login.title") }}</h1>
            <div v-if="error" class="alert alert-danger py-2">{{ error }}</div>

            <form v-if="step === 'password'" @submit.prevent="submitPassword">
                <div class="mb-3">
                    <label class="form-label" for="username">{{ $t("login.username") }}</label>
                    <input id="username" class="form-control" v-model.trim="username" autocomplete="username" required autofocus />
                </div>
                <div class="mb-3">
                    <label class="form-label" for="password">{{ $t("login.password") }}</label>
                    <input id="password" type="password" class="form-control" v-model="password" autocomplete="current-password" required />
                </div>
                <button class="btn btn-secondary" :disabled="busy">{{ $t("login.continue") }}</button>
            </form>

            <form v-else-if="step === 'totp'" @submit.prevent="submitCode">
                <label class="form-label" for="code">{{ $t("login.code") }}</label>
                <input id="code" class="form-control mono mb-1" v-model.trim="code" autocomplete="one-time-code" inputmode="numeric" required autofocus />
                <div class="form-text mb-3">{{ $t("login.recoveryHelp") }}</div>
                <TrustBrowser v-model="trust" />
                <button class="btn btn-secondary" :disabled="busy">{{ $t("login.logIn") }}</button>
            </form>

            <form v-else-if="step === 'totp_setup'" @submit.prevent="confirmSetup">
                <p>{{ $t("login.setupHelp") }}</p>
                <div class="text-center mb-2" v-if="qr"><img :src="qr" class="qr" :alt="$t('login.qr')" /></div>
                <p class="small">{{ $t("login.secret") }} <span class="mono user-select-all">{{ setup && setup.secret }}</span></p>
                <label class="form-label" for="setup-code">{{ $t("login.setupCode") }}</label>
                <input id="setup-code" class="form-control mono mb-3" v-model.trim="code" autocomplete="one-time-code" inputmode="numeric" required />
                <TrustBrowser v-model="trust" />
                <button class="btn btn-secondary" :disabled="busy">{{ $t("login.enable") }}</button>
            </form>

            <div v-else-if="step === 'recovery'">
                <div class="alert alert-warning">{{ $t("login.recoveryCodes") }}</div>
                <pre class="bg-light p-3 rounded mono text-center user-select-all">{{ recoveryCodes.join("\n") }}</pre>
                <div class="form-check mb-3">
                    <input class="form-check-input" type="checkbox" id="kept" v-model="kept" />
                    <label class="form-check-label" for="kept">{{ $t("login.kept") }}</label>
                </div>
                <button class="btn btn-secondary" :disabled="!kept" @click="finish">{{ $t("login.continue") }}</button>
            </div>
        </div>
    </div>
</template>

<script setup>
import QRCode from "qrcode";
import { onMounted, ref } from "vue";
import { useRouter } from "vue-router";
import { api } from "../api";
import TrustBrowser from "../components/TrustBrowser.vue";
import { loadSession, session } from "../session";

const router = useRouter();
const step = ref("password");
const username = ref("");
const password = ref("");
const code = ref("");
const error = ref(null);
const busy = ref(false);
const setup = ref(null);
const qr = ref(null);
const recoveryCodes = ref([]);
const kept = ref(false);
const trust = ref(false);

async function attempt(action) {
    busy.value = true;
    error.value = null;
    try {
        await action();
    } catch (requestError) {
        error.value = requestError.message;
        if (requestError.message && requestError.message.includes("log in again")) {
            step.value = "password";
        }
    } finally {
        busy.value = false;
    }
}

async function startSetup() {
    setup.value = await api.get("login/totp_setup");
    qr.value = await QRCode.toDataURL(setup.value.uri, { margin: 1, width: 240 });
    step.value = "totp_setup";
}

const submitPassword = () =>
    attempt(async () => {
        const result = await api.post("login", { username: username.value, password: password.value });
        password.value = "";
        code.value = "";
        if (result.next === "done") {
            await finish(); // a browser trusted for the second factor
        } else if (result.next === "totp_setup") {
            await startSetup();
        } else {
            step.value = "totp";
        }
    });

async function finish() {
    await loadSession();
    router.push(session.mustChangePassword ? "/password" : "/");
}

const submitCode = () =>
    attempt(async () => {
        await api.post("login/totp", { code: code.value, trust: trust.value });
        await finish();
    });

const confirmSetup = () =>
    attempt(async () => {
        const result = await api.post("login/totp_setup", { code: code.value, trust: trust.value });
        recoveryCodes.value = result.recovery_codes;
        step.value = "recovery";
    });

onMounted(async () => {
    if (session.authenticated) {
        router.push("/");
    } else if (session.stage === "totp_setup") {
        await attempt(startSetup);
    } else if (session.stage === "password") {
        step.value = "totp";
    }
});
</script>
