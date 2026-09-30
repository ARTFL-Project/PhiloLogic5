<template>
    <div>
        <h1 class="h3 mb-3">{{ $t("users.title") }}</h1>
        <ul class="nav nav-tabs mb-3">
            <li class="nav-item"><button class="nav-link" :class="{ active: tab === 'users' }" type="button" @click="tab = 'users'">{{ $t("users.accounts") }}</button></li>
            <li class="nav-item"><button class="nav-link" :class="{ active: tab === 'audit' }" type="button" @click="loadAudit">{{ $t("users.audit") }}</button></li>
        </ul>
        <div v-if="error" class="alert alert-danger py-2">{{ error }}</div>
        <div v-if="secret" class="alert alert-warning">
            {{ $t("users.temporaryPassword", { name: secret.name }) }} <span class="mono user-select-all fw-bold">{{ secret.password }}</span>
            <button class="btn-close float-end" type="button" @click="secret = null" :aria-label="$t('users.close')"></button>
        </div>

        <template v-if="tab === 'users'">
            <form class="d-flex flex-wrap gap-2 mb-3" @submit.prevent="addUser">
                <input class="form-control w-auto" v-model.trim="newName" :placeholder="$t('users.name')" :aria-label="$t('users.name')" required />
                <select class="form-select w-auto" v-model="newRole" :aria-label="$t('users.role')">
                    <option value="loader">{{ $t("users.roles.loader") }}</option>
                    <option value="admin">{{ $t("users.roles.admin") }}</option>
                </select>
                <button class="btn btn-secondary">{{ $t("users.add") }}</button>
            </form>
            <div class="table-responsive">
                <table class="table align-middle">
                    <thead>
                        <tr>
                            <th>{{ $t("users.name") }}</th>
                            <th>{{ $t("users.role") }}</th>
                            <th>{{ $t("users.status") }}</th>
                            <th>{{ $t("users.grants") }}</th>
                            <th></th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr v-for="user in users" :key="user.name">
                            <td class="mono">{{ user.name }}</td>
                            <td>
                                <select class="form-select form-select-sm w-auto" :value="user.role" @change="update(user, { role: $event.target.value })" :aria-label="$t('users.role')">
                                    <option value="loader">{{ $t("users.roles.loader") }}</option>
                                    <option value="admin">{{ $t("users.roles.admin") }}</option>
                                </select>
                            </td>
                            <td class="small">
                                <span v-if="user.disabled" class="badge text-bg-danger me-1">{{ $t("users.disabled") }}</span>
                                <span v-if="!user.totp_enabled" class="badge text-bg-warning me-1">{{ $t("users.noSecondFactor") }}</span>
                                <span v-if="user.must_change_password" class="badge text-bg-info me-1">{{ $t("users.temporary") }}</span>
                                <span v-if="user.trusted_browsers" class="badge text-bg-light me-1">{{ $t("users.trustedBrowsers", { count: user.trusted_browsers }) }}</span>
                            </td>
                            <td>
                                <span v-for="database in user.grants" :key="database" class="badge text-bg-light me-1">
                                    {{ database }} <button class="btn btn-sm p-0 ms-1" type="button" @click="action(user, 'revoke', { database })" :aria-label="$t('editors.remove')"><i class="bi bi-x"></i></button>
                                </span>
                                <select v-if="user.role !== 'admin'" class="form-select form-select-sm d-inline w-auto" @change="grant(user, $event)" :aria-label="$t('users.grant')">
                                    <option value="">{{ $t("users.grant") }}</option>
                                    <option v-for="database in databases.filter((name) => !user.grants.includes(name))" :key="database" :value="database">{{ database }}</option>
                                </select>
                            </td>
                            <td class="text-end text-nowrap">
                                <button class="btn btn-sm btn-outline-secondary" type="button" @click="action(user, 'reset_password')">{{ $t("users.resetPassword") }}</button>
                                <button class="btn btn-sm btn-outline-secondary ms-1" type="button" @click="action(user, 'reset_totp')">{{ $t("users.resetSecondFactor") }}</button>
                                <button class="btn btn-sm btn-outline-secondary ms-1" type="button" @click="update(user, { disabled: !user.disabled })">{{ user.disabled ? $t("users.enable") : $t("users.disable") }}</button>
                                <button class="btn btn-sm btn-outline-danger ms-1" type="button" :disabled="user.name === session.user" @click="remove(user)" :aria-label="$t('users.remove')"><i class="bi bi-trash"></i></button>
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </template>

        <div v-else class="table-responsive">
            <table class="table table-sm small">
                <thead>
                    <tr>
                        <th>{{ $t("users.time") }}</th>
                        <th>{{ $t("users.name") }}</th>
                        <th>{{ $t("users.address") }}</th>
                        <th>{{ $t("users.action") }}</th>
                        <th>{{ $t("users.detail") }}</th>
                    </tr>
                </thead>
                <tbody>
                    <tr v-for="(entry, index) in audit" :key="index">
                        <td class="text-nowrap">{{ formatDate(entry.time, locale) }}</td>
                        <td class="mono">{{ entry.user }}</td>
                        <td class="mono">{{ entry.ip }}</td>
                        <td>{{ entry.action }}</td>
                        <td class="mono">{{ entry.detail ? JSON.stringify(entry.detail) : "" }}</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
</template>

<script setup>
import { onMounted, ref } from "vue";
import { useI18n } from "vue-i18n";
import { api } from "../api";
import { session } from "../session";
import { formatDate } from "../utils";

const { t, locale } = useI18n();
const tab = ref("users");
const users = ref([]);
const databases = ref([]);
const audit = ref([]);
const error = ref(null);
const secret = ref(null);
const newName = ref("");
const newRole = ref("loader");

async function run(request) {
    error.value = null;
    try {
        const result = await request();
        users.value = (await api.get("users")).users;
        return result;
    } catch (requestError) {
        error.value = requestError.message;
        return null;
    }
}

async function addUser() {
    const result = await run(() => api.post("users", { name: newName.value, role: newRole.value }));
    if (result) {
        secret.value = { name: newName.value, password: result.temporary_password };
        newName.value = "";
    }
}

const update = (user, changes) => run(() => api.patch(`users/${user.name}`, changes));

async function action(user, name, data = {}) {
    if (["reset_password", "reset_totp"].includes(name) && !window.confirm(t(`users.confirm.${name}`, { name: user.name }))) {
        return;
    }
    const result = await run(() => api.post(`users/${user.name}/${name}`, data));
    if (result && result.temporary_password) {
        secret.value = { name: user.name, password: result.temporary_password };
    }
}

function grant(user, event) {
    const database = event.target.value;
    event.target.value = "";
    if (database) {
        action(user, "grant", { database });
    }
}

async function remove(user) {
    if (window.confirm(t("users.confirm.remove", { name: user.name }))) {
        await run(() => api.delete(`users/${user.name}`));
    }
}

async function loadAudit() {
    tab.value = "audit";
    try {
        audit.value = (await api.get("audit")).entries;
    } catch (requestError) {
        error.value = requestError.message;
    }
}

onMounted(async () => {
    await run(async () => {
        databases.value = (await api.get("databases")).databases.map((database) => database.name);
    });
});
</script>
