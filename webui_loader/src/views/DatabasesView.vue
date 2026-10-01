<template>
    <div>
        <div class="d-flex flex-wrap align-items-center gap-2 mb-3">
            <h1 class="h3 mb-0 me-3">{{ $t("databases.title") }}</h1>
            <input class="form-control w-auto" v-model="filter" :placeholder="$t('databases.filter')" :aria-label="$t('databases.filter')" />
            <router-link class="btn btn-secondary ms-auto" to="/load"><i class="bi bi-plus-lg me-1"></i>{{ $t("nav.newLoad") }}</router-link>
        </div>
        <p v-if="system" class="small text-body-secondary">
            {{ $t("databases.root", { root: system.database_root }) }} &middot; {{ $t("databases.free", { size: formatSize(system.disk_free) }) }} &middot; {{ $t("databases.cores", { count: system.cpu_count }) }}
        </p>
        <div v-if="error" class="alert alert-danger">{{ error }}</div>
        <div v-if="loading" class="text-center my-5"><span class="spinner-border"></span></div>
        <div v-else class="table-responsive">
            <table class="table table-hover align-middle">
                <thead>
                    <tr>
                        <th>{{ $t("databases.name") }}</th>
                        <th class="text-end">{{ $t("databases.documents") }}</th>
                        <th>{{ $t("databases.loaded") }}</th>
                        <th>{{ $t("databases.owner") }}</th>
                        <th>{{ $t("databases.status") }}</th>
                        <th></th>
                    </tr>
                </thead>
                <tbody>
                    <tr v-for="database in shown" :key="database.name">
                        <td>
                            <div class="fw-semibold">{{ database.title || database.name }}</div>
                            <div v-if="database.title && database.title !== database.name" class="small text-body-secondary mono">{{ database.name }}</div>
                        </td>
                        <td class="text-end">{{ database.documents ?? "" }}</td>
                        <td class="small">{{ formatDate(database.loaded, locale) }}</td>
                        <td class="small">{{ database.owner }}<span class="text-body-secondary"> / {{ database.group }}</span><div v-if="database.loaded_by" class="text-body-secondary">{{ $t("databases.loadedBy", { user: database.loaded_by }) }}</div></td>
                        <td>
                            <router-link v-if="database.loading" class="badge text-bg-secondary text-decoration-none" :to="`/jobs/${database.loading.job}`">{{ $t("databases.loading", { user: database.loading.user || "?" }) }}</router-link>
                            <span v-else-if="!database.loaded" class="badge text-bg-warning" :title="$t('databases.incompleteHelp')">{{ $t("databases.incomplete") }}</span>
                            <span v-if="!database.web_config_writable || !database.may_edit" class="badge text-bg-light" :title="database.may_edit ? database.web_config_reason : $t('databases.notAllowed')">
                                <i class="bi bi-lock me-1"></i>{{ $t("databases.readOnly") }}
                            </span>
                        </td>
                        <td class="text-end text-nowrap">
                            <a class="btn btn-sm btn-outline-secondary" :href="database.url" target="_blank" rel="noopener"><i class="bi bi-box-arrow-up-right me-1"></i>{{ $t("databases.open") }}</a>
                            <router-link class="btn btn-sm btn-outline-secondary ms-1" :to="`/databases/${database.name}/web_config`"><i class="bi bi-sliders me-1"></i>{{ $t("databases.webConfig") }}</router-link>
                            <div class="btn-group ms-1" v-if="database.has_load_config">
                                <router-link class="btn btn-sm btn-outline-secondary" :class="{ disabled: !database.may_edit || !database.replaceable || database.loading }" :to="{ path: '/load', query: { from: database.name, again: 1 } }" :title="database.replaceable ? '' : database.replace_reason">{{ $t("databases.loadAgain") }}</router-link>
                                <router-link class="btn btn-sm btn-outline-secondary" :to="{ path: '/load', query: { from: database.name } }">{{ $t("databases.loadLike") }}</router-link>
                            </div>
                        </td>
                    </tr>
                    <tr v-if="!shown.length">
                        <td colspan="6" class="text-center text-body-secondary">{{ $t("databases.none") }}</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
</template>

<script setup>
import { computed, onMounted, ref } from "vue";
import { useI18n } from "vue-i18n";
import { api } from "../api";
import { formatDate, formatSize } from "../utils";

const { locale } = useI18n();
const databases = ref([]);
const system = ref(null);
const filter = ref("");
const loading = ref(true);
const error = ref(null);

const shown = computed(() => {
    const text = filter.value.toLowerCase();
    return databases.value.filter(
        (database) => !text || database.name.toLowerCase().includes(text) || (database.title || "").toLowerCase().includes(text),
    );
});

onMounted(async () => {
    try {
        const [result, systemResult] = await Promise.all([api.get("databases"), api.get("system")]);
        databases.value = result.databases;
        system.value = systemResult;
    } catch (requestError) {
        error.value = requestError.message;
    } finally {
        loading.value = false;
    }
});
</script>
