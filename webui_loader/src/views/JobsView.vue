<template>
    <div>
        <h1 class="h3 mb-3">{{ $t("jobs.title") }}</h1>
        <div v-if="error" class="alert alert-danger">{{ error }}</div>
        <div class="table-responsive">
            <table class="table table-hover align-middle">
                <thead>
                    <tr>
                        <th>{{ $t("jobs.database") }}</th>
                        <th>{{ $t("jobs.state") }}</th>
                        <th>{{ $t("jobs.started") }}</th>
                        <th>{{ $t("jobs.duration") }}</th>
                        <th class="text-end">{{ $t("jobs.files") }}</th>
                        <th v-if="showUsers">{{ $t("jobs.user") }}</th>
                    </tr>
                </thead>
                <tbody>
                    <tr v-for="job in jobs" :key="job.id" role="button" @click="$router.push(`/jobs/${job.id}`)">
                        <td class="mono">{{ job.dbname }}</td>
                        <td><JobState :state="job.state" /></td>
                        <td class="small">{{ formatDate(job.created, locale) }}</td>
                        <td class="small">{{ formatDuration((job.ended || now) - (job.started || job.created)) }}</td>
                        <td class="text-end">{{ job.files }}</td>
                        <td v-if="showUsers" class="small">{{ job.user }}</td>
                    </tr>
                    <tr v-if="!jobs.length && !loading">
                        <td colspan="6" class="text-center text-body-secondary">{{ $t("jobs.none") }}</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref } from "vue";
import { useI18n } from "vue-i18n";
import { api } from "../api";
import JobState from "../components/JobState.vue";
import { session } from "../session";
import { formatDate, formatDuration } from "../utils";

const { locale } = useI18n();
const jobs = ref([]);
const loading = ref(true);
const error = ref(null);
const now = ref(Date.now() / 1000);
const showUsers = computed(() => session.mode === "service" && session.role === "admin");
let timer = null;

async function refresh() {
    try {
        jobs.value = (await api.get("jobs")).jobs;
        error.value = null;
    } catch (requestError) {
        error.value = requestError.message;
    } finally {
        loading.value = false;
    }
    now.value = Date.now() / 1000;
    timer = setTimeout(refresh, jobs.value.some((job) => job.state === "running") ? 3000 : 15000);
}

onMounted(refresh);
onUnmounted(() => clearTimeout(timer));
</script>
