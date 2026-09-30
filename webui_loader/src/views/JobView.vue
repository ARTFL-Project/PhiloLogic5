<template>
    <div v-if="job">
        <div class="d-flex flex-wrap align-items-center gap-2 mb-3">
            <h1 class="h3 mb-0 font-monospace">{{ job.dbname }}</h1>
            <JobState :state="job.state" />
            <span class="small text-body-secondary">{{ $t("job.startedBy", { date: formatDate(job.created, locale), user: job.user || "" }) }}</span>
            <button v-if="job.state === 'running'" class="btn btn-sm btn-outline-danger ms-auto" type="button" :disabled="job.cancel_requested || cancelling" @click="cancel">
                <i class="bi bi-stop-circle me-1"></i>{{ job.cancel_requested ? $t("job.cancelling") : $t("job.cancel") }}
            </button>
        </div>
        <div v-if="error" class="alert alert-danger">{{ error }}</div>
        <div class="row g-3">
            <div class="col-lg-4">
                <div class="card mb-3">
                    <div class="card-body">
                        <StageStepper :stages="job.stages" :now="now" />
                        <div v-if="job.progress" class="mt-3">
                            <div class="small mb-1">{{ job.progress.description }}: {{ job.progress.done }} / {{ job.progress.total }}</div>
                            <div class="progress" role="progressbar" :aria-valuenow="job.progress.percent" aria-valuemin="0" aria-valuemax="100">
                                <div class="progress-bar progress-bar-striped progress-bar-animated" :style="{ width: `${job.progress.percent}%` }"></div>
                            </div>
                        </div>
                        <div class="small text-body-secondary mt-3">{{ $t("job.elapsed", { time: formatDuration((job.ended || now) - (job.started || job.created)) }) }}</div>
                    </div>
                </div>
                <div v-if="job.state === 'succeeded'" class="alert alert-success">
                    <i class="bi bi-check-circle me-2"></i>{{ $t("job.done") }}
                    <a :href="job.application_url || job.url" target="_blank" rel="noopener" class="d-block mt-1">{{ job.application_url || job.url }}</a>
                    <router-link class="btn btn-sm btn-outline-secondary mt-2" :to="`/databases/${job.dbname}/web_config`">{{ $t("databases.webConfig") }}</router-link>
                </div>
                <div v-if="['failed', 'interrupted'].includes(job.state)" class="alert alert-danger">
                    <i class="bi bi-x-octagon me-2"></i>{{ $t(`job.${job.state}`, { code: job.exit_code }) }}
                </div>
                <div v-if="job.removed_files && job.removed_files.length" class="card mb-3">
                    <div class="card-header small">{{ $t("job.removedFiles", { count: job.removed_files.length }) }}</div>
                    <ul class="list-group list-group-flush small file-list">
                        <li v-for="file in job.removed_files" :key="file.name" class="list-group-item"><span class="mono">{{ file.name }}</span>: {{ file.cause }}</li>
                    </ul>
                </div>
            </div>
            <div class="col-lg-8">
                <div class="d-flex align-items-center mb-1">
                    <h2 class="h6 mb-0">{{ $t("job.log") }}</h2>
                    <div class="form-check form-switch ms-auto small">
                        <input class="form-check-input" type="checkbox" id="follow" v-model="follow" />
                        <label class="form-check-label" for="follow">{{ $t("job.follow") }}</label>
                    </div>
                </div>
                <pre ref="logElement" class="log p-2 rounded mono">{{ log }}</pre>
                <details class="mt-3">
                    <summary class="small">{{ $t("job.command") }}</summary>
                    <pre class="bg-light p-2 rounded mono mt-2">{{ job.command }}</pre>
                    <pre v-if="config" class="bg-light p-2 rounded mono">{{ config }}</pre>
                </details>
            </div>
        </div>
    </div>
    <div v-else-if="error" class="alert alert-danger">{{ error }}</div>
    <div v-else class="text-center my-5"><span class="spinner-border"></span></div>
</template>

<script setup>
import { nextTick, onMounted, onUnmounted, ref } from "vue";
import { useI18n } from "vue-i18n";
import { api } from "../api";
import JobState from "../components/JobState.vue";
import StageStepper from "../components/StageStepper.vue";
import { formatDate, formatDuration } from "../utils";

const props = defineProps({ id: { type: String, required: true } });
const { locale, t } = useI18n();
const job = ref(null);
const log = ref("");
const config = ref(null);
const error = ref(null);
const follow = ref(true);
const cancelling = ref(false);
const logElement = ref(null);
const now = ref(Date.now() / 1000);
let offset = 0;
let timer = null;
let stopped = false;

// Append a part of the log: progress bars rewrite their line (carriage returns)
function appendLog(text) {
    const lines = (log.value + text).split("\n");
    log.value = lines.map((line) => line.slice(line.lastIndexOf("\r") + 1)).join("\n");
}

async function readLog() {
    let more = true;
    while (more) {
        const part = await api.get(`jobs/${props.id}/log`, { offset });
        if (part.text) {
            appendLog(part.text);
        }
        offset = part.offset;
        more = part.more;
    }
    if (follow.value && logElement.value) {
        await nextTick();
        logElement.value.scrollTop = logElement.value.scrollHeight;
    }
}

async function refresh() {
    try {
        job.value = await api.get(`jobs/${props.id}`);
        await readLog();
        if (!config.value) {
            config.value = (await api.get(`jobs/${props.id}/load_config`)).config;
        }
        error.value = null;
    } catch (requestError) {
        error.value = requestError.message;
    }
    now.value = Date.now() / 1000;
    if (!stopped && (!job.value || job.value.state === "running")) {
        timer = setTimeout(refresh, 2000);
    }
}

async function cancel() {
    if (!window.confirm(t("job.confirmCancel", { name: job.value.dbname }))) {
        return;
    }
    cancelling.value = true;
    try {
        await api.post(`jobs/${props.id}/cancel`);
        await refresh();
    } catch (requestError) {
        error.value = requestError.message;
    } finally {
        cancelling.value = false;
    }
}

onMounted(refresh);
onUnmounted(() => {
    stopped = true;
    clearTimeout(timer);
});
</script>
