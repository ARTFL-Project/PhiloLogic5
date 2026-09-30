<template>
    <div>
        <p class="small text-body-secondary">{{ $t("uploads.help") }}</p>
        <div class="d-flex flex-wrap gap-2 mb-2">
            <label class="btn btn-outline-secondary btn-sm mb-0">
                <i class="bi bi-file-zip me-1"></i>{{ $t("uploads.archive") }}
                <input type="file" class="d-none" accept=".zip,.tar,.tgz,.gz,.bz2,.xz" :disabled="busy" @change="pickArchive" />
            </label>
            <label class="btn btn-outline-secondary btn-sm mb-0">
                <i class="bi bi-folder-plus me-1"></i>{{ $t("uploads.folder") }}
                <input type="file" class="d-none" webkitdirectory multiple :disabled="busy" @change="pickFolder" />
            </label>
            <span class="small ms-auto align-self-center" v-if="usage">{{ usage.quota ? $t("uploads.usage", { used: formatSize(usage.size), quota: formatSize(usage.quota) }) : $t("uploads.used", { used: formatSize(usage.size) }) }}</span>
        </div>
        <div v-if="current" class="mb-2">
            <div class="small mb-1">{{ current.name }} — {{ formatSize(sent) }} / {{ formatSize(current.size) }} <span v-if="current.resumed" class="badge text-bg-info">{{ $t("uploads.resumed") }}</span></div>
            <div class="progress" role="progressbar" :aria-valuenow="percent" aria-valuemin="0" aria-valuemax="100">
                <div class="progress-bar" :style="{ width: `${percent}%` }">{{ percent }}%</div>
            </div>
            <div v-if="extracting" class="small mt-1"><span class="spinner-border spinner-border-sm me-1"></span>{{ $t("uploads.extracting") }}</div>
        </div>
        <div v-if="error" class="alert alert-danger py-1 small">{{ error }} <button v-if="retry" class="btn btn-sm btn-link" @click="retry">{{ $t("uploads.retry") }}</button></div>
        <ul class="list-group list-group-flush small" v-if="uploads.length">
            <li v-for="upload in uploads" :key="upload.id" class="list-group-item d-flex align-items-center gap-2">
                <i class="bi" :class="upload.kind === 'archive' ? 'bi-file-zip' : 'bi-folder'"></i>
                <span>{{ upload.name }}</span>
                <span class="text-body-secondary">{{ upload.file_count }} {{ $t("uploads.files") }}, {{ formatSize(upload.size) }}</span>
                <span v-if="upload.state !== 'ready'" class="badge" :class="upload.state === 'error' ? 'text-bg-danger' : 'text-bg-warning'" :title="upload.error || ''">{{ $t(`uploads.state.${upload.state}`) }}</span>
                <button v-if="upload.state === 'ready'" class="btn btn-sm btn-secondary ms-auto" type="button" @click="emit('ready', upload.path)">{{ $t("uploads.use") }}</button>
                <button class="btn btn-sm btn-outline-danger" :class="{ 'ms-auto': upload.state !== 'ready' }" type="button" @click="remove(upload)" :aria-label="$t('editors.remove')"><i class="bi bi-trash"></i></button>
            </li>
        </ul>
    </div>
</template>

<script setup>
import { computed, onMounted, ref } from "vue";
import { api, request } from "../api";
import { formatSize } from "../utils";

const CHUNK = 4 * 1024 * 1024; // under the limit of the server (8 MB), and of reverse proxies configured for it
const PARALLEL_FILES = 3;
const PENDING_KEY = "philologic5-webui-loader-uploads";

const emit = defineEmits(["ready"]);
const uploads = ref([]);
const usage = ref(null);
const current = ref(null);
const sent = ref(0);
const busy = ref(false);
const extracting = ref(false);
const error = ref(null);
const retry = ref(null);
const percent = computed(() => (current.value && current.value.size ? Math.floor((100 * sent.value) / current.value.size) : 0));

const hidden = (path) => path.split("/").some((part) => part.startsWith(".")) || path.startsWith("__MACOSX/");

async function refresh() {
    const result = await api.get("uploads");
    uploads.value = result.uploads;
    usage.value = result.usage;
}

function pending() {
    try {
        return JSON.parse(localStorage.getItem(PENDING_KEY) || "{}");
    } catch {
        return {};
    }
}

function remember(key, uploadId) {
    const all = pending();
    if (uploadId) {
        all[key] = uploadId;
    } else {
        delete all[key];
    }
    localStorage.setItem(PENDING_KEY, JSON.stringify(all));
}

async function sendFile(uploadId, path, file, offset) {
    let position = offset;
    if (file.size === 0) {
        // An empty file is created by an empty chunk
        await request("PUT", `uploads/${uploadId}/content`, { params: { path, offset: 0 }, raw: new Blob([]) });
        return;
    }
    while (position < file.size) {
        const chunk = file.slice(position, Math.min(position + CHUNK, file.size));
        const result = await request("PUT", `uploads/${uploadId}/content`, { params: { path, offset: position }, raw: chunk });
        sent.value += result.received - position;
        position = result.received;
    }
}

async function upload(kind, name, entries) {
    busy.value = true;
    error.value = null;
    retry.value = null;
    const size = entries.reduce((total, entry) => total + entry.file.size, 0);
    const key = `${kind}:${name}:${size}:${entries.length}`;
    current.value = { name, size, resumed: false };
    sent.value = 0;
    try {
        let uploadId = pending()[key];
        let received = {};
        if (uploadId) {
            try {
                const status = await api.get(`uploads/${uploadId}`);
                if (status.state === "receiving") {
                    received = status.files || {};
                    current.value.resumed = true;
                } else {
                    uploadId = null;
                }
            } catch {
                uploadId = null;
            }
        }
        if (!uploadId) {
            uploadId = (await api.post("uploads", { name, kind, size, file_count: entries.length })).id;
            remember(key, uploadId);
        }
        sent.value = Object.values(received).reduce((total, bytes) => total + bytes, 0);
        const queue = [...entries];
        const worker = async () => {
            while (queue.length) {
                const entry = queue.shift();
                const offset = received[entry.path] || 0;
                await sendFile(uploadId, entry.path, entry.file, Math.min(offset, entry.file.size));
            }
        };
        await Promise.all(Array.from({ length: kind === "files" ? PARALLEL_FILES : 1 }, worker));
        extracting.value = kind === "archive";
        const status = await api.post(`uploads/${uploadId}/complete`);
        remember(key, null);
        await refresh();
        emit("ready", status.path);
    } catch (uploadError) {
        error.value = uploadError.message;
        retry.value = () => upload(kind, name, entries);
    } finally {
        busy.value = false;
        extracting.value = false;
    }
}

function pickArchive(event) {
    const file = event.target.files[0];
    event.target.value = "";
    if (file) {
        upload("archive", file.name, [{ path: file.name, file }]);
    }
}

function pickFolder(event) {
    const entries = Array.from(event.target.files)
        .map((file) => ({ path: file.webkitRelativePath || file.name, file }))
        .filter((entry) => !hidden(entry.path));
    event.target.value = "";
    if (entries.length) {
        upload("files", entries[0].path.split("/")[0], entries);
    }
}

async function remove(item) {
    await api.delete(`uploads/${item.id}`);
    await refresh();
}

onMounted(refresh);
</script>
