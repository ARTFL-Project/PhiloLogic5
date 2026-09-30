<template>
    <div class="border rounded">
        <div class="d-flex flex-wrap gap-2 align-items-center p-2 border-bottom bg-body-tertiary">
            <button class="btn btn-sm btn-outline-secondary" type="button" :disabled="!listing || !listing.parent" @click="open(listing.parent)" :aria-label="$t('files.up')">
                <i class="bi bi-arrow-up"></i>
            </button>
            <select v-if="roots.length" class="form-select form-select-sm w-auto" @change="open($event.target.value)" :aria-label="$t('files.roots')">
                <option v-for="root in roots" :key="root" :value="root" :selected="listing && listing.path.startsWith(root)">{{ root }}</option>
            </select>
            <input class="form-control form-control-sm mono flex-grow-1" v-model="typedPath" @keyup.enter="open(typedPath)" :aria-label="$t('files.path')" />
            <input v-if="mode === 'directory'" class="form-control form-control-sm mono w-auto" v-model="pattern" @change="refresh" :title="$t('files.pattern')" :aria-label="$t('files.pattern')" size="8" />
        </div>
        <div v-if="error" class="alert alert-danger m-2 py-1">{{ error }}</div>
        <ul v-if="listing" class="list-group list-group-flush file-list">
            <li v-for="name in listing.directories" :key="`d-${name}`" class="list-group-item list-group-item-action" role="button" @click="open(join(name))">
                <i class="bi bi-folder me-2 text-warning"></i>{{ name }}
            </li>
            <li v-for="file in listing.files" :key="`f-${file.name}`" class="list-group-item d-flex" :class="{ 'list-group-item-action': mode === 'file', active: mode === 'file' && join(file.name) === selected }" :role="mode === 'file' ? 'button' : undefined" @click="mode === 'file' && choose(join(file.name))">
                <span><i class="bi bi-file-earmark me-2"></i>{{ file.name }}</span>
                <span class="ms-auto small text-body-secondary">{{ formatSize(file.size) }}</span>
            </li>
            <li v-if="listing.truncated" class="list-group-item small text-body-secondary">{{ $t("files.truncated", { count: listing.file_count }) }}</li>
        </ul>
        <div v-if="listing && mode === 'directory'" class="d-flex align-items-center gap-2 p-2 border-top">
            <span class="small">{{ $t("files.matching", { count: listing.matching, pattern, size: formatSize(listing.matching_size) }) }} <span class="mono text-body-secondary">{{ listing.path }}</span></span>
            <button class="btn btn-sm btn-secondary ms-auto" type="button" :disabled="!listing.matching" @click="choose(listing.path)">{{ $t("files.useFolder") }}</button>
        </div>
    </div>
</template>

<script setup>
import { onMounted, ref } from "vue";
import { api } from "../api";
import { formatSize } from "../utils";

// Browse the files of the machine of the UI server (in the service, only the directories which can be loaded from)
const props = defineProps({
    mode: { type: String, default: "directory" }, // choose a directory (with a pattern), or a file
    start: { type: String, default: "" },
    initialPattern: { type: String, default: "*.xml" },
    selected: { type: String, default: "" },
});
const emit = defineEmits(["choose"]);
const listing = ref(null);
const roots = ref([]);
const error = ref(null);
const typedPath = ref(props.start);
const pattern = ref(props.initialPattern);

const join = (name) => (listing.value.path.endsWith("/") ? listing.value.path + name : `${listing.value.path}/${name}`);

async function open(path) {
    try {
        const result = await api.get("files", { path, pattern: props.mode === "directory" ? pattern.value : "*" });
        listing.value = result;
        roots.value = result.roots || [];
        typedPath.value = result.path;
        error.value = null;
    } catch (requestError) {
        error.value = requestError.message;
    }
}

const refresh = () => open(listing.value ? listing.value.path : props.start);

function choose(path) {
    emit("choose", props.mode === "directory" ? { directory: path, pattern: pattern.value } : path);
}

onMounted(() => open(props.start || undefined));
</script>
