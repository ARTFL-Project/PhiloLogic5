<template>
    <div class="medium mx-auto" v-if="schema">
        <div class="d-flex align-items-center mb-3">
            <h1 class="h3 mb-0">{{ $t("load.title") }}</h1>
            <button class="btn btn-sm btn-outline-secondary ms-auto" type="button" @click="startOver">{{ $t("load.startOver") }}</button>
        </div>
        <ul class="nav nav-pills mb-3">
            <li class="nav-item" v-for="(step, index) in steps" :key="step">
                <button class="nav-link" :class="{ active: state.step === step }" type="button" :disabled="index > 0 && !sourceReady" @click="go(step)">
                    {{ index + 1 }}. {{ $t(`load.steps.${step}`) }}
                </button>
            </li>
        </ul>

        <!-- 1. Source -->
        <div v-show="state.step === 'source'">
            <div class="card mb-3">
                <div class="card-header">{{ $t("load.startFrom") }}</div>
                <div class="card-body">
                    <div class="form-check" v-for="choice in ['defaults', 'database', 'config']" :key="choice">
                        <input class="form-check-input" type="radio" :id="`start-${choice}`" :value="choice" v-model="state.startFrom" @change="changeStart" />
                        <label class="form-check-label" :for="`start-${choice}`">{{ $t(`load.start.${choice}`) }}</label>
                    </div>
                    <div v-if="state.startFrom === 'database'" class="mt-2">
                        <select class="form-select w-auto" v-model="state.startDatabase" @change="readDatabaseConfig(state.startDatabase)" :aria-label="$t('load.start.database')">
                            <option value="" disabled>{{ $t("load.chooseDatabase") }}</option>
                            <option v-for="database in databases.filter((db) => db.has_load_config)" :key="database.name" :value="database.name">{{ database.name }}</option>
                        </select>
                    </div>
                    <div v-if="state.startFrom === 'config'" class="mt-2">
                        <div class="input-group mb-2">
                            <input class="form-control mono" v-model="state.startConfigPath" :placeholder="$t('load.configPath')" :aria-label="$t('load.configPath')" />
                            <button class="btn btn-outline-secondary" type="button" @click="readConfigFile(state.startConfigPath)">{{ $t("load.read") }}</button>
                            <button class="btn btn-outline-secondary" type="button" @click="browseConfig = !browseConfig"><i class="bi bi-folder2-open"></i></button>
                        </div>
                        <FileBrowser v-if="browseConfig" mode="file" :start="startDirectory" :selected="state.startConfigPath" @choose="(path) => { state.startConfigPath = path; browseConfig = false; readConfigFile(path); }" />
                    </div>
                    <div v-if="baseError" class="alert alert-danger py-1 mt-2">{{ baseError }}</div>
                    <BaseNotes v-if="state.baseInfo" :info="state.baseInfo" class="mt-2" />
                </div>
            </div>

            <div class="card mb-3">
                <div class="card-header">{{ $t("load.files") }}</div>
                <div class="card-body">
                    <ul class="nav nav-tabs mb-2">
                        <li class="nav-item">
                            <button class="nav-link" :class="{ active: state.filesMode !== 'upload' }" type="button" @click="state.filesMode = state.filesMode === 'upload' ? 'directory' : state.filesMode">
                                <i class="bi bi-hdd me-1"></i>{{ $t("load.filesOnHost") }}
                            </button>
                        </li>
                        <li class="nav-item" v-if="session.server.uploads">
                            <button class="nav-link" :class="{ active: state.filesMode === 'upload' }" type="button" @click="state.filesMode = 'upload'">
                                <i class="bi bi-upload me-1"></i>{{ $t("load.uploadFromComputer") }}
                            </button>
                        </li>
                    </ul>
                    <div v-if="state.filesMode !== 'upload'" class="mb-2">
                        <div class="form-check form-check-inline" v-for="mode in ['directory', 'file_list']" :key="mode">
                            <input class="form-check-input" type="radio" :id="`files-${mode}`" :value="mode" v-model="state.filesMode" />
                            <label class="form-check-label" :for="`files-${mode}`">{{ $t(`load.filesModes.${mode}`) }}</label>
                        </div>
                    </div>
                    <div v-if="state.files" class="alert alert-success py-2">
                        <i class="bi bi-check-circle me-2"></i>
                        <span v-if="state.files.directory">{{ $t("load.chosenFolder", { directory: state.files.directory, pattern: state.files.pattern }) }}</span>
                        <span v-else>{{ $t("load.chosenList", { path: state.files.file_list }) }}</span>
                        <div v-if="state.files.directory" class="form-check mt-1">
                            <input class="form-check-input" type="checkbox" id="recursive" v-model="state.files.recursive" />
                            <label class="form-check-label" for="recursive">{{ $t("load.recursive") }}</label>
                        </div>
                    </div>
                    <FileBrowser v-if="state.filesMode === 'directory'" mode="directory" :start="state.files && state.files.directory ? state.files.directory : startDirectory" :initial-pattern="state.files && state.files.pattern ? state.files.pattern : '*.xml'" @choose="(spec) => (state.files = { ...spec, recursive: false })" />
                    <div v-else-if="state.filesMode === 'file_list'">
                        <p class="small text-body-secondary">{{ $t("load.fileListHelp") }}</p>
                        <FileBrowser mode="file" :start="startDirectory" @choose="(path) => (state.files = { file_list: path })" />
                    </div>
                    <Uploader v-else-if="state.filesMode === 'upload'" @ready="(path) => (state.files = { directory: path, pattern: '*', recursive: true })" />
                </div>
            </div>

            <div class="card mb-3">
                <div class="card-header">{{ $t("load.texts") }}</div>
                <div class="card-body row g-3">
                    <div class="col-sm-4">
                        <label class="form-label" for="file_type">{{ $t("fields.file_type") }}</label>
                        <select id="file_type" class="form-select" v-model="state.file_type">
                            <option value="xml">{{ $t("load.xml") }}</option>
                            <option value="plain_text">{{ $t("load.plainText") }}</option>
                        </select>
                    </div>
                    <div class="col-sm-4" v-if="state.file_type === 'xml'">
                        <label class="form-label" for="header">{{ $t("fields.header") }}</label>
                        <select id="header" class="form-select" v-model="state.header">
                            <option value="tei">TEI</option>
                            <option value="dc">Dublin Core</option>
                        </select>
                    </div>
                    <div class="col-12">
                        <label class="form-label" for="bibliography">{{ $t("fields.bibliography") }} <span class="text-body-secondary small">({{ state.file_type === "plain_text" ? $t("load.required") : $t("load.optional") }})</span></label>
                        <div class="input-group">
                            <input id="bibliography" class="form-control mono" v-model="state.bibliography" :placeholder="$t('load.bibliographyHelp')" />
                            <button class="btn btn-outline-secondary" type="button" @click="browseBibliography = !browseBibliography"><i class="bi bi-folder2-open"></i></button>
                        </div>
                        <FileBrowser v-if="browseBibliography" class="mt-2" mode="file" :start="startDirectory" :selected="state.bibliography" @choose="(path) => { state.bibliography = path; browseBibliography = false; }" />
                    </div>
                </div>
            </div>

            <div class="card mb-3">
                <div class="card-header">{{ $t("load.database") }}</div>
                <div class="card-body row g-3">
                    <div class="col-sm-6">
                        <label class="form-label" for="dbname">{{ $t("fields.dbname") }}</label>
                        <input id="dbname" class="form-control mono" :class="{ 'is-invalid': state.dbname && !validName }" v-model.trim="state.dbname" />
                        <div class="invalid-feedback">{{ $t("load.nameHelp") }}</div>
                        <div class="form-text" v-if="system && system.databases_url">{{ $t("load.urlWillBe", { url: absoluteUrl(`${system.databases_url}${state.dbname || "\u2026"}/`) }) }}</div>
                    </div>
                    <div class="col-sm-3">
                        <label class="form-label" for="cores">{{ $t("fields.cores") }}</label>
                        <input id="cores" type="number" min="1" :max="maxCores" class="form-control" v-model.number="state.cores" />
                        <div class="form-text" v-if="system">{{ system.max_cores ? $t("load.coresLimit", { count: system.max_cores }) : $t("load.coresHelp", { count: system.cpu_count }) }}</div>
                    </div>
                    <div class="col-12" v-if="existing">
                        <div class="alert alert-warning py-2 mb-0">
                            <i class="bi bi-exclamation-triangle me-2"></i>{{ $t("load.exists", { name: state.dbname }) }}
                            <div class="form-check mt-1">
                                <input class="form-check-input" type="checkbox" id="overwrite" v-model="state.overwrite" />
                                <label class="form-check-label" for="overwrite">{{ $t("load.overwrite") }}</label>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 2. Options -->
        <div v-show="state.step === 'options'">
            <div class="d-flex align-items-center mb-2">
                <ul class="nav nav-tabs flex-grow-1">
                    <li class="nav-item" v-for="group in optionGroups" :key="group">
                        <button class="nav-link" :class="{ active: optionGroup === group }" type="button" @click="optionGroup = group">
                            {{ $t(`load.groups.${group}`) }}<span v-if="groupChanged(group)" class="text-primary"> &bull;</span><span v-if="groupErrors(group)" class="text-danger"> !</span>
                        </button>
                    </li>
                </ul>
                <div class="form-check form-switch ms-3">
                    <input class="form-check-input" type="checkbox" id="advanced" v-model="state.showAdvanced" />
                    <label class="form-check-label small" for="advanced">{{ $t("load.showAdvanced") }}</label>
                </div>
            </div>
            <OptionField
                v-for="option in groupOptions"
                :key="option.key"
                :option="option"
                v-model="state.options[option.key]"
                :error="fieldErrors[option.key]"
                :readonly-source="state.baseInfo && state.baseInfo.code[option.key]"
                :spacy-models="system ? system.spacy_models : []"
            />
            <p v-if="!groupOptions.length" class="text-body-secondary">{{ $t("load.noBasicOptions") }}</p>
        </div>

        <!-- 3. Previews -->
        <div v-if="state.step === 'previews'">
            <PreviewPanel :files="state.files" :options="optionsPayload" :file-names="sampleNames" />
        </div>

        <!-- 4. Review -->
        <div v-if="state.step === 'review'">
            <div v-if="checking" class="text-center my-4"><span class="spinner-border"></span></div>
            <template v-else-if="plan">
                <Messages :errors="plan.errors" :warnings="plan.warnings" />
                <div v-if="plan.ok" class="alert alert-success py-2"><i class="bi bi-check-circle me-2"></i>{{ $t("load.ready") }}</div>
                <div class="card mb-3" v-if="plan.summary">
                    <div class="card-body small">
                        <div>{{ $t("load.summary", { count: plan.summary.count, size: formatSize(plan.summary.size) }) }}</div>
                        <div class="text-body-secondary">{{ Object.entries(plan.summary.extensions).map(([ext, n]) => `${ext}: ${n}`).join(", ") }}</div>
                    </div>
                </div>
                <h2 class="h6">{{ $t("load.command") }}</h2>
                <pre class="bg-light p-2 rounded mono">{{ plan.command }}</pre>
                <h2 class="h6" v-if="plan.config">{{ $t("load.config") }}</h2>
                <pre v-if="plan.config" class="bg-light p-2 rounded mono">{{ plan.config }}</pre>
            </template>
            <div v-if="launchError" class="alert alert-danger">{{ launchError }}</div>
        </div>

        <div class="sticky-actions d-flex gap-2">
            <button class="btn btn-outline-secondary" type="button" v-if="stepIndex > 0" @click="go(steps[stepIndex - 1])"><i class="bi bi-arrow-left me-1"></i>{{ $t("load.back") }}</button>
            <span class="align-self-center small text-danger" v-if="state.step === 'source' && !sourceReady">{{ sourceMissing }}</span>
            <button class="btn btn-secondary ms-auto" type="button" v-if="stepIndex < steps.length - 1" :disabled="!sourceReady" @click="go(steps[stepIndex + 1])">{{ $t("load.next") }}<i class="bi bi-arrow-right ms-1"></i></button>
            <button class="btn btn-secondary ms-auto" type="button" v-else :disabled="!plan || !plan.ok || launching" @click="launch">
                <span v-if="launching" class="spinner-border spinner-border-sm me-1"></span><i v-else class="bi bi-play-fill me-1"></i>{{ $t("load.launch") }}
            </button>
        </div>
    </div>
    <div v-else class="text-center my-5"><span class="spinner-border"></span></div>
</template>

<script setup>
import { computed, onMounted, reactive, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import { useRoute, useRouter } from "vue-router";
import { api } from "../api";
import BaseNotes from "../components/BaseNotes.vue";
import FileBrowser from "../components/FileBrowser.vue";
import Messages from "../components/Messages.vue";
import OptionField from "../components/OptionField.vue";
import PreviewPanel from "../components/PreviewPanel.vue";
import Uploader from "../components/Uploader.vue";
import { session } from "../session";
import { absoluteUrl, clone, deepEqual, formatSize } from "../utils";

const STATE_KEY = "philologic5-webui-loader-load";
const NAME = /^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$/;
const SOURCE_OPTIONS = ["header", "file_type"];

const { t } = useI18n();
const route = useRoute();
const router = useRouter();
const steps = ["source", "options", "previews", "review"];
const schema = ref(null);
const system = ref(null);
const databases = ref([]);
const plan = ref(null);
const checking = ref(false);
const launching = ref(false);
const launchError = ref(null);
const baseError = ref(null);
const browseConfig = ref(false);
const browseBibliography = ref(false);
const optionGroup = ref("structure");

function freshState() {
    return {
        step: "source",
        startFrom: "defaults",
        startDatabase: "",
        startConfigPath: "",
        base: null,
        baseInfo: null,
        filesMode: "directory",
        files: null,
        dbname: "",
        overwrite: false,
        header: "tei",
        file_type: "xml",
        bibliography: "",
        cores: 4,
        options: {},
        showAdvanced: false,
    };
}

const saved = (() => {
    try {
        return JSON.parse(sessionStorage.getItem(STATE_KEY) || "null");
    } catch {
        return null;
    }
})();
const state = reactive(route.query.from ? freshState() : saved || freshState());
watch(state, () => sessionStorage.setItem(STATE_KEY, JSON.stringify(state)), { deep: true });

const stepIndex = computed(() => steps.indexOf(state.step));
const startDirectory = computed(() => (system.value ? (system.value.roots && system.value.roots[0]) || system.value.home || "" : ""));
const validName = computed(() => NAME.test(state.dbname) && !state.dbname.includes(".."));
const existing = computed(() => databases.value.some((database) => database.name === state.dbname));
const sourceMissing = computed(() => {
    if (!state.files) return t("load.missing.files");
    if (!validName.value) return t("load.missing.name");
    if (existing.value && !state.overwrite) return t("load.missing.overwrite");
    if (state.file_type === "plain_text" && !state.bibliography) return t("load.missing.bibliography");
    return "";
});
const sourceReady = computed(() => !sourceMissing.value);
const editableOptions = computed(() => schema.value.options.filter((option) => !SOURCE_OPTIONS.includes(option.key)));
const optionGroups = computed(() => schema.value.groups.filter((group) => group !== "source" && editableOptions.value.some((option) => option.group === group)));
const groupOptions = computed(() => editableOptions.value.filter((option) => option.group === optionGroup.value && (option.basic || state.showAdvanced || changed(option))));
const fieldErrors = computed(() => {
    const errors = {};
    for (const error of (plan.value && plan.value.errors) || []) {
        errors[error.field] = error.message;
    }
    return errors;
});
const sampleNames = computed(() => (plan.value && plan.value.summary ? plan.value.summary.sample : []));

// Options sent with a load: all but those which the load config sets by code
const optionsPayload = computed(() => {
    const payload = {};
    for (const option of editableOptions.value) {
        if (!(state.baseInfo && state.baseInfo.code[option.key])) {
            payload[option.key] = state.options[option.key];
        }
    }
    return payload;
});

const requestPayload = computed(() => ({
    dbname: state.dbname,
    files: state.files,
    header: state.header,
    file_type: state.file_type,
    bibliography: state.bibliography || null,
    options: optionsPayload.value,
    base: state.base,
    cores: state.cores,
    overwrite: existing.value && state.overwrite,
}));

function changed(option) {
    return !deepEqual(state.options[option.key], option.default) && !(option.kind === "path" && !state.options[option.key] && !option.default);
}
const groupChanged = (group) => editableOptions.value.some((option) => option.group === group && changed(option));
const groupErrors = (group) => editableOptions.value.some((option) => option.group === group && fieldErrors.value[option.key]);

function defaults() {
    const options = {};
    for (const option of schema.value.options) {
        options[option.key] = clone(option.default);
    }
    return options;
}

// Start from what a load config sets
function applyBase(info, base) {
    const options = defaults();
    for (const [key, value] of Object.entries(info.options)) {
        if (key in options) {
            options[key] = value;
        }
    }
    state.options = options;
    state.header = info.options.header || "tei";
    state.file_type = info.options.file_type || "xml";
    state.base = base;
    state.baseInfo = info;
    baseError.value = null;
}

async function readDatabaseConfig(name) {
    try {
        const database = await api.get(`databases/${name}`);
        if (!database.load_config || database.load_config.error) {
            throw new Error(database.load_config ? database.load_config.error : t("load.noSavedConfig"));
        }
        applyBase(database.load_config, { database: name });
    } catch (error) {
        baseError.value = error.message;
    }
}

async function readConfigFile(path) {
    try {
        applyBase(await api.post("load_config", { path }), { config_path: path });
    } catch (error) {
        baseError.value = error.message;
    }
}

function changeStart() {
    if (state.startFrom === "defaults") {
        state.options = defaults();
        state.base = null;
        state.baseInfo = null;
        baseError.value = null;
    }
}

async function check() {
    checking.value = true;
    try {
        plan.value = await api.post("preflight", { request: requestPayload.value });
    } catch (error) {
        plan.value = { errors: [{ field: null, message: error.message }], warnings: [], ok: false };
    } finally {
        checking.value = false;
    }
}

async function go(step) {
    state.step = step;
    launchError.value = null;
    if (step === "review" || (step === "previews" && !plan.value)) {
        await check();
    }
}

async function launch() {
    launching.value = true;
    launchError.value = null;
    try {
        const result = await api.post("jobs", { request: requestPayload.value });
        sessionStorage.removeItem(STATE_KEY);
        router.push(`/jobs/${result.id}`);
    } catch (error) {
        if (error.data && error.data.errors) {
            plan.value = { ...plan.value, ...error.data };
        }
        launchError.value = error.message;
    } finally {
        launching.value = false;
    }
}

function startOver() {
    sessionStorage.removeItem(STATE_KEY);
    Object.assign(state, freshState(), { options: defaults(), cores: defaultCores() });
    plan.value = null;
    router.replace("/load");
}

const maxCores = computed(() => (system.value ? system.value.max_cores || system.value.cpu_count : 64));
const defaultCores = () => (system.value ? Math.max(1, Math.min(8, Math.floor(system.value.cpu_count / 2), maxCores.value)) : 4);

onMounted(async () => {
    const [schemaResult, systemResult, databasesResult] = await Promise.all([api.get("load_options"), api.get("system"), api.get("databases")]);
    schema.value = schemaResult;
    system.value = systemResult;
    databases.value = databasesResult.databases;
    if (!Object.keys(state.options).length) {
        state.options = defaults();
        state.cores = defaultCores();
    }
    if (route.query.from) {
        state.startFrom = "database";
        state.startDatabase = route.query.from;
        await readDatabaseConfig(route.query.from);
        if (route.query.again) {
            state.dbname = route.query.from;
            state.overwrite = true;
        }
    }
});
</script>
