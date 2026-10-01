<template>
    <div>
        <div class="form-text mt-0 mb-1">{{ query ? $t("replacements.queryHelp") : $t("replacements.htmlHelp") }}</div>
        <table class="table table-sm align-middle mb-1" v-if="rows.length">
            <thead>
                <tr>
                    <th class="number">#</th>
                    <th>{{ $t("replacements.find") }}</th>
                    <th class="arrow"></th>
                    <th>{{ $t("replacements.replace") }}</th>
                    <th v-if="!readonly"></th>
                </tr>
            </thead>
            <tbody>
                <tr v-for="(row, index) in rows" :key="index">
                    <td class="number small text-body-secondary">{{ index + 1 }}</td>
                    <td>
                        <input class="form-control form-control-sm mono" :value="row[0]" :readonly="readonly" :aria-label="`${$t('replacements.find')} ${index + 1}`" @change="set(index, 0, $event.target.value)" />
                        <div v-if="visibleCharacters(row[0])" class="visible small mono text-body-secondary">{{ visibleCharacters(row[0]) }}</div>
                    </td>
                    <td class="arrow text-body-secondary"><i class="bi bi-arrow-right"></i></td>
                    <td>
                        <input class="form-control form-control-sm mono" :value="row[1]" :readonly="readonly" :placeholder="$t('replacements.nothing')" :aria-label="`${$t('replacements.replace')} ${index + 1}`" @change="set(index, 1, $event.target.value)" />
                        <div v-if="visibleCharacters(row[1])" class="visible small mono text-body-secondary">{{ visibleCharacters(row[1]) }}</div>
                    </td>
                    <td v-if="!readonly" class="text-end text-nowrap">
                        <button class="btn btn-sm btn-link p-0 me-2" type="button" :disabled="index === 0" @click="move(index, -1)" :aria-label="$t('editors.up')"><i class="bi bi-arrow-up"></i></button>
                        <button class="btn btn-sm btn-link p-0 me-2" type="button" :disabled="index === rows.length - 1" @click="move(index, 1)" :aria-label="$t('editors.down')"><i class="bi bi-arrow-down"></i></button>
                        <button class="btn btn-sm btn-link p-0 text-danger" type="button" @click="remove(index)" :aria-label="$t('editors.remove')"><i class="bi bi-x-lg"></i></button>
                    </td>
                </tr>
            </tbody>
        </table>
        <button v-if="!readonly" class="btn btn-sm btn-outline-secondary" type="button" @click="add"><i class="bi bi-plus"></i> {{ $t("editors.add") }}</button>
        <div class="tester border rounded p-2 mt-2">
            <label class="form-label small mb-1" :for="`${id}-text`">{{ query ? $t("replacements.tryQuery") : $t("replacements.tryHtml") }}</label>
            <div class="input-group input-group-sm">
                <input :id="`${id}-text`" class="form-control mono" v-model="sample" :placeholder="query ? 'rousseau-emile OR contrat' : '<note>\u2026</note>'" @keyup.enter="test" />
                <button class="btn btn-outline-secondary" type="button" :disabled="!sample || testing" @click="test">{{ $t("replacements.try") }}</button>
            </div>
            <div v-if="testError" class="small text-danger mt-1">{{ testError }}</div>
            <div v-else-if="result" class="mt-2">
                <ol v-if="shownSteps.length" class="small mono steps mb-1">
                    <li v-for="step in shownSteps" :key="step.number" :value="step.number">
                        <span v-if="step.error" class="text-danger">{{ $t("replacements.failed", { error: step.error }) }}</span>
                        <span v-else>{{ step.text }}</span>
                    </li>
                </ol>
                <div v-else class="small text-body-secondary mb-1">{{ $t("replacements.noMatch") }}</div>
                <div class="small">
                    {{ $t("replacements.result") }} <span class="mono fw-semibold result">{{ result.result }}</span>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import { computed, ref, watch } from "vue";
import { api } from "../api";
import { deepEqual } from "../utils";
import { visibleCharacters } from "../webConfigForms";

// A list of (pattern, replacement) which the runtime applies in order: to queries (query_parser_regex), or to the HTML
// of results (the formatting regexes). The tester applies them on the server, as the runtime does.
const props = defineProps({
    modelValue: { type: Array, default: () => [] },
    query: Boolean,
    readonly: Boolean,
});
const emit = defineEmits(["update:modelValue"]);
const id = `replacements-${Math.random().toString(36).slice(2)}`;
const rows = computed(() => (props.modelValue || []).map((row) => [row[0] ?? "", row[1] ?? ""]));
const sample = ref("");
const result = ref(null);
const testError = ref(null);
const testing = ref(false);

function emitRows(newRows) {
    emit("update:modelValue", newRows);
}

function set(index, part, value) {
    emitRows(rows.value.map((row, rowIndex) => (rowIndex === index ? (part === 0 ? [value, row[1]] : [row[0], value]) : row)));
}

function remove(index) {
    emitRows(rows.value.filter((row, rowIndex) => rowIndex !== index));
}

function move(index, step) {
    const newRows = [...rows.value];
    [newRows[index], newRows[index + step]] = [newRows[index + step], newRows[index]];
    emitRows(newRows);
}

function add() {
    emitRows([...rows.value, ["", ""]]);
}

// The steps which changed the text, or failed, with their number
const shownSteps = computed(() => {
    if (!result.value) {
        return [];
    }
    return result.value.steps
        .map((step, index) => ({ ...step, number: index + 1, before: index === 0 ? result.value.sample : result.value.steps[index - 1].text }))
        .filter((step) => step.error || step.text !== step.before);
});

let tested = null;
async function test() {
    if (!sample.value) {
        return;
    }
    testing.value = true;
    testError.value = null;
    tested = { rows: rows.value, sample: sample.value };
    try {
        result.value = await api.post("replacements", { replacements: rows.value, text: sample.value, query: props.query });
        result.value.sample = sample.value;
    } catch (requestError) {
        result.value = null;
        testError.value = requestError.message;
    } finally {
        testing.value = false;
    }
}

// Once tried, the result follows the changes of the replacements
watch(rows, (newRows) => {
    if (tested && !deepEqual(newRows, tested.rows) && sample.value) {
        test();
    }
});
</script>

<style scoped>
.number {
    width: 2rem;
}
.arrow {
    width: 1.5rem;
    text-align: center;
}
.visible {
    white-space: pre;
    overflow-x: auto;
}
.steps {
    padding-left: 1.5rem;
    white-space: pre-wrap;
    word-break: break-all;
}
.result {
    white-space: pre-wrap;
    word-break: break-all;
}
</style>
