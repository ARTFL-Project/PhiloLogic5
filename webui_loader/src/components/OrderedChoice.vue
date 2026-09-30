<template>
    <div>
        <ul class="list-group mb-2" v-if="(modelValue || []).length">
            <li v-for="(item, index) in modelValue" :key="`${item}-${index}`" class="list-group-item d-flex align-items-center py-1">
                <span class="mono">{{ item }}</span>
                <span v-if="choices.length && !choices.includes(item)" class="badge text-bg-light ms-2">{{ $t("editors.notInDatabase") }}</span>
                <span class="ms-auto" v-if="!readonly">
                    <button class="btn btn-sm btn-link p-0 me-2" type="button" :disabled="index === 0" @click="move(index, -1)" :aria-label="$t('editors.up')"><i class="bi bi-arrow-up"></i></button>
                    <button class="btn btn-sm btn-link p-0 me-2" type="button" :disabled="index === modelValue.length - 1" @click="move(index, 1)" :aria-label="$t('editors.down')"><i class="bi bi-arrow-down"></i></button>
                    <button class="btn btn-sm btn-link p-0 text-danger" type="button" @click="remove(index)" :aria-label="$t('editors.remove')"><i class="bi bi-x-lg"></i></button>
                </span>
            </li>
        </ul>
        <div class="input-group input-group-sm w-auto" v-if="!readonly">
            <select v-if="available.length" class="form-select" v-model="picked" :aria-label="$t('editors.add')">
                <option value="" disabled>{{ $t("editors.choose") }}</option>
                <option v-for="choice in available" :key="choice" :value="choice">{{ choice }}</option>
            </select>
            <input v-if="allowCustom" class="form-control mono" v-model="custom" :placeholder="$t('editors.other')" @keyup.enter="add" />
            <button class="btn btn-outline-secondary" type="button" @click="add"><i class="bi bi-plus"></i> {{ $t("editors.add") }}</button>
        </div>
    </div>
</template>

<script setup>
import { computed, ref } from "vue";

// An ordered selection, from choices (and other values if allowCustom)
const props = defineProps({
    modelValue: { type: Array, default: () => [] },
    choices: { type: Array, default: () => [] },
    allowCustom: { type: Boolean, default: false },
    readonly: Boolean,
});
const emit = defineEmits(["update:modelValue"]);
const picked = ref("");
const custom = ref("");
const available = computed(() => props.choices.filter((choice) => !(props.modelValue || []).includes(choice)));

function add() {
    const value = (custom.value || picked.value).trim();
    if (value && !(props.modelValue || []).includes(value)) {
        emit("update:modelValue", [...(props.modelValue || []), value]);
    }
    picked.value = "";
    custom.value = "";
}

function remove(index) {
    emit("update:modelValue", props.modelValue.filter((item, itemIndex) => itemIndex !== index));
}

function move(index, step) {
    const items = [...props.modelValue];
    [items[index], items[index + step]] = [items[index + step], items[index]];
    emit("update:modelValue", items);
}
</script>
