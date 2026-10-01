<template>
    <div>
        <div v-for="(order, index) in orders" :key="index" class="d-flex flex-wrap align-items-center gap-1 mb-2">
            <template v-for="(field, position) in order" :key="position">
                <span v-if="position" class="small text-body-secondary mx-1">{{ $t("sortOrders.then") }}</span>
                <span class="input-group input-group-sm w-auto">
                    <select class="form-select form-select-sm mono" :value="field" :disabled="readonly" :aria-label="$t('sortOrders.field')" @change="setField(index, position, $event.target.value)">
                        <option v-for="choice in choices(field)" :key="choice" :value="choice">{{ choice }}</option>
                    </select>
                    <button v-if="!readonly && order.length > 1" class="btn btn-outline-secondary" type="button" @click="removeField(index, position)" :aria-label="$t('editors.remove')"><i class="bi bi-x"></i></button>
                </span>
            </template>
            <button v-if="!readonly" class="btn btn-sm btn-link" type="button" @click="addField(index)"><i class="bi bi-plus"></i> {{ $t("sortOrders.addField") }}</button>
            <span v-if="!readonly" class="ms-auto text-nowrap">
                <button class="btn btn-sm btn-link p-0 me-2" type="button" :disabled="index === 0" @click="move(index, -1)" :aria-label="$t('editors.up')"><i class="bi bi-arrow-up"></i></button>
                <button class="btn btn-sm btn-link p-0 me-2" type="button" :disabled="index === orders.length - 1" @click="move(index, 1)" :aria-label="$t('editors.down')"><i class="bi bi-arrow-down"></i></button>
                <button class="btn btn-sm btn-link p-0 text-danger" type="button" @click="remove(index)" :aria-label="$t('editors.remove')"><i class="bi bi-x-lg"></i></button>
            </span>
        </div>
        <button v-if="!readonly" class="btn btn-sm btn-outline-secondary" type="button" :disabled="!metadataFields.length" @click="add"><i class="bi bi-plus"></i> {{ $t("sortOrders.add") }}</button>
    </div>
</template>

<script setup>
import { computed } from "vue";

// Ways to sort results: each a sequence of metadata fields, such as author then title
const props = defineProps({
    modelValue: { type: Array, default: () => [] },
    metadataFields: { type: Array, default: () => [] },
    readonly: Boolean,
});
const emit = defineEmits(["update:modelValue"]);
const orders = computed(() => (props.modelValue || []).map((order) => [...order]));
const choices = (field) => (props.metadataFields.includes(field) ? props.metadataFields : [field, ...props.metadataFields]);

function emitOrders(newOrders) {
    emit("update:modelValue", newOrders);
}

function setField(index, position, field) {
    emitOrders(orders.value.map((order, orderIndex) => (orderIndex === index ? order.map((other, otherPosition) => (otherPosition === position ? field : other)) : order)));
}

function addField(index) {
    const order = orders.value[index];
    const next = props.metadataFields.find((field) => !order.includes(field)) || props.metadataFields[0];
    emitOrders(orders.value.map((other, orderIndex) => (orderIndex === index ? [...other, next] : other)));
}

function removeField(index, position) {
    emitOrders(orders.value.map((order, orderIndex) => (orderIndex === index ? order.filter((field, otherPosition) => otherPosition !== position) : order)));
}

function move(index, step) {
    const newOrders = [...orders.value];
    [newOrders[index], newOrders[index + step]] = [newOrders[index + step], newOrders[index]];
    emitOrders(newOrders);
}

function remove(index) {
    emitOrders(orders.value.filter((order, orderIndex) => orderIndex !== index));
}

function add() {
    emitOrders([...orders.value, [props.metadataFields[0]]]);
}
</script>
