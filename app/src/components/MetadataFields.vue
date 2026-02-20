<template>
    <div class="input-group pb-2" v-for="field in fields" :key="field.value">
        <!-- Text input: scoped slot with plain-input fallback -->
        <div class="input-group pb-2" :id="field.value + '-group'"
            v-if="inputStyles[field.value] === 'text'">
            <slot name="text-input" :field="field">
                <label class="btn btn-outline-secondary" :for="field.value + '-input-filter'">
                    {{ field.label }}
                </label>
                <input type="text" class="form-control" :id="field.value + '-input-filter'"
                    :name="field.value" :placeholder="field.example"
                    v-model="modelValue[field.value]"
                    :aria-label="`${$t('collocation.filterBy')} ${field.label}`" />
            </slot>
        </div>

        <!-- Checkbox input -->
        <div class="input-group pb-2" :id="field.value + '-group'"
            v-if="inputStyles[field.value] === 'checkbox'">
            <label class="btn btn-outline-secondary me-2"
                style="border-top-right-radius: 0; border-bottom-right-radius: 0">
                {{ field.label }}
            </label>
            <div class="d-inline-block">
                <div class="form-check d-inline-block ms-3" style="padding-top: 0.35rem"
                    :id="field.value"
                    v-for="choice in choiceValues[field.value]"
                    :key="choice.value">
                    <input class="form-check-input" type="checkbox" :id="choice.value"
                        v-model="checkedValues[choice.value]"
                        :aria-label="`${$t('collocation.filterBy')} ${field.label}: ${choice.text}`" />
                    <label class="form-check-label" :for="choice.value">
                        {{ choice.text }}
                    </label>
                </div>
            </div>
        </div>

        <!-- Dropdown input -->
        <div class="input-group pb-2" :id="field.value + '-group'"
            v-if="inputStyles[field.value] === 'dropdown'">
            <label class="btn btn-outline-secondary" :id="field.value + '-label'"
                :for="field.value + '-select'">
                {{ field.label }}
            </label>
            <select class="form-select" :id="field.value + '-select'"
                v-model="selectedValues[field.value]"
                :aria-labelledby="field.value + '-label'">
                <option v-for="option in choiceValues[field.value]"
                    :key="option.value" :value="option.value">
                    {{ option.text }}
                </option>
            </select>
        </div>

        <!-- Date / integer range input -->
        <div class="input-group pb-2" :id="field.value + '-group'"
            v-if="['date', 'int'].includes(inputStyles[field.value])">
            <label class="btn btn-outline-secondary" :id="field.value + '-date-label'"
                style="border-top-right-radius: 0; border-bottom-right-radius: 0">
                {{ field.label }}
            </label>
            <div class="btn-group" role="group">
                <button class="btn btn-secondary dropdown-toggle"
                    style="border-top-left-radius: 0; border-bottom-left-radius: 0"
                    type="button" :id="field.value + '-selector'"
                    data-bs-toggle="dropdown" aria-expanded="false"
                    :aria-label="$t(`searchForm.${dateType[field.value]}DateLabel`, { field: field.label })">
                    {{ $t(`searchForm.${dateType[field.value]}Date`) }}
                </button>
                <ul class="dropdown-menu" :aria-labelledby="field.value + '-selector'">
                    <li>
                        <button type="button" class="dropdown-item"
                            @click="dateTypeToggle(field.value, 'exact')">
                            {{ $t("searchForm.exactDate") }}
                        </button>
                    </li>
                    <li>
                        <button type="button" class="dropdown-item"
                            @click="dateTypeToggle(field.value, 'range')">
                            {{ $t("searchForm.rangeDate") }}
                        </button>
                    </li>
                </ul>
            </div>
            <input type="text" class="form-control" :id="field.value + '-input-filter'"
                :name="field.value" :placeholder="field.example"
                v-model="modelValue[field.value]"
                :aria-labelledby="field.value + '-date-label'"
                v-if="dateType[field.value] === 'exact'" />
            <span class="d-inline-block" v-if="dateType[field.value] === 'range'">
                <div class="input-group ms-3">
                    <label class="btn btn-outline-secondary"
                        :id="field.value + '-from-label'"
                        :for="field.value + '-start-input-filter'">
                        {{ $t("searchForm.dateFrom") }}
                    </label>
                    <input type="text" class="form-control date-range"
                        :id="field.value + '-start-input-filter'"
                        :name="field.value + '-start'" :placeholder="field.example"
                        v-model="dateRange[field.value].start"
                        :aria-labelledby="`${field.value}-date-label ${field.value}-from-label`" />
                    <label class="btn btn-outline-secondary ms-3"
                        :id="field.value + '-to-label'"
                        :for="field.value + '-end-input-filter'">
                        {{ $t("searchForm.dateTo") }}
                    </label>
                    <input type="text" class="form-control date-range"
                        :id="field.value + '-end-input-filter'"
                        :name="field.value + '-end'" :placeholder="field.example"
                        v-model="dateRange[field.value].end"
                        :aria-labelledby="`${field.value}-date-label ${field.value}-to-label`" />
                </div>
            </span>
        </div>
    </div>
</template>

<script setup>
const props = defineProps({
    fields: {
        type: Array,
        required: true,
    },
    inputStyles: {
        type: Object,
        required: true,
    },
    choiceValues: {
        type: Object,
        default: () => ({}),
    },
    modelValue: {
        type: Object,
        required: true,
    },
    checkedValues: {
        type: Object,
        default: () => ({}),
    },
    selectedValues: {
        type: Object,
        default: () => ({}),
    },
    dateType: {
        type: Object,
        required: true,
    },
    dateRange: {
        type: Object,
        required: true,
    },
});

function dateTypeToggle(fieldName, newDateType) {
    props.dateRange[fieldName] = { start: "", end: "" };
    props.modelValue[fieldName] = "";
    props.dateType[fieldName] = newDateType;
}
</script>
