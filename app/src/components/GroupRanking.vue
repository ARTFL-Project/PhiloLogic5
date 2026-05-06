<template>
    <div :class="embedded ? '' : 'card'" v-if="items.length > 0" role="region"
        :aria-label="regionAriaLabel || title">
        <h2 v-if="hiddenHeader" class="visually-hidden">{{ hiddenHeader }}</h2>
        <h3 v-if="!embedded && title" class="group-ranking-header">{{ title }}</h3>
        <ul class="list-group list-group-flush" :aria-label="title || regionAriaLabel">
            <li v-for="(item, index) in items" :key="item[0]">
                <button type="button"
                    class="list-group-item position-relative w-100 text-start border-0 pb-1"
                    style="text-align: justify"
                    @click="$emit('select', item[0])"
                    @keydown.enter="$emit('select', item[0])"
                    @keydown.space.prevent="$emit('select', item[0])"
                    :aria-label="buildAriaLabel(item)">
                    <span class="group-name">{{ item[0] }}</span>
                    <span class="badge text-bg-secondary position-absolute"
                        style="right: 1rem; top: 0.5rem" aria-hidden="true">
                        {{ formatScore(item[1]) }}
                    </span>
                    <br v-if="item[2] && item[2].length > 0">
                    <small v-if="item[2] && item[2].length > 0" class="explainer-line"
                        aria-hidden="true">
                        <strong>{{ explainerLabel }}:</strong> {{ item[2].join(', ') }}
                    </small>
                </button>
                <hr v-if="index < items.length - 1" class="my-0 group-ranking-divider"
                    aria-hidden="true">
            </li>
        </ul>
    </div>
</template>

<script setup>
const props = defineProps({
    title: { type: String, default: "" },
    items: { type: Array, required: true },
    explainerLabel: { type: String, default: "" },
    selectAriaLabelPrefix: { type: String, default: "" },
    scoreAriaLabel: { type: String, default: "score" },
    regionAriaLabel: { type: String, default: "" },
    hiddenHeader: { type: String, default: "" },
    formatScore: { type: Function, default: (n) => n },
    embedded: { type: Boolean, default: false },
});

defineEmits(["select"]);

function buildAriaLabel(item) {
    const [name, score, explainers] = item;
    const parts = [];
    parts.push(props.selectAriaLabelPrefix ? `${props.selectAriaLabelPrefix} ${name}` : name);
    parts.push(`${props.scoreAriaLabel}: ${props.formatScore(score)}`);
    if (explainers && explainers.length > 0 && props.explainerLabel) {
        parts.push(`${props.explainerLabel}: ${explainers.join(', ')}`);
    }
    return parts.join(', ');
}
</script>

<style lang="scss" scoped>
@use "../assets/styles/theme.module.scss" as theme;

.group-ranking-header {
    text-align: center;
    font-variant: small-caps;
    font-size: 1rem;
    color: #fff;
    background-color: theme.$link-color;
    padding: 0.5rem;
    margin-bottom: 0;
}

.group-name {
    display: inline-block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: calc(100% - 4rem);
    vertical-align: bottom;
    padding-bottom: 0.15rem;
    color: theme.$link-color;
    font-weight: 500;
}

.explainer-line {
    font-size: 0.9em;
    color: #495057;
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: calc(100% - 4rem);
}

.group-ranking-divider {
    opacity: 1;
    border-color: rgba(0, 0, 0, 0.125);
}

.list-group-item {
    cursor: pointer;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    transform: scale(1);
}

.list-group-item:hover {
    transform: scale(1.01);
    background-color: rgba(theme.$link-color, 0.15) !important;
    border-color: rgba(theme.$link-color, 0.3) !important;
    box-shadow: inset 0 0 8px rgba(theme.$link-color, 0.1);
    z-index: 1;
}

.list-group-item:active {
    transform: scale(0.98);
    background-color: rgba(theme.$link-color, 0.2) !important;
}

.list-group-item:hover .badge {
    background-color: theme.$link-color !important;
    transform: scale(1.1);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

.list-group-item:focus {
    outline: 2px solid theme.$link-color;
    outline-offset: -2px;
    box-shadow: inset 0 0 0 0.2rem rgba(theme.$link-color, 0.25);
    z-index: 2;
}

.badge {
    font-size: 0.75rem;
}
</style>
