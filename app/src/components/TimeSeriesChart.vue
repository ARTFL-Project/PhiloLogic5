<template>
    <!-- room on the right for half a label, the last bar's -->
    <div class="ts-chart" :style="{ paddingRight: `${labelWidth / 2}px` }">
        <div class="ts-y-title" aria-hidden="true">{{ yTitle }}</div>
        <div class="ts-y-ticks" aria-hidden="true">
            <span v-for="tick in yTicks" :key="tick" class="ts-y-tick" :style="{ bottom: `${percent(tick)}%` }">
                {{ format(tick) }}
            </span>
            <!-- the longest, for the column's width -->
            <span class="ts-y-probe">{{ longestTick }}</span>
        </div>

        <div class="ts-plot" ref="plot">
            <div v-for="tick in yTicks" :key="tick" class="ts-grid" :style="{ bottom: `${percent(tick)}%` }"
                aria-hidden="true"></div>
            <!-- A button per period, as tall as the plot so as to be easy to point at, its bar inside: one tab stop,
                 the arrow keys moving between them -->
            <div class="ts-columns" role="group" :aria-label="label" :aria-describedby="describedby"
                :style="{ '--bar-width': barWidth }">
                <button v-for="(value, index) in values" :key="index" type="button" class="ts-column"
                    :class="{ active: index === active }" :tabindex="index === current ? 0 : -1"
                    :aria-label="`${ranges[index]}: ${format(value)} ${unit}`" :ref="(el) => (columns[index] = el)"
                    @click="emit('select', index)" @focus="onFocus(index)" @blur="onBlur(index)"
                    @keydown="onKeydown(index, $event)" @pointerenter="onPointerEnter(index)"
                    @pointerleave="onPointerLeave">
                    <span class="ts-bar" :style="{ height: `${percent(value)}%` }"></span>
                </button>
            </div>

            <!-- What the button says, on hover and focus: it stays while pointed at, and Escape dismisses it -->
            <div v-if="tip !== null" class="ts-tooltip" ref="tooltip" aria-hidden="true" :style="tipStyle"
                @pointerenter="keepTip" @pointerleave="onPointerLeave" @click="emit('select', tip)">
                <div class="ts-tooltip-title">{{ ranges[tip] }}</div>
                <div class="ts-tooltip-value">{{ format(values[tip]) }} {{ unit }}</div>
            </div>
        </div>

        <div class="ts-x-ticks" aria-hidden="true">
            <span v-for="index in xTicks" :key="index" class="ts-x-tick"
                :style="{ left: `${((index + 0.5) / labels.length) * 100}%` }">{{ labels[index] }}</span>
            <!-- the longest label, measured for how many fit side by side -->
            <span class="ts-x-tick ts-probe" ref="probe">{{ longestLabel }}</span>
        </div>
        <div class="ts-x-title" aria-hidden="true">{{ xTitle }}</div>
        <!-- the keys, for whoever came by the keyboard (screen readers have the description), in a line kept for it -->
        <div v-if="hint" class="ts-hint" :class="{ shown: focused !== null }" aria-hidden="true">{{ hint }}</div>
    </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, useTemplateRef, watch } from "vue";
import { useI18n } from "vue-i18n";

// Bars of a value by period, the periods' labels under them (as many as fit), their names (ranges) and values in
// the bars' buttons and tooltips. Choosing a bar emits its index.
const props = defineProps({
    values: { type: Array, required: true },
    labels: { type: Array, required: true }, // the periods' starts
    ranges: { type: Array, required: true }, // the periods' names
    unit: { type: String, required: true },
    xTitle: { type: String, required: true },
    yTitle: { type: String, required: true },
    label: { type: String, required: true }, // the chart's accessible name
    describedby: { type: String, default: null },
    hint: { type: String, default: null }, // shown while a period has the focus
});
const emit = defineEmits(["select"]);
const { locale } = useI18n();

const plot = useTemplateRef("plot");
const probe = useTemplateRef("probe");
const tooltip = useTemplateRef("tooltip");
const columns = [];
const measured = ref(0); // the plot's width, in px
const height = ref(0);
const width = computed(() => measured.value || 800); // before it's laid out
const labelWidth = ref(0); // the longest label's
const tipWidth = ref(0);

const current = ref(0); // the period of the chart's tab stop
const focused = ref(null);
const hovered = ref(null);
const dismissed = ref(false); // the tooltip, until another period is pointed at or focused
const active = computed(() => hovered.value ?? focused.value);
const tip = computed(() => (dismissed.value ? null : active.value));

function format(value) {
    return Number(value).toLocaleString(locale.value, { maximumFractionDigits: 2 });
}

// ---- Scales ----
// Round steps (1, 2 or 5 times a power of ten) from 0, about one per 50 px of height
const yTicks = computed(() => {
    const max = Math.max(0, ...props.values);
    const count = Math.max(2, Math.floor((height.value || 400) / 50));
    const raw = max > 0 ? max / count : 1;
    const power = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 5, 10].map((m) => m * power).find((s) => s >= raw);
    const top = Math.max(step, Math.ceil(max / step) * step);
    const ticks = [];
    for (let i = 0; i * step <= top * (1 + 1e-9); i++) ticks.push(Number((i * step).toPrecision(12)));
    return ticks;
});

const longestTick = computed(() => yTicks.value.map(format).reduce((a, b) => (b.length > a.length ? b : a)));

function percent(value) {
    const top = yTicks.value[yTicks.value.length - 1];
    return (Math.max(0, value) / top) * 100;
}

// The bars as Chart.js drew them, 72% of their column, or touching when the gaps would be under a pixel
const barWidth = computed(() => {
    const gap = (0.28 * width.value) / props.values.length;
    return gap * (window.devicePixelRatio || 1) < 1 ? "100%" : "72%";
});

// Every nth label, n as small as leaves an em between them, each centered on its bar
const longestLabel = computed(() => props.labels.reduce((a, b) => (`${b}`.length > `${a}`.length ? b : a), ""));
const xTicks = computed(() => {
    const size = (labelWidth.value || `${longestLabel.value}`.length * 8) + 12;
    const every = Math.max(1, Math.ceil((size * props.labels.length) / width.value));
    return props.labels.map((_, i) => i).filter((i) => i % every === 0);
});

// Above its bar, its arrow on the bar: in the plot, or past its end by as little as that takes
const ARROW = 12; // from the tooltip's side to its arrow's middle, off the rounded corners
const tipStyle = computed(() => {
    const center = ((tip.value + 0.5) / props.values.length) * width.value;
    const half = tipWidth.value / 2;
    const left = Math.max(half - ARROW, Math.min(width.value + ARROW - half, center));
    return {
        left: `${left}px`,
        bottom: `${percent(props.values[tip.value])}%`,
        "--arrow-shift": `${center - left}px`,
    };
});
watch(
    () => tip.value !== null && [props.ranges[tip.value], props.values[tip.value], props.unit],
    () => (tipWidth.value = tooltip.value?.offsetWidth ?? 0),
    { flush: "post" }
);

// ---- Pointer and keyboard ----
let hideTimer = null;
function keepTip() {
    clearTimeout(hideTimer);
}
function onPointerEnter(index) {
    keepTip();
    hovered.value = index;
    dismissed.value = false;
}
// With time to move onto the tooltip
function onPointerLeave() {
    clearTimeout(hideTimer);
    hideTimer = setTimeout(() => (hovered.value = null), 250);
}
function onFocus(index) {
    current.value = index;
    focused.value = index;
    dismissed.value = false;
}
function onBlur(index) {
    if (focused.value === index) focused.value = null;
}
function onKeydown(index, event) {
    const last = props.values.length - 1;
    const next = { ArrowRight: index + 1, ArrowLeft: index - 1, Home: 0, End: last }[event.key];
    if (next !== undefined) {
        event.preventDefault();
        columns[Math.min(last, Math.max(0, next))]?.focus();
    }
}
// Escape dismisses the tooltip, whether pointed at or focused, wherever the focus is
function onDocumentKeydown(event) {
    if (event.key === "Escape" && tip.value !== null) {
        dismissed.value = true;
        event.preventDefault();
    }
}

// Another series: the tab stop back on its first period
watch(
    () => props.values.length,
    () => {
        current.value = 0;
        hovered.value = null;
        focused.value = null;
    }
);

// ---- Sizes ----
let observer = null;
function measure() {
    measured.value = plot.value.clientWidth;
    height.value = plot.value.clientHeight;
    labelWidth.value = probe.value.getBoundingClientRect().width;
}
onMounted(() => {
    measure();
    if (typeof ResizeObserver !== "undefined") {
        observer = new ResizeObserver(measure);
        observer.observe(plot.value);
        observer.observe(probe.value); // the label's own size: other labels, zoom, text spacing
    }
    document.addEventListener("keydown", onDocumentKeydown);
});
onBeforeUnmount(() => {
    observer?.disconnect();
    clearTimeout(hideTimer);
    document.removeEventListener("keydown", onDocumentKeydown);
});
</script>

<style scoped lang="scss">
@use "../assets/styles/theme.module.scss" as theme;

.ts-chart {
    display: grid;
    grid-template-columns: auto auto minmax(0, 1fr);
    grid-template-rows: 400px auto auto;
    font-size: 12px;
    color: #595959;
}

.ts-y-title {
    grid-row: 1;
    grid-column: 1;
    writing-mode: vertical-rl;
    transform: rotate(180deg);
    text-align: center;
    padding-left: 0.5em;
}

.ts-y-ticks {
    grid-row: 1;
    grid-column: 2;
    position: relative;
    margin-right: 0.5em;
}

.ts-y-probe {
    display: block;
    height: 0;
    overflow: hidden;
    visibility: hidden;
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
}

.ts-y-tick {
    position: absolute;
    right: 0;
    transform: translateY(50%);
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
}

.ts-plot {
    grid-row: 1;
    grid-column: 3;
    position: relative;
    border-bottom: 1px solid #949494;
}

/* Recessive: the values are on the axis, in the tooltips, and in the table */
.ts-grid {
    position: absolute;
    left: 0;
    right: 0;
    border-top: 1px solid #e2e2e2;
}

.ts-columns {
    position: absolute;
    inset: 0;
    display: flex;
}

.ts-column {
    flex: 1 1 0;
    min-width: 0;
    height: 100%;
    display: flex;
    align-items: flex-end;
    justify-content: center;
    padding: 0;
    border: 0;
    background: transparent;
    cursor: pointer;
}

/* Over its neighbors, for the app's focus ring */
.ts-column:focus-visible {
    position: relative;
    z-index: 1;
}

/* Thin, rounded at its end, square at the baseline */
.ts-bar {
    width: var(--bar-width);
    max-width: 24px;
    background-color: theme.$link-color;
    border-radius: 4px 4px 0 0;
}

.ts-column.active .ts-bar {
    background-color: rgba(theme.$link-color, 0.7);
}

.ts-x-ticks {
    grid-row: 2;
    grid-column: 3;
    position: relative;
    height: 2em;
}

.ts-x-tick {
    position: absolute;
    top: 0.4em;
    transform: translateX(-50%);
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
}

.ts-probe {
    visibility: hidden;
    left: 0;
    transform: none;
}

.ts-x-title {
    grid-row: 3;
    grid-column: 3;
    text-align: center;
}

.ts-hint {
    grid-row: 4;
    grid-column: 1 / -1;
    margin-top: 0.5em;
    text-align: center;
    visibility: hidden;
}

.ts-hint.shown {
    visibility: visible;
}

.ts-tooltip {
    position: absolute;
    z-index: 2;
    margin-bottom: 10px;
    transform: translateX(-50%);
    background: theme.$button-color;
    color: #fff;
    border-radius: 4px;
    padding: 0.2rem 0.5rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3), 0 2px 4px rgba(0, 0, 0, 0.2);
    font-size: 14px;
    line-height: 1.5;
    white-space: nowrap;
    cursor: pointer;
}

.ts-tooltip::after {
    content: "";
    position: absolute;
    top: 100%;
    left: calc(50% + var(--arrow-shift, 0px));
    transform: translateX(-50%);
    border: 8px solid transparent;
    border-top-color: theme.$button-color;
}

.ts-tooltip-title {
    font-weight: bold;
}

.ts-tooltip-value {
    font-size: 13px;
}

@media (max-width: 768px) {
    .ts-chart {
        grid-template-rows: 230px auto auto;
    }
}

/* In forced colors, the bars in the text's color, the one pointed at or focused in the highlight's */
@media (forced-colors: active) {
    .ts-bar {
        forced-color-adjust: none;
        background-color: CanvasText;
    }

    .ts-column.active .ts-bar {
        background-color: Highlight;
    }

    .ts-grid {
        border-top-color: GrayText;
    }

    .ts-tooltip {
        border: 1px solid CanvasText;
    }

    .ts-tooltip::after {
        display: none;
    }
}
</style>
