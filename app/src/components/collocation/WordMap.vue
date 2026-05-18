<template>
    <div>
        <!-- Detection auto-runs on mount; the min-words knob below triggers
             re-runs. No standalone Detect button is needed — its only role
             was redundant re-runs. -->

        <div v-if="loading && !result" class="text-center py-5">
            <progress-spinner :lg="true" :message="$t('threads.detecting')" />
        </div>

        <div v-if="result && result.threads && result.threads.length > 0" class="mx-2 my-3">
            <!-- Everything (count, explanation, controls, viz) lives inside the
                 result card so the layout reads as one self-contained unit. -->
            <div v-if="result.graph" class="card shadow-sm p-3 mb-3 overview-card">
                <p class="text-muted mb-1">
                    {{ $t('threads.summary', { threads: result.threads.length }) }}
                    <span class="info-tip" tabindex="0"
                        :data-tip="$t('threads.networkSubtitle')"
                        :aria-label="$t('threads.networkSubtitle')">ⓘ</span>
                </p>
                <p class="small text-muted mb-3">{{ $t('threads.networkHint') }}</p>

                <div class="overview-body d-flex flex-wrap gap-3 align-items-start">
                    <aside class="legend-aside">
                        <!-- Legend chips first (they describe what each thread is),
                             then the controls grouped together below. -->
                        <div class="legend-list">
                            <button v-for="thread in result.threads" :key="`leg-${thread.id}`" type="button"
                                class="btn btn-sm legend-chip"
                                :style="{ borderColor: threadColor(thread.id - 1, 1), color: threadColor(thread.id - 1, 1) }"
                                @click="onViewPassages(thread)"
                                :title="thread.words.slice(0, 10).map((w) => w.word).join(', ')">
                                <span class="legend-swatch" :style="{ backgroundColor: threadColor(thread.id - 1, 1) }"></span>
                                <span class="legend-text">T{{ thread.id }}: {{ thread.label }}</span>
                            </button>
                        </div>
                        <div class="control-stack mt-3">
                            <div class="mb-3">
                                <label class="small text-muted d-block mb-1" for="min-words-net">
                                    {{ $t('threads.minWords') }}
                                </label>
                                <select id="min-words-net" v-model="minWordsPerThread"
                                    @change="onGrainChange" class="form-select form-select-sm">
                                    <option value="auto">{{ $t('threads.minWordsAuto') }}{{ minWordsPerThread === 'auto' && result?.min_words_resolved ? ` (${result.min_words_resolved})` : '' }}</option>
                                    <option v-for="n in [4, 5, 6, 8, 10, 12, 15]" :key="n" :value="n">{{ n }}</option>
                                </select>
                                <p class="control-hint">{{ $t('threads.minWordsHint') }}</p>
                            </div>
                            <div v-if="result.graph.nodes.length > result.graph.n_members" class="words-slider">
                                <label class="small text-muted d-block mb-1" for="net-word-count">
                                    {{ $t('threads.wordsShown') }}: <strong>{{ networkWordCount }}</strong>
                                    <span class="text-muted">/ {{ result.graph.nodes.length }}</span>
                                </label>
                                <input type="range" id="net-word-count" class="form-range form-range-sm w-100"
                                    :min="result.graph.n_members" :max="result.graph.nodes.length" step="1"
                                    v-model.number="networkWordCount" />
                                <p class="control-hint">{{ $t('threads.wordsShownHint') }}</p>
                            </div>
                        </div>
                    </aside>

                    <div class="viz-area">
                        <div class="thread-network">
                            <div class="zoom-controls">
                                <button type="button" @click="zoomIn" :disabled="!canZoomIn"
                                    :title="$t('threads.zoomIn')" :aria-label="$t('threads.zoomIn')">+</button>
                                <button type="button" @click="zoomOut" :disabled="!canZoomOut"
                                    :title="$t('threads.zoomOut')" :aria-label="$t('threads.zoomOut')">−</button>
                                <button type="button" @click="resetView" :disabled="!isManualView"
                                    :title="$t('threads.zoomReset')" :aria-label="$t('threads.zoomReset')">
                                    <i class="bi bi-arrows-fullscreen" aria-hidden="true"></i>
                                </button>
                            </div>
                            <canvas ref="canvasRef" class="network-canvas"
                                @mousedown="onMouseDown" @mousemove="onMouseMove"
                                @mouseleave="onMouseLeave"></canvas>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Backend didn't ship graph data (older server) — friendly fallback. -->
            <div v-else class="text-center py-4 text-muted">
                {{ $t('threads.noGraph') }}
            </div>
        </div>

        <div v-else-if="result && (!result.threads || result.threads.length === 0)"
            class="text-center py-5 text-muted">
            {{ $t('threads.noResults') }}
        </div>

        <DistinctivePassagesModal
            :group-name="modal.groupName" :signature="modal.signature"
            :passages="modal.passages" :loading="modal.loading"
            :has-more="modal.hasMore" :view-all-url="modal.viewAllUrl"
            @load-more="loadMorePassages" @view-all="onViewAllPassages" />
    </div>
</template>

<script setup>
import { computed, inject, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import { useRouter } from "vue-router";
import { Modal } from "bootstrap";
import { useMainStore } from "../../stores/main";
import { concordanceMethod, debug, paramsFilter, paramsToRoute } from "../../utils.js";
import DistinctivePassagesModal from "../DistinctivePassagesModal.vue";
import ProgressSpinner from "../ProgressSpinner";

const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const router = useRouter();
const store = useMainStore();
const { formData } = storeToRefs(store);

// ---- Data fetch state ----
const loading = ref(false);
const result = ref(null);
const networkWordCount = ref(0);
const minWordsPerThread = ref("auto");
let fetchToken = 0;

// Same color palette as the streamgraph so thread colors are consistent across tabs.
const threadHues = [205, 30, 145, 280, 0, 90, 165, 235, 50, 315, 120, 260, 15, 60, 200, 320];
function threadColor(i, alpha) {
    const h = threadHues[i % threadHues.length];
    return `hsla(${h}, 55%, 50%, ${alpha})`;
}

function runDetection(opts = {}) {
    const myToken = ++fetchToken;
    loading.value = true;
    if (!opts.keepResult) result.value = null;
    const params = paramsFilter(formData.value);
    if (minWordsPerThread.value !== "auto") {
        params.min_words_per_thread = minWordsPerThread.value;
    }
    $http.get(`${$dbUrl}/scripts/get_threads.py`, { params }).then((resp) => {
        if (myToken !== fetchToken) return;
        result.value = resp.data;
        networkWordCount.value = resp.data?.graph?.n_members || 0;
    }).catch((error) => {
        debug({ $options: { name: "word-map" } }, error);
    }).finally(() => {
        if (myToken === fetchToken) loading.value = false;
    });
}

function onGrainChange() {
    runDetection({ keepResult: true });
}

function reset() {
    result.value = null;
    fetchToken++;
}

// ---- Passages modal ----
const modal = ref({
    groupName: "", yearRange: "", signature: [], passages: [],
    loading: false, hasMore: false, offset: 0, pageSize: 25, viewAllUrl: "",
});
let modalInstance = null;
let passagesFetchToken = 0;

function onViewPassages(thread) {
    // Use the full result year range; the thread's signature tokens do the
    // semantic filtering, so a broader window surfaces more relevant hits
    // than a tight peak-centric slice would.
    const [yMin, yMax] = result.value.year_range;
    const yearRange = `${yMin}-${yMax}`;
    modal.value = {
        groupName: `${formData.value.q} · T${thread.id}: ${thread.label}`,
        yearRange,
        signature: thread.words.slice(0, 20).map((w) => ({ word: w.word, z: null })),
        passages: [],
        loading: true,
        hasMore: false,
        offset: 0,
        pageSize: 25,
        viewAllUrl: paramsToRoute({
            ...formData.value,
            report: "concordance",
            method: concordanceMethod(formData.value.colloc_within),
            year: yearRange,
        }),
    };
    showModal();
    fetchPassages({ append: false });
}

function showModal() {
    const el = document.getElementById("distinctive-passages-modal");
    if (!el) return;
    if (!modalInstance) modalInstance = new Modal(el);
    modalInstance.show();
}

async function fetchPassages({ append }) {
    const myToken = ++passagesFetchToken;
    modal.value.loading = true;
    try {
        const params = {
            ...paramsFilter(formData.value),
            restrict_to_field: "year",
            restrict_to_value: modal.value.yearRange,
            signature_tokens: modal.value.signature.map((s) => s.word).join(","),
            top_n: modal.value.pageSize,
            offset: modal.value.offset,
        };
        const resp = await $http.get(`${$dbUrl}/scripts/get_representative_passages.py`, { params });
        if (myToken !== passagesFetchToken) return;
        const data = resp.data || {};
        modal.value.passages = append
            ? modal.value.passages.concat(data.passages || [])
            : (data.passages || []);
        modal.value.hasMore = !!data.has_more;
    } catch (error) {
        debug({ $options: { name: "word-map" } }, error);
    } finally {
        if (myToken === passagesFetchToken) modal.value.loading = false;
    }
}

function loadMorePassages() {
    modal.value.offset += modal.value.pageSize;
    fetchPassages({ append: true });
}

function onViewAllPassages() {
    const dest = modal.value.viewAllUrl;
    if (modalInstance) modalInstance.hide();
    if (dest) router.push(dest);
}

defineExpose({ runDetection, reset });

// =====================================================================
// Canvas force-layout visualization (formerly ThreadNetwork.vue).
// Operates on result.value.graph + result.value.threads. Non-reactive
// internal state (nodes, edges, camera) so the per-frame draw is a
// straight imperative loop with no Vue diffing for 500+ elements.
// =====================================================================

// Logical layout space (canvas scales to fit, single scale factor).
const W = 1000;
const H = 560;
const labelThreshold = 7;

const canvasRef = ref(null);
let ctx = null;
let dpr = 1;
let cssW = 0;
let cssH = 0;
// Camera: an auto-fit transform that maps node coords (W×H logical) into
// canvas px. Lerped each tick toward the bounding box of the current node
// set so the graph always fills the canvas instead of sitting in a blob.
let viewScale = 1;
let viewOffX = 0;
let viewOffY = 0;

// User zoom/pan: when active, auto-fit is suspended and the user controls
// the camera via the +/-/reset buttons, click-drag pan, and mouse wheel.
// userZoom is a continuous multiplier on top of auto-fit (1.0 = baseline);
// buttons snap to ZOOM_STEPS, the wheel slides continuously between them.
const ZOOM_STEPS = [1, 1.5, 2, 3, 4];
const MIN_ZOOM = ZOOM_STEPS[0];
const MAX_ZOOM = ZOOM_STEPS[ZOOM_STEPS.length - 1];
const userZoom = ref(1);
const isManualView = ref(false);
const canZoomIn = computed(() => userZoom.value < MAX_ZOOM - 1e-3);
const canZoomOut = computed(() => userZoom.value > MIN_ZOOM + 1e-3 || isManualView.value);

let nodes = [];
let edges = [];
let clusterAnchors = {};   // cluster id -> { x, y } target position
let hovered = -1;
let raf = null;
let resizeObs = null;
let setupForCanvas = null;  // which canvas element we've wired listeners to

// Pan/click disambiguation state.
let isPanning = false;
let panActive = false;
let panStartX = 0, panStartY = 0;
let panOffStartX = 0, panOffStartY = 0;

// Animated zoom: an rAF loop lerps viewScale/viewOff toward the target
// over ~320ms. Animations compose — clicking + mid-animation retargets
// from the current visual position.
let zoomAnim = null;
let zoomRaf = null;

function showLabel(n) {
    return n.member ? (n.active || n.r >= labelThreshold) : n.active;
}

function threadById(id) {
    const ts = result.value?.threads || [];
    for (const t of ts) if (t.id === id) return t;
    return null;
}

// ---- Canvas setup (DPR-aware, viewport-capped) ----
function setupCanvas() {
    const c = canvasRef.value;
    if (!c) return;
    dpr = window.devicePixelRatio || 1;
    const parent = c.parentElement;
    const containerW = parent ? parent.getBoundingClientRect().width : 800;
    // Cap height so the graph doesn't push the legend off-screen on tall
    // displays. The viz wants room to breathe so dense clusters don't pile
    // into a ball — 60vh gives the simulation more vertical canvas to use.
    const maxH = Math.max(320, window.innerHeight * 0.60);
    const naturalH = (containerW * H) / W;
    if (naturalH <= maxH) {
        cssW = containerW;
        cssH = naturalH;
    } else {
        cssH = maxH;
        cssW = (maxH * W) / H;
    }
    c.width = Math.max(1, Math.round(cssW * dpr));
    c.height = Math.max(1, Math.round(cssH * dpr));
    c.style.width = cssW + "px";
    c.style.height = cssH + "px";
    ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    viewScale = cssW / W;
    viewOffX = 0;
    viewOffY = 0;
    if (nodes.length > 0) applyFit(false);
}

// ---- Auto-fit camera ----
function computeFitTarget() {
    if (nodes.length === 0 || cssW === 0) return null;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        if (n.x - n.r < minX) minX = n.x - n.r;
        if (n.y - n.r < minY) minY = n.y - n.r;
        if (n.x + n.r > maxX) maxX = n.x + n.r;
        if (n.y + n.r > maxY) maxY = n.y + n.r;
    }
    // Logical padding — extra room on top for labels above the node circles.
    minX -= 16; maxX += 16; minY -= 22; maxY += 16;
    const dx = maxX - minX;
    const dy = maxY - minY;
    if (dx <= 0 || dy <= 0) return null;
    const s = Math.min(cssW / dx, cssH / dy);
    return {
        scale: s,
        offX: (cssW - dx * s) / 2 - minX * s,
        offY: (cssH - dy * s) / 2 - minY * s,
    };
}

function applyFit(damped = true) {
    if (isManualView.value) return;
    const t = computeFitTarget();
    if (!t) return;
    if (damped) {
        const k = 0.08;
        viewScale = viewScale * (1 - k) + t.scale * k;
        viewOffX = viewOffX * (1 - k) + t.offX * k;
        viewOffY = viewOffY * (1 - k) + t.offY * k;
    } else {
        viewScale = t.scale;
        viewOffX = t.offX;
        viewOffY = t.offY;
    }
}

// ---- Hit testing ----
function nearestNode(mx, my) {
    let best = -1;
    let bestD = Infinity;
    for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        const dx = mx - (n.x * viewScale + viewOffX);
        const dy = my - (n.y * viewScale + viewOffY);
        const r = n.r * viewScale + 2;
        const d2 = dx * dx + dy * dy;
        if (d2 <= r * r && d2 < bestD) { bestD = d2; best = i; }
    }
    return best;
}

function setActive(idx) {
    for (const n of nodes) n.active = false;
    if (idx < 0) return;
    nodes[idx].active = true;
    for (const e of edges) {
        if (e.source === idx) nodes[e.target].active = true;
        else if (e.target === idx) nodes[e.source].active = true;
    }
}

// ---- Mouse/pan handlers ----
function onMouseDown(ev) {
    if (ev.button !== 0) return;
    panActive = true;
    isPanning = false;
    panStartX = ev.clientX;
    panStartY = ev.clientY;
    panOffStartX = viewOffX;
    panOffStartY = viewOffY;
    window.addEventListener("mousemove", onWindowMouseMove);
    window.addEventListener("mouseup", onWindowMouseUp);
}

// Listen on window during drag so the pan continues if the cursor leaves
// the canvas (and we don't lose the mouseup if it happens elsewhere).
function onWindowMouseMove(ev) {
    if (!panActive) return;
    const dx = ev.clientX - panStartX;
    const dy = ev.clientY - panStartY;
    if (!isPanning && (dx * dx + dy * dy) > 16) {
        isPanning = true;
        isManualView.value = true;
        const c = canvasRef.value;
        if (c) c.style.cursor = "grabbing";
    }
    if (isPanning) {
        viewOffX = panOffStartX + dx;
        viewOffY = panOffStartY + dy;
        if (!raf) draw();
    }
}

function onWindowMouseUp() {
    window.removeEventListener("mousemove", onWindowMouseMove);
    window.removeEventListener("mouseup", onWindowMouseUp);
    if (!panActive) return;
    panActive = false;
    const c = canvasRef.value;
    if (isPanning) {
        isPanning = false;
        if (c) c.style.cursor = hovered >= 0 ? "pointer" : "default";
        return;  // suppress click after drag
    }
    if (hovered >= 0) {
        const t = threadById(nodes[hovered].cluster);
        if (t) onViewPassages(t);
    }
}

function onMouseMove(ev) {
    if (isPanning) return;
    const c = canvasRef.value;
    if (!c) return;
    const rect = c.getBoundingClientRect();
    const idx = nearestNode(ev.clientX - rect.left, ev.clientY - rect.top);
    if (idx !== hovered) {
        hovered = idx;
        setActive(idx);
        c.style.cursor = idx >= 0 ? "pointer" : "default";
        if (!raf) draw();
    }
}

function onMouseLeave() {
    if (isPanning) return;
    if (hovered !== -1) {
        hovered = -1;
        setActive(-1);
        const c = canvasRef.value; if (c) c.style.cursor = "default";
        if (!raf) draw();
    }
}

// ---- Zoom controls (buttons step, wheel slides, both animate where it helps) ----
const ZOOM_ANIM_MS = 320;

// Quintic ease-in-out — smoother ramps at the ends than cubic.
function easeInOutQuintic(t) {
    return t < 0.5 ? 16 * t * t * t * t * t : 1 - Math.pow(-2 * t + 2, 5) / 2;
}

function animateView(toScale, toOffX, toOffY, { afterReset = false, duration = ZOOM_ANIM_MS } = {}) {
    zoomAnim = {
        fromScale: viewScale,
        fromOffX: viewOffX,
        fromOffY: viewOffY,
        toScale, toOffX, toOffY,
        startedAt: performance.now(),
        duration,
        afterReset,
    };
    if (zoomRaf == null) zoomRaf = requestAnimationFrame(stepZoom);
}

function stepZoom(now) {
    zoomRaf = null;
    if (!zoomAnim) return;
    const t = Math.min(1, (now - zoomAnim.startedAt) / zoomAnim.duration);
    const e = easeInOutQuintic(t);
    viewScale = zoomAnim.fromScale + (zoomAnim.toScale - zoomAnim.fromScale) * e;
    viewOffX = zoomAnim.fromOffX + (zoomAnim.toOffX - zoomAnim.fromOffX) * e;
    viewOffY = zoomAnim.fromOffY + (zoomAnim.toOffY - zoomAnim.fromOffY) * e;
    if (!raf) draw();
    if (t < 1) {
        zoomRaf = requestAnimationFrame(stepZoom);
    } else {
        const wasReset = zoomAnim.afterReset;
        zoomAnim = null;
        // Release the manual-view freeze AFTER the reset animation lands,
        // so applyFit doesn't yank the view mid-animation.
        if (wasReset) isManualView.value = false;
    }
}

// Compute the (scale, offX, offY) you'd get by zooming `viewScale/viewOff*`
// by `factor` about the screen point (cx, cy). Anchor algebra:
//   world.x = (cx - viewOffX) / viewScale  (stays fixed)
//   new viewOffX = cx - world.x * (viewScale * factor)
function zoomTarget(cx, cy, factor) {
    return {
        scale: viewScale * factor,
        offX: cx * (1 - factor) + viewOffX * factor,
        offY: cy * (1 - factor) + viewOffY * factor,
    };
}

// Find the next zoom step strictly above the current zoom (epsilon avoids
// no-op clicks when the wheel left us right at a step boundary).
function nextStepAbove(z) {
    for (let i = 0; i < ZOOM_STEPS.length; i++) {
        if (ZOOM_STEPS[i] > z + 1e-3) return ZOOM_STEPS[i];
    }
    return MAX_ZOOM;
}
function nextStepBelow(z) {
    for (let i = ZOOM_STEPS.length - 1; i >= 0; i--) {
        if (ZOOM_STEPS[i] < z - 1e-3) return ZOOM_STEPS[i];
    }
    return MIN_ZOOM;
}

function applyUserZoom(newZoom, cx, cy, { animated = true } = {}) {
    newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, newZoom));
    const factor = newZoom / userZoom.value;
    userZoom.value = newZoom;
    isManualView.value = true;
    const t = zoomTarget(cx, cy, factor);
    if (animated) {
        animateView(t.scale, t.offX, t.offY);
    } else {
        // Cancel any in-flight animation — wheel is the authoritative input.
        if (zoomRaf != null) { cancelAnimationFrame(zoomRaf); zoomRaf = null; }
        zoomAnim = null;
        viewScale = t.scale;
        viewOffX = t.offX;
        viewOffY = t.offY;
        if (!raf) draw();
    }
}

function zoomIn() {
    if (!canZoomIn.value) return;
    applyUserZoom(nextStepAbove(userZoom.value), cssW / 2, cssH / 2);
}

function zoomOut() {
    if (!canZoomOut.value) return;
    const next = nextStepBelow(userZoom.value);
    if (next <= MIN_ZOOM + 1e-3) {
        resetView();
        return;
    }
    applyUserZoom(next, cssW / 2, cssH / 2);
}

function resetView() {
    userZoom.value = 1;
    const t = computeFitTarget();
    if (t) {
        animateView(t.scale, t.offX, t.offY, { afterReset: true });
    } else {
        isManualView.value = false;
    }
}

// Wheel: continuous zoom about the cursor. Skips animation — it would
// lag behind rapid scrolling. Exponential factor keeps perceived speed
// roughly constant across zoom levels.
function onWheel(ev) {
    ev.preventDefault();
    const c = canvasRef.value;
    if (!c) return;
    const rect = c.getBoundingClientRect();
    const cx = ev.clientX - rect.left;
    const cy = ev.clientY - rect.top;
    // Trackpad gives much smaller deltaY than a wheel-mouse "click"; the
    // 0.0015 factor calibrates so a single wheel notch (~100) ≈ 1.16×.
    const factor = Math.exp(-ev.deltaY * 0.0015);
    const newZoom = userZoom.value * factor;
    if (newZoom <= MIN_ZOOM + 1e-3 && userZoom.value <= MIN_ZOOM + 1e-3) return;
    if (newZoom <= MIN_ZOOM + 1e-3) {
        userZoom.value = MIN_ZOOM;
        resetView();
        return;
    }
    applyUserZoom(newZoom, cx, cy, { animated: false });
}

// ---- Render ----
function draw() {
    if (!ctx || cssW === 0) return;
    ctx.clearRect(0, 0, cssW, cssH);

    // edges
    const anyHover = hovered >= 0;
    ctx.lineWidth = 0.6;
    for (let i = 0; i < edges.length; i++) {
        const e = edges[i];
        const a = nodes[e.source], b = nodes[e.target];
        const active = anyHover && (e.source === hovered || e.target === hovered);
        ctx.strokeStyle = anyHover && !active
            ? "rgba(185,185,185,0.06)"
            : "rgba(185,185,185,0.5)";
        ctx.beginPath();
        ctx.moveTo(a.x * viewScale + viewOffX, a.y * viewScale + viewOffY);
        ctx.lineTo(b.x * viewScale + viewOffX, b.y * viewScale + viewOffY);
        ctx.stroke();
    }

    // nodes
    for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        const dim = anyHover && !n.active;
        const fa = (n.member ? 0.85 : 0.30) * (dim ? 0.18 : 1.0);
        ctx.fillStyle = threadColor(n.cluster - 1, fa);
        ctx.strokeStyle = n.member
            ? (dim ? "rgba(255,255,255,0.4)" : "#fff")
            : threadColor(n.cluster - 1, dim ? 0.12 : 0.55);
        ctx.lineWidth = n.member ? 0.6 : 0.5;
        ctx.beginPath();
        ctx.arc(n.x * viewScale + viewOffX, n.y * viewScale + viewOffY,
            n.r * viewScale, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
    }

    // Labels (after nodes so they sit on top). Drawn in priority order with
    // bounding-box collision skipping — important labels win, the rest are
    // dropped if they would overlap an already-drawn one. Hover/zoom always
    // reveal what's hidden.
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    const labelOrder = [];
    for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        if (!showLabel(n)) continue;
        const tier = n.active ? 0 : (n.member ? 1 : 2);
        labelOrder.push({ idx: i, tier, size: n.r });
    }
    labelOrder.sort((a, b) => a.tier - b.tier || b.size - a.size);
    const drawnLabels = [];
    const PAD_X = 2;
    for (const item of labelOrder) {
        const n = nodes[item.idx];
        // When hovering: labels outside the active subgraph render very
        // faint so the hovered node's network reads as a spotlight subset.
        const dim = anyHover && !n.active;
        let font, fill, hPx;
        if (n.active) { font = "600 14px sans-serif"; fill = "#111"; hPx = 14; }
        else if (n.member) {
            font = "13px sans-serif";
            fill = dim ? "rgba(120,120,120,0.22)" : "#444";
            hPx = 13;
        } else {
            font = "11px sans-serif";
            fill = dim ? "rgba(120,120,120,0.18)" : "#777";
            hPx = 11;
        }
        const sx = n.x * viewScale + viewOffX;
        const sy = (n.y - n.r - 3) * viewScale + viewOffY;
        ctx.font = font;
        const w = ctx.measureText(n.word).width;
        const left = sx - w / 2 - PAD_X;
        const right = sx + w / 2 + PAD_X;
        const top = sy - hPx;
        const bottom = sy;
        let collision = false;
        for (let k = 0; k < drawnLabels.length; k++) {
            const r = drawnLabels[k];
            if (left < r.right && right > r.left && top < r.bottom && bottom > r.top) {
                collision = true;
                break;
            }
        }
        if (collision && !n.active) continue;
        drawnLabels.push({ left, top, right, bottom });
        ctx.fillStyle = fill;
        ctx.fillText(n.word, sx, sy);
    }
}

// ---- Cluster ring ordering ----
// Place clusters around the anchor ring so neighbours on the ring share
// as many bridge edges as possible. For typical N (3-8) brute-force over
// permutations is cheap (≤ 5040 perms × O(N) score = sub-ms). Above N=8
// we fall back to a greedy insertion heuristic.
function orderClustersOnRing(clusters, slice, allEdges, count) {
    const N = clusters.length;
    if (N <= 2) return clusters.slice();
    const idx = new Map();
    clusters.forEach((c, i) => idx.set(c, i));
    const aff = Array.from({ length: N }, () => new Array(N).fill(0));
    for (const e of allEdges) {
        if (e.source >= count || e.target >= count) continue;
        const ca = slice[e.source].cluster;
        const cb = slice[e.target].cluster;
        if (ca === cb) continue;
        const i = idx.get(ca), j = idx.get(cb);
        const w = e.weight || 1;
        aff[i][j] += w;
        aff[j][i] += w;
    }
    function ringScore(order) {
        let s = 0;
        for (let k = 0; k < order.length; k++) {
            s += aff[idx.get(order[k])][idx.get(order[(k + 1) % order.length])];
        }
        return s;
    }
    if (N > 8) return greedyRingOrder(clusters, aff, idx);
    const tail = clusters.slice(1);
    let best = clusters.slice(), bestScore = ringScore(best);
    function permute(arr, k) {
        if (k === arr.length - 1) {
            const order = [clusters[0], ...arr];
            const s = ringScore(order);
            if (s > bestScore) { bestScore = s; best = order.slice(); }
            return;
        }
        for (let i = k; i < arr.length; i++) {
            [arr[k], arr[i]] = [arr[i], arr[k]];
            permute(arr, k + 1);
            [arr[k], arr[i]] = [arr[i], arr[k]];
        }
    }
    permute(tail, 0);
    return best;
}

function greedyRingOrder(clusters, aff, idx) {
    const N = clusters.length;
    let bestI = 0, bestJ = 1, bestPair = aff[0][1];
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
        if (aff[i][j] > bestPair) { bestPair = aff[i][j]; bestI = i; bestJ = j; }
    }
    const order = [clusters[bestI], clusters[bestJ]];
    const placed = new Set([bestI, bestJ]);
    while (placed.size < N) {
        let bestC = -1, bestPos = 0, bestGain = -Infinity;
        for (let c = 0; c < N; c++) {
            if (placed.has(c)) continue;
            for (let p = 0; p < order.length; p++) {
                const a = idx.get(order[p]);
                const b = idx.get(order[(p + 1) % order.length]);
                const gain = aff[c][a] + aff[c][b] - aff[a][b];
                if (gain > bestGain) { bestGain = gain; bestC = c; bestPos = p + 1; }
            }
        }
        order.splice(bestPos, 0, clusters[bestC]);
        placed.add(bestC);
    }
    return order;
}

// ---- Init + sim ----
function initLayout() {
    if (raf) { cancelAnimationFrame(raf); raf = null; }
    const g = result.value?.graph;
    if (!g || !g.nodes || g.nodes.length === 0) {
        nodes = []; edges = []; hovered = -1;
        draw();
        return;
    }
    const floor = g.n_members || g.nodes.length;
    const count = Math.min(g.nodes.length, Math.max(floor, networkWordCount.value || floor));
    const slice = g.nodes.slice(0, count);
    const maxW = Math.max(...slice.map((n) => n.weight), 1);
    const clusters = [...new Set(slice.map((n) => n.cluster))];
    // Order clusters around the anchor ring so pairs sharing many bridge
    // edges sit at adjacent positions — preserves the "related clusters
    // are visually close" property of force-only layouts while still
    // getting the spatial separation of anchors.
    const orderedClusters = orderClustersOnRing(clusters, slice, g.edges, count);
    // Cluster anchors: each cluster gets a fixed target position so the
    // simulation pulls clusters apart by design rather than relying on
    // repulsion alone.
    clusterAnchors = {};
    const anchorRad = Math.min(W, H) * 0.38;
    orderedClusters.forEach((cid, i) => {
        const ang = (i / orderedClusters.length) * 2 * Math.PI - Math.PI / 2;
        clusterAnchors[cid] = {
            x: W / 2 + Math.cos(ang) * anchorRad,
            y: H / 2 + Math.sin(ang) * anchorRad,
        };
    });
    nodes = slice.map((n) => {
        const anchor = clusterAnchors[n.cluster];
        const base = Math.sqrt(n.weight / maxW);
        return {
            word: n.word,
            cluster: n.cluster,
            member: n.member,
            r: n.member ? 3 + base * 13 : 2 + base * 6,
            x: anchor.x + (Math.random() - 0.5) * 60,
            y: anchor.y + (Math.random() - 0.5) * 60,
            vx: 0, vy: 0, fx: 0, fy: 0,
            active: false,
        };
    });
    edges = g.edges
        .filter((e) => e.source < count && e.target < count)
        .map((e) => ({ source: e.source, target: e.target, weight: e.weight, intra: e.intra }));
    hovered = -1;
    viewScale = cssW > 0 ? cssW / W : 1;
    viewOffX = 0;
    viewOffY = 0;
    // Re-engage auto-fit on every (re)init: any prior user zoom/pan is
    // meaningless for the new layout.
    isManualView.value = false;
    userZoom.value = 1;
    if (zoomRaf != null) { cancelAnimationFrame(zoomRaf); zoomRaf = null; }
    zoomAnim = null;
    runSim();
}

function runSim() {
    const N = nodes.length;
    if (N === 0) { draw(); return; }
    // Start cooler than 1.0 — at full alpha the accumulated repulsion of
    // many nodes flings outliers to the wall faster than the inward forces
    // can catch them. A gentler ramp lets the system settle from a compact
    // state instead of recovering from an explosion.
    let alpha = 0.6;
    // Repulsion now only shapes spacing — the discrete overlap-resolution
    // pass below handles strict non-overlap, so we no longer have to crank
    // this up to brute-force readability.
    const REPULSION = Math.max(350, 180000 / N);
    // Intra-cluster repulsion stays dampened: members of the same thread
    // should sit closer than members of different threads.
    const INTRA_REPEL = 0.55;
    const SPRING_INTRA = 0.030;
    // Inter-cluster (bridge) springs need to be weak — otherwise they yank
    // cluster members across the canvas toward sibling anchors. Keep them
    // just strong enough to bias related clusters to sit closer together.
    const SPRING_INTER = 0.0025;
    const TARGET_LEN = 75;
    // CENTER is a weak global pull to keep the whole graph centered; the
    // bulk of positioning comes from the anchor force below.
    const CENTER = 0.004;
    // Each node feels a pull toward its cluster's fixed anchor — this is
    // what separates clusters into distinct regions. Cranked high enough
    // that weakly-connected members don't drift toward a sibling cluster's
    // gap, but the overlap-resolution pass means we don't need it to
    // double as "keep nodes from touching".
    const ANCHOR_FORCE = 0.055;
    // Cluster gravity keeps members tight around the cluster centroid.
    const CLUSTER_GRAVITY = 0.045;
    const DAMP = 0.82;
    // Soft wall starts well inside the canvas so context outliers get
    // caught long before the hard clamp pins them in a corner.
    const WALL_MARGIN = 100;
    const WALL_STIFF = 0.30;

    function tick() {
        // Member centroids — context words follow the cores, can't drag them.
        const cents = {};
        for (let i = 0; i < N; i++) {
            const a = nodes[i];
            if (!a.member) continue;
            const c = cents[a.cluster] || (cents[a.cluster] = { x: 0, y: 0, n: 0 });
            c.x += a.x; c.y += a.y; c.n += 1;
        }
        for (const k in cents) { cents[k].x /= cents[k].n; cents[k].y /= cents[k].n; }
        // Repulsion + centering + cluster gravity + cluster anchor + soft wall — O(N²)
        for (let i = 0; i < N; i++) {
            const a = nodes[i];
            let fx = (W / 2 - a.x) * CENTER;
            let fy = (H / 2 - a.y) * CENTER;
            const c = cents[a.cluster];
            if (c) { fx += (c.x - a.x) * CLUSTER_GRAVITY; fy += (c.y - a.y) * CLUSTER_GRAVITY; }
            const anchor = clusterAnchors[a.cluster];
            if (anchor) {
                fx += (anchor.x - a.x) * ANCHOR_FORCE;
                fy += (anchor.y - a.y) * ANCHOR_FORCE;
            }
            if (a.x < WALL_MARGIN) fx += (WALL_MARGIN - a.x) * WALL_STIFF;
            else if (a.x > W - WALL_MARGIN) fx -= (a.x - (W - WALL_MARGIN)) * WALL_STIFF;
            if (a.y < WALL_MARGIN) fy += (WALL_MARGIN - a.y) * WALL_STIFF;
            else if (a.y > H - WALL_MARGIN) fy -= (a.y - (H - WALL_MARGIN)) * WALL_STIFF;
            for (let j = 0; j < N; j++) {
                if (i === j) continue;
                const b = nodes[j];
                let dx = a.x - b.x;
                let dy = a.y - b.y;
                let d2 = dx * dx + dy * dy || 0.01;
                const d = Math.sqrt(d2);
                const rep = a.cluster === b.cluster ? REPULSION * INTRA_REPEL : REPULSION;
                const f = rep / d2;
                fx += (dx / d) * f;
                fy += (dy / d) * f;
            }
            a.fx = fx; a.fy = fy;
        }
        // Springs — same-thread hard, bridges slack.
        for (let i = 0; i < edges.length; i++) {
            const e = edges[i];
            const a = nodes[e.source];
            const b = nodes[e.target];
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            const d = Math.sqrt(dx * dx + dy * dy) || 0.01;
            const stiff = e.intra ? SPRING_INTRA : SPRING_INTER;
            const force = stiff * (d - TARGET_LEN) * (0.3 + e.weight);
            const ux = dx / d, uy = dy / d;
            a.fx += ux * force; a.fy += uy * force;
            b.fx -= ux * force; b.fy -= uy * force;
        }
        // Integrate.
        for (let i = 0; i < N; i++) {
            const a = nodes[i];
            a.vx = (a.vx + a.fx * alpha) * DAMP;
            a.vy = (a.vy + a.fy * alpha) * DAMP;
            a.x = Math.max(a.r, Math.min(W - a.r, a.x + a.vx * alpha));
            a.y = Math.max(a.r, Math.min(H - a.r, a.y + a.vy * alpha));
        }
        // Discrete overlap resolution: any two nodes whose circles (plus a
        // small padding for label legibility) intersect get pushed apart,
        // splitting the correction half-and-half. Positional constraint, not
        // a force — lets us tune repulsion for *spacing* rather than for
        // *non-overlap*.
        const PAD = 3;
        for (let i = 0; i < N; i++) {
            const a = nodes[i];
            for (let j = i + 1; j < N; j++) {
                const b = nodes[j];
                const dx = b.x - a.x;
                const dy = b.y - a.y;
                const minDist = a.r + b.r + PAD;
                const d2 = dx * dx + dy * dy;
                if (d2 < minDist * minDist && d2 > 0) {
                    const d = Math.sqrt(d2);
                    const push = (minDist - d) * 0.5;
                    const ux = dx / d, uy = dy / d;
                    a.x = Math.max(a.r, Math.min(W - a.r, a.x - ux * push));
                    a.y = Math.max(a.r, Math.min(H - a.r, a.y - uy * push));
                    b.x = Math.max(b.r, Math.min(W - b.r, b.x + ux * push));
                    b.y = Math.max(b.r, Math.min(H - b.r, b.y + uy * push));
                }
            }
        }
        applyFit(true);
        draw();
        alpha *= 0.985;
        if (alpha > 0.012) raf = requestAnimationFrame(tick);
        else raf = null;
    }
    tick();
}

function onViewportResize() { setupCanvas(); draw(); }

// ---- Canvas lifecycle ----
// The canvas lives inside v-if="result.graph", so it mounts when graph
// data arrives and may unmount on a fresh query. Wire listeners per
// element instance, tear down the prior set when the element changes.
async function ensureCanvasSetup() {
    await nextTick();
    const c = canvasRef.value;
    if (!c || setupForCanvas === c) return;
    if (setupForCanvas) setupForCanvas.removeEventListener("wheel", onWheel);
    if (resizeObs) { resizeObs.disconnect(); resizeObs = null; }
    setupCanvas();
    if (typeof ResizeObserver !== "undefined" && c.parentElement) {
        resizeObs = new ResizeObserver(() => { setupCanvas(); draw(); });
        resizeObs.observe(c.parentElement);
    }
    // Wheel needs an explicit non-passive listener so preventDefault works
    // — modern browsers default wheel listeners to passive otherwise.
    c.addEventListener("wheel", onWheel, { passive: false });
    setupForCanvas = c;
}

// When graph data arrives or the displayed-node count changes, (re-)build
// the layout. flush: 'post' so the DOM is updated (canvas mounted) before
// we touch it.
watch([() => result.value?.graph, networkWordCount], async ([g]) => {
    if (!g) return;
    await ensureCanvasSetup();
    initLayout();
}, { flush: "post" });

onMounted(() => {
    window.addEventListener("resize", onViewportResize);
});

onBeforeUnmount(() => {
    if (raf) cancelAnimationFrame(raf);
    if (zoomRaf != null) cancelAnimationFrame(zoomRaf);
    if (resizeObs) resizeObs.disconnect();
    window.removeEventListener("resize", onViewportResize);
    window.removeEventListener("mousemove", onWindowMouseMove);
    window.removeEventListener("mouseup", onWindowMouseUp);
    if (setupForCanvas) setupForCanvas.removeEventListener("wheel", onWheel);
});
</script>

<style lang="scss" scoped>
.overview-card {
    background-color: #fff;
}

/* Two-column body: viz on the left, legend aside on the right. Below ~890px
   the row's flex-basis won't fit and the legend wraps under the viz. */
.viz-area {
    flex: 999 1 600px;
    min-width: 0;
}

.legend-aside {
    flex: 1 0 290px;
    min-width: 290px;
    max-width: 340px;
}

.legend-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.control-stack {
    padding-top: 0.75rem;
    border-top: 1px solid #eee;
}

.control-hint {
    margin: 0.25rem 0 0;
    font-size: 0.7rem;
    line-height: 1.3;
    color: #888;
}

.legend-chip {
    border: 1px solid;
    background-color: #fff;
    padding: 0.2rem 0.5rem;
    font-size: 0.75rem;
    line-height: 1.25;
    width: 100%;
    display: flex;
    align-items: flex-start;
    text-align: left;
}

.legend-text {
    word-break: break-word;
}

.legend-swatch {
    display: inline-block;
    flex-shrink: 0;
    width: 0.65rem;
    height: 0.65rem;
    border-radius: 0.15rem;
    margin-right: 0.3rem;
    margin-top: 0.2rem;
    vertical-align: middle;
}

.info-tip {
    position: relative;
    display: inline-block;
    cursor: help;
    color: #999;
    margin-left: 0.25rem;
    font-size: 0.95em;
    user-select: none;
    outline: none;
}

.info-tip:hover::after,
.info-tip:focus::after {
    content: attr(data-tip);
    position: absolute;
    top: calc(100% + 6px);
    left: 0;
    z-index: 1000;
    background: rgba(33, 37, 41, 0.95);
    color: #fff;
    padding: 0.5rem 0.75rem;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    line-height: 1.4;
    width: max-content;
    max-width: 320px;
    white-space: normal;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
    pointer-events: none;
}

/* ---- Canvas + zoom controls ---- */
.thread-network {
    width: 100%;
    position: relative;
}

.network-canvas {
    display: block;
    margin: 0 auto;       /* JS sets explicit width/height; centered when narrower than container */
    cursor: default;
}

.zoom-controls {
    position: absolute;
    top: 6px;
    right: 6px;
    z-index: 5;
    display: flex;
    flex-direction: column;
    gap: 3px;
}

.zoom-controls button {
    width: 28px;
    height: 28px;
    padding: 0;
    border: 1px solid #ddd;
    background: rgba(255, 255, 255, 0.92);
    color: #444;
    border-radius: 4px;
    cursor: pointer;
    font-size: 15px;
    line-height: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06);
    transition: background 0.12s ease, color 0.12s ease;
}

.zoom-controls button:hover:not(:disabled) {
    background: #fff;
    color: #111;
}

.zoom-controls button:disabled {
    opacity: 0.35;
    cursor: not-allowed;
}
</style>
