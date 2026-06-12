<template>
    <div>
        <div v-if="loading && !result" class="text-center py-5">
            <progress-spinner :lg="true" :message="$t('usagePatterns.detecting')" />
        </div>

        <div v-if="result && result.patterns && result.patterns.length > 0" class="mx-2 my-3">
            <!-- Everything (count, explanation, controls, viz) lives inside the
                 result card so the layout reads as one self-contained unit. -->
            <div v-if="result.graph" class="card shadow-sm p-3 mb-3 overview-card">
                <div class="overview-body d-flex flex-wrap gap-3 align-items-start">
                    <aside class="legend-aside">
                        <p class="text-muted mb-1 d-flex align-items-center flex-wrap gap-1">
                            <i18n-t keypath="usagePatterns.summary" tag="span">
                                <template #patterns>
                                    <select v-model.number="patternCount" @change="onGrainChange"
                                        class="form-select form-select-sm summary-dropdown"
                                        :aria-label="$t('usagePatterns.patternCount')">
                                        <option v-for="n in (result?.available_pattern_counts || [])" :key="n" :value="n">
                                            {{ n }}</option>
                                    </select>
                                </template>
                            </i18n-t>
                            <span class="info-tip" tabindex="0" :data-tip="$t('usagePatterns.networkSubtitle')"
                                :aria-label="$t('usagePatterns.networkSubtitle')">ⓘ</span>
                        </p>
                        <div class="legend-list mt-2">
                            <button v-for="pattern in result.patterns" :key="`leg-${pattern.id}`" type="button"
                                class="btn btn-sm legend-chip" :class="{ 'cluster-off': hiddenClusters.has(pattern.id) }"
                                :style="{ borderColor: patternColor(pattern.id - 1, 1), color: patternColor(pattern.id - 1, 1) }"
                                @click.stop="openLegendMenu(pattern, $event)"
                                @mouseenter="onLegendChipEnter(pattern.id)"
                                @mouseleave="onLegendChipLeave"
                                :title="pattern.words.slice(0, 10).map((w) => w.word).join(', ')">
                                <span class="legend-swatch"
                                    :style="{ backgroundColor: patternColor(pattern.id - 1, 1) }"></span>
                                <span class="legend-text">{{ pattern.label }}</span>
                            </button>
                        </div>
                        <div class="graph-search mt-3">
                            <input v-model="searchQuery" @input="onSearchInput" type="text"
                                class="form-control form-control-sm" :placeholder="$t('usagePatterns.searchPlaceholder')" />
                        </div>
                    </aside>

                    <div class="viz-area">
                        <div class="pattern-network">
                            <div class="zoom-controls">
                                <button type="button" @click="zoomIn" :title="$t('usagePatterns.zoomIn')"
                                    :aria-label="$t('usagePatterns.zoomIn')">+</button>
                                <button type="button" @click="zoomOut" :title="$t('usagePatterns.zoomOut')"
                                    :aria-label="$t('usagePatterns.zoomOut')">−</button>
                                <button type="button" @click="resetView" :title="$t('usagePatterns.zoomReset')"
                                    :aria-label="$t('usagePatterns.zoomReset')">
                                    <i class="bi bi-arrows-fullscreen" aria-hidden="true"></i>
                                </button>
                            </div>
                            <!-- Dim node rings drawn UNDERNEATH sigma so labels on
                                 highlighted nodes (rendered inside sigma) sit on top
                                 instead of being covered by the rings. -->
                            <canvas ref="dimRingsRef" class="dim-rings-overlay"></canvas>
                            <div ref="containerRef" class="sigma-container"
                                :style="{ height: sigmaHeight + 'px' }"></div>
                            <!-- Soft cluster hulls drawn on top of sigma (low alpha, no
                                 pointer events). -->
                            <canvas ref="hullsRef" class="hulls-overlay"></canvas>
                            <!-- In-place spinner during grain change (result still
                                 populated, so the main spinner above is hidden). -->
                            <div v-if="loading" class="viz-loading-overlay">
                                <progress-spinner :message="$t('usagePatterns.detecting')" />
                            </div>
                            <!-- Right-click context menu: actions depend on whether
                                 user clicked a node or empty stage. -->
                            <div v-if="contextMenu" class="ctx-menu"
                                :style="{ left: contextMenu.x + 'px', top: contextMenu.y + 'px' }" @click.stop>
                                <template v-if="contextMenu.kind === 'node'">
                                    <button type="button" class="ctx-item" @click="ctxPassagesForNode">
                                        {{ $t('usagePatterns.ctxPassagesNode') }}
                                    </button>
                                    <button type="button" class="ctx-item" @click="ctxPassagesForCluster">
                                        {{ $t('usagePatterns.ctxPassagesCluster') }}
                                    </button>
                                    <div class="ctx-sep"></div>
                                    <button type="button" class="ctx-item" @click="ctxFlyToCluster">
                                        {{ $t('usagePatterns.ctxFlyCluster') }}
                                    </button>
                                    <button type="button" class="ctx-item" @click="ctxSoloCluster">
                                        {{ $t('usagePatterns.ctxSoloCluster') }}
                                    </button>
                                    <button type="button" class="ctx-item" @click="ctxHideCluster">
                                        {{ $t('usagePatterns.ctxHideCluster') }}
                                    </button>
                                </template>
                                <template v-else>
                                    <button type="button" class="ctx-item" @click="ctxResetView">
                                        {{ $t('usagePatterns.ctxResetView') }}
                                    </button>
                                    <button type="button" class="ctx-item" :disabled="hiddenClusters.size === 0"
                                        @click="ctxShowAll">
                                        {{ $t('usagePatterns.ctxShowAll') }}
                                    </button>
                                </template>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Backend didn't ship graph data (older server) — friendly fallback. -->
            <div v-else class="text-center py-4 text-muted">
                {{ $t('usagePatterns.noGraph') }}
            </div>
        </div>

        <div v-else-if="result && (!result.patterns || result.patterns.length === 0)" class="text-center py-5 text-muted">
            {{ $t('usagePatterns.noResults') }}
        </div>

        <DistinctivePassagesModal :group-name="modal.groupName" :signature="modal.signature" :passages="modal.passages"
            :loading="modal.loading" :has-more="modal.hasMore" :view-all-url="modal.viewAllUrl"
            @load-more="loadMorePassages" @view-all="onViewAllPassages" />

        <!-- Legend chip popup: 3 actions. Teleported to body so the absolute
             positioning isn't clipped by ancestor overflow. -->
        <Teleport to="body">
            <div v-if="legendMenu" class="ctx-menu legend-popup"
                :style="{ left: legendMenu.x + 'px', top: legendMenu.y + 'px', position: 'fixed' }" @click.stop>
                <div class="legend-popup-header">
                    <span class="legend-swatch"
                        :style="{ backgroundColor: patternColor(legendMenu.pattern.id - 1, 1) }"></span>
                    <span class="legend-popup-label">T{{ legendMenu.pattern.id }}: {{ legendMenu.pattern.label }}</span>
                </div>
                <div class="ctx-sep"></div>
                <button type="button" class="ctx-item" @click="legendGetPassages">
                    {{ $t('usagePatterns.legendPopupPassages') }}
                </button>
                <button type="button" class="ctx-item" @click="legendFocusCluster">
                    {{ $t('usagePatterns.legendPopupFocus') }}
                </button>
                <button type="button" class="ctx-item" @click="legendToggleHideCluster">
                    {{ hiddenClusters.has(legendMenu.pattern.id)
                        ? $t('usagePatterns.legendPopupShow')
                        : $t('usagePatterns.legendPopupHide') }}
                </button>
            </div>
        </Teleport>
    </div>
</template>

<script setup>
import EdgeCurveProgram from "@sigma/edge-curve";
import { createNodeBorderProgram } from "@sigma/node-border";
import { Modal } from "bootstrap";
import Graph from "graphology";
import forceAtlas2 from "graphology-layout-forceatlas2";
import noverlap from "graphology-layout-noverlap";
import { storeToRefs } from "pinia";
import Sigma from "sigma";
import { inject, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import { useMainStore } from "../../stores/main";
import { concordanceMethod, debug, paramsFilter, paramsToRoute } from "../../utils.js";
import DistinctivePassagesModal from "../DistinctivePassagesModal.vue";
import ProgressSpinner from "../ProgressSpinner";

// Outline-only node program: the outer band takes the node's `color` (the
// cluster color), the inner is driven by the `innerColor` attribute so we
// can flip it transparent on dimmed nodes (lets the highlighted edges show
// through them instead of being clipped by a solid white centre).
const NodeRingProgram = createNodeBorderProgram({
    borders: [
        { size: { value: 0.12, mode: "auto" }, color: { attribute: "color" } },
        { size: { fill: true }, color: { attribute: "innerColor" } },
    ],
});

const emit = defineEmits(["filterList"]);

const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const router = useRouter();
const store = useMainStore();
const { formData } = storeToRefs(store);

// ---- Data fetch state ----
const loading = ref(false);
const result = ref(null);
// Seeded at 4, but a fresh query defers to the backend's smart default (the
// hub-strength knee) and adopts that count; only a user dropdown pick sends an
// explicit count. Mirrors UsagePatternsTimeline.
const patternCount = ref(4);
let userChoseCount = false;
// Cluster IDs the user has muted via legend chips. Plain click toggles a single
// cluster; shift-click solos (hides all others). A node is hidden iff its
// cluster is muted.
const hiddenClusters = ref(new Set());
let fetchToken = 0;

// Same color palette as the streamgraph so pattern colors are consistent across
// tabs. Emitted as rgba() — sigma's color parser doesn't accept hsla().
const patternHues = [205, 30, 145, 280, 0, 90, 165, 235, 50, 315, 120, 260, 15, 60, 200, 320];
function patternColor(i, alpha = 1) {
    const h = patternHues[i % patternHues.length];
    const s = 0.55, l = 0.50;
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
    const m = l - c / 2;
    const [r1, g1, b1] = h < 60 ? [c, x, 0] : h < 120 ? [x, c, 0]
        : h < 180 ? [0, c, x] : h < 240 ? [0, x, c]
            : h < 300 ? [x, 0, c] : [c, 0, x];
    const r = Math.round((r1 + m) * 255);
    const g = Math.round((g1 + m) * 255);
    const b = Math.round((b1 + m) * 255);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function runDetection(opts = {}) {
    // Fresh run (mount, new query) defers to the backend smart default; only a
    // user dropdown change sends an explicit count.
    if (!opts.grainChange) userChoseCount = false;
    const myToken = ++fetchToken;
    loading.value = true;
    if (!opts.keepResult) result.value = null;
    const params = paramsFilter(formData.value);
    if (userChoseCount) params.n_clusters = patternCount.value;
    $http.get(`${$dbUrl}/scripts/get_usage_patterns.py`, { params }).then((resp) => {
        if (myToken !== fetchToken) return;
        result.value = resp.data;
        const avail = resp.data?.available_pattern_counts || [];
        if (!userChoseCount) {
            // Adopt the backend's smart default so the dropdown reflects it.
            const n = resp.data?.patterns?.length;
            if (n) patternCount.value = n;
        } else if (avail.length && !avail.includes(patternCount.value)) {
            // Keep the selection valid on thin queries that yield fewer senses
            // than requested (backend already truncated; reflect it here).
            patternCount.value = avail[avail.length - 1];
        }
        emit("filterList", resp.data?.filter_list || []);
    }).catch((error) => {
        debug({ $options: { name: "word-map" } }, error);
    }).finally(() => {
        if (myToken === fetchToken) loading.value = false;
    });
}

function onGrainChange() {
    if (G) {
        pendingPrevPositions = new Map();
        G.forEachNode((id, attrs) => {
            pendingPrevPositions.set(attrs.word, { x: attrs.x, y: attrs.y });
        });
    }
    userChoseCount = true;
    runDetection({ keepResult: true, grainChange: true });
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

function openPassages(groupName, words) {
    const [yMin, yMax] = result.value.year_range;
    const yearRange = `${yMin}-${yMax}`;
    modal.value = {
        groupName,
        yearRange,
        signature: words.slice(0, 20).map((word) => ({ word, z: null })),
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

// Legend chip click opens a small popup with explicit actions (passages /
// focus / hide-or-show).
const legendMenu = ref(null);    // { pattern, x, y } viewport-relative, or null

function openLegendMenu(pattern, event) {
    const rect = event.currentTarget.getBoundingClientRect();
    const popupWidth = 220;
    let x = rect.right + 8;
    if (x + popupWidth > window.innerWidth) x = Math.max(8, rect.left - popupWidth - 8);
    legendMenu.value = { pattern, x, y: rect.top };
}
function closeLegendMenu() { legendMenu.value = null; }

// Hover a legend chip → focus that cluster in the graph: members stay bright,
// everything else gets the standard "dim" treatment (same path used by node
// hover + search). Reuses the dim-ring overlay, so no extra plumbing.
function onLegendChipEnter(clusterId) {
    hoveredCluster = clusterId;
    if (renderer) renderer.refresh();
}
function onLegendChipLeave() {
    if (hoveredCluster === null) return;
    hoveredCluster = null;
    if (renderer) renderer.refresh();
}

function legendGetPassages() {
    const t = legendMenu.value?.pattern;
    if (t) openPassages(`${formData.value.q} · ${t.label}`,
        t.words.slice(0, 20).map((w) => w.word));
    closeLegendMenu();
}
function legendFocusCluster() {
    const t = legendMenu.value?.pattern;
    if (!t) return;
    // Zoom-only: other clusters stay visible. Auto-unhide first if this
    // cluster was hidden — zooming to invisible nodes would do nothing.
    if (hiddenClusters.value.has(t.id)) {
        const next = new Set(hiddenClusters.value);
        next.delete(t.id);
        hiddenClusters.value = next;
        applyVisibility();
        renderer?.refresh();
    }
    zoomToCluster(t.id);
    closeLegendMenu();
}
function legendToggleHideCluster() {
    const t = legendMenu.value?.pattern;
    if (!t || !result.value) return;
    const next = new Set(hiddenClusters.value);
    if (next.has(t.id)) {
        next.delete(t.id);
    } else {
        next.add(t.id);
        if (next.size >= result.value.patterns.length) {
            closeLegendMenu();
            return;
        }
    }
    hiddenClusters.value = next;
    applyVisibility();
    renderer?.refresh();
    closeLegendMenu();
}

// Outside-click dismiss for the legend popup. Chip clicks use @click.stop, so
// they don't reach this handler; clicks inside the popup itself are caught by
// its own @click.stop.
function onDocClickForLegend(e) {
    if (!legendMenu.value) return;
    const el = document.querySelector(".legend-popup");
    if (el && !el.contains(e.target)) closeLegendMenu();
}

// ---- Right-click context menu actions ----
function closeContextMenu() { contextMenu.value = null; }
function ctxPatternFor(nodeId) {
    if (!G || !result.value) return null;
    const cid = G.getNodeAttribute(nodeId, "cluster");
    return result.value.patterns.find((t) => t.id === cid) || null;
}
function ctxPassagesForNode() {
    if (!contextMenu.value) return;
    onViewNode(contextMenu.value.nodeId);
    closeContextMenu();
}
function ctxPassagesForCluster() {
    if (!contextMenu.value) return;
    const pattern = ctxPatternFor(contextMenu.value.nodeId);
    if (pattern) openPassages(`${formData.value.q} · ${pattern.label}`,
        pattern.words.slice(0, 20).map((w) => w.word));
    closeContextMenu();
}
function ctxFlyToCluster() {
    if (!contextMenu.value) return;
    const pattern = ctxPatternFor(contextMenu.value.nodeId);
    if (pattern) flyToCluster(pattern.id);
    closeContextMenu();
}
function ctxSoloCluster() {
    if (!contextMenu.value || !result.value) return;
    const pattern = ctxPatternFor(contextMenu.value.nodeId);
    if (pattern) onLegendClick(pattern, { shiftKey: true });
    closeContextMenu();
}
function ctxHideCluster() {
    if (!contextMenu.value) return;
    const pattern = ctxPatternFor(contextMenu.value.nodeId);
    if (pattern && !hiddenClusters.value.has(pattern.id))
        onLegendClick(pattern, { shiftKey: false });
    closeContextMenu();
}
function ctxShowAll() {
    hiddenClusters.value = new Set();
    applyVisibility();
    renderer?.refresh();
    closeContextMenu();
}
function ctxResetView() { resetView(); closeContextMenu(); }

// Node click → the node + its NPMI neighbours (which may span clusters) are
// the words the user actually saw lit up, so they (not the full pattern) seed
// the passage signature.
function onViewNode(nodeId) {
    if (!G) return;
    const word = G.getNodeAttribute(nodeId, "word");
    const words = [word];
    G.forEachNeighbor(nodeId, (nb) => words.push(G.getNodeAttribute(nb, "word")));
    openPassages(`${formData.value.q} · ${word}`, [...new Set(words)]);
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
// Sigma + graphology rendering. Operates on result.value.graph.
// FA2 (Barnes-Hut) for layout, sigma for WebGL rendering, label LOD,
// and hover/click events.
// =====================================================================

const containerRef = ref(null);
const hullsRef = ref(null);     // cluster-hull overlay canvas (above sigma)
const dimRingsRef = ref(null);  // dim-node rings overlay canvas (below sigma)
// Sigma container height — computed to fill the viewport from the container's
// top down to a small bottom margin. Updated on mount, window resize, and
// whenever a new result loads (in case chrome above the graph shifts).
const sigmaHeight = ref(540);
const searchQuery = ref("");    // search-in-graph input
// Right-click menu: { x, y, kind: 'node'|'stage', nodeId? } or null when closed.
// Coordinates are container-relative (sigma's viewport space).
const contextMenu = ref(null);
let renderer = null;            // Sigma instance
let G = null;                   // graphology Graph (data + layout positions)
let hoveredNode = null;         // hover-highlight target
let matchedNodes = new Set();   // search matches (highlighted, others dimmed)
let hoveredCluster = null;      // legend-chip hover focus (entire cluster vs dimmed others)
let LABELS_PER_CLUSTER = 5;     // top-N anchor words per cluster get forced labels
// Position transition state — captured pre-change so the new layout can tween
// in from the previous one (keyed by word so shared words flow to new spots).
let pendingPrevPositions = null;
let animFrameId = null;

function buildGraph(g, seedPositions = null) {
    const graph = new Graph();
    const maxAnchor = Math.max(0.001, ...g.nodes.map((n) => n.anchor || 0));
    // Seed each cluster's members in its own region of a ring. FA2 starts from
    // a clustered arrangement and just *refines* it (rather than discovering
    // clusters from random init, which 500 nodes in a tiny box can't do in
    // 300 iterations). Order is arbitrary — only matters that clusters start apart.
    const clusterIds = [...new Set(g.nodes.map((n) => n.cluster))];
    const ringRad = 30;             // small ring — a hint, not a lock
    const seed = {};
    clusterIds.forEach((cid, i) => {
        const a = (i / clusterIds.length) * 2 * Math.PI;
        seed[cid] = { x: Math.cos(a) * ringRad, y: Math.sin(a) * ringRad };
    });
    // For continuity across pattern-count changes: place new words near the
    // centroid of cluster siblings that we DO have prior positions for.
    const prevClusterCentroid = new Map();
    if (seedPositions) {
        const acc = new Map();
        g.nodes.forEach((n) => {
            const p = seedPositions.get(n.word);
            if (!p) return;
            const a = acc.get(n.cluster) || { sx: 0, sy: 0, n: 0 };
            a.sx += p.x; a.sy += p.y; a.n += 1;
            acc.set(n.cluster, a);
        });
        acc.forEach((a, c) => prevClusterCentroid.set(c, { x: a.sx / a.n, y: a.sy / a.n }));
    }
    g.nodes.forEach((n, i) => {
        const prev = seedPositions?.get(n.word);
        const centroid = prevClusterCentroid.get(n.cluster);
        const base = prev || centroid || seed[n.cluster];
        const jitter = prev ? 0 : 8;
        graph.addNode(String(i), {
            x: base.x + (Math.random() - 0.5) * jitter,
            y: base.y + (Math.random() - 0.5) * jitter,
            // Size scales with within-sense anchor (the legend's centrality), so
            // the biggest circles ARE the label words. Sqrt damps extremes.
            size: 2 + 10 * Math.sqrt((n.anchor || 0) / maxAnchor),
            color: patternColor(n.cluster - 1, 1),
            innerColor: "#ffffff",
            label: n.word,
            word: n.word,
            cluster: n.cluster,
            member: n.member,
            anchor: n.anchor || 0,
            weight: n.weight,
            // Visible unless its cluster is muted (see applyVisibility).
            hidden: false,
        });
    });
    // FA2 has no native cluster concept, so we encode "cluster cohesion" via
    // edge weights: intra-cluster edges keep their full NPMI weight, inter-
    // cluster bridges are scaled down to INTER_WEIGHT_SCALE of their nominal
    // weight in FA2's eyes. Intra-pull dominates → tight cluster blobs; inter
    // still pulls related clusters near each other (so the layout encodes
    // sense-affinity, not just "clusters spread evenly around the center").
    //
    // Rendering: intra edges stay straight (clean inside clusters); inter
    // bridges render as gentle curves so the bundle of A-to-B edges arcs
    // through the gap instead of crossing at the cluster boundaries.
    const INTER_WEIGHT_SCALE = 0.15;
    g.edges.forEach((e) => {
        const w = e.weight || 0;
        graph.addEdge(String(e.source), String(e.target), {
            size: 0.2 + 0.7 * w,
            // Opaque colours below white. Sigma's curve edge program doesn't
            // reliably honour alpha, so we use solid colours instead.
            color: e.intra ? "#d4d8df" : "#e6e8ec",
            weight: e.intra ? w : INTER_WEIGHT_SCALE * w,
            intra: e.intra,
            type: e.intra ? "line" : "curve",
            curvature: e.intra ? 0 : 0.35,
        });
    });
    // Top-N per cluster get forceLabel (always rendered). Other labels appear
    // automatically as you zoom in and node screen-size crosses the threshold.
    const byCluster = {};
    graph.forEachNode((id, attrs) => {
        if (!attrs.member) return;
        (byCluster[attrs.cluster] ||= []).push({ id, anchor: attrs.anchor });
    });
    for (const arr of Object.values(byCluster)) {
        arr.sort((a, b) => b.anchor - a.anchor);
        arr.slice(0, LABELS_PER_CLUSTER).forEach(({ id }) =>
            graph.setNodeAttribute(id, "forceLabel", true));
    }
    return graph;
}

function applyVisibility() {
    if (!G) return;
    // All graph words (members + the context the backend padded in up to its
    // cap) are shown; only muted clusters hide nodes. Decluttering is by cluster
    // toggle (legend chips), not a per-word count.
    const offClusters = hiddenClusters.value;
    G.forEachNode((id, attrs) => {
        G.setNodeAttribute(id, "hidden", offClusters.has(attrs.cluster));
    });
}

// ---- Search-in-graph ----
function onSearchInput() {
    matchedNodes.clear();
    const q = searchQuery.value.trim().toLowerCase();
    if (q && G) {
        G.forEachNode((id, attrs) => {
            if (!attrs.hidden && attrs.word.toLowerCase().startsWith(q)) matchedNodes.add(id);
        });
    }
    renderer?.refresh();
    // Fit the camera to ALL matches rather than diving to the first one. When the
    // matches sit in one zone the bounding box is small, so it zooms in; when they
    // are scattered across the graph the box spans most of it, so the camera stays
    // near the overview and every match stays visible (highlighted).
    if (matchedNodes.size > 0 && renderer) {
        // camera.animate() expects FRAMED-graph coords (camera-independent);
        // compose graph → viewport → framed since sigma 3 has no direct map.
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        matchedNodes.forEach((id) => {
            const a = G.getNodeAttributes(id);
            const p = renderer.viewportToFramedGraph(renderer.graphToViewport({ x: a.x, y: a.y }));
            if (p.x < minX) minX = p.x;
            if (p.x > maxX) maxX = p.x;
            if (p.y < minY) minY = p.y;
            if (p.y > maxY) maxY = p.y;
        });
        const spread = Math.max(maxX - minX, maxY - minY);
        // ratio 1 ≈ whole graph; smaller = more zoomed in. Pad the box by 1.3, and
        // clamp: never zoom past a single tight cluster, never zoom out beyond the
        // overview.
        const ratio = Math.min(1, Math.max(0.4, spread * 1.3));
        renderer.getCamera().animate(
            { x: (minX + maxX) / 2, y: (minY + maxY) / 2, ratio }, { duration: 400 }
        );
    }
}

// ---- Cluster hulls + mini-map ----
// Convex hull (Andrew's monotone chain). Returns the hull as a closed loop of points.
function convexHull(pts) {
    if (pts.length < 3) return pts.slice();
    const p = pts.slice().sort((a, b) => a.x - b.x || a.y - b.y);
    const cross = (O, A, B) => (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
    const lower = [];
    for (const q of p) {
        while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop();
        lower.push(q);
    }
    const upper = [];
    for (let i = p.length - 1; i >= 0; i--) {
        const q = p[i];
        while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop();
        upper.push(q);
    }
    lower.pop(); upper.pop();
    return lower.concat(upper);
}

function sizeOverlayCanvas(canvas) {
    if (!canvas) return null;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== w * dpr) canvas.width = w * dpr;
    if (canvas.height !== h * dpr) canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
}

function drawHulls() {
    const ov = sizeOverlayCanvas(hullsRef.value);
    if (!ov || !G || !renderer) return;
    const { ctx } = ov;
    // Group all visible nodes (members + context) by cluster, in screen coords,
    // so the hull encloses every word in the cluster.
    const byCluster = {};
    G.forEachNode((id, attrs) => {
        if (attrs.hidden) return;
        const s = renderer.graphToViewport(attrs);
        (byCluster[attrs.cluster] ||= { pts: [], color: attrs.color }).pts.push(s);
    });
    for (const cid in byCluster) {
        const { pts, color } = byCluster[cid];
        if (pts.length < 3) continue;
        const hull = convexHull(pts);
        // Inflate the hull outward by ~18 px so it doesn't sit on the outer nodes.
        const cx = hull.reduce((a, p) => a + p.x, 0) / hull.length;
        const cy = hull.reduce((a, p) => a + p.y, 0) / hull.length;
        const pad = 18;
        const padded = hull.map((p) => {
            const dx = p.x - cx, dy = p.y - cy;
            const len = Math.sqrt(dx * dx + dy * dy) || 1;
            return { x: p.x + (dx / len) * pad, y: p.y + (dy / len) * pad };
        });
        // Soft fill at low alpha — matches cluster color so the region is recognisable.
        const fill = color.replace(/,\s*1\s*\)$/, ", 0.13)");
        ctx.fillStyle = fill;
        ctx.beginPath();
        ctx.moveTo(padded[0].x, padded[0].y);
        for (let i = 1; i < padded.length; i++) ctx.lineTo(padded[i].x, padded[i].y);
        ctx.closePath();
        ctx.fill();
    }
    drawDimNodes();
}

// Paint dim node rings on the BELOW-sigma overlay so overlap pixels are
// identical to non-overlap pixels (stroked circles vs the WebGL ring
// program's alpha-cutout artefacts) AND labels rendered inside sigma sit on
// top of them. Sigma hides these nodes via the reducer; this is the only
// thing rendering them while a hover or search is active.
function drawDimNodes() {
    // Always size + clear the canvas so leftover rings don't linger when
    // the user clears their hover / search.
    const ov = sizeOverlayCanvas(dimRingsRef.value);
    if (!ov || !G || !renderer) return;
    const { ctx } = ov;
    const clusterFocus = hoveredCluster !== null;
    const searching = matchedNodes.size > 0;
    const hovering = hoveredNode !== null;
    if (!clusterFocus && !searching && !hovering) return;
    const ratio = renderer.getCamera().getState().ratio;
    ctx.strokeStyle = "#cad0d8";
    ctx.lineWidth = 1;
    G.forEachNode((id, attrs) => {
        if (attrs.hidden) return;     // user-hidden (cluster-off)
        let isDim;
        if (clusterFocus) isDim = attrs.cluster !== hoveredCluster;
        else if (hovering) isDim = id !== hoveredNode && !G.areNeighbors(id, hoveredNode);
        else isDim = !matchedNodes.has(id);
        if (!isDim) return;
        const pos = renderer.graphToViewport(attrs);
        const r = (attrs.size || 4) / ratio;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
        ctx.stroke();
    });
}

function setupReducers() {
    if (!renderer) return;
    // Dim colour for non-matches / non-neighbours: a medium gray-blue so the
    // ring is clearly visible on white while still receding behind the
    // highlighted (full-color) nodes.
    const DIM = "#b0b6c0";
    renderer.setSetting("nodeReducer", (node, attrs) => {
        // Legend chip hover takes priority: show only the focused cluster.
        if (hoveredCluster !== null) {
            return attrs.cluster === hoveredCluster
                ? { ...attrs, zIndex: 1 }
                : { ...attrs, hidden: true };
        }
        // Node hover takes priority over search, so hovering a match reveals its
        // connections — its neighbours (even non-matches) light up. Only visible
        // nodes can be hovered, so during a search this fires on a match.
        if (hoveredNode !== null) {
            if (node === hoveredNode || G.areNeighbors(node, hoveredNode)) {
                return { ...attrs, zIndex: 1, forceLabel: true };
            }
            // Non-neighbour: hidden in sigma; drawDimNodes paints the ring.
            return { ...attrs, hidden: true };
        }
        // Search (not hovering): highlight matches, hide the rest. (drawDimNodes
        // paints the dim ones as Canvas2D rings on the overlay so overlaps render
        // cleanly — see drawHulls.)
        if (matchedNodes.size > 0) {
            return matchedNodes.has(node)
                ? { ...attrs, zIndex: 1, forceLabel: true }
                : { ...attrs, hidden: true };
        }
        return attrs;
    });
    renderer.setSetting("edgeReducer", (edge, attrs) => {
        if (hoveredCluster !== null) {
            const [s, t] = G.extremities(edge);
            const sIn = G.getNodeAttribute(s, "cluster") === hoveredCluster;
            const tIn = G.getNodeAttribute(t, "cluster") === hoveredCluster;
            // Only intra-cluster edges remain — bridges to other clusters get hidden.
            return sIn && tIn ? attrs : { ...attrs, hidden: true };
        }
        if (hoveredNode !== null) {
            const [s, t] = G.extremities(edge);
            if (s === hoveredNode || t === hoveredNode) return { ...attrs, color: "#888", size: attrs.size * 1.5 };
            return { ...attrs, hidden: true };
        }
        if (matchedNodes.size > 0) {
            const [s, t] = G.extremities(edge);
            if (matchedNodes.has(s) || matchedNodes.has(t)) return attrs;
            return { ...attrs, hidden: true };
        }
        return attrs;
    });
}

function flyToCluster(clusterId) {
    if (!G || !renderer) return;
    const xs = [], ys = [];
    G.forEachNode((id, attrs) => {
        if (attrs.cluster === clusterId && attrs.member) {
            xs.push(attrs.x); ys.push(attrs.y);
        }
    });
    if (!xs.length) return;
    const cx = xs.reduce((a, b) => a + b, 0) / xs.length;
    const cy = ys.reduce((a, b) => a + b, 0) / ys.length;
    const fg = renderer.viewportToFramedGraph(renderer.graphToViewport({ x: cx, y: cy }));
    renderer.getCamera().animate({ x: fg.x, y: fg.y, ratio: 0.4 }, { duration: 600 });
}

// Camera centroid + tight fixed ratio (about 7× zoom from the default ~1.0
// natural fit). Used by the legend popup's "Focus" action.
function zoomToCluster(clusterId) {
    if (!G || !renderer) return;
    const xs = [], ys = [];
    G.forEachNode((id, attrs) => {
        if (attrs.cluster === clusterId && attrs.member) {
            xs.push(attrs.x); ys.push(attrs.y);
        }
    });
    if (!xs.length) return;
    const cx = xs.reduce((a, b) => a + b, 0) / xs.length;
    const cy = ys.reduce((a, b) => a + b, 0) / ys.length;
    const fg = renderer.viewportToFramedGraph(renderer.graphToViewport({ x: cx, y: cy }));
    renderer.getCamera().animate({ x: fg.x, y: fg.y, ratio: 0.15 }, { duration: 600 });
}

function zoomIn() { renderer?.getCamera().animatedZoom({ duration: 200 }); }
function zoomOut() { renderer?.getCamera().animatedUnzoom({ duration: 200 }); }
function resetView() {
    // Match the same comfortable ratio used on first render — animatedReset
    // would snap to ratio 1 and crop nodes/hulls at the edges.
    renderer?.getCamera().animate({ x: 0.5, y: 0.5, ratio: 1.05 }, { duration: 400 });
}

// Compute the sigma container's height so it extends from its current top
// to a small margin from the bottom of the viewport. Idempotent; safe to
// call on mount, resize, or after a result loads.
function updateSigmaHeight() {
    if (!containerRef.value) return;
    const top = containerRef.value.getBoundingClientRect().top;
    const bottomMargin = 24;
    const minHeight = 480;
    const available = window.innerHeight - top - bottomMargin;
    sigmaHeight.value = Math.max(minHeight, Math.floor(available));
}

function destroyRenderer() {
    cancelTween();
    if (renderer) { renderer.kill(); renderer = null; }
    G = null;
    hoveredNode = null;
    hoveredCluster = null;
    matchedNodes.clear();
    searchQuery.value = "";
    hiddenClusters.value = new Set();
    contextMenu.value = null;
    legendMenu.value = null;
}

async function renderGraph(g) {
    const prevPos = pendingPrevPositions;
    pendingPrevPositions = null;
    destroyRenderer();
    if (!g || !containerRef.value) return;
    G = buildGraph(g, prevPos);
    // Snapshot of pre-FA2 positions: these are the tween's START frame, so the
    // user sees the layout reorganize from where it was rather than snap.
    const tweenStart = prevPos ? new Map() : null;
    if (tweenStart) {
        G.forEachNode((id, attrs) => tweenStart.set(id, { x: attrs.x, y: attrs.y }));
    }
    // Standard FA2: strong repulsion + weak gravity spreads the clusters into
    // distinct regions. (LinLog mode was tried and collapses these graphs into a
    // central mass — the senses share too many bridge words to separate under
    // logarithmic attraction.)
    forceAtlas2.assign(G, {
        iterations: 500,
        settings: {
            barnesHutOptimize: G.order > 200,
            barnesHutTheta: 0.5,
            gravity: 0.4,           // gentle gravity → layout has room to breathe
            scalingRatio: 25,       // strong repulsion → meaningful gaps between nodes
            strongGravityMode: false,
            linLogMode: false,
            edgeWeightInfluence: 1,
            slowDown: 5,
            adjustSizes: false,     // noverlap handles overlap as a post-pass
            outboundAttractionDistribution: false,
        },
    });
    // Post-process: nudge any remaining overlaps apart. Preserves cluster structure
    // (only resolves local collisions); much cleaner than relying on FA2 alone.
    noverlap.assign(G, {
        maxIterations: 300,
        settings: { margin: 7, ratio: 1.4, speed: 3, gridSize: 20 },
    });
    // Park each node at its tween-start spot (the previous layout) and stash
    // the post-FA2 position as the target — the rAF loop interpolates between.
    if (tweenStart) {
        G.forEachNode((id, attrs) => {
            G.setNodeAttribute(id, "tx", attrs.x);
            G.setNodeAttribute(id, "ty", attrs.y);
            const s = tweenStart.get(id);
            if (s) {
                G.setNodeAttribute(id, "x", s.x);
                G.setNodeAttribute(id, "y", s.y);
            }
        });
    }
    applyVisibility();
    renderer = new Sigma(G, containerRef.value, {
        renderEdgeLabels: false,
        defaultEdgeColor: "#dddfe4",
        // Screen-px size threshold for showing a node's label. As you zoom in,
        // more nodes cross this threshold — that's the automatic label LOD.
        // (forceLabel-tagged nodes — top-N per cluster — bypass this.)
        labelRenderedSizeThreshold: 8,
        labelFont: "sans-serif",
        labelSize: 12,
        labelWeight: "500",
        labelColor: { color: "#222" },
        zIndex: true,
        // Outline-only nodes: see NodeRingProgram definition near the top.
        defaultNodeType: "ring",
        nodeProgramClasses: { ring: NodeRingProgram },
        // Edges with type:"curve" render via @sigma/edge-curve; intra-cluster
        // edges stay type:"line" so the inside of a cluster reads cleanly.
        edgeProgramClasses: { curve: EdgeCurveProgram },
    });
    setupReducers();
    // Pull the camera back a touch so node radii (in screen px) and hull
    // padding (18px past nodes) don't get clipped at the canvas edges on
    // first render. Sigma 3's default state is {x: 0.5, y: 0.5, ratio: 1}
    // — the graph centre lives at framed (0.5, 0.5), NOT (0, 0).
    renderer.getCamera().setState({ x: 0.5, y: 0.5, ratio: 1.05 });
    renderer.on("enterNode", ({ node }) => {
        hoveredNode = node;
        renderer.refresh();
    });
    renderer.on("leaveNode", () => {
        hoveredNode = null;
        renderer.refresh();
    });
    renderer.on("clickNode", ({ node }) => {
        onViewNode(node);
    });
    renderer.on("rightClickNode", ({ node, event }) => {
        event.original?.preventDefault?.();
        contextMenu.value = { x: event.x, y: event.y, kind: "node", nodeId: node };
    });
    renderer.on("rightClickStage", ({ event }) => {
        event.original?.preventDefault?.();
        contextMenu.value = { x: event.x, y: event.y, kind: "stage" };
    });
    // Any left-click on the stage closes an open context menu (the menu's own
    // buttons stop propagation, so they fire first).
    renderer.on("clickStage", () => { contextMenu.value = null; });
    // Redraw the hull overlay every time sigma paints — it needs to follow
    // camera pan/zoom + reducer-driven node attribute changes.
    renderer.on("afterRender", drawHulls);
    if (tweenStart) animateNodesToTarget();
}

function cancelTween() {
    if (animFrameId != null) {
        cancelAnimationFrame(animFrameId);
        animFrameId = null;
    }
}

// rAF tween from the parked x/y → the stashed tx/ty. Smoothstep easing.
function animateNodesToTarget() {
    cancelTween();
    if (!G || !renderer) return;
    const DURATION = 700;
    const startTime = performance.now();
    const startPos = new Map();
    G.forEachNode((id, attrs) => startPos.set(id, { x: attrs.x, y: attrs.y }));
    const step = (now) => {
        if (!G || !renderer) { animFrameId = null; return; }
        const t = Math.min(1, (now - startTime) / DURATION);
        const e = t * t * (3 - 2 * t);
        G.forEachNode((id, attrs) => {
            const s = startPos.get(id);
            const tx = attrs.tx, ty = attrs.ty;
            if (!s || tx == null || ty == null) return;
            G.setNodeAttribute(id, "x", s.x + (tx - s.x) * e);
            G.setNodeAttribute(id, "y", s.y + (ty - s.y) * e);
        });
        renderer.refresh();
        if (t < 1) {
            animFrameId = requestAnimationFrame(step);
        } else {
            G.forEachNode((id) => {
                G.removeNodeAttribute(id, "tx");
                G.removeNodeAttribute(id, "ty");
            });
            animFrameId = null;
        }
    };
    animFrameId = requestAnimationFrame(step);
}

// Rebuild sigma whenever a new graph arrives.
watch(() => result.value?.graph, async (g) => {
    if (!g) return;
    await nextTick();
    updateSigmaHeight();
    renderGraph(g);
}, { flush: "post" });

onMounted(() => {
    // Async-loaded: parent's nextTick-then-call may miss our mount; self-fetch.
    if (!result.value) runDetection();
    document.addEventListener("click", onDocClickForLegend);
    window.addEventListener("resize", updateSigmaHeight);
    nextTick(updateSigmaHeight);
});

onBeforeUnmount(() => {
    destroyRenderer();
    document.removeEventListener("click", onDocClickForLegend);
    window.removeEventListener("resize", updateSigmaHeight);
    closeLegendMenu();
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
    transition: opacity 0.18s ease, background-color 0.18s ease;
}

.legend-chip.cluster-off {
    opacity: 0.45;
    background-color: #f6f6f6;
}

.legend-chip.cluster-off .legend-text {
    text-decoration: line-through;
}

.legend-chip.cluster-off .legend-swatch {
    opacity: 0.4;
}

.summary-dropdown {
    display: inline-block;
    width: auto;
    min-width: 3.2rem;
    padding: 0.1rem 1.5rem 0.1rem 0.4rem;
    font-weight: 600;
    vertical-align: baseline;
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

/* ---- Sigma container + zoom controls ---- */
.pattern-network {
    width: 100%;
    position: relative;
    /* White bg + rounded corners live HERE (not on sigma-container) so the
       dim-rings overlay at z 0 inside pattern-network can paint visibly. If
       sigma-container had the white bg, its z 1 layer would cover the rings. */
    background: #fff;
    border-radius: 0.375rem;
}

.sigma-container {
    position: relative;
    z-index: 1;       /* above dim-rings overlay (z 0), below hulls (z 3) */
    width: 100%;
    /* height is set inline (see updateSigmaHeight) — computed to fit the
       viewport. min-height kicks in if measurement fails / very tall chrome. */
    min-height: 480px;
    cursor: default;
}

/* Dim node rings overlay — drawn UNDERNEATH the sigma canvas so node labels
   (rendered by sigma) stay on top of any ring that passes behind them. */
.dim-rings-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 0;
}
/* Inset shadow is drawn on top of the sigma WebGL canvas via a pseudo-element
   on the parent — putting it on .sigma-container itself doesn't work because
   the canvas covers the parent's bg painting (where inset shadow lives). */
.pattern-network::after {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 0.375rem;
    pointer-events: none;
    z-index: 3;
    box-shadow:
        inset 0 0 0 1px #d8dee8,
        inset 0 4px 10px rgba(0, 0, 0, 0.10),
        inset 0 -1px 3px rgba(0, 0, 0, 0.04);
}

/* Cluster hulls overlay — soft translucent shapes drawn on top of sigma at
   low alpha, so node colours show through. Doesn't intercept mouse events. */
.hulls-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 3;
}

.viz-loading-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255, 255, 255, 0.65);
    z-index: 4;
    pointer-events: all;
}

.ctx-menu {
    position: absolute;
    z-index: 5;
    background: #fff;
    border: 1px solid rgba(0, 0, 0, 0.12);
    border-radius: 0.25rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    min-width: 11rem;
    padding: 0.25rem 0;
}

.ctx-item {
    display: block;
    width: 100%;
    text-align: left;
    border: 0;
    background: none;
    padding: 0.35rem 0.75rem;
    font-size: 0.85rem;
    color: #222;
    cursor: pointer;
}

.ctx-item:hover:not(:disabled) {
    background: #f0f0f0;
}

.ctx-item:disabled {
    color: #aaa;
    cursor: default;
}

.ctx-sep {
    height: 1px;
    background: rgba(0, 0, 0, 0.08);
    margin: 0.25rem 0;
}

.legend-popup {
    min-width: 13rem;
    z-index: 1040;
}

.legend-popup-header {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.4rem 0.75rem 0.25rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: #333;
}

.legend-popup-label {
    word-break: break-word;
}


.zoom-controls {
    position: absolute;
    top: 6px;
    left: 6px;
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
