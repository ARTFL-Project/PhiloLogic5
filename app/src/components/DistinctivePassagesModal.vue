<template>
    <div class="modal fade" tabindex="-1" id="distinctive-passages-modal" aria-hidden="true"
        aria-labelledby="distinctive-passages-modal-title">
        <div class="modal-dialog modal-xl modal-dialog-scrollable">
            <div class="modal-content">
                <div class="card-header text-center position-relative py-2">
                    <h2 class="modal-title mb-0 h6" id="distinctive-passages-modal-title"
                        style="font-variant: small-caps; font-size: 1.2rem; font-weight: 700;">
                        {{ groupName
                            ? $t('distinctivePassages.titleFor', { group: groupName })
                            : $t('distinctivePassages.title') }}
                    </h2>
                    <button type="button" class="btn btn-secondary btn-sm close-box"
                        data-bs-dismiss="modal" :aria-label="$t('common.closeModal')"
                        @click="$event.target.blur()"
                        style="position: absolute; top: 1px; right: 0;">
                        <span class="icon-x"></span>
                    </button>
                </div>

                <div class="modal-body">
                    <div v-if="signature && signature.length > 0" class="signature-strip mb-2">
                        <strong>{{ $t('distinctivePassages.signature') }}:</strong>
                        <span v-for="s in signature" :key="s.word" class="sig-chip">{{ s.word }}</span>
                    </div>

                    <p v-if="signature && signature.length > 0" class="ordering-note mb-3">
                        {{ $t('distinctivePassages.orderingNote') }}
                    </p>

                    <div v-if="loading && passages.length === 0" class="text-center py-5">
                        <progress-spinner :lg="true" />
                    </div>

                    <div v-else-if="passages.length > 0" class="passage-list">
                        <article v-for="(p, i) in passages" :key="`${p.philo_id.join('-')}-${i}`"
                            class="card philologic-occurrence text-view mb-3 shadow-sm passage-card">
                            <div class="row citation-container g-0">
                                <div class="col">
                                    <span class="cite">
                                        <span class="number">{{ i + 1 }}</span>
                                        <citations :citation="p.citation" :result-number="i + 1"></citations>
                                    </span>
                                </div>
                            </div>
                            <div class="row">
                                <div class="col m-3 concordance-text passage-context" v-html="p.context"></div>
                            </div>
                        </article>
                    </div>

                    <div v-else class="text-center py-4 text-muted">
                        {{ $t('distinctivePassages.noResults') }}
                    </div>

                    <div v-if="hasMore" class="text-center mt-2">
                        <button type="button" class="btn btn-outline-secondary"
                            :disabled="loading" @click="$emit('load-more')">
                            <span v-if="loading">…</span>
                            <span v-else>{{ $t('distinctivePassages.loadMore') }}</span>
                        </button>
                    </div>
                </div>

                <div v-if="viewAllUrl" class="modal-footer justify-content-center">
                    <button type="button" class="btn btn-link"
                        @click="$emit('view-all')">
                        {{ groupName
                            ? $t('distinctivePassages.viewAllFor', { group: groupName })
                            : $t('distinctivePassages.viewAll') }}
                    </button>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import Citations from "./Citations";  // eslint-disable-line no-unused-vars
import ProgressSpinner from "./ProgressSpinner";  // eslint-disable-line no-unused-vars

defineProps({
    groupName: { type: String, default: "" },
    signature: { type: Array, default: () => [] },
    passages: { type: Array, default: () => [] },
    loading: { type: Boolean, default: false },
    hasMore: { type: Boolean, default: false },
    viewAllUrl: { type: [String, Object], default: "" },
});

defineEmits(["load-more", "view-all"]);
</script>

<style lang="scss" scoped>
@use "../assets/styles/theme.module.scss" as theme;

.signature-strip {
    font-size: 0.9rem;
    color: #495057;
    line-height: 1.8;
}

.sig-chip {
    display: inline-block;
    padding: 0.1rem 0.5rem;
    margin-left: 0.25rem;
    background-color: rgba(theme.$link-color, 0.1);
    border: 1px solid rgba(theme.$link-color, 0.25);
    border-radius: 0.5rem;
    color: theme.$link-color;
    font-size: 0.85rem;
}

.ordering-note {
    font-size: 0.85rem;
    color: #6c757d;
    font-style: italic;
}

.passage-card .passage-context {
    line-height: 1.55;
}

.passage-context :deep(.colloc-explainer) {
    background-color: rgba(theme.$passage-color, 0.07);
    color: theme.$passage-color;
    font-weight: 500;
    padding: 0.05em 0.15em;
    border-radius: 0.15em;
}

.close-box .icon-x {
    background-color: #fff !important;
}
</style>
