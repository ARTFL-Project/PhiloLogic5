<template>
    <div class="modal-dialog" role="document">
        <div class="modal-content" role="dialog" aria-labelledby="export-modal-title"
            aria-describedby="export-modal-description">

            <div class="modal-header">
                <h5 class="modal-title" id="export-modal-title">
                    <i class="bi bi-download me-2"></i>
                    {{ $t("resultsSummary.exportResults") }}
                </h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal" :aria-label="$t('common.closeModal')">
                </button>
            </div>

            <div class="modal-body">
                <div class="alert alert-info mb-3" role="alert"
                    v-if="report == 'concordance' || report == 'kwic' || report == 'bibliography'">
                    <i class="bi bi-info-circle me-2"></i>
                    <span id="export-modal-description">{{ $t("exportResults.currentPage") }}</span>
                </div>

                <!-- HTML Export Section -->
                <div v-if="report == 'concordance' || report == 'kwic'" class="export-section mb-4">
                    <h6 class="section-header">
                        <i class="bi bi-code-slash me-2"></i>
                        {{ $t("exportResults.html") }}
                    </h6>
                    <p class="text-muted small mb-2">{{ $t("exportResults.htmlDescription") }}</p>
                    <div class="btn-group w-100" role="group">
                        <button type="button" class="btn btn-outline-secondary export-btn"
                            @click="getResults('json', false)"
                            :aria-label="`${$t('exportResults.exportAs')} JSON ${$t('exportResults.html')}`">
                            <i class="bi bi-filetype-json me-2"></i>
                            JSON
                        </button>
                        <button type="button" class="btn btn-outline-secondary export-btn"
                            @click="getResults('csv', false)"
                            :aria-label="`${$t('exportResults.exportAs')} CSV ${$t('exportResults.html')}`">
                            <i class="bi bi-filetype-csv me-2"></i>
                            CSV
                        </button>
                    </div>
                </div>

                <!-- Plain Text Export Section -->
                <div class="export-section">
                    <h6 class="section-header" v-if="report == 'concordance' || report == 'kwic'">
                        <i class="bi bi-file-text me-2"></i>
                        {{ $t("exportResults.plain") }}
                    </h6>
                    <p class="text-muted small mb-2" v-if="report == 'concordance' || report == 'kwic'">
                        {{ $t("exportResults.plainDescription") }}
                    </p>
                    <div class="btn-group w-100" role="group">
                        <button type="button" class="btn btn-outline-secondary export-btn"
                            @click="getResults('json', true)"
                            :aria-label="`${$t('exportResults.exportAs')} JSON ${$t('exportResults.plain')}`">
                            <i class="bi bi-filetype-json me-2"></i>
                            JSON
                        </button>
                        <button type="button" class="btn btn-outline-secondary export-btn"
                            @click="getResults('csv', true)"
                            :aria-label="`${$t('exportResults.exportAs')} CSV ${$t('exportResults.plain')}`">
                            <i class="bi bi-filetype-csv me-2"></i>
                            CSV
                        </button>
                    </div>
                </div>
            </div>

            <div class="modal-footer">
                <small class="text-muted">
                    <i class="bi bi-info-circle me-1"></i>
                    {{ $t("exportResults.downloadNote") }}
                </small>
            </div>
        </div>
    </div>
</template>

<script>
import { mapFields } from "vuex-map-fields";

export default {
    name: "ExportResults",
    computed: {
        ...mapFields(["formData.report"]),
    },
    inject: ["$http"],
    methods: {
        getResults(format, html) {
            // Add loading state to button
            const buttons = document.querySelectorAll('.export-btn');
            buttons.forEach(btn => {
                btn.disabled = true;
                btn.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status"></span>Exporting...`;
            });

            this.$http
                .get(
                    `${this.$dbUrl}/scripts/export_results.py?${this.paramsToUrlString({
                        ...this.$store.state.formData,
                        filter_html: html.toString(),
                        output_format: format,
                        report: "",
                    })}&report=${this.report}`
                )
                .then((response) => {
                    let text = "";
                    let element = document.createElement("a");
                    let filename = `${this.paramsToUrlString({ ...this.$store.state.formData })}.${format}`;
                    if (format == "json") {
                        text = JSON.stringify(response.data);
                    } else if (format == "csv") {
                        text = response.data;
                    }
                    element.setAttribute("href", "data:text/plain;charset=utf-8," + encodeURIComponent(text));
                    element.setAttribute("download", filename);
                    element.style.display = "none";
                    document.body.appendChild(element);
                    element.click();
                    document.body.removeChild(element);
                })
                .catch((error) => {
                    console.error('Export failed:', error);
                })
                .finally(() => {
                    // Reset button states
                    setTimeout(() => {
                        location.reload(); // Simple way to reset modal state
                    }, 1000);
                });
        },
    },
};
</script>

<style scoped>
.export-section {
    padding: 1rem;
    border: 1px solid #dee2e6;
    border-radius: 0.375rem;
    background-color: #f8f9fa;
}

.section-header {
    color: #495057;
    margin-bottom: 0.5rem;
    font-weight: 600;
}

.export-btn {
    padding: 0.75rem 1rem;
    font-weight: 500;
    transition: all 0.2s ease-in-out;
}

.export-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.export-btn:active {
    transform: translateY(0);
}

.export-btn i {
    opacity: 0.8;
}

.modal-header {
    background-color: #f8f9fa;
    border-bottom: 2px solid #dee2e6;
}

.modal-title i {
    color: #6c757d;
}

.modal-footer {
    background-color: #f8f9fa;
    border-top: 1px solid #dee2e6;
}

.alert-info {
    border-left: 4px solid #0dcaf0;
}
</style>