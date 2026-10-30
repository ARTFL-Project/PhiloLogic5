<template>
    <footer class="footer-bar">
        <div class="d-flex flex-column flex-sm-row justify-content-sm-between align-items-center gap-2">
            <span class="d-flex align-items-center gap-2">
                <span>{{ $t("common.poweredBy") }}</span>
                <a href="https://artfl-project.uchicago.edu/" :title="$t('common.philoTitle')"
                    :aria-label="$t('common.philoLinkLabel')">
                    <img src="../assets/philo-white.png" :alt="$t('common.philoAlt')" height="32" width="88" />
                </a>
            </span>
            <!-- A broken web_config.cfg sends no link -->
            <a :href="philoConfig.accessibility_link" v-if="philoConfig.accessibility_link">
                <i class="bi bi-universal-access me-1" aria-hidden="true"></i>{{ $t("common.accessibility") }}
            </a>
        </div>
    </footer>
</template>

<script setup>
import { inject } from "vue";

const philoConfig = inject("$philoConfig");
</script>

<style lang="scss" scoped>
@use "../assets/styles/theme.module.scss" as theme;

.footer-bar {
    // at the bottom of the window on pages shorter than it (#app is a column)
    margin-top: auto;
    padding: 0.5rem 1.25rem;
}

a {
    text-decoration: none;
}

a:hover,
a:focus {
    background-color: rgba(255, 255, 255, 0.12);
    box-shadow: none;
}

a:hover {
    text-decoration: underline;
}

a:focus-visible {
    outline: 2px solid theme.$focus-on-dark-color;
    outline-offset: 2px;
}
</style>
