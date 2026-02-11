<template>
    <div class="spinner-container d-inline-block px-1">
        <div class="spinner-border progress-spinner"
            :class="{ 'spinner-border-sm': sm, 'spinner-large': lg, 'spinner-xl': xl }">
        </div>
        <div role="status" aria-live="polite" aria-atomic="true" class="visually-hidden">{{ statusMessage }}</div>
        <div class="spinner-text" v-if="progress > 0" :class="{ 'spinner-text-large': lg }">
            {{ progress }}%
        </div>
        <div class="spinner-text" v-else-if="text" :class="{ 'spinner-text-large': lg }">
            {{ text }}
        </div>
    </div>
</template>
<script>
export default {
    name: 'ProgressSpinner',
    data() {
        return {
            statusMessage: '',
            announceTimeout: null
        }
    },
    mounted() {
        this.announceTimeout = setTimeout(() => {
            this.statusMessage = (this.message || this.$t("common.loading")) + '...';
        }, 100);
    },
    beforeUnmount() {
        if (this.announceTimeout) {
            clearTimeout(this.announceTimeout);
        }
    },
    props: {
        progress: {
            default: 0
        },
        text: {
            type: String,
            default: ''
        },
        sm: {
            type: Boolean,
            default: false
        },
        lg: {
            type: Boolean,
            default: false
        },
        xl: {
            type: Boolean,
            default: false
        },
        message: {
            type: String,
            default: ''
        }
    },
}
</script>

<style lang="scss" scoped>
@use "../assets/styles/theme.module.scss" as theme;

.spinner-container {
    position: relative;
}

.spinner-text {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: .75rem;
    color: theme.$link-color;
}

.spinner-text-large {
    font-size: 1.25rem;
}

.spinner-large {
    width: 4rem;
    height: 4rem;
    border-width: .25em;
}

.spinner-xl {
    width: 8rem;
    height: 8rem;
    border-width: .35em;
}

.progress-spinner {
    color: theme.$link-color;
}
</style>