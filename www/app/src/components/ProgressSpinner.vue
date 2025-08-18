<template>
    <div class="spinner-container d-inline-block px-1">
        <div class="spinner-border text-secondary" role="status"
            :class="{ 'spinner-border-sm': sm, 'spinner-large': lg }"><span class="visually-hidden">{{
                $t("common.loading") }}...</span>
        </div>
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
</style>