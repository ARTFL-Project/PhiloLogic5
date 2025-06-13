<template>
    <div style="margin-top: 0.2rem">
        <div class="dropdown">
            <button class="btn btn-sm btn-outline-secondary dropdown-toggle" type="button" id="language-selector"
                data-bs-toggle="dropdown" aria-expanded="false" :aria-label="$t('common.changeLanguage')">
                <img src="../assets/language.png" height="18" width="18" :alt="$t('common.languageIcon')" />
            </button>
            <ul class="dropdown-menu" aria-labelledby="language-selector">
                <li v-for="locale in $i18n.availableLocales" :key="`locale-${locale}`">
                    <button type="button" class="dropdown-item" @click="changeLocale(locale)">
                        {{ localNames[locale] }}
                    </button>
                </li>
            </ul>
        </div>
    </div>
</template>
<script>
export default {
    name: "locale-changer",
    data() {
        return {
            localNames: {
                en: "English",
                fr: "Français",
            },
        };
    },
    methods: {
        changeLocale(locale) {
            localStorage.setItem("lang", locale);
            this.$i18n.locale = locale;
        },
    },
};
</script>
<style lang="scss" scoped>
@import "../assets/styles/theme.module.scss";

button {
    font-size: inherit !important;
    border-width: 0 !important;
}


.dropdown-menu {
    left: initial !important;
    right: 0;
    min-width: initial;
    padding: 0;
}

.dropdown-item {
    font-size: 80% !important;
    font-variant: small-caps;
    padding-left: 0.5rem;
    padding-right: 0.5rem;
    border: none;
    background: none;
    width: 100%;
    text-align: left;
}

.dropdown-item:hover {
    color: #fff !important;
    background: $link-color !important;
    cursor: pointer;
}

.dropdown-toggle::after {
    color: #fff;
}
</style>