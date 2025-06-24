<template>
    <cite class="philologic-cite ps-2" role="group" :aria-label="$t('citations.citationGroup')" tabindex="0">
        <span class="citation text-view" v-for="(cite, citeIndex) in citation" :key="citeIndex">
            <span v-html="cite.prefix" v-if="cite.prefix"></span>

            <router-link :to="cite.href" :style="cite.style" v-if="cite.href"
                :aria-label="`${$t('citations.viewText')}: ${cite.label}`">
                {{ cite.label }}
            </router-link>
            <span :style="cite.style" v-else>{{ cite.label }}</span>

            <span v-html="cite.suffix" v-if="cite.suffix"></span>

            <!-- Hide separators from screen readers -->
            <span v-if="citeIndex != citation.length - 1" aria-hidden="true">
                <span class="separator px-2" v-if="!separator">&#9679;</span>
                <span v-else>{{ separator }}</span>
            </span>
        </span>
    </cite>
</template>
<script>
export default {
    name: "citations-generator",
    props: ["citation", "separator"],
};
</script>
<style scoped lang="scss">
@import "../assets/styles/theme.module.scss";

cite {
    font-style: normal;
}

.separator {
    font-size: 0.75rem;
    vertical-align: 0.05rem;
}

.citation {
    font-weight: 600;
}

.philologic-cite a:focus {
    outline: 2px solid $link-color;
    outline-offset: 2px;
    border-radius: 2px;
}
</style>
