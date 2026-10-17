<template>
    <!-- The sections of a table of contents, each with the sections within it: what an entry shows is the slot's -->
    <ul :class="child ? 'toc-children' : 'toc-tree'">
        <li v-for="(element, index) in nodes" :key="index"
            :class="`toc-item${child ? ' toc-child' : ''} toc-${element.philo_type}`">
            <div class="toc-content-wrapper">
                <span v-if="!child && element.philo_type === 'div1'" class="div1-marker" aria-hidden="true">※</span>
                <span v-else :class="'bullet-point-' + element.philo_type" aria-hidden="true"></span>
                <slot :element="element"></slot>
            </div>
            <TocTree v-if="element.children && element.children.length > 0" :nodes="element.children" child>
                <template #default="{ element: section }">
                    <slot :element="section"></slot>
                </template>
            </TocTree>
        </li>
    </ul>
</template>

<script setup>
defineProps({
    // The sections, with their children, as buildTocTree makes them
    nodes: { type: Array, required: true },
    // The sections within one: not the top of the tree
    child: { type: Boolean, default: false },
});
</script>
