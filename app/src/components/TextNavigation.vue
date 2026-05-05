<template>
    <div class="container-fluid mt-4">
        <header class="row">
            <div class="col-8 offset-2">
                <div id="object-title" class="text-center pt-4">
                    <h1 class="h5">
                        <citations :citation="textNavigationCitation"></citations>
                    </h1>
                </div>
            </div>
        </header>
        <section class="row text-center mt-4" id="toc-wrapper" v-if="navBar === true || loading === false"
            :aria-label="$t('textNav.navigationControls')">
            <div id="toc-top-bar">
                <div id="nav-buttons" v-scroll="handleScroll">
                    <button type="button" class="btn btn-secondary btn-sm" id="back-to-top" @click="backToTop()"
                        :aria-label="$t('textNav.backToTop')" v-show="navBarVisible">
                        <span class="d-none d-sm-inline-block">{{ $t("textNav.backToTop") }}</span>
                        <span class="d-inline-block d-sm-none">{{ $t("textNav.top") }}</span>
                    </button>
                    <div class="btn-group btn-group-sm" role="group" :aria-label="$t('textNav.navigationControls')">
                        <button type="button" class="btn btn-secondary" v-if="textObject.prev" id="prev-obj"
                            @click="goToTextObject(textObject.prev)" :aria-label="$t('textNav.previousSection')">
                            <span aria-hidden="true">&lt;</span>
                            <span class="visually-hidden">{{ $t('textNav.previous') }}</span>
                        </button>
                        <button type="button" class="btn btn-secondary" id="show-toc" @click="toggleTableOfContents()"
                            :aria-expanded="tocOpen" :aria-controls="tocOpen ? 'toc-content' : null"
                            :aria-label="$t('textNav.toggleToc')">
                            {{ $t("textNav.toc") }}
                        </button>
                        <button type="button" class="btn btn-secondary" v-if="textObject.next" id="next-obj"
                            @click="goToTextObject(textObject.next)" :aria-label="$t('textNav.nextSection')">
                            <span aria-hidden="true">&gt;</span>
                            <span class="visually-hidden">{{ $t('textNav.next') }}</span>
                        </button>
                    </div>
                    <a id="report-error" class="btn btn-secondary btn-sm position-absolute" target="_blank"
                        rel="noopener noreferrer" :href="philoConfig.report_error_link"
                        v-if="philoConfig.report_error_link != ''" :aria-label="$t('common.reportError')">{{
                            $t("common.reportError")
                        }}</a>
                </div>
                <nav id="toc" role="navigation" :aria-label="$t('textNav.tableOfContents')">
                    <transition name="slide-fade">
                        <div class="card py-3 shadow" id="toc-content" :style="tocHeight" v-if="tocOpen" role="region"
                            :aria-label="$t('textNav.tocContent')">
                            <ul class="toc-tree">
                                <li v-for="(element, elIndex) in processedTocElements" :key="elIndex"
                                    :class="'toc-item toc-' + element.philo_type">

                                    <!-- Section content -->
                                    <div class="toc-content-wrapper">
                                        <span v-if="element.philo_type === 'div1'" class="div1-marker"
                                            aria-hidden="true">※</span>
                                        <span v-else :class="'bullet-point-' + element.philo_type"
                                            aria-hidden="true"></span>
                                        <button type="button"
                                            :class="{ 'current-obj': element.philo_id === currentPhiloId }"
                                            class="btn btn-link toc-link"
                                            @click="textObjectSelection(element.philo_id, elIndex, $event)"
                                            :aria-current="element.philo_id === currentPhiloId ? 'page' : null">
                                            {{ element.label }}
                                        </button>
                                    </div>

                                    <!-- Child sections -->
                                    <ul v-if="element.children && element.children.length > 0" class="toc-children">
                                        <li v-for="(child, childIndex) in element.children" :key="childIndex"
                                            :class="'toc-item toc-child toc-' + child.philo_type">
                                            <div class="toc-content-wrapper">
                                                <span :class="'bullet-point-' + child.philo_type"
                                                    aria-hidden="true"></span>
                                                <button type="button"
                                                    :class="{ 'current-obj': child.philo_id === currentPhiloId }"
                                                    class="btn btn-link toc-link"
                                                    @click="textObjectSelection(child.philo_id, childIndex, $event)"
                                                    :aria-current="child.philo_id === currentPhiloId ? 'page' : null">
                                                    {{ child.label }}
                                                </button>
                                            </div>

                                            <!-- Grandchildren (div3 level) -->
                                            <ul v-if="child.children && child.children.length > 0" class="toc-children">
                                                <li v-for="(grandchild, grandchildIndex) in child.children"
                                                    :key="grandchildIndex"
                                                    :class="'toc-item toc-child toc-' + grandchild.philo_type">
                                                    <div class="toc-content-wrapper">
                                                        <span :class="'bullet-point-' + grandchild.philo_type"
                                                            aria-hidden="true"></span>
                                                        <button type="button"
                                                            :class="{ 'current-obj': grandchild.philo_id === currentPhiloId }"
                                                            class="btn btn-link toc-link"
                                                            @click="textObjectSelection(grandchild.philo_id, grandchildIndex, $event)"
                                                            :aria-current="grandchild.philo_id === currentPhiloId ? 'page' : null">
                                                            {{ grandchild.label }}
                                                        </button>
                                                    </div>
                                                </li>
                                            </ul>
                                        </li>
                                    </ul>
                                </li>
                            </ul>
                        </div>
                    </transition>
                </nav>
            </div>
        </section>
        <div class="text-center" style="font-size: 85%" v-if="philoConfig.dictionary_lookup.url_root != ''"
            :aria-label="$t('textNav.dictionaryLookup')">
            <p>{{ $t("textNav.dicoLookUp") }}.</p>
        </div>
        <div role="region" class="row" id="all-content" :aria-label="$t('textNav.textContent')">
            <div class="col-12 col-sm-10 offset-sm-1 col-lg-8 offset-lg-2" id="center-content" v-if="textObject.text"
                style="text-align: center">
                <article class="card text-view mt-2 mb-4 p-4 shadow d-inline-block">
                    <div id="book-page">
                        <section id="previous-pages" v-if="beforeObjImgs"
                            :aria-label="$t('textNav.previousPageImages')">
                            <span class="xml-pb-image">
                                <a :href="img[0]" :large-img="img[1]" class="page-image-link"
                                    v-for="(img, imageIndex) in beforeObjImgs" :key="imageIndex" data-gallery
                                    :aria-label="$t('textNav.viewPageImage', { number: imageIndex + 1 })"></a>
                            </span>
                        </section>
                        <section id="previous-graphics" v-if="beforeGraphicsImgs"
                            :aria-label="$t('textNav.previousGraphics')">
                            <a :href="img[0]" :large-img="img[1]" class="d-none inline-img"
                                v-for="(img, beforeIndex) in beforeGraphicsImgs" :key="beforeIndex" data-gallery
                                :aria-label="$t('textNav.viewGraphic', { number: beforeIndex + 1 })"></a>
                        </section>
                        <section id="text-obj-content" class="text-content-area" v-html="textObject.text"
                            :philo-id="philoID" @keydown="dicoLookup($event)" tabindex="0" role="document"
                            :aria-label="$t('textNav.mainTextContent')"></section>
                        <section id="next-pages" v-if="afterObjImgs" :aria-label="$t('textNav.nextPageImages')">
                            <span class="xml-pb-image">
                                <a :href="img[0]" :large-img="img[1]" class="page-image-link"
                                    v-for="(img, afterIndex) in afterObjImgs" :key="afterIndex" data-gallery
                                    :aria-label="$t('textNav.viewPageImage', { number: afterIndex + 1 })"></a>
                            </span>
                        </section>
                        <section id="next-graphics" v-if="afterGraphicsImgs" :aria-label="$t('textNav.nextGraphics')">
                            <a :href="img[0]" :large-img="img[1]" class="inline-img"
                                v-for="(img, afterGraphIndex) in afterGraphicsImgs" :key="afterGraphIndex" data-gallery
                                :aria-label="$t('textNav.viewGraphic', { number: afterGraphIndex + 1 })"></a>
                        </section>
                    </div>
                </article>
            </div>
        </div>
        <div id="gallery-template" aria-hidden="true">
        </div>
    </div>
</template>
<script setup>
import { Popover } from "bootstrap";
import GLightbox from "glightbox";
import "glightbox/dist/css/glightbox.css";
import { storeToRefs } from "pinia";
import { computed, inject, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import vueScrollTo from "vue-scrollto";
import { useMainStore } from "../stores/main";
import { buildTocTree, debug, deepEqual, paramsToUrlString } from "../utils.js";
import citations from "./Citations";

//  Injects, route, router, store
const $http = inject("$http");
const $dbUrl = inject("$dbUrl");
const philoConfig = inject("$philoConfig");
const route = useRoute();
const router = useRouter();
const store = useMainStore();
const {
    formData,
    textNavigationCitation,
    navBar,
    tocElements,
    byte,
    searching,
} = storeToRefs(store);

const logError = (error) => debug({ $options: { name: "textNavigation" } }, error);

//  Reactive state ─
const textObject = ref({});
const beforeObjImgs = ref([]);
const afterObjImgs = ref([]);
const beforeGraphicsImgs = ref([]);
const afterGraphicsImgs = ref([]);
const tocOpen = ref(false);
const philoID = ref("");
const currentPhiloId = ref("");
const highlight = ref(false);
const start = ref(0);
const end = ref(0);
const tocPosition = ref(0);
const navButtonPosition = ref(0);
const navBarVisible = ref(false);
const gallery = ref(null);
const images = ref([]);
const loading = ref(false);
const textObjectURL = ref("");

//  Computed ─
const processedTocElements = computed(() =>
    buildTocTree(tocElements.value.elements.slice(start.value, end.value))
);
const tocHeight = computed(() => `max-height: ${window.innerHeight - 200}`);
const whiteSpace = computed(() =>
    philoConfig.respect_text_line_breaks ? "pre" : "normal"
);

//  Image link handling
function buildImageBucket(allImgs, currentObjImgs, before, after) {
    if (currentObjImgs.length === 0) return;
    const root = philoConfig.page_images_url_root;
    let beforeIndex = 0;
    for (let i = 0; i < allImgs.length; i++) {
        const img = allImgs[i];
        if (currentObjImgs.indexOf(img[0]) === -1) {
            const second = img.length === 2 ? img[1] : img[0];
            before.push([`${root}/${img[0]}`, `${root}/${second}`]);
        } else {
            beforeIndex = i;
            break;
        }
    }
    for (let i = beforeIndex; i < allImgs.length; i++) {
        const img = allImgs[i];
        if (currentObjImgs.indexOf(img[0]) === -1) {
            const second = img.length === 2 ? img[1] : img[0];
            after.push([`${root}/${img[0]}`, `${root}/${second}`]);
        }
    }
}

function insertPageLinks(imgObj) {
    beforeObjImgs.value = [];
    afterObjImgs.value = [];
    buildImageBucket(imgObj.all_imgs, imgObj.current_obj_img, beforeObjImgs.value, afterObjImgs.value);
}

function insertInlineImgs(imgObj) {
    beforeGraphicsImgs.value = [];
    afterGraphicsImgs.value = [];
    buildImageBucket(imgObj.graphics, imgObj.current_graphic_img, beforeGraphicsImgs.value, afterGraphicsImgs.value);
}

//  Post-render setup helpers (extracted from the fetchText().then() block)
function setUpNotePopovers() {
    const notes = document.getElementsByClassName("note");
    if (notes.length === 0) return;
    Array.from(notes).forEach((note, index) => {
        const noteContent = note.nextElementSibling?.innerHTML;
        if (!noteContent) return;
        const noteId = `note-content-${index}`;
        note.setAttribute("role", "button");
        note.setAttribute("tabindex", "0");
        note.setAttribute("aria-label", `Note ${note.textContent || index + 1}`);
        note.setAttribute("aria-describedby", noteId);

        const popoverInstance = new Popover(note, {
            html: true,
            content: noteContent,
            trigger: "focus",
            customClass: "custom-popover shadow-lg",
        });
        note.addEventListener("shown.bs.popover", () => {
            const popoverElement = document.querySelector(".popover");
            if (popoverElement) {
                popoverElement.setAttribute("id", noteId);
                popoverElement.setAttribute("role", "tooltip");
            }
        });
        note.addEventListener("keydown", (e) => {
            if (e.key === "Escape") {
                popoverInstance.hide();
                note.focus();
            }
        });
    });
}

function setUpNoteRefs() {
    const noteRefs = document.getElementsByClassName("note-ref");
    if (noteRefs.length === 0) return;
    Array.from(noteRefs).forEach((noteRef, index) => {
        const noteRefId = `note-ref-content-${index}`;
        noteRef.setAttribute("role", "button");
        noteRef.setAttribute("tabindex", "0");
        noteRef.setAttribute("aria-label", `Note reference ${noteRef.textContent || index + 1}`);
        noteRef.setAttribute("aria-describedby", noteRefId);

        // Lazy: fetch the note's content on first focus only. The focus event
        // that triggered this fetch already fired before the Popover existed,
        // so we manually call .show() once it's constructed; subsequent focus
        // cycles are handled by Bootstrap's trigger:"focus".
        let initialized = false;
        const initializePopover = () => {
            if (initialized) return;
            initialized = true;
            $http.get(`${$dbUrl}/scripts/get_notes.py?`, {
                params: {
                    target: noteRef.getAttribute("target"),
                    philo_id: route.params.pathInfo.split("/").join(" "),
                },
            }).then((response) => {
                const popoverInstance = new Popover(noteRef, {
                    html: true,
                    content: response.data.text,
                    trigger: "focus",
                    customClass: "custom-popover shadow-lg",
                });
                noteRef.addEventListener("shown.bs.popover", () => {
                    const popoverElement = document.querySelector(".popover");
                    if (popoverElement) {
                        popoverElement.setAttribute("id", noteRefId);
                        popoverElement.setAttribute("role", "tooltip");
                    }
                });
                noteRef.addEventListener("keydown", (e) => {
                    if (e.key === "Escape") {
                        popoverInstance.hide();
                        noteRef.focus();
                    }
                });
                // Show now if the user is still on this element; otherwise let
                // the next focus event trigger it normally.
                if (document.activeElement === noteRef) {
                    popoverInstance.show();
                }
            });
        };
        noteRef.addEventListener("focus", initializePopover);
    });
}

function setUpHeadingSemantics() {
    const headwords = document.querySelectorAll(".philologic-fragment .headword");
    headwords.forEach((headword) => {
        let current = headword.parentElement;
        let headingDepth = 0;
        // Count parent divs that contain a .headword child (other than ours)
        while (current && current.className !== "philologic-fragment") {
            if (current.tagName.toLowerCase() === "div") {
                const divHeadword = current.querySelector(":scope > .headword");
                if (divHeadword && divHeadword !== headword) {
                    headingDepth++;
                }
            }
            current = current.parentElement;
        }
        headword.setAttribute("role", "heading");
        headword.setAttribute("aria-level", headingDepth + 2);
    });
}

function setUpInternalLinks() {
    Array.from(document.getElementsByClassName("link-back")).forEach((el) => {
        const goToNote = () => {
            const link = el.getAttribute("link");
            router.push(link);
            el.removeEventListener("click", goToNote);
        };
        el.addEventListener("click", goToNote);
    });

    Array.from(document.querySelectorAll("a[type='search']")).forEach((el) => {
        const [metadata, metadataValue] = el.getAttribute("target").split(":");
        el.href = `${$dbUrl.replace(/\/+$/, "")}/bibliography?${metadata}=${metadataValue}`;
    });
}

function scrollToTarget() {
    if (byte.value !== "") {
        const element = document.getElementsByClassName("highlight")[0];
        if (!element) return;
        const parent = element.parentElement;
        if (parent.classList.contains("note-content")) {
            const note = parent.previousSibling;
            vueScrollTo.scrollTo(note, 250, {
                easing: "ease-out",
                offset: -150,
                onDone: () => setTimeout(() => note.focus(), 500),
            });
        } else {
            vueScrollTo.scrollTo(element, 250, { easing: "ease-out", offset: -150 });
        }
    } else if (formData.value.start_byte !== "") {
        vueScrollTo.scrollTo(document.querySelector(".passage-marker"), 250, {
            easing: "ease-out",
            offset: -150,
        });
    } else if (route.hash) {
        const note = document.getElementById(route.hash.slice(1));
        if (!note) return;
        vueScrollTo.scrollTo(note, 250, {
            easing: "ease-out",
            offset: -250,
            onDone: () => setTimeout(() => note.focus(), 500),
        });
    }
}

//  Text fetch ─
function fetchText() {
    searching.value = true;
    textObjectURL.value = route.params;
    philoID.value = textObjectURL.value.pathInfo.split("/").join(" ");

    // Block doc-level rendering — redirect to table of contents
    const nonZeroParts = philoID.value.split(" ").filter((n) => n !== "0");
    if (nonZeroParts.length <= 1) {
        const docId = philoID.value.split(" ")[0];
        router.replace(`/navigate/${docId}/table-of-contents`);
        return;
    }

    let byteString = "";
    if ("byte" in route.query) {
        byte.value = route.query.byte;
        byteString = typeof route.query.byte === "object"
            ? `byte=${byte.value.join("&byte=")}`
            : `byte=${byte.value}`;
    } else {
        byte.value = "";
    }

    const navigationParams = { report: "navigation", philo_id: philoID.value };
    if (formData.value.start_byte !== "") {
        navigationParams.start_byte = formData.value.start_byte;
        navigationParams.end_byte = formData.value.end_byte;
    }
    const urlQuery = `${byteString}&${paramsToUrlString(navigationParams)}`;

    $http.get(`${$dbUrl}/reports/navigation.py?${urlQuery}`)
        .then((response) => {
            textObject.value = response.data;
            textNavigationCitation.value = response.data.citation;
            navBar.value = true;
            highlight.value = byte.value.length > 0;

            if (!deepEqual(response.data.imgs, {})) {
                insertPageLinks(response.data.imgs);
                insertInlineImgs(response.data.imgs);
            }

            nextTick(() => {
                setUpNotePopovers();
                setUpNoteRefs();
                setUpHeadingSemantics();
                setUpInternalLinks();
                scrollToTarget();
                setUpGallery();
                searching.value = false;
            });
        })
        .catch((error) => {
            logError(error);
            loading.value = false;
        });
}

//  TOC fetch + scroll ─
function fetchToC() {
    tocPosition.value = "";
    const philoId = route.params.pathInfo.split("/").join(" ");
    const docId = philoId.split(" ")[0];
    currentPhiloId.value = philoId;

    if (docId !== tocElements.value.docId) {
        $http.get(`${$dbUrl}/scripts/get_table_of_contents.py`, {
            params: { philo_id: currentPhiloId.value },
        }).then((response) => {
            const elements = response.data.toc;
            start.value = Math.max(0, response.data.current_obj_position - 100);
            end.value = response.data.current_obj_position + 100;
            tocElements.value = {
                docId,
                elements,
                start: start.value,
                end: end.value,
            };
            const tocButton = document.querySelector("#show-toc");
            if (tocButton) {
                tocButton.removeAttribute("disabled");
                tocButton.classList.remove("disabled");
            }
        }).catch(logError);
    } else {
        start.value = tocElements.value.start;
        end.value = tocElements.value.end;
        nextTick(() => {
            const tocButton = document.querySelector("#show-toc");
            if (tocButton) {
                tocButton.removeAttribute("disabled");
                tocButton.classList.remove("disabled");
            }
        });
    }
}

function loadBefore() {
    if (tocElements.value.elements && start.value < tocElements.value.elements.length) {
        const firstElement = tocElements.value.elements[start.value]?.philo_id;
        if (firstElement) tocPosition.value = firstElement;
    }
    start.value = Math.max(0, start.value - 200);
}

function loadAfter() {
    end.value += 200;
}

function handleTocScroll() {
    const tocContent = document.getElementById("toc-content");
    if (!tocContent || !tocElements.value.elements) return;

    const scrollTop = tocContent.scrollTop;
    const scrollHeight = tocContent.scrollHeight;
    const clientHeight = tocContent.clientHeight;

    if (scrollTop <= 100 && start.value > 0) {
        const oldScrollHeight = scrollHeight;
        loadBefore();
        nextTick(() => {
            const newScrollHeight = tocContent.scrollHeight;
            tocContent.scrollTop = scrollTop + (newScrollHeight - oldScrollHeight);
        });
    }

    if (scrollTop + clientHeight >= scrollHeight - 100 && end.value < tocElements.value.elements.length) {
        loadAfter();
    }
}

function toggleTableOfContents() {
    tocOpen.value = !tocOpen.value;
    if (tocOpen.value) {
        nextTick(() => {
            const currentElement = document.querySelector(".current-obj");
            if (currentElement) {
                currentElement.scrollIntoView({ behavior: "smooth", block: "center" });
                currentElement.focus();
            }
            // Slight delay so the panel finishes rendering before we attach
            setTimeout(() => {
                const tocContent = document.getElementById("toc-content");
                if (tocContent) {
                    tocContent.addEventListener("scroll", handleTocScroll, { passive: true });
                }
            }, 100);
        });
    } else {
        const tocContent = document.getElementById("toc-content");
        if (tocContent) {
            tocContent.removeEventListener("scroll", handleTocScroll);
        }
    }
}

//  Navigation handlers
function backToTop() {
    window.scrollTo({ top: 0, behavior: "smooth" });
}

function goToTextObject(philoIDArg) {
    const path = philoIDArg.split(/[- ]/).join("/");
    if (tocOpen.value) tocOpen.value = false;
    router.push({ path: `/navigate/${path}` });
}

function textObjectSelection(philoId, index, event) {
    event.preventDefault();
    const newStart = Math.max(0, tocElements.value.start + index - 100);
    tocElements.value = {
        ...tocElements.value,
        start: newStart,
        end: tocElements.value.end - index + 100,
    };
    goToTextObject(philoId);
}

function handleScroll() {
    if (!navBarVisible.value) {
        if (window.scrollY > navButtonPosition.value) {
            navBarVisible.value = true;
            const topBar = document.getElementById("toc-top-bar");
            topBar.style.top = 0;
            topBar.classList.add("visible", "shadow");
            const tocWrapper = document.getElementById("toc-wrapper");
            tocWrapper.style.top = "31px";
            const navButtons = document.getElementById("nav-buttons");
            navButtons.classList.add("visible");
            const reportError = document.getElementById("report-error");
            if (reportError != null) reportError.classList.add("visible");
        }
    } else if (window.scrollY < navButtonPosition.value) {
        navBarVisible.value = false;
        const topBar = document.getElementById("toc-top-bar");
        topBar.style.top = "initial";
        topBar.classList.remove("visible", "shadow");
        const tocWrapper = document.getElementById("toc-wrapper");
        tocWrapper.style.top = "0px";
        const navButtons = document.getElementById("nav-buttons");
        navButtons.style.top = "initial";
        navButtons.classList.remove("visible");
        const reportError = document.getElementById("report-error");
        if (reportError != null) reportError.classList.remove("visible");
    }
}

function dicoLookup(event) {
    if (event.key !== "d") return;
    const selection = window.getSelection().toString();
    let link;
    if (philoConfig.dictionary_lookup.keywords) {
        link = `${philoConfig.dictionary_lookup.url_root}?${philoConfig.dictionary_lookup_keywords.selected_keyword}=${selection}&`;
        const keyValues = [];
        for (const [key, value] of Object.entries(
            philoConfig.dictionary_lookup_keywords.immutable_key_values
        )) {
            keyValues.push(`${key}=${value}`);
        }
        for (const [key, value] of Object.entries(
            philoConfig.dictionary_lookup_keywords.variable_key_values
        )) {
            const fieldValue = textObject.value.metadata_fields[value] || "";
            keyValues.push(`${key}=${fieldValue}`);
        }
        link += keyValues.join("&");
    } else {
        link = `${philoConfig.dictionary_lookup.url_root}/${selection}`;
    }
    window.open(link);
}

//  Image gallery
function setUpGallery() {
    for (const imageType of ["page-image-link", "inline-img", "external-img"]) {
        Array.from(document.getElementsByClassName(imageType)).forEach((item) => {
            item.addEventListener("click", (event) => {
                event.preventDefault();
                const items = Array.from(document.getElementsByClassName(imageType));
                images.value = items.map((it) => it.getAttribute("href") || it.getAttribute("src"));
                const imageIndex = items.indexOf(event.target);
                gallery.value = GLightbox({
                    elements: items.map((it) => ({
                        href: it.getAttribute("href") || it.getAttribute("src"),
                        type: "image",
                    })),
                    openEffect: "fade",
                    closeEffect: "fade",
                    startAt: imageIndex,
                });
                gallery.value.open();
                gallery.value.on("slide_changed", () => {
                    // TODO: only create anchor tag the first time.
                    const img = items[gallery.value.getActiveSlideIndex()].getAttribute("large-img");
                    const newNode = document.createElement("a");
                    newNode.style.cssText =
                        "position: absolute; top: 20px; font-size: 15px; right: 70px; opacity: 0.7; border: solid; padding: 0px 5px; color: #fff !important";
                    newNode.href = img;
                    newNode.id = "large-img-link";
                    newNode.target = "_blank";
                    newNode.innerHTML = "&nearr;";
                    const closeBtn = document.getElementsByClassName("gclose")[0];
                    closeBtn.parentNode.insertBefore(newNode, closeBtn);
                });
            });
        });
    }
}

//  Popover cleanup
function destroyPopovers() {
    document.querySelectorAll(".note, .note-ref").forEach((note) => {
        const popover = Popover.getInstance(note);
        if (popover != null) popover.dispose();
        // Strip event listeners by cloning and replacing the node
        const clone = note.cloneNode(true);
        note.parentNode.replaceChild(clone, note);
    });
}

//  Watcher
watch(
    () => route.params,
    () => {
        if (route.name === "textNavigation") {
            destroyPopovers();
            fetchText();
            fetchToC();
        }
    }
);

//  Lifecycle
onMounted(() => {
    const tocButton = document.querySelector("#show-toc");
    if (tocButton) {
        navButtonPosition.value = tocButton.getBoundingClientRect().top;
    }
});

onBeforeUnmount(() => {
    if (gallery.value) gallery.value.close();
    destroyPopovers();
    const tocContent = document.getElementById("toc-content");
    if (tocContent) {
        tocContent.removeEventListener("scroll", handleTocScroll);
    }
});

//  Initial dispatch ─
formData.value.report = "textNavigation";
fetchToC();
fetchText();
</script>
<style lang="scss" scoped>
@use "../assets/styles/theme.module.scss" as theme;

.separator {
    padding: 5px;
    font-size: 60%;
    display: inline-block;
    vertical-align: middle;
}

#toc-content {
    display: inline-block;
    position: relative;
    max-height: 90vh;
    min-width: 250px;
    max-width: 66.667%;
    overflow: scroll;
    text-align: justify;
    line-height: 180%;
    z-index: 50;
    background: #fff;
    padding: 0 1rem;
}

@media (max-width: 991.98px) {
    #toc-content {
        max-width: 83.333%;
    }
}

@media (max-width: 575.98px) {
    #toc-content {
        max-width: 100%;
    }
}

#toc-wrapper {
    position: relative;
    z-index: 49;
    pointer-events: all;
    margin-left: -1.5rem;
}

#toc-top-bar {
    height: 31px;
    width: 100%;
}

#toc {
    margin-top: 31px;
    pointer-events: all;
}

#toc-top-bar.visible {
    position: fixed;
}

#nav-buttons.visible {
    position: fixed;
    backdrop-filter: blur(0.5rem);
    background-color: rgba(255, 255, 255, 0.3);
    pointer-events: all;
}

#back-to-top {
    position: absolute;
    left: 0;
}

#report-error {
    position: absolute;
    right: 0;
    opacity: 0;
    transition: opacity 0.25s;
    pointer-events: none;
}

#report-error.visible {
    opacity: 0.95;
    pointer-events: all;
}

#nav-buttons {
    position: absolute;
    width: 100%;
}

#toc-nav-bar {
    background-color: #ddd;
    opacity: 0.95;
    backdrop-filter: blur(5px) contrast(0.8);
}

/* TextNavigation-specific TOC styling */
.toc-link {
    text-decoration: none;
    font-size: 0.95rem;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    transition: all 0.15s ease-in-out;
}

.toc-link:hover,
.toc-link:focus {
    background: rgba(theme.$link-color, 0.1);
    transform: scale(1.02);
}

.toc-link.current-obj {
    background: rgba(theme.$link-color, 0.15);
    font-weight: 600;
    border-left: 3px solid theme.$link-color;
    padding-left: calc(0.5rem - 3px);
}

.toc-tree {
    margin-left: 1rem;
}

.div1-marker {
    margin-left: -1.25rem;
}

#book-page {
    text-align: justify;
    white-space: v-bind("whiteSpace");
}

:deep(.xml-pb) {
    display: block;
    text-align: center;
    margin: 10px;
}

:deep(.xml-pb::before) {
    content: "-" attr(n) "-";
    white-space: pre;
}

:deep(p) {
    margin-bottom: 0.5rem;
}

:deep(.highlight) {
    background-color: theme.$passage-color;
    color: #fff !important;
    padding: 0 0.3rem;
    border-radius: 0.2rem;
}

:deep(.xml-div1::after),
/* clear floats from inline images */
:deep(.xml-div2::after),
:deep(.xml-div3::after) {
    content: "";
    display: block;
    clear: right;
}

/* Styling for theater */

:deep(.xml-castitem::after) {
    content: "\A";
    white-space: pre;
}

:deep(.xml-castlist > .xml-castitem:first-of-type::before) {
    content: "\A";
    white-space: pre;
}

:deep(.xml-castgroup::before) {
    content: "\A";
    white-space: pre;
}

:deep(b.headword) {
    font-weight: 700 !important;
    font-size: 130%;
    font-variant: small-caps;
    display: block;
    margin-top: 20px;
}

:deep(b.headword::before) {
    content: "\A";
    white-space: pre;
}

:deep(#bibliographic-results b.headword) {
    font-weight: 400 !important;
    font-size: 100%;
    display: inline;
}

:deep(.xml-lb),
:deep(.xml-l) {
    text-align: justify;
    display: block;
}

:deep(.xml-sp .xml-lb:first-of-type) {
    content: "";
    white-space: normal;
}

:deep(.xml-lb[type="hyphenInWord"]) {
    display: inline;
}

#book-page .xml-sp {
    display: block;
}

:deep(.xml-sp::before) {
    content: "\A";
    white-space: pre;
}

:deep(.xml-stage + .xml-sp:nth-of-type(n + 2)::before) {
    content: "";
}

:deep(.xml-fw, .xml-join) {
    display: none;
}

:deep(.xml-speaker + .xml-stage::before) {
    content: "";
    white-space: normal;
}

:deep(.xml-stage) {
    font-style: italic;
}

:deep(.xml-stage::after) {
    content: "\A";
    white-space: pre;
}

:deep(div1 div2::before) {
    content: "\A";
    white-space: pre;
}

:deep(.xml-speaker) {
    font-weight: 700;
}

:deep(.xml-pb) {
    display: block;
    text-align: center;
    margin: 10px;
}

:deep(.xml-pb::before) {
    content: "-" attr(n) "-";
    white-space: pre;
}

:deep(#main-text .xml-pb-image a) {
    text-decoration: none;
    border: 1px solid theme.$link-color;
    border-radius: 0.25rem;
    padding: 0.15rem;
    font-weight: 700;
}

:deep(#main-text .xml-pb-image a:hover),
:deep(#main-text .xml-pb-image a:focus) {
    background-color: rgba(theme.$link-color, 0.025);
    border: 2px solid theme.$link-color;
}

:deep(.xml-lg) {
    display: block;
}

:deep(.xml-lg::after) {
    content: "\A";
    white-space: pre;
}

:deep(.xml-lg:first-of-type::before) {
    content: "\A";
    white-space: pre;
}

:deep(.xml-castList) :deep(.xml-front),
:deep(.xml-castItem),
:deep(.xml-docTitle),
:deep(.xml-docImprint),
:deep(.xml-performance),
:deep(.xml-docAuthor),
:deep(.xml-docDate),
:deep(.xml-premiere),
:deep(.xml-casting),
:deep(.xml-recette),
:deep(.xml-nombre) {
    display: block;
}

:deep(.xml-docTitle) {
    font-style: italic;
    font-weight: bold;
}

:deep(.xml-docAuthor),
:deep(.xml-docTitle),
:deep(.xml-docDate) {
    text-align: center;
}

:deep(.xml-docTitle span[type="main"]) {
    font-size: 150%;
    display: block;
}

:deep(.xml-docTitle span[type="sub"]) {
    font-size: 120%;
    display: block;
}

:deep(.xml-performance),
:deep(.xml-docImprint) {
    margin-top: 10px;
}

:deep(.xml-set) {
    display: block;
    font-style: italic;
    margin-top: 10px;
}

/*Dictionary formatting*/

body {
    counter-reset: section;
    /* Set the section counter to 0 */
}

:deep(.xml-prononciation::before) {
    content: "(";
}

:deep(.xml-prononciation::after) {
    content: ")\A";
}

:deep(.xml-nature) {
    font-style: italic;
}

:deep(.xml-indent),
:deep(.xml-variante) {
    display: block;
}

:deep(.xml-variante) {
    padding-top: 10px;
    padding-bottom: 10px;
    text-indent: -1.3em;
    padding-left: 1.3em;
}

:deep(.xml-variante::before) {
    counter-increment: section;
    content: counter(section) ")\00a0";
    font-weight: 700;
}

:deep(:not(.xml-rubrique) + .xml-indent) {
    padding-top: 10px;
}

:deep(.xml-indent) {
    padding-left: 1.3em;
}

:deep(.xml-cit) {
    padding-left: 2.3em;
    display: block;
    text-indent: -1.3em;
}

:deep(.xml-indent > .xml-cit) {
    padding-left: 1em;
}

:deep(.xml-cit::before) {
    content: "\2012\00a0\00ab\00a0";
}

:deep(.xml-cit::after) {
    content: "\00a0\00bb\00a0(" attr(aut) "\00a0" attr(ref) ")";
    font-variant: small-caps;
}

:deep(.xml-rubrique) {
    display: block;
    margin-top: 20px;
}

:deep(.xml-rubrique::before) {
    content: attr(nom);
    font-variant: small-caps;
    font-weight: 700;
}

:deep(.xml-corps + .xml-rubrique) {
    margin-top: 10px;
}

/*Methodique styling*/

:deep(div[type="article"] .headword) {
    display: inline-block;
    margin-bottom: 10px;
}

:deep(.headword + p) {
    display: inline;
}

:deep(.headword + p + p) {
    margin-top: 10px;
}

/*Note handling*/
:deep(.popover-content .xml-p:not(:first-of-type)) {
    display: block;
    margin-top: 1em;
    margin-bottom: 1em;
}

:deep(.note-content) {
    display: none;
}

:deep(.note),
:deep(.note-ref) {
    vertical-align: 0.3em;
    font-size: 0.7em;
    margin-left: 0.1rem;
    padding: 0 0.2rem;
    border-radius: 50%;
}

:deep(.note:hover),
:deep(.note-ref:hover) {
    cursor: pointer;
    text-decoration: none;
}

:deep(div[type="notes"] .xml-note) {
    margin: 15px 0px;
    display: block;
}

:deep(.xml-note::before) {
    content: "note\00a0" attr(n) "\00a0:\00a0";
    font-weight: 700;
}

/*Page images*/

:deep(.xml-pb-image) {
    display: block;
    text-align: center;
    margin: 10px;
}

:deep(.page-image-link) {
    margin-top: 10px;
    /*display: block;*/
    text-align: center;
}

/*Inline images*/
:deep(.inline-img) {
    max-width: 40%;
    float: right;
    height: auto;
    padding-left: 15px;
    padding-top: 15px;
}

:deep(.inline-img:hover) {
    cursor: pointer;
}

:deep(.link-back) {
    margin-left: 10px;
    line-height: initial;
}

:deep(.xml-add) {
    color: #ef4500;
}

:deep(.xml-seg) {
    display: block;
}

/*Table display*/

:deep(.xml-table) {
    display: table;
    position: relative;
    text-align: center;
    border-collapse: collapse;
}

:deep(.xml-table .xml-pb-image) {
    position: absolute;
    width: 100%;
    margin-top: 15px;
}

:deep(.xml-row) {
    display: table-row;
    font-weight: 700;
    text-align: left;
    min-height: 50px;
    font-variant: small-caps;
    padding-top: 10px;
    padding-bottom: 10px;
    padding-right: 20px;
    border-bottom: #ddd 1px solid;
}

:deep(.xml-row ~ .xml-row) {
    font-weight: inherit;
    text-align: justify;
    font-variant: inherit;
}

:deep(.xml-pb-image + .xml-row) {
    padding-top: 50px;
    padding-bottom: 10px;
    border-top-width: 0px;
}

:deep(.xml-cell) {
    display: table-cell;
    padding-top: inherit;
    /*inherit padding when image is above */
    padding-bottom: inherit;
}

:deep(s) {
    text-decoration: none;
}

:deep(h5 .text-view) {
    font-size: inherit !important;
}

.slide-fade-enter-active {
    transition: all 0.3s ease-out;
}

.slide-fade-leave-active {
    transition: all 0.3s ease-out;
}

.slide-fade-enter-from,
.slide-fade-leave-to {
    transform: translateY(-30px);
    opacity: 0;
}

/* Image button styling */
.img-buttons {
    font-size: 45px !important;
    color: #fff !important;
}

:deep(#full-size-image) {
    right: 90px;
    font-weight: 700 !important;
    font-size: 20px !important;
    left: auto;
    margin: -15px;
    text-decoration: none;
    cursor: pointer;
    position: absolute;
    top: 28px;
    color: #fff;
    opacity: 0.8;
    border: 3px solid;
    padding: 0 0.25rem;
}

#full-size-image:hover {
    opacity: 1;
}

:deep([class*="passage-"]) {
    color: theme.$passage-color;
    font-weight: 700;
}

// Support for rend attribute of TEI files
:deep([rend="bold"]) {
    font-weight: 700;
}

:deep([rend="italic"]) {
    font-style: italic;
}

:deep([rend="smallcaps"]) {
    font-variant: small-caps;
}

:deep(p[rend="center"]) {
    text-align: center;
    display: block;
    margin: 1rem 0 0.25rem 0;
}

:deep(b.headword[rend="center"]) {
    margin-bottom: 30px;
    text-align: center;
}
</style>
