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
                        :aria-label="$t('textNav.backToTop')">
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
                    <button type="button" class="btn btn-secondary visually-hidden" id="hide-toc"
                        @click="toggleTableOfContents()" :aria-label="$t('textNav.closeToc')">
                        <span class="icon-x"></span>
                    </button>
                    <transition name="slide-fade">
                        <div class="card py-3 shadow" id="toc-content" :style="tocHeight" v-if="tocOpen" role="region"
                            :aria-label="$t('textNav.tocContent')">
                            <ul class="toc-tree" role="tree">
                                <li v-for="(element, elIndex) in processedTocElements" :key="elIndex"
                                    :class="'toc-item toc-' + element.philo_type" role="treeitem"
                                    :aria-level="element.level || 1">

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
                                            :aria-label="$t('textNav.goToSection', { title: element.label })"
                                            :aria-current="element.philo_id === currentPhiloId ? 'page' : null">
                                            {{ element.label }}
                                        </button>
                                    </div>

                                    <!-- Child sections -->
                                    <ul v-if="element.children && element.children.length > 0" class="toc-children"
                                        role="group">
                                        <li v-for="(child, childIndex) in element.children" :key="childIndex"
                                            :class="'toc-item toc-child toc-' + child.philo_type" role="treeitem"
                                            :aria-level="child.level || 2">
                                            <div class="toc-content-wrapper">
                                                <span :class="'bullet-point-' + child.philo_type"
                                                    aria-hidden="true"></span>
                                                <button type="button"
                                                    :class="{ 'current-obj': child.philo_id === currentPhiloId }"
                                                    class="btn btn-link toc-link"
                                                    @click="textObjectSelection(child.philo_id, childIndex, $event)"
                                                    :aria-label="$t('textNav.goToSection', { title: child.label })"
                                                    :aria-current="child.philo_id === currentPhiloId ? 'page' : null">
                                                    {{ child.label }}
                                                </button>
                                            </div>

                                            <!-- Grandchildren (div3 level) -->
                                            <ul v-if="child.children && child.children.length > 0" class="toc-children"
                                                role="group">
                                                <li v-for="(grandchild, grandchildIndex) in child.children"
                                                    :key="grandchildIndex"
                                                    :class="'toc-item toc-child toc-' + grandchild.philo_type"
                                                    role="treeitem" :aria-level="grandchild.level || 3">
                                                    <div class="toc-content-wrapper">
                                                        <span :class="'bullet-point-' + grandchild.philo_type"
                                                            aria-hidden="true"></span>
                                                        <button type="button"
                                                            :class="{ 'current-obj': grandchild.philo_id === currentPhiloId }"
                                                            class="btn btn-link toc-link"
                                                            @click="textObjectSelection(grandchild.philo_id, grandchildIndex, $event)"
                                                            :aria-label="$t('textNav.goToSection', { title: grandchild.label })"
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
<script>
import { Popover } from "bootstrap";
import GLightbox from 'glightbox';
import 'glightbox/dist/css/glightbox.css';
import { mapStores, mapWritableState } from "pinia";
import mixins from "../mixins";
import { useMainStore } from "../stores/main";
import citations from "./Citations";

export default {
    name: "textNavigation",
    mixins: [mixins],
    components: {
        citations
    },
    inject: ["$http"],
    computed: {
        ...mapWritableState(useMainStore, [
            "formData",
            "textNavigationCitation",
            "navBar",
            "tocElements",
            "byte",
            "searching",
            "accessAuthorized"
        ]),
        ...mapStores(useMainStore),

        tocElementsToDisplay: function () {
            return this.tocElements.elements.slice(this.start, this.end);
        },
        processedTocElements() {
            const elementsToDisplay = this.tocElements.elements.slice(this.start, this.end);
            return this.buildTocTree(elementsToDisplay);
        },
        tocHeight() {
            return `max-height: ${window.innerHeight - 200}`;
        },
        whiteSpace() {
            if (this.$philoConfig.respect_text_line_breaks) {
                return "pre";
            }
            return "normal";
        },
    },
    data() {
        const store = useMainStore();
        return {
            store,
            philoConfig: this.$philoConfig,
            textObject: {},
            beforeObjImgs: [],
            afterObjImgs: [],
            beforeGraphicsImgs: [],
            afterGraphicsImgs: [],
            navbar: null,
            loading: false,
            tocOpen: false,
            done: false,
            authorized: true,
            textRendered: false,
            textObjectURL: "",
            philoID: "",
            highlight: false,
            start: 0,
            end: 0,
            tocPosition: 0,
            navButtonPosition: 0,
            navBarVisible: false,
            timeToRender: 0,
            gallery: null,
            images: [],
            imageIndex: null
        };
    },
    created() {
        this.formData.report = "textNavigation";
        this.fetchToC();
        this.fetchText();
    },
    watch: {
        $route() {
            if (this.$route.name == "textNavigation") {
                this.destroyPopovers();
                this.fetchText();
                this.fetchToC();
            }
        },
    },
    mounted() {
        let tocButton = document.querySelector("#show-toc");
        this.navButtonPosition = tocButton.getBoundingClientRect().top;
    },
    unmounted() {
        if (this.gallery) {
            this.gallery.close();
        }
        this.destroyPopovers();

        // Clean up TOC scroll listener
        const tocContent = document.getElementById('toc-content');
        if (tocContent) {
            tocContent.removeEventListener('scroll', this.handleTocScroll);
        }
    },
    methods: {
        fetchText() {
            this.searching = true;
            this.textRendered = false;
            this.textObjectURL = this.$route.params;
            this.philoID = this.textObjectURL.pathInfo.split("/").join(" ");
            let byteString = "";
            if ("byte" in this.$route.query) {
                this.byte = this.$route.query.byte;
                if (typeof this.$route.query.byte == "object") {
                    byteString = `byte=${this.byte.join("&byte=")}`;
                } else {
                    byteString = `byte=${this.byte}`;
                }
            } else {
                this.byte = "";
            }
            let navigationParams = {
                report: "navigation",
                philo_id: this.philoID,
            };
            if (this.formData.start_byte !== "") {
                navigationParams.start_byte = this.formData.start_byte;
                navigationParams.end_byte = this.formData.end_byte;
            }
            let urlQuery = `${byteString}&${this.paramsToUrlString(navigationParams)}`;
            this.timeToRender = new Date().getTime();
            this.$http
                .get(`${this.$dbUrl}/reports/navigation.py?${urlQuery}`)
                .then((response) => {
                    this.textObject = response.data;
                    this.textNavigationCitation = response.data.citation;
                    this.navBar = true;
                    if (this.byte.length > 0) {
                        this.highlight = true;
                    } else {
                        this.highlight = false;
                    }

                    if (!this.deepEqual(response.data.imgs, {})) {
                        this.insertPageLinks(response.data.imgs);
                        this.insertInlineImgs(response.data.imgs);
                    }
                    this.setUpNavBar();
                    this.$nextTick(() => {
                        // Handle inline notes if there are any
                        let notes = document.getElementsByClassName("note");
                        if (notes.length > 0) {
                            Array.from(notes).forEach((note, index) => {
                                let innerHTML = note.nextElementSibling.innerHTML;
                                const noteId = `note-content-${index}`;

                                // Add ARIA attributes for screen reader accessibility
                                note.setAttribute('role', 'button');
                                note.setAttribute('tabindex', '0');
                                note.setAttribute('aria-label', `Note ${note.textContent || index + 1}`);
                                note.setAttribute('aria-describedby', noteId);

                                new Popover(note, {
                                    html: true,
                                    content: innerHTML,
                                    trigger: "focus",
                                    customClass: "custom-popover shadow-lg",
                                });

                                // Set ID on popover when it shows so aria-describedby works
                                note.addEventListener('shown.bs.popover', () => {
                                    const popoverElement = document.querySelector(`.popover`);
                                    if (popoverElement) {
                                        popoverElement.setAttribute('id', noteId);
                                        popoverElement.setAttribute('role', 'tooltip');
                                    }
                                });
                            });
                        }

                        // Handle ref notes if there are any
                        let noteRefs = document.getElementsByClassName("note-ref");
                        if (noteRefs.length > 0) {
                            Array.from(noteRefs).forEach((noteRef, index) => {
                                const noteRefId = `note-ref-content-${index}`;

                                // Add ARIA attributes for screen reader accessibility
                                noteRef.setAttribute('role', 'button');
                                noteRef.setAttribute('tabindex', '0');
                                noteRef.setAttribute('aria-label', `Note reference ${noteRef.textContent || index + 1}`);
                                noteRef.setAttribute('aria-describedby', noteRefId);

                                let getNotes = () => {
                                    this.$http
                                        .get(`${this.$dbUrl}/scripts/get_notes.py?`, {
                                            params: {
                                                target: noteRef.getAttribute("target"),
                                                philo_id: this.$route.params.pathInfo.split("/").join(" "),
                                            },
                                        })
                                        .then((response) => {
                                            new Popover(noteRef, {
                                                html: true,
                                                content: response.data.text,
                                                trigger: "focus",
                                                customClass: "custom-popover shadow-lg",
                                            });

                                            // Set ID on popover when it shows so aria-describedby works
                                            noteRef.addEventListener('shown.bs.popover', () => {
                                                const popoverElement = document.querySelector(`.popover`);
                                                if (popoverElement) {
                                                    popoverElement.setAttribute('id', noteRefId);
                                                    popoverElement.setAttribute('role', 'tooltip');
                                                }
                                            });

                                            noteRef.removeEventListener("click", getNotes);
                                        });
                                };
                                noteRef.addEventListener("click", getNotes());
                            });
                        }

                        // Add heading semantics to headword elements for accessibility
                        let headwords = document.querySelectorAll("#text-obj-content .headword");
                        headwords.forEach((headword) => {
                            let current = headword.parentElement;
                            let depth = 0;

                            // Count every <div> ancestor until we hit the container
                            while (current && current.id !== 'text-obj-content') {
                                if (current.tagName.toLowerCase() === 'div') {
                                    depth++;
                                }
                                current = current.parentElement;
                            }

                            // Start at level 2:
                            // depth 1 (one div) = level 2
                            // depth 2 (nested div) = level 3, etc.
                            const level = Math.min(depth + 1, 6);

                            headword.setAttribute('role', 'heading');
                            headword.setAttribute('aria-level', level - 2); // Two parents before div1
                        });

                        let linkBack = document.getElementsByClassName("link-back");
                        if (linkBack.length > 0) {
                            Array.from(linkBack).forEach((el) => {
                                if (el) {
                                    var goToNote = () => {
                                        let link = el.getAttribute("link");
                                        this.$router.push(link);
                                        el.removeEventListener("click", goToNote);
                                    };
                                    el.addEventListener("click", goToNote);
                                }
                            });
                        }

                        let innerSearchLinks = document.querySelectorAll("a[type='search']");
                        if (innerSearchLinks.length > 0) {
                            Array.from(innerSearchLinks).forEach((el) => {
                                let metadata, metadataValue;
                                [metadata, metadataValue] = el.getAttribute("target").split(":");
                                el.href = `${this.$dbUrl.replace(
                                    /\/+$/,
                                    ""
                                )}/bibliography?${metadata}=${metadataValue}`;
                            });
                        }

                        // Scroll to highlight
                        if (this.byte != "") {
                            let element = document.getElementsByClassName("highlight")[0];
                            let parent = element.parentElement;
                            if (parent.classList.contains("note-content")) {
                                let note = parent.previousSibling;
                                this.$scrollTo(note, 250, {
                                    easing: "ease-out",
                                    offset: -150,
                                    onDone: function () {
                                        setTimeout(() => {
                                            note.focus();
                                        }, 500);
                                    },
                                });
                            } else {
                                this.$scrollTo(element, 250, {
                                    easing: "ease-out",
                                    offset: -150,
                                });
                            }
                        } else if (this.formData.start_byte != "") {
                            this.$scrollTo(document.querySelector(".passage-marker"), 250, {
                                easing: "ease-out",
                                offset: -150,
                            });
                        } else if (this.$route.hash) {
                            // for note link back
                            let note = document.getElementById(this.$route.hash.slice(1));
                            this.$scrollTo(note, 250, {
                                easing: "ease-out",
                                offset: -250,
                                onDone: () => {
                                    setTimeout(() => {
                                        note.focus();
                                    }, 500);
                                },
                            });
                        }

                        this.setUpGallery();
                        this.searching = false;
                    });
                })
                .catch((error) => {
                    this.debug(this, error);
                    this.loading = false;
                });
        },
        insertPageLinks(imgObj) {
            let currentObjImgs = imgObj.current_obj_img;
            let allImgs = imgObj.all_imgs;
            this.beforeObjImgs = [];
            this.afterObjImgs = [];
            if (currentObjImgs.length > 0) {
                let beforeIndex = 0;
                for (let i = 0; i < allImgs.length; i++) {
                    let img = allImgs[i];
                    if (currentObjImgs.indexOf(img[0]) === -1) {
                        if (img.length == 2) {
                            this.beforeObjImgs.push([
                                `${this.philoConfig.page_images_url_root}/${img[0]}`,
                                `${this.philoConfig.page_images_url_root}/${img[1]}`,
                            ]);
                        } else {
                            this.beforeObjImgs.push([
                                `${this.philoConfig.page_images_url_root}/${img[0]}`,
                                `${this.philoConfig.page_images_url_root}/${img[0]}`,
                            ]);
                        }
                    } else {
                        beforeIndex = i;
                        break;
                    }
                }
                for (let i = beforeIndex; i < allImgs.length; i++) {
                    let img = allImgs[i];
                    if (currentObjImgs.indexOf(img[0]) === -1) {
                        if (img.length == 2) {
                            this.afterObjImgs.push([
                                `${this.philoConfig.page_images_url_root}/${img[0]}`,
                                `${this.philoConfig.page_images_url_root}/${img[1]}`,
                            ]);
                        } else {
                            this.afterObjImgs.push([
                                `${this.philoConfig.page_images_url_root}/${img[0]}`,
                                `${this.philoConfig.page_images_url_root}/${img[0]}`,
                            ]);
                        }
                    }
                }
            }
        },
        insertInlineImgs(imgObj) {
            var currentObjImgs = imgObj.current_graphic_img;
            var allImgs = imgObj.graphics;
            var img;
            this.beforeGraphicsImgs = [];
            this.afterGraphicsImgs = [];
            if (currentObjImgs.length > 0) {
                var beforeIndex = 0;
                for (let i = 0; i < allImgs.length; i++) {
                    img = allImgs[i];
                    if (currentObjImgs.indexOf(img[0]) === -1) {
                        if (img.length == 2) {
                            this.beforeGraphicsImgs.push([
                                `${this.philoConfig.page_images_url_root}/${img[0]}`,
                                `${this.philoConfig.page_images_url_root}/${img[1]}`,
                            ]);
                        } else {
                            this.beforeGraphicsImgs.push([
                                `${this.philoConfig.page_images_url_root}/${img[0]}`,
                                `${this.philoConfig.page_images_url_root}/${img[0]}`,
                            ]);
                        }
                    } else {
                        beforeIndex = i;
                        break;
                    }
                }
                for (let i = beforeIndex; i < allImgs.length; i++) {
                    img = allImgs[i];
                    if (currentObjImgs.indexOf(img[0]) === -1) {
                        if (img.length == 2) {
                            this.afterGraphicsImgs.push([
                                `${this.philoConfig.page_images_url_root}/${img[0]}`,
                                `${this.philoConfig.page_images_url_root}/${img[1]}`,
                            ]);
                        } else {
                            this.afterGraphicsImgs.push([
                                `${this.philoConfig.page_images_url_root}/${img[0]}`,
                                `${this.philoConfig.page_images_url_root}/${img[0]}`,
                            ]);
                        }
                    }
                }
            }
        },
        fetchToC() {
            this.tocPosition = "";
            var philoId = this.$route.params.pathInfo.split("/").join(" ");
            let docId = philoId.split(" ")[0];
            this.currentPhiloId = philoId;
            if (docId !== this.tocElements.docId) {
                this.$http
                    .get(`${this.$dbUrl}/scripts/get_table_of_contents.py`, {
                        params: {
                            philo_id: this.currentPhiloId,
                        },
                    })
                    .then((response) => {
                        let tocElements = response.data.toc;
                        this.start = response.data.current_obj_position - 100;
                        if (this.start < 0) {
                            this.start = 0;
                        }
                        this.end = response.data.current_obj_position + 100;

                        this.tocElements = {
                            docId: philoId.split(" ")[0],
                            elements: tocElements,
                            start: this.start,
                            end: this.end,
                        };
                        let tocButton = document.querySelector("#show-toc");
                        tocButton.removeAttribute("disabled");
                        tocButton.classList.remove("disabled");
                    })
                    .catch((error) => {
                        this.debug(this, error);
                    });
            } else {
                this.start = this.tocElements.start;
                this.end = this.tocElements.end;
                this.$nextTick(function () {
                    let tocButton = document.querySelector("#show-toc");
                    tocButton.removeAttribute("disabled");
                    tocButton.classList.remove("disabled");
                });
            }
        },
        setUpGallery() {
            // Image Gallery handling
            for (let imageType of ["page-image-link", "inline-img", "external-img"]) {
                Array.from(document.getElementsByClassName(imageType)).forEach((item) => {
                    item.addEventListener("click", (event) => {
                        event.preventDefault();
                        this.images = [...document.getElementsByClassName(imageType)].map(
                            (item) => item.getAttribute("href") || item.getAttribute("src"))
                        let imageIndex = Array.from(document.getElementsByClassName(imageType)).indexOf(event.target)
                        this.gallery = GLightbox({
                            elements: [...document.getElementsByClassName(imageType)].map(
                                (item) => { return { href: item.getAttribute("href") || item.getAttribute("src"), type: "image" } }),
                            openEffect: "fade", closeEffect: "fade", startAt: imageIndex
                        })
                        this.gallery.open()
                        this.gallery.on('slide_changed', () => {
                            // TODO: only create anchor tag the first time.
                            let img = Array.from(document.getElementsByClassName(imageType))[
                                this.gallery.getActiveSlideIndex()
                            ].getAttribute("large-img");
                            const newNode = document.createElement("a");
                            newNode.style.cssText = `position: absolute; top: 20px; font-size: 15px; right: 70px; opacity: 0.7; border: solid; padding: 0px 5px; color: #fff !important`
                            newNode.href = img
                            newNode.id = "large-img-link"
                            newNode.target = "_blank"
                            newNode.innerHTML = "&nearr;"
                            let closeBtn = document.getElementsByClassName("gclose")[0]
                            closeBtn.parentNode.insertBefore(newNode, closeBtn)

                        });
                    });
                });
            }
        },
        loadBefore() {
            // Store current first visible element to maintain position
            if (this.tocElements.elements && this.start < this.tocElements.elements.length) {
                var firstElement = this.tocElements.elements[this.start]?.philo_id;
                if (firstElement) {
                    this.tocPosition = firstElement;
                }
            }

            // Load 200 more entries before current start
            this.start -= 200;
            if (this.start < 0) {
                this.start = 0;
            }
        },
        loadAfter() {
            this.end += 200;
        },
        handleTocScroll() {
            const tocContent = document.getElementById('toc-content');
            if (!tocContent || !this.tocElements.elements) return;

            const scrollTop = tocContent.scrollTop;
            const scrollHeight = tocContent.scrollHeight;
            const clientHeight = tocContent.clientHeight;

            // Load previous entries when scrolled to within 100px of the top
            if (scrollTop <= 100 && this.start > 0) {
                const oldScrollHeight = scrollHeight;
                this.loadBefore();

                // Maintain scroll position after loading previous entries
                this.$nextTick(() => {
                    const newScrollHeight = tocContent.scrollHeight;
                    const heightDifference = newScrollHeight - oldScrollHeight;
                    tocContent.scrollTop = scrollTop + heightDifference;
                });
            }

            // Load more entries when scrolled to within 100px of the bottom
            if (scrollTop + clientHeight >= scrollHeight - 100 && this.end < this.tocElements.elements.length) {
                this.loadAfter();
            }
        },
        toggleTableOfContents() {
            this.tocOpen = !this.tocOpen;
            if (this.tocOpen) {
                this.$nextTick(() => {
                    const currentElement = this.$el.querySelector(".current-obj");
                    if (currentElement) {
                        currentElement.scrollIntoView({ behavior: "smooth", block: "center" });
                        currentElement.focus();
                    }

                    // Add scroll listener for automatic loading with a slight delay
                    setTimeout(() => {
                        const tocContent = document.getElementById('toc-content');
                        if (tocContent) {
                            tocContent.addEventListener('scroll', this.handleTocScroll, { passive: true });
                        }
                    }, 100);
                });
            } else {
                // Remove scroll listener when TOC is closed
                const tocContent = document.getElementById('toc-content');
                if (tocContent) {
                    tocContent.removeEventListener('scroll', this.handleTocScroll);
                }
            }
        },
        backToTop() {
            window.scrollTo({ top: 0, behavior: "smooth" });
        },
        goToTextObject(philoID) {
            philoID = philoID.split(/[- ]/).join("/");
            if (this.tocOpen) {
                this.tocOpen = false;
            }
            this.$router.push({ path: `/navigate/${philoID}` });
        },
        textObjectSelection(philoId, index, event) {
            event.preventDefault();
            let newStart = this.tocElements.start + index - 100;
            if (newStart < 0) {
                newStart = 0;
            }
            this.tocElements = {
                ...this.tocElements,
                start: newStart,
                end: this.tocElements.end - index + 100,
            };
            this.goToTextObject(philoId);
        },
        setUpNavBar() {
            // let prevButton = document.querySelector("#prev-obj");
            // let nextButton = document.querySelector("#next-obj");
            // if (this.textObject.next === "" || typeof this.textObject.next === "undefined") {
            //     nextButton.classList.add("disabled");
            // } else {
            //     nextButton.removeAttribute("disabled");
            //     nextButton.classList.remove("disabled");
            // }
            // if (this.textObject.prev === "" || typeof this.textObject.prev === "undefined") {
            //     prevButton.classList.add("disabled");
            // } else {
            //     prevButton.removeAttribute("disabled");
            //     prevButton.classList.remove("disabled");
            // }
            this.textObject = { ...this.textObject };
        },
        handleScroll() {
            if (!this.navBarVisible) {
                if (window.scrollY > this.navButtonPosition) {
                    this.navBarVisible = true;
                    let topBar = document.getElementById("toc-top-bar");
                    topBar.style.top = 0;
                    topBar.classList.add("visible", "shadow");
                    let tocWrapper = document.getElementById("toc-wrapper");
                    tocWrapper.style.top = "31px";
                    let navButtons = document.getElementById("nav-buttons");
                    navButtons.classList.add("visible");
                    let backToTop = document.getElementById("back-to-top");
                    backToTop.classList.add("visible");
                    let reportError = document.getElementById("report-error");
                    if (reportError != null) {
                        reportError.classList.add("visible");
                    }
                }
            } else if (window.scrollY < this.navButtonPosition) {
                this.navBarVisible = false;
                let topBar = document.getElementById("toc-top-bar");
                topBar.style.top = "initial";
                topBar.classList.remove("visible", "shadow");
                let tocWrapper = document.getElementById("toc-wrapper");
                tocWrapper.style.top = "0px";
                let navButtons = document.getElementById("nav-buttons");
                navButtons.style.top = "initial";
                navButtons.classList.remove("visible");
                let backToTop = document.getElementById("back-to-top");
                backToTop.classList.remove("visible");
                let reportError = document.getElementById("report-error");
                if (reportError != null) {
                    reportError.classList.remove("visible");
                }
            }
        },
        dicoLookup(event) {
            if (event.key === "d") {
                let selection = window.getSelection().toString();
                let link;
                if (this.$philoConfig.dictionary_lookup.keywords) {
                    link = `${this.philoConfig.dictionary_lookup.url_root}?${this.philoConfig.dictionary_lookup_keywords.selected_keyword}=${selection}&`;
                    let keyValues = [];
                    for (const [key, value] of Object.entries(
                        this.philoConfig.dictionary_lookup_keywords.immutable_key_values
                    )) {
                        keyValues.push(`${key}=${value}`);
                    }
                    for (const [key, value] of Object.entries(
                        this.philoConfig.dictionary_lookup_keywords.variable_key_values
                    )) {
                        let fieldValue = this.textObject.metadata_fields[value] || "";
                        keyValues.push(`${key}=${fieldValue}`);
                    }
                    link += keyValues.join("&");
                } else {
                    link = `${this.philoConfig.dictionary_lookup.url_root}/${selection}`;
                }
                window.open(link);
            }
        },
        destroyPopovers() {
            document.querySelectorAll(".note, .note-ref").forEach((note) => {
                let popover = Popover.getInstance(note);
                if (popover != null) {
                    Popover.getInstance(note).dispose();
                }
            });
        },
    },
};
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
    overflow: scroll;
    text-align: justify;
    line-height: 180%;
    z-index: 50;
    background: #fff;
    padding: 0 1rem;
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
    opacity: 0;
    transition: opacity 0.25s;
    pointer-events: none;
}

#report-error {
    position: absolute;
    right: 0;
    opacity: 0;
    transition: opacity 0.25s;
    pointer-events: none;
}

#back-to-top.visible,
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
    background: rgba(theme.$link-color, 0.1);
    font-weight: 600;
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
