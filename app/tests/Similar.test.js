import { describe, it, expect } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { createGlobalConfig, createMockHttp } from "./helpers.js";
import { useMainStore } from "../src/stores/main.js";
import Similar from "../src/components/collocation/Similar.vue";

const fields = [{ label: "Author", value: "author" }];

function mountSimilar(biblio, formData) {
    const http = createMockHttp({
        "collocation.py": { file_path: "/hitlists/groups.npz" },
        "get_similar_collocate_distributions.py": { similar: [["Corneille", 0.081, ["rome", "hymen"]]] },
    });
    const global = createGlobalConfig({
        http,
        stubs: { ProgressSpinner: { template: "<div />" }, BibliographyCriteria: { template: "<div />" } },
    });
    const store = useMainStore();
    store.formData = { ...store.formData, q: "lemma:amour", report: "collocation", ...formData };
    const wrapper = mount(Similar, {
        props: { biblio, collocatesFilePath: "/hitlists/search.pickle", fieldsToCompare: fields },
        global,
    });
    return { wrapper, http };
}

describe("Similar", () => {
    it("asks for a filter of the search, the part of the corpus to compare with, without one", async () => {
        const { wrapper, http } = mountSimilar([], {});
        wrapper.vm.runSimilar("author");
        await flushPromises();
        expect(wrapper.find("[role=note]").text()).toContain("filter your search");
        expect(wrapper.find("#similarity-field-select").exists()).toBe(false);
        expect(http.get).not.toHaveBeenCalled();
    });

    it("compares the search with each group counted inside its metadata and in the whole corpus", async () => {
        const biblio = [{ key: "author", alias: "Author", value: "Voltaire" }];
        const { wrapper, http } = mountSimilar(biblio, { author: "Voltaire" });
        wrapper.vm.runSimilar("author");
        await flushPromises();
        const [inside, whole, similar] = http.get.mock.calls;
        expect(inside[0]).toContain("/reports/collocation.py");
        expect(inside[1].params).toMatchObject({ q: "lemma:amour", author: "Voltaire", map_field: "author" });
        expect(whole[0]).toContain("/reports/collocation.py");
        expect(whole[1].params.map_field).toBe("author");
        expect(whole[1].params).not.toHaveProperty("author");
        expect(similar[0]).toContain("get_similar_collocate_distributions.py");
        expect(similar[1].params).toEqual({
            primary_file_path: "/hitlists/search.pickle",
            file_path: "/hitlists/groups.npz",
            inside_file_path: "/hitlists/groups.npz",
        });
        expect(wrapper.text()).toContain("Corneille");
    });
});
