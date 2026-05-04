import { describe, it, expect } from "vitest";
import {
    copyObject,
    dateRangeHandler,
    deepEqual,
    extractSurfaceFromCollocate,
    sortResults,
    isOnlyFacetChange,
    buildTocTree,
} from "../src/utils.js";

// ---------------------------------------------------------------------------
// copyObject
// ---------------------------------------------------------------------------
describe("copyObject", () => {
    it("returns a deep copy", () => {
        const original = { a: 1, b: { c: 2 } };
        const copy = copyObject(original);
        expect(copy).toEqual(original);
        copy.b.c = 99;
        expect(original.b.c).toBe(2);
    });

    it("handles arrays", () => {
        const original = [1, [2, 3]];
        const copy = copyObject(original);
        copy[1][0] = 99;
        expect(original[1][0]).toBe(2);
    });
});

// ---------------------------------------------------------------------------
// dateRangeHandler
// ---------------------------------------------------------------------------
describe("dateRangeHandler", () => {
    it("combines start and end into range string for int fields", () => {
        const result = dateRangeHandler(
            { year: "int" },
            { year: { start: "1800", end: "1900" } },
            { year: "range" },
            { year: "" }
        );
        expect(result.year).toBe("1800-1900");
    });

    it("combines start and end with <=> separator for date fields", () => {
        const result = dateRangeHandler(
            { pub_date: "date" },
            { pub_date: { start: "1800", end: "1900" } },
            { pub_date: "range" },
            { pub_date: "" }
        );
        expect(result.pub_date).toBe("1800<=>1900");
    });

    it("handles start-only range for int fields", () => {
        const result = dateRangeHandler(
            { year: "int" },
            { year: { start: "1800", end: "" } },
            { year: "range" },
            { year: "" }
        );
        expect(result.year).toBe("1800-");
    });

    it("handles end-only range for int fields", () => {
        const result = dateRangeHandler(
            { year: "int" },
            { year: { start: "", end: "1900" } },
            { year: "range" },
            { year: "" }
        );
        expect(result.year).toBe("-1900");
    });

    it("does not modify exact date fields", () => {
        const values = { year: "1850" };
        const result = dateRangeHandler(
            { year: "int" },
            { year: { start: "", end: "" } },
            { year: "exact" },
            values
        );
        expect(result.year).toBe("1850");
    });

    it("ignores non-date/int fields", () => {
        const values = { author: "Voltaire" };
        const result = dateRangeHandler(
            { author: "text" },
            { author: { start: "", end: "" } },
            { author: "exact" },
            values
        );
        expect(result.author).toBe("Voltaire");
    });

    it("handles multiple fields", () => {
        const result = dateRangeHandler(
            { year: "int", pub_date: "date" },
            {
                year: { start: "1800", end: "1900" },
                pub_date: { start: "1850", end: "1860" },
            },
            { year: "range", pub_date: "range" },
            { year: "", pub_date: "" }
        );
        expect(result.year).toBe("1800-1900");
        expect(result.pub_date).toBe("1850<=>1860");
    });
});

// ---------------------------------------------------------------------------
// deepEqual
// ---------------------------------------------------------------------------
describe("deepEqual", () => {
    it("returns true for equal primitives", () => {
        expect(deepEqual(1, 1)).toBe(true);
        expect(deepEqual("a", "a")).toBe(true);
        expect(deepEqual(null, null)).toBe(true);
    });

    it("returns false for different primitives", () => {
        expect(deepEqual(1, 2)).toBe(false);
        expect(deepEqual("a", "b")).toBe(false);
    });

    it("returns true for deeply equal objects", () => {
        expect(deepEqual({ a: 1, b: { c: 2 } }, { a: 1, b: { c: 2 } })).toBe(true);
    });

    it("returns false for different objects", () => {
        expect(deepEqual({ a: 1 }, { a: 2 })).toBe(false);
        expect(deepEqual({ a: 1 }, { a: 1, b: 2 })).toBe(false);
    });

    it("returns true for equal arrays", () => {
        expect(deepEqual([1, 2, 3], [1, 2, 3])).toBe(true);
    });

    it("returns false for different arrays", () => {
        expect(deepEqual([1, 2], [1, 3])).toBe(false);
    });
});

// ---------------------------------------------------------------------------
// extractSurfaceFromCollocate
// ---------------------------------------------------------------------------
describe("extractSurfaceFromCollocate", () => {
    it("extracts plain collocates", () => {
        const result = extractSurfaceFromCollocate([["word", 10], ["other", 5]]);
        expect(result).toEqual([
            { collocate: "word", surfaceForm: "word", count: 10 },
            { collocate: "other", surfaceForm: "other", count: 5 },
        ]);
    });

    it("strips lemma: prefix from display but preserves in surfaceForm", () => {
        const result = extractSurfaceFromCollocate([["lemma:faire", 20]]);
        expect(result[0].collocate).toBe("faire");
        expect(result[0].surfaceForm).toBe("lemma:faire");
        expect(result[0].count).toBe(20);
    });

    it("strips attribute notation from display", () => {
        const result = extractSurfaceFromCollocate([["word:pos:noun", 15]]);
        expect(result[0].collocate).toBe("word");
        expect(result[0].surfaceForm).toBe("word:pos:noun");
    });
});

// ---------------------------------------------------------------------------
// sortResults
// ---------------------------------------------------------------------------
describe("sortResults", () => {
    const data = {
        a: { count: 10, metadata: {} },
        b: { count: 30, metadata: {} },
        c: { count: 20, metadata: {} },
    };

    it("sorts by count descending by default", () => {
        const result = sortResults(data, "count");
        expect(result[0].label).toBe("b");
        expect(result[1].label).toBe("c");
        expect(result[2].label).toBe("a");
    });

    it("sorts by label when sortKey is 'label'", () => {
        const result = sortResults(data, "label");
        expect(result[0].label).toBe("a");
    });
});

// ---------------------------------------------------------------------------
// isOnlyFacetChange
// ---------------------------------------------------------------------------
describe("isOnlyFacetChange", () => {
    it("returns true when only facet param changed", () => {
        const oldUrl = { q: "test", facet: "author" };
        const newUrl = { q: "test", facet: "title" };
        expect(isOnlyFacetChange(newUrl, oldUrl)).toBe(true);
    });

    it("returns false when non-facet param changed", () => {
        const oldUrl = { q: "test", facet: "author" };
        const newUrl = { q: "other", facet: "author" };
        expect(isOnlyFacetChange(newUrl, oldUrl)).toBe(false);
    });

    it("returns false when nothing changed", () => {
        const url = { q: "test", facet: "author" };
        expect(isOnlyFacetChange(url, url)).toBe(false);
    });

    it("handles relative_frequency and word_property as facet params", () => {
        const oldUrl = { q: "test" };
        const newUrl = { q: "test", relative_frequency: "true", word_property: "lemma" };
        expect(isOnlyFacetChange(newUrl, oldUrl)).toBe(true);
    });

    it("handles collocation_method and similarity_by as facet params", () => {
        const oldUrl = { q: "test" };
        const newUrl = { q: "test", collocation_method: "compare", similarity_by: "author" };
        expect(isOnlyFacetChange(newUrl, oldUrl)).toBe(true);
    });
});

// ---------------------------------------------------------------------------
// buildTocTree
// ---------------------------------------------------------------------------
describe("buildTocTree", () => {
    it("creates flat tree from div1 elements", () => {
        const elements = [
            { philo_type: "div1", label: "Ch1" },
            { philo_type: "div1", label: "Ch2" },
        ];
        const tree = buildTocTree(elements);
        expect(tree.length).toBe(2);
        expect(tree[0].children.length).toBe(0);
    });

    it("nests div2 under div1", () => {
        const elements = [
            { philo_type: "div1", label: "Ch1" },
            { philo_type: "div2", label: "Sec1" },
            { philo_type: "div2", label: "Sec2" },
            { philo_type: "div1", label: "Ch2" },
        ];
        const tree = buildTocTree(elements);
        expect(tree.length).toBe(2);
        expect(tree[0].children.length).toBe(2);
        expect(tree[0].children[0].label).toBe("Sec1");
        expect(tree[1].children.length).toBe(0);
    });

    it("nests div3 under div2 under div1", () => {
        const elements = [
            { philo_type: "div1", label: "Ch1" },
            { philo_type: "div2", label: "Sec1" },
            { philo_type: "div3", label: "Sub1" },
        ];
        const tree = buildTocTree(elements);
        expect(tree.length).toBe(1);
        expect(tree[0].children[0].children[0].label).toBe("Sub1");
    });

    it("handles empty input", () => {
        expect(buildTocTree([])).toEqual([]);
    });
});
