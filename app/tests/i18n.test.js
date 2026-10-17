import { afterEach, describe, it, expect } from "vitest";
import i18n from "../src/i18n.js";

const { t, locale } = i18n.global;
afterEach(() => (locale.value = "en"));

describe("i18n", () => {
    it("counts as each language does", () => {
        const said = (n) => t("timeSeries.occurrences", { n }, n);
        expect([0, 1, 2].map(said)).toEqual(["0 occurrences", "1 occurrence", "2 occurrences"]);
        locale.value = "fr";
        expect([0, 1, 1.5, 2].map(said)).toEqual(["0 occurrence", "1 occurrence", "1.5 occurrence", "2 occurrences"]);
        locale.value = "es";
        expect([0, 1, 2].map(said)).toEqual(["0 ocurrencias", "1 ocurrencia", "2 ocurrencias"]);
    });
});
