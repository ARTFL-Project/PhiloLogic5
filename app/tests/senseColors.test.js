import { describe, it, expect } from "vitest";
import { senseInk } from "../src/components/collocation/senseColors.js";

// WCAG's contrast ratio between two colors, from their sRGB channels
function luminance([r, g, b]) {
    const c = (v) => (v / 255 <= 0.04045 ? v / 255 / 12.92 : ((v / 255 + 0.055) / 1.055) ** 2.4);
    return 0.2126 * c(r) + 0.7152 * c(g) + 0.0722 * c(b);
}
function contrast(a, b) {
    const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
    return (hi + 0.05) / (lo + 0.05);
}
function hslToRgb(h, s, l) {
    const a = s * Math.min(l, 1 - l);
    const f = (n) => {
        const k = (n + h / 30) % 12;
        return Math.round(255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))));
    };
    return [f(0), f(8), f(4)];
}

describe("senseInk", () => {
    it("picks white or near black, whichever stands out more from the sense's color", () => {
        const white = [255, 255, 255];
        const dark = [0x21, 0x25, 0x29];
        for (const [sense, hue] of [["1", 205], ["2", 30], ["3", 145]]) {
            for (const shade of [26, 40, 54, 68, 82]) {
                const fill = hslToRgb(hue, 0.62, shade / 100);
                const ink = senseInk(sense, { [sense]: shade });
                const best = contrast(fill, white) >= contrast(fill, dark) ? "#fff" : "#212529";
                expect(ink, `${sense} at ${shade}%`).toBe(best);
            }
        }
        expect(senseInk(null, {})).toBe("#212529"); // the grey of the rest
    });
});
