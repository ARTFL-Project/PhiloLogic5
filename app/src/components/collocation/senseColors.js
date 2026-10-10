// The colors of the senses: each sense numbered at the top (1, 2, ...) has its hue, the same as the usage patterns',
// and the senses carved out of it (2.1, 2.1.1, 2.2, ...) are shades of it; the rest of the sentences, which no sense
// takes, is grey.
const HUES = [205, 30, 145, 280, 0, 90, 165, 235, 50, 315, 120, 260, 15, 60, 200, 320];
// The shades of a hue, far apart from one another and alternating light and dark
const LIGHTNESS = [42, 72, 24, 58, 32, 82];
const SATURATION = [62, 72, 66, 55, 70, 60];

// For each sense of a report's nodes, its shade: the sum of its number's parts after the first (1: 0; 1.1: 1; 1.2 and
// 1.1.1: 2), so that a sense and those carved out of it, and the senses carved out of the same one, differ in shade,
// whatever the detail shown
export function senseShades(nodes) {
    const out = {};
    for (const node of nodes) {
        if (node.sense) out[node.sense] = node.sense.split(".").slice(1).reduce((s, x) => s + Number(x), 0);
    }
    return out;
}

export function senseColor(sense, alpha, shades) {
    if (!sense) return `hsla(0, 0%, 62%, ${alpha})`;
    const top = Number(sense.split(".")[0]);
    const k = (shades[sense] ?? 0) % LIGHTNESS.length;
    return `hsla(${HUES[(top - 1) % HUES.length]}, ${SATURATION[k]}%, ${LIGHTNESS[k]}%, ${alpha})`;
}

// Its color for text: the same hue, dark enough to read on white whatever the shade
export function senseTextColor(sense, shades) {
    if (!sense) return "#6c757d";
    const top = Number(sense.split(".")[0]);
    const k = (shades[sense] ?? 0) % LIGHTNESS.length;
    return `hsl(${HUES[(top - 1) % HUES.length]}, ${SATURATION[k]}%, ${Math.min(LIGHTNESS[k], 36)}%)`;
}
