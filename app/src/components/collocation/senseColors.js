// The colors of the senses: each sense numbered at the top (1, 2, ...) has its hue, the same as the usage patterns',
// and the senses carved out of it (2.1, 2.1.1, 2.2, ...) are shades of it, in the order of their numbers; the rest of
// the sentences, which no sense takes, is grey.
const HUES = [205, 30, 145, 280, 0, 90, 165, 235, 50, 315, 120, 260, 15, 60, 200, 320];
const LIGHTNESS = [42, 58, 30, 70, 50, 36]; // the shades of a hue, told apart from one another

// For each sense of a report's nodes, its place among the senses of its family (0 for the sense at the top)
export function senseShades(nodes) {
    const ranks = new Map();
    for (const node of nodes) if (node.sense) ranks.set(node.sense, node.rank);
    const out = {};
    const count = {};
    for (const [sense] of [...ranks.entries()].sort((a, b) => a[1] - b[1])) {
        const top = sense.split(".")[0];
        count[top] = (count[top] ?? -1) + 1;
        out[sense] = count[top];
    }
    return out;
}

export function senseColor(sense, alpha, shades) {
    if (!sense) return `hsla(0, 0%, 62%, ${alpha})`;
    const top = Number(sense.split(".")[0]);
    const light = LIGHTNESS[(shades[sense] ?? 0) % LIGHTNESS.length];
    return `hsla(${HUES[(top - 1) % HUES.length]}, 58%, ${light}%, ${alpha})`;
}

// Its color for text: the same hue, dark enough to read on white whatever the shade
export function senseTextColor(sense, shades) {
    if (!sense) return "#6c757d";
    const top = Number(sense.split(".")[0]);
    const light = Math.min(LIGHTNESS[(shades[sense] ?? 0) % LIGHTNESS.length], 36);
    return `hsl(${HUES[(top - 1) % HUES.length]}, 58%, ${light}%)`;
}
