// The colors of the senses: each sense numbered at the top (1, 2, ...) has its hue, the same as the usage patterns',
// and the senses carved out of it (2.1, 2.1.1, 2.2, ...) are shades of it; the rest of the sentences, which no sense
// takes, is grey.
const HUES = [205, 30, 145, 280, 0, 90, 165, 235, 50, 315, 120, 260, 15, 60, 200, 320];
// The shades of a family are chosen from the end of the tree up: its final senses (where each of its branches ends,
// at the finest detail), in the order the tree draws them, share the range from dark to light evenly, so that the
// whole range serves however many there are, and senses next to one another in the tree, which split from one
// another, have neighboring shades. A sense keeps the shade of the branch its number ends in, at any detail.
const LIGHTEST = 82;
const DARKEST = 26;

export function senseShades(nodes) {
    const out = {};
    if (!nodes.length) return out;
    const first = (node) => (node.children.length
        ? Math.min(...node.children.map((c) => first(nodes[c])))
        : node.sense === null ? Infinity : node.rank);
    const ends = {}; // for each sense at the top, the senses its branches end in, in the tree's order
    const walk = (node) => {
        if (node.children.length) {
            node.children.map((c) => nodes[c]).sort((a, b) => first(a) - first(b)).forEach(walk);
        } else if (node.sense) {
            (ends[node.sense.split(".")[0]] ||= []).push(node.sense);
        }
    };
    walk(nodes[0]);
    for (const senses of Object.values(ends)) {
        senses.forEach((sense, i) => {
            out[sense] = senses.length > 1 ? DARKEST + ((LIGHTEST - DARKEST) * i) / (senses.length - 1) : 46;
        });
    }
    for (const node of nodes) if (node.sense && out[node.sense] === undefined) out[node.sense] = 46;
    return out;
}

export function senseColor(sense, alpha, shades) {
    if (!sense) return `hsla(0, 0%, 62%, ${alpha})`;
    const top = Number(sense.split(".")[0]);
    return `hsla(${HUES[(top - 1) % HUES.length]}, 62%, ${shades[sense] ?? 46}%, ${alpha})`;
}

// Its color for text: the same hue, dark enough to read on white whatever the shade
export function senseTextColor(sense, shades) {
    if (!sense) return "#6c757d";
    const top = Number(sense.split(".")[0]);
    return `hsl(${HUES[(top - 1) % HUES.length]}, 62%, ${Math.min(shades[sense] ?? 46, 36)}%)`;
}
