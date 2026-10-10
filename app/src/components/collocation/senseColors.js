// The colors of the senses: each sense numbered at the top (1, 2, ...) has its hue, the same as the usage patterns',
// and the senses carved out of it (2.1, 2.1.1, 2.2, ...) are shades of it; the rest of the sentences, which no sense
// takes, is grey.
const HUES = [205, 30, 145, 280, 0, 90, 165, 235, 50, 315, 120, 260, 15, 60, 200, 320];
// A sense's shade derives from that of the sense it was carved out of, which its number tells (1.1.1 out of 1.1, 1.2
// out of 1): the first carved out of a sense goes to the other side of the middle (a dark sense gives a light one, a
// light one a dark one), the second the other way, by steps that shrink with depth, so that each stands apart from
// the sense it comes from, from its siblings and from its grandparent
const STEPS = [26, 16, 10];
const LIGHTEST = 84;
const DARKEST = 22;

export function senseShades(nodes) {
    const out = {};
    const shade = (sense) => {
        if (out[sense] !== undefined) return out[sense];
        const parts = sense.split(".");
        if (parts.length === 1) return (out[sense] = 46);
        const parent = shade(parts.slice(0, -1).join("."));
        const i = Number(parts[parts.length - 1]);
        const step = STEPS[Math.min(parts.length - 2, STEPS.length - 1)] * Math.ceil(i / 2);
        const away = parent < 50 ? 1 : -1;
        return (out[sense] = Math.max(DARKEST, Math.min(LIGHTEST, parent + (i % 2 ? away : -away) * step)));
    };
    for (const node of nodes) if (node.sense) shade(node.sense);
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
