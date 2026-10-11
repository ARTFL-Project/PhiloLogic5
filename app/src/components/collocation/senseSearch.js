// Finding a word among the senses: the senses whose words (or those next to the query, or those of the collocation
// filter that alone mark a sense) include it.
const GROUPS = [["words", 0], ["before", 1], ["after", 2], ["hidden", 0]];

// Whether a word of a sense is the one looked for: the word itself (its lemma, when lemmas are counted), or, next to
// the query, the form the text has (sa in "sa volonté", for the lemma son)
export function matches(w, term, queryWord = "") {
    const looked = (term || "").trim().toLocaleLowerCase();
    if (!looked) return false;
    if (w.word.toLocaleLowerCase() === looked) return true;
    const phrase = (w.phrase || "").toLocaleLowerCase();
    if (!phrase) return false;
    if (phrase === looked) return true;
    return looked !== queryWord.toLocaleLowerCase() && phrase.split(/[\s'’-]+/).includes(looked);
}

// The word of a sense that is the one looked for, and how markedly more frequent it is there, or null
export function inSense(node, term, queryWord = "") {
    for (const [group, at] of GROUPS) {
        const w = (node[group] || []).find((x) => matches(x, term, queryWord));
        if (w) return { word: w, group: at, ratio: w.ratio };
    }
    return null;
}

// For each sense whose words include the one looked for, the node it does at its most markedly (a sense goes on
// through several nodes as it is refined), in the order of the senses' numbers
export function findWord(nodes, term, queryWord = "") {
    const best = new Map();
    for (const node of nodes) {
        if (!node.sense) continue;
        const hit = inSense(node, term, queryWord);
        if (hit && (!best.has(node.sense) || hit.ratio > best.get(node.sense).ratio)) {
            best.set(node.sense, { sense: node.sense, rank: node.rank, node, ...hit });
        }
    }
    return [...best.values()].sort((a, b) => a.rank - b.rank);
}

// The words a search can find: those the senses list, and next to the query, the forms the text has
export function searchableWords(nodes) {
    const out = new Set();
    for (const node of nodes) {
        if (!node.sense) continue;
        for (const [group] of GROUPS) for (const w of node[group] || []) out.add(w.phrase || w.word);
    }
    return [...out].sort((a, b) => a.localeCompare(b));
}
