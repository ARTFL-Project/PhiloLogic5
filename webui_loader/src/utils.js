export function formatSize(bytes) {
    if (bytes === null || bytes === undefined) {
        return "";
    }
    const units = ["B", "KB", "MB", "GB", "TB"];
    let size = bytes;
    let unit = 0;
    while (size >= 1024 && unit < units.length - 1) {
        size /= 1024;
        unit += 1;
    }
    return unit === 0 ? `${size} ${units[unit]}` : `${size.toFixed(1)} ${units[unit]}`;
}

export function formatDate(seconds, locale) {
    if (!seconds) {
        return "";
    }
    return new Date(seconds * 1000).toLocaleString(locale);
}

export function formatDuration(seconds) {
    if (seconds === null || seconds === undefined || seconds < 0) {
        return "";
    }
    const total = Math.round(seconds);
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const rest = total % 60;
    if (hours) {
        return `${hours} h ${String(minutes).padStart(2, "0")} min`;
    }
    if (minutes) {
        return `${minutes} min ${String(rest).padStart(2, "0")} s`;
    }
    return `${rest} s`;
}

export function clone(value) {
    return value === undefined ? undefined : JSON.parse(JSON.stringify(value));
}

export function deepEqual(a, b) {
    return JSON.stringify(a) === JSON.stringify(b);
}

const HTML_ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" };
const escapeHtml = (text) => text.replace(/[&<>"]/g, (character) => HTML_ESCAPES[character]);
const JSON_TOKEN = /("(?:\\.|[^"\\\n])*")(\s*:)?|\b(true|false|null)\b|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|([{}[\],])/g;

// JSON text as HTML with its keys, strings, numbers, literals and punctuation in spans (classes json-*), all of it
// escaped. Text which isn't valid JSON (being typed) is highlighted as far as it goes.
export function highlightJson(text) {
    let html = "";
    let last = 0;
    for (const match of (text || "").matchAll(JSON_TOKEN)) {
        html += escapeHtml(text.slice(last, match.index));
        const [token, string, colon, literal, number, punctuation] = match;
        if (string !== undefined) {
            html += `<span class="${colon ? "json-key" : "json-string"}">${escapeHtml(string)}</span>${colon ? `<span class="json-punctuation">${escapeHtml(colon)}</span>` : ""}`;
        } else if (literal !== undefined) {
            html += `<span class="json-literal">${literal}</span>`;
        } else if (number !== undefined) {
            html += `<span class="json-number">${number}</span>`;
        } else if (punctuation !== undefined) {
            html += `<span class="json-punctuation">${escapeHtml(punctuation)}</span>`;
        } else {
            html += escapeHtml(token);
        }
        last = match.index + token.length;
    }
    return html + escapeHtml((text || "").slice(last));
}

const MAX_DIFF_CELLS = 1000000;

// The lines of two texts which differ, with context lines around them: rows of {type: same, removed, added or gap,
// text}. Lines are matched by their longest common subsequence, except in long texts, where the lines between the
// first and the last difference are all shown as changed.
export function lineDiff(before, after, context = 2) {
    const a = before.split("\n");
    const b = after.split("\n");
    let start = 0;
    while (start < a.length && start < b.length && a[start] === b[start]) {
        start += 1;
    }
    let endA = a.length;
    let endB = b.length;
    while (endA > start && endB > start && a[endA - 1] === b[endB - 1]) {
        endA -= 1;
        endB -= 1;
    }
    const middleA = a.slice(start, endA);
    const middleB = b.slice(start, endB);
    let middle;
    if (middleA.length * middleB.length > MAX_DIFF_CELLS) {
        middle = [...middleA.map((text) => ({ type: "removed", text })), ...middleB.map((text) => ({ type: "added", text }))];
    } else {
        // lengths[i][j]: the longest common subsequence of middleA[i:] and middleB[j:]
        const lengths = Array.from({ length: middleA.length + 1 }, () => new Uint32Array(middleB.length + 1));
        for (let i = middleA.length - 1; i >= 0; i -= 1) {
            for (let j = middleB.length - 1; j >= 0; j -= 1) {
                lengths[i][j] = middleA[i] === middleB[j] ? lengths[i + 1][j + 1] + 1 : Math.max(lengths[i + 1][j], lengths[i][j + 1]);
            }
        }
        middle = [];
        let i = 0;
        let j = 0;
        while (i < middleA.length || j < middleB.length) {
            if (i < middleA.length && j < middleB.length && middleA[i] === middleB[j]) {
                middle.push({ type: "same", text: middleA[i] });
                i += 1;
                j += 1;
            } else if (i < middleA.length && (j === middleB.length || lengths[i + 1][j] >= lengths[i][j + 1])) {
                middle.push({ type: "removed", text: middleA[i] });
                i += 1;
            } else {
                middle.push({ type: "added", text: middleB[j] });
                j += 1;
            }
        }
    }
    const rows = [...a.slice(0, start).map((text) => ({ type: "same", text })), ...middle, ...a.slice(endA).map((text) => ({ type: "same", text }))];
    // Only the context of the changes, the rest as gaps
    const near = rows.map(() => false);
    rows.forEach((row, index) => {
        if (row.type !== "same") {
            for (let other = Math.max(0, index - context); other <= Math.min(rows.length - 1, index + context); other += 1) {
                near[other] = true;
            }
        }
    });
    const shown = [];
    rows.forEach((row, index) => {
        if (near[index]) {
            shown.push(row);
        } else if (shown.length === 0 || shown[shown.length - 1].type !== "gap") {
            shown.push({ type: "gap", text: "\u22ef" }); // midline ellipsis
        }
    });
    return shown;
}
