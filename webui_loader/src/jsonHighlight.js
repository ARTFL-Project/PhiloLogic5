import { classHighlighter, highlightCode } from "@lezer/highlight";
import { parser } from "@lezer/json";
import { escapeHtml } from "./utils";

// JSON text as lines of HTML, its tokens in spans of the classes of their kinds (tok-propertyName, tok-string,
// tok-number, tok-bool, tok-keyword for null, tok-punctuation: classHighlighter, as in the JSON editor), escaped. The
// text is parsed as a whole, so that keys are told from other strings; it needn't be valid.
export function highlightJsonLines(text) {
    const lines = [""];
    highlightCode(
        text,
        parser.parse(text),
        classHighlighter,
        (code, classes) => {
            const html = escapeHtml(code);
            lines[lines.length - 1] += classes ? `<span class="${classes}">${html}</span>` : html;
        },
        () => lines.push(""),
    );
    return lines;
}
