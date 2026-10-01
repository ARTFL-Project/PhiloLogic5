import { defaultKeymap, history, historyKeymap } from "@codemirror/commands";
import { json, jsonParseLinter } from "@codemirror/lang-json";
import { bracketMatching, foldGutter, foldKeymap, indentOnInput, syntaxHighlighting } from "@codemirror/language";
import { linter, lintGutter } from "@codemirror/lint";
import { Compartment, EditorState } from "@codemirror/state";
import { EditorView, highlightActiveLineGutter, keymap, lineNumbers } from "@codemirror/view";
import { classHighlighter } from "@lezer/highlight";

// The JSON editor (CodeMirror), loaded only where one is shown. Its tokens have the classes of classHighlighter, which
// the styles color (tok-*), as the JSON shown by jsonHighlight.js.

const theme = EditorView.theme({
    "&": {
        maxHeight: "24rem",
        border: "var(--bs-border-width) solid var(--bs-border-color)",
        borderRadius: "var(--bs-border-radius)",
        backgroundColor: "var(--bs-body-bg)",
        fontSize: "0.85rem",
    },
    "&.cm-focused": { outline: "none", boxShadow: "0 0 0 0.2rem rgba(128, 128, 128, 0.2)" },
    ".cm-scroller": { overflow: "auto", fontFamily: 'SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace' },
    ".cm-content": { minHeight: "4.5rem" },
    ".cm-gutters": {
        backgroundColor: "var(--bs-tertiary-bg)",
        color: "var(--bs-secondary-color)",
        borderRight: "var(--bs-border-width) solid var(--bs-border-color)",
    },
});

const editable = new Compartment();
const readOnlyExtensions = (readonly) => [EditorState.readOnly.of(readonly), EditorView.editable.of(!readonly)];

// An editor of JSON text in parent: onChange(text) after each change of its text
export function createJsonEditor(parent, { doc, readonly, onChange }) {
    return new EditorView({
        parent,
        state: EditorState.create({
            doc,
            extensions: [
                lineNumbers(),
                foldGutter(),
                highlightActiveLineGutter(),
                history(),
                indentOnInput(),
                bracketMatching(),
                json(),
                syntaxHighlighting(classHighlighter),
                linter(jsonParseLinter(), { delay: 300 }),
                lintGutter(),
                keymap.of([...defaultKeymap, ...historyKeymap, ...foldKeymap]),
                editable.of(readOnlyExtensions(readonly)),
                EditorView.updateListener.of((update) => {
                    if (update.docChanged) {
                        onChange(update.state.doc.toString());
                    }
                }),
                theme,
            ],
        }),
    });
}

export function replaceText(view, text) {
    view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: text } });
}

export function setReadonly(view, readonly) {
    view.dispatch({ effects: editable.reconfigure(readOnlyExtensions(readonly)) });
}
