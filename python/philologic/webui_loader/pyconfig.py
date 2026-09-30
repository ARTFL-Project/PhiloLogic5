"""Reading and editing PhiloLogic's Python config files (load_config.py, web_config.cfg, db.locals.py) without running
them. Assignments of literal values are evaluated, along with references to the names assigned before them (such as
citations["author"] in web configs); anything else is code, whose value can't be known. Edits only replace the lines
of the assignments they change, keeping comments, order and code."""

import ast
import re

from black import FileMode, format_str

# Lines as the Python tokenizer counts them (str.splitlines also breaks on form feeds and such, inside strings)
SOURCE_LINE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+$")

# Calls allowed in literal values, as the loader writes empty sets as set()
SAFE_CALLS = {"set": set, "frozenset": frozenset, "tuple": tuple, "list": list, "dict": dict}


class Code:
    """Value of a name bound by code: known only by running the file"""

    def __init__(self, source):
        self.source = source

    def __eq__(self, other):
        return isinstance(other, Code) and other.source == self.source

    def __repr__(self):
        return f"Code({self.source!r})"


class NotLiteral(Exception):
    """An expression which can't be evaluated without running code"""


def evaluate(node, names):
    """Value of a literal expression, in which names assigned literal values before it can be used"""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        items = [evaluate(item, names) for item in node.elts]
        return {ast.List: list, ast.Tuple: tuple, ast.Set: set}[type(node)](items)
    if isinstance(node, ast.Dict):
        if any(key is None for key in node.keys):  # ** unpacking
            raise NotLiteral
        return {evaluate(key, names): evaluate(value, names) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        raise NotLiteral
    if isinstance(node, ast.Subscript):
        container = evaluate(node.value, names)
        key = evaluate(node.slice, names)
        try:
            return container[key]
        except (KeyError, IndexError, TypeError) as error:
            raise NotLiteral from error
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = evaluate(node.operand, names)
        if isinstance(operand, (int, float)) and not isinstance(operand, bool):
            return -operand if isinstance(node.op, ast.USub) else operand
        raise NotLiteral
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in SAFE_CALLS
        and node.func.id not in names
        and not node.keywords
        and len(node.args) <= 1
    ):
        return SAFE_CALLS[node.func.id](*(evaluate(arg, names) for arg in node.args))
    raise NotLiteral


class Entry:
    """The last binding of a name in a config file"""

    def __init__(self, name, value, statement=None, assigned=True):
        self.name = name
        self.value = value  # a Code if not a literal value
        self.statement = statement  # the assignment, if it can be replaced by an edit
        self.assigned = assigned  # False when bound by another statement: import, def, for...

    @property
    def is_code(self):
        return isinstance(self.value, Code)


def bound_names(statement):
    """Names bound by a statement which isn't a simple assignment"""
    names = set()
    for node in ast.walk(statement):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[0])
    return names


class ConfigFile:
    """The names of a config file with their values, as far as they can be known without running it. Names given in
    initial_names can be used by the file without being assigned by it (the runtime runs web configs with the defaults
    already set). Raises SyntaxError if the text isn't valid Python."""

    def __init__(self, text, initial_names=None):
        self.text = text
        self.entries = {}
        self.code_statements = []  # statements other than assignments of literals and docstrings
        tree = ast.parse(text)
        lines_used = {}
        for statement in tree.body:
            for line in range(statement.lineno, statement.end_lineno + 1):
                lines_used[line] = lines_used.get(line, 0) + 1
        names = dict(initial_names or {})
        for statement in tree.body:
            source = ast.get_source_segment(text, statement)
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
                continue  # docstring
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                name = statement.targets[0].id
                alone = all(lines_used[line] == 1 for line in range(statement.lineno, statement.end_lineno + 1))
                try:
                    value = evaluate(statement.value, names)
                except NotLiteral:
                    value = Code(ast.get_source_segment(text, statement.value))
                    names.pop(name, None)
                    self.code_statements.append(source)
                else:
                    names[name] = value
                self.entries[name] = Entry(name, value, statement if alone else None)
                continue
            self.code_statements.append(source)
            for name in bound_names(statement):
                names.pop(name, None)
                self.entries[name] = Entry(name, Code(source), assigned=False)

    @property
    def values(self):
        """The names which have a literal value, with it"""
        return {name: entry.value for name, entry in self.entries.items() if not entry.is_code}

    def edited(self, assignments, comments=None, removed=()):
        """Text of the file with these assignments ({name: source of the whole assignment}): an existing assignment of
        the name is replaced (keeping a comment after it on its last line), otherwise the assignment is added at the
        end, after its comment in comments if any. The assignments of the removed names are deleted. Raises
        ValueError for a name bound by code."""
        comments = comments or {}
        lines = SOURCE_LINE.findall(self.text)
        replacements = []
        appended = []
        for name in removed:
            entry = self.entries.get(name)
            if entry is not None and entry.statement is not None and name not in assignments:
                replacements.append((entry.statement, None))
        for name, assignment in assignments.items():
            assignment = assignment.rstrip("\n")
            entry = self.entries.get(name)
            if entry is None:
                comment = "".join(f"# {line}".rstrip() + "\n" for line in comments.get(name, "").splitlines())
                appended.append(f"{comment}{assignment}\n")
            elif entry.statement is None or entry.is_code:
                raise ValueError(f"{name} is set by code, which can't be edited")
            else:
                replacements.append((entry.statement, assignment))
        for statement, assignment in sorted(replacements, key=lambda item: item[0].lineno, reverse=True):
            first, last = statement.lineno - 1, statement.end_lineno - 1
            # Offsets are in bytes of UTF-8
            if assignment is None:
                lines[first : last + 1] = []
                continue
            before = lines[first].encode("utf8")[: statement.col_offset].decode("utf8")
            after = lines[last].encode("utf8")[statement.end_col_offset :].decode("utf8")
            if not after.endswith("\n"):
                after += "\n"
            lines[first : last + 1] = [before + assignment + after]
        text = "".join(lines)
        if appended:
            if text and not text.endswith("\n"):
                text += "\n"
            text += "".join(f"\n{assignment}" for assignment in appended)
        return text


def string_source(string):
    """Source of a string, raw if it has backslashes (regexes) and can be written raw: printable (no line breaks,
    control characters...), without double quotes, and not ending with a backslash"""
    if "\\" in string and string.isprintable() and '"' not in string and not string.endswith("\\"):
        return f'r"{string}"'
    return repr(string)


def to_source(value, references=()):
    """Python source of a literal value. A dict equal to the value of one of the references ((expression, value)
    pairs) is written as its expression, such as citations["author"]."""
    if isinstance(value, dict):
        for expression, referenced in references:
            if value == referenced:
                return expression
        items = ", ".join(f"{to_source(key)}: {to_source(item, references)}" for key, item in value.items())
        return "{" + items + "}"
    if isinstance(value, list):
        return "[" + ", ".join(to_source(item, references) for item in value) + "]"
    if isinstance(value, tuple):
        items = ", ".join(to_source(item, references) for item in value)
        return f"({items},)" if len(value) == 1 else f"({items})"
    if isinstance(value, (set, frozenset)):
        if not value:
            return "set()"
        return "{" + ", ".join(sorted(to_source(item) for item in value)) + "}"
    if isinstance(value, str):
        return string_source(value)
    if value is None or isinstance(value, (bool, int, float, bytes)):
        return repr(value)
    raise TypeError(f"{type(value).__name__} values can't be written in a config file")


def assignment_source(name, value, references=(), line_length=88):
    """Source of an assignment of a literal value, formatted by black (at the line length of the web configs the
    loader writes)"""
    source = f"{name} = {to_source(value, references)}\n"
    try:
        return format_str(source, mode=FileMode(line_length=line_length))
    except Exception:  # black can't format it: keep it on one line
        return source
