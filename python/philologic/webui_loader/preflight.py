"""Checks of a load before it is launched, and what it will run: prepare() resolves the files, writes the load config
and checks everything philoload5 would stumble on, and the load is only launched from what it prepared."""

import csv
import os
import shutil
from dataclasses import dataclass, field

from philologic.loadtime.Loader import npm_node_dir, tei_header
from philologic.webui_loader import databases, files, load_config_io, load_schema
from philologic.webui_loader.files import FilesError

NPM = "/var/lib/philologic5/bin/npm"
WEB_APP = "/var/lib/philologic5/web_app"
HEADER_SAMPLE = 20  # files whose TEI header is checked
MAX_HEADER_CHECK = 20 * 1024**2  # larger files aren't read for it
DISK_FACTOR = 10  # databases take up to about 10 times the size of their files (more with word attributes)


@dataclass
class Plan:
    """A load as prepared: its errors (which prevent launching it), warnings, files and load config"""

    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    files: list = field(default_factory=list)
    summary: dict = None
    config_text: str = None
    cwd: str = None
    base: dict = None  # what the load config was read from

    def error(self, field_name, message):
        self.errors.append({"field": field_name, "message": message})

    def warning(self, field_name, message):
        self.warnings.append({"field": field_name, "message": message})

    def to_dict(self):
        return {
            "errors": self.errors,
            "warnings": self.warnings,
            "summary": self.summary,
            "config": self.config_text,
            "ok": not self.errors,
        }


def read_base(settings, base, roots):
    """Text of the load config which a load starts from, and the directory to run it in: {"config_path": path} or
    {"database": name} (the copy saved in the database)"""
    if not base:
        return None, None
    if base.get("database"):
        db_path = databases.database_path(settings, base["database"])
        if db_path is None:
            raise FilesError(f"no database {base['database']}")
        path = load_config_io.database_config_path(db_path)
        cwd = None
    elif base.get("config_path"):
        path = base["config_path"]
        files.check_allowed(path, roots)
        cwd = os.path.dirname(os.path.abspath(path))
    else:
        return None, None
    try:
        with open(path, encoding="utf8") as config_file:
            return config_file.read(), cwd
    except OSError as error:
        raise FilesError(f"{path} can't be read") from error


def memory_available():
    """Available memory in bytes, where /proc/meminfo tells it"""
    try:
        with open("/proc/meminfo", encoding="utf8") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def spacy_model_size(model):
    """Size of the files of an installed spaCy model, or None if it isn't installed"""
    try:
        import importlib.util

        spec = importlib.util.find_spec(model)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    total = 0
    for location in spec.submodule_search_locations:
        for dirpath, dirnames, filenames in os.walk(location):
            total += sum(os.path.getsize(os.path.join(dirpath, name)) for name in filenames)
    return total


_spacy_models = None


def installed_spacy_models():
    """The spaCy models installed (spaCy is imported once, the first time)"""
    global _spacy_models
    if _spacy_models is None:
        try:
            import spacy.util

            _spacy_models = sorted(spacy.util.get_installed_models())
        except Exception:
            _spacy_models = []
    return _spacy_models


def human_size(size):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024


def check_bibliography(plan, path, paths):
    """The bibliography must have a filename column naming the files (as Loader.parse_bibliography_file reads it)"""
    delimiter = "\t" if path.endswith((".tab", ".tsv")) else ","
    try:
        with open(path, encoding="utf8", newline="") as bibliography:
            rows = list(csv.DictReader(bibliography, delimiter=delimiter, skipinitialspace=True))
    except OSError:
        plan.error("bibliography", f"{path} can't be read")
        return
    except (UnicodeDecodeError, csv.Error) as error:
        plan.error("bibliography", f"{path} can't be read as UTF-8 CSV: {error}")
        return
    if not rows:
        plan.error("bibliography", f"{path} is empty")
        return
    columns = [column.strip() for column in rows[0].keys() if column]
    if "filename" not in columns:
        plan.error("bibliography", f"{path} has no filename column (its columns: {', '.join(columns)})")
        return
    listed = {row.get("filename", row.get(" filename", "")).strip() for row in rows}
    names = {os.path.basename(path) for path in paths}
    missing = sorted(names - listed)
    if missing:
        plan.warning(
            "bibliography",
            f"{len(missing)} files are not in the bibliography, such as {', '.join(missing[:5])}",
        )
    plan.summary["bibliography_columns"] = columns


def prepare(settings, jobs, request, roots, user=None, can_replace=None):
    """Check a load and prepare what it will run. request: dbname, files (spec, see files.resolve), header,
    file_type, bibliography, options ({key: JSON value}), base (see read_base), cores, overwrite. can_replace(dbname)
    says whether this user may replace an existing database (the service's permissions)."""
    plan = Plan()
    dbname = request.get("dbname") or ""
    header = request.get("header") or "tei"
    file_type = request.get("file_type") or "xml"
    options = dict(request.get("options") or {})
    cores = request.get("cores") or 4
    if header not in ("tei", "dc"):
        plan.error("header", "the header is tei or dc")
    if file_type not in ("xml", "plain_text"):
        plan.error("file_type", "the file type is xml or plain_text")

    # Global config and installation
    if not os.path.isdir(settings.database_root):
        plan.error("global", f"database_root ({settings.database_root}) doesn't exist")
    elif not os.access(settings.database_root, os.W_OK):
        plan.error("global", f"database_root ({settings.database_root}) can't be written")
    if not os.path.exists(NPM):
        plan.error("global", f"{NPM} is missing: the web client of the database can't be built")
    elif not os.path.exists(os.path.join(npm_node_dir(NPM), "node")):
        plan.error("global", f"node isn't in {npm_node_dir(NPM)}: the web client of the database can't be built")
    if not os.path.isdir(WEB_APP):
        plan.error("global", f"{WEB_APP} is missing: PhiloLogic is not installed")

    # Database
    if not databases.valid_name(dbname):
        plan.error(
            "dbname", "a database name is made of letters, digits, '.', '_' and '-', and starts with a letter or digit"
        )
    else:
        db_path = os.path.join(settings.database_root, dbname)
        if os.path.exists(db_path):
            if not request.get("overwrite"):
                plan.error("dbname", f"{dbname} already exists: confirm that it is to be replaced")
            else:
                replaceable, reason = databases.can_replace(settings, db_path)
                if not replaceable:
                    plan.error("dbname", f"{dbname} can't be replaced: {reason}")
                elif can_replace is not None and not can_replace(dbname):
                    plan.error("dbname", f"you are not allowed to replace {dbname}")
                else:
                    plan.warning("dbname", f"{dbname} will be deleted and loaded again")
        lock = jobs.running_load(dbname)
        if lock is not None:
            plan.error("dbname", f"{dbname} is being loaded (by {lock.get('user') or 'someone'})")

    # Files
    try:
        plan.files = files.resolve(request.get("files") or {}, roots)
    except FilesError as error:
        plan.error("files", str(error))
    plan.summary = files.summary(plan.files)
    if not plan.files:
        plan.error("files", "no files to load")
    if plan.summary["missing_count"]:
        plan.error(
            "files", f"{plan.summary['missing_count']} files can't be read, such as {plan.summary['missing'][0]}"
        )
    if plan.summary["duplicate_names"]:
        plan.error(
            "files",
            "several files have the same name, and only one of each would be loaded: "
            + ", ".join(plan.summary["duplicate_names"][:5]),
        )
    readable = [path for path in plan.files if path not in set(plan.summary["missing"])]
    if file_type == "xml" and header == "tei" and readable:
        step = max(1, len(readable) // HEADER_SAMPLE)
        without_header = []
        for path in readable[::step][:HEADER_SAMPLE]:
            if os.path.getsize(path) > MAX_HEADER_CHECK:
                continue
            try:
                with open(path, encoding="utf8") as text_file:
                    if tei_header(text_file.read()) is None:
                        without_header.append(os.path.basename(path))
            except (OSError, UnicodeDecodeError):
                without_header.append(os.path.basename(path))
        if without_header:
            plan.warning(
                "files",
                f"{len(without_header)} of the {min(HEADER_SAMPLE, len(readable))} files checked have no TEI header "
                f"or aren't UTF-8, and would be left out: {', '.join(without_header[:5])}",
            )
    bibliography = request.get("bibliography")
    if file_type == "plain_text" and not bibliography:
        plan.error("bibliography", "plain text files need a bibliography giving their metadata")
    if bibliography:
        try:
            files.check_allowed(bibliography, roots)
        except FilesError as error:
            plan.error("bibliography", str(error))
        else:
            check_bibliography(plan, bibliography, plan.files)

    # Load config
    config_ok = True
    try:
        base_text, plan.cwd = read_base(settings, request.get("base"), roots)
    except FilesError as error:
        plan.error("base", str(error))
        base_text, config_ok = None, False
    plan.base = request.get("base")
    if base_text is not None:
        try:
            base = load_config_io.read(base_text)
        except SyntaxError as error:
            plan.error("base", f"the load config isn't valid Python: {error}")
            config_ok = False
        else:
            if base["custom_code"] and settings.service:
                plan.error("base", "load configs with code of their own can't be used here: load from a shell")
                config_ok = False
            elif base["custom_code"]:
                plan.warning(
                    "base", "the load config has code of its own, which is kept: " + base["custom_code"][0][:200]
                )
    for key, error in load_config_io.validate(options).items():
        plan.error(key, error)
        config_ok = False
    navigable = options.get("navigable_objects", list(load_schema.default("navigable_objects")))
    level = options.get("default_object_level", load_schema.default("default_object_level"))
    if level not in navigable:
        plan.error("default_object_level", f"{level} must be one of the navigable objects")
    for key in ("lemma_file", "words_to_index"):
        path = options.get(key)
        if path:
            if not os.path.isabs(path):
                plan.error(key, "must be an absolute path")
            elif not files.within(path, roots):
                plan.error(key, f"{path} is outside the directories whose files can be loaded")
            elif not os.path.isfile(path) or not os.access(path, os.R_OK):
                plan.error(key, f"{path} can't be read")
    model = options.get("spacy_model")
    if model and load_schema.validate("spacy_model", model) is None:
        size = spacy_model_size(model)
        if settings.service and model not in installed_spacy_models():
            plan.error("spacy_model", f"{model} is not an installed spaCy model")
        elif size is None:
            plan.error("spacy_model", f"the spaCy model {model} is not installed")
        else:
            available = memory_available()
            if available is not None and size * 3 * cores > available:
                plan.warning(
                    "cores",
                    f"each of the {cores} parse processes loads its own copy of {model} (unless it runs on a GPU): "
                    f"about {human_size(size * 3 * cores)} for {human_size(available)} of available memory",
                )
    if config_ok:
        try:
            plan.config_text = load_config_io.render(options, base_text)
        except ValueError as error:
            plan.error("base", str(error))

    # Resources
    if not isinstance(cores, int) or isinstance(cores, bool) or cores < 1:
        plan.error("cores", "must be a whole number, at least 1")
    elif settings.max_cores and cores > settings.max_cores:
        plan.error(
            "cores", f"a load can use at most {settings.max_cores} cores on this machine (webui_loader_max_cores)"
        )
    elif cores > (os.cpu_count() or 1):
        plan.warning("cores", f"more than the {os.cpu_count()} processors of this machine")
    if os.path.isdir(settings.database_root) and plan.summary["size"]:
        free = shutil.disk_usage(settings.database_root).free
        needed = plan.summary["size"] * DISK_FACTOR
        if free < plan.summary["size"] * 2:
            plan.error("files", f"only {human_size(free)} free in {settings.database_root}")
        elif free < needed:
            plan.warning(
                "files",
                f"{human_size(free)} free in {settings.database_root}: databases can take up to about "
                f"{DISK_FACTOR} times the size of their files ({human_size(needed)})",
            )
    return plan
