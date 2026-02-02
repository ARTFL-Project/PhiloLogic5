"""
Corpus Manager for PhiloLogic5 Tests

Handles:
- On-demand corpus building
- Corpus caching with hash-based invalidation
- Subset corpus generation for fast tests

Test databases are built in the standard PhiloLogic location defined in
/etc/philologic/philologic5.cfg under a 'test_dbs' subdirectory.
"""

import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.loadtime.LoadOptions import CONFIG_FILE


@dataclass
class CorpusConfig:
    """Configuration for a test corpus."""

    name: str
    source_dir: Path
    subset_files: Optional[list[str]] = None  # None means all files
    load_config: Optional[Path] = None
    header: str = "tei"
    cores: int = 32

    def get_source_files(self) -> list[str]:
        """Get list of source files to include in corpus."""
        if self.subset_files:
            return self.subset_files
        source_path = Path(self.source_dir)
        return sorted([f.name for f in source_path.iterdir() if f.is_file() and f.suffix == ".xml"])

    def get_hash(self) -> str:
        """Generate hash based on source files and config for cache invalidation."""
        hasher = hashlib.sha256()

        # Hash corpus name and configuration
        hasher.update(self.name.encode())
        hasher.update(self.header.encode())

        # Hash source files (names and mtimes)
        source_files = self.get_source_files()
        source_path = Path(self.source_dir)
        for f in sorted(source_files):
            file_path = source_path / f
            hasher.update(f.encode())
            if file_path.exists():
                hasher.update(str(file_path.stat().st_mtime).encode())

        # Hash load config if present
        if self.load_config and self.load_config.exists():
            hasher.update(self.load_config.read_bytes())

        # Hash relevant PhiloLogic source files for cache invalidation
        # This ensures cache is invalidated when parser/loader code changes
        philo_files = [
            "python/philologic/loadtime/Loader.py",
            "python/philologic/loadtime/Parser.py",
            "python/philologic/loadtime/LoadFilters.py",
            "python/philologic/loadtime/PostFilters.py",
            "python/philologic/loadtime/OHCOVector.py",
        ]
        for pf in philo_files:
            full_path = REPO_ROOT / pf
            if full_path.exists():
                hasher.update(full_path.read_bytes())

        return hasher.hexdigest()[:16]


class CorpusManager:
    """Manages test corpus building and caching.

    Test databases are built in the standard PhiloLogic database_root
    from /etc/philologic/philologic5.cfg under a 'test_dbs' subdirectory.
    """

    MANIFEST_FILE = "manifest.json"

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize corpus manager.

        Args:
            cache_dir: Optional custom cache directory. If not provided,
                      uses database_root/test_dbs from philologic5.cfg
        """
        if cache_dir:
            self.database_root = cache_dir
        else:
            self.database_root = Path(CONFIG_FILE.database_root) / "test_dbs"
        self.url_root = CONFIG_FILE.url_root.rstrip("/") + "/test_dbs"
        self.database_root.mkdir(parents=True, exist_ok=True)

    def get_corpus(self, config: CorpusConfig, force_rebuild: bool = False) -> Path:
        """Get or build a test corpus.

        Args:
            config: Corpus configuration
            force_rebuild: Force rebuild even if cached

        Returns:
            Path to the corpus data directory
        """
        corpus_hash = config.get_hash()
        corpus_path = self.database_root / f"{config.name}_{corpus_hash}"

        if not force_rebuild and self._is_valid_corpus(corpus_path, corpus_hash):
            return corpus_path / "data"

        # Build the corpus
        self._build_corpus(config, corpus_path)
        return corpus_path / "data"

    def _is_valid_corpus(self, corpus_path: Path, expected_hash: str) -> bool:
        """Check if cached corpus is valid."""
        manifest_path = corpus_path / self.MANIFEST_FILE
        if not manifest_path.exists():
            return False

        try:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("hash") != expected_hash:
                return False
            # Also verify the database files exist
            data_path = corpus_path / "data"
            if not (data_path / "toms.db").exists():
                return False
            if not (data_path / "words.lmdb").exists():
                return False
            return True
        except (json.JSONDecodeError, KeyError):
            return False

    def _build_corpus(self, config: CorpusConfig, dest_path: Path) -> None:
        """Build a PhiloLogic corpus.

        Args:
            config: Corpus configuration
            dest_path: Destination path for the built corpus
        """
        # Import here to avoid circular imports
        from philologic.loadtime import LoadFilters, Parser, PostFilters
        from philologic.loadtime.Loader import (
            Loader,
            setup_db_dir,
            DEFAULT_OBJECT_LEVEL,
            NAVIGABLE_OBJECTS,
            ASCII_CONVERSION,
        )

        # Clean any existing partial build
        if dest_path.exists():
            shutil.rmtree(dest_path)

        dest_path.mkdir(parents=True)
        db_destination = dest_path
        data_destination = dest_path / "data"

        # Setup database directory
        setup_db_dir(str(db_destination), force_delete=True)

        # Prepare source files
        source_files = config.get_source_files()
        file_paths = [str(Path(config.source_dir) / f) for f in source_files]

        # Build configuration values
        load_values = {
            "dbname": config.name,
            "files": file_paths,
            "database_root": str(self.database_root),
            "url_root": self.url_root,
            "db_destination": str(db_destination),
            "data_destination": str(data_destination),
            "destination": "./",
            "default_object_level": DEFAULT_OBJECT_LEVEL,
            "navigable_objects": NAVIGABLE_OBJECTS,
            "load_filters": LoadFilters.set_load_filters(),
            "post_filters": PostFilters.DefaultPostFilters,
            "plain_text_obj": [],
            "parser_factory": Parser.XMLParser,
            "token_regex": Parser.TOKEN_REGEX,
            "ascii_conversion": ASCII_CONVERSION,
            "doc_xpaths": Parser.DEFAULT_DOC_XPATHS,
            "tag_to_obj_map": Parser.DEFAULT_TAG_TO_OBJ_MAP,
            "metadata_to_parse": Parser.DEFAULT_METADATA_TO_PARSE,
            "pseudo_empty_tags": [],
            "suppress_tags": [],
            "break_apost": True,
            "chars_not_to_index": Parser.CHARS_NOT_TO_INDEX,
            "break_sent_in_line_group": False,
            "tag_exceptions": Parser.TAG_EXCEPTIONS,
            "join_hyphen_in_words": True,
            "abbrev_expand": True,
            "long_word_limit": 200,
            "flatten_ligatures": True,
            "lowercase_index": True,
            "cores": config.cores,
            "sort_order": ["year", "author", "title", "filename"],
            "header": config.header,
            "debug": False,
            "words_to_index": set(),
            "suppress_word_attributes": set(),
            "file_type": "xml",
            "sentence_breakers": [],
            "punctuation": Parser.PUNCTUATION,
            "pos": "",
            "metadata_sql_types": {},
            "lemma_file": None,
            "spacy_model": None,
            "load_config": "",
        }

        # Load external config if provided
        if config.load_config and config.load_config.exists():
            from philologic.utils import load_module

            external_config = load_module("external_load_config", str(config.load_config))
            for attr in dir(external_config):
                if not attr.startswith("__"):
                    value = getattr(external_config, attr)
                    if not callable(value) and attr in load_values:
                        load_values[attr] = value

        # Execute the load pipeline
        loader = Loader.set_class_attributes(load_values)
        loader.add_files(file_paths)
        load_metadata = loader.parse_metadata(load_values["sort_order"], header=config.header)
        loader.set_file_data(load_metadata, loader.textdir, loader.workdir)
        loader.parse_files(config.cores)
        loader.merge_objects()
        loader.count_words()
        loader.build_inverted_index()
        loader.setup_sql_load()
        loader.post_processing()
        loader.finish()

        # Write manifest
        manifest = {
            "hash": config.get_hash(),
            "name": config.name,
            "source_files": source_files,
            "source_dir": str(config.source_dir),
            "build_timestamp": datetime.now().isoformat(),
            "file_count": len(source_files),
        }
        (dest_path / self.MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))

    def clear_cache(self) -> None:
        """Clear all cached corpora."""
        if self.database_root.exists():
            shutil.rmtree(self.database_root)
        self.database_root.mkdir(parents=True)

    def list_cached(self) -> list[dict]:
        """List all cached corpora.

        Returns:
            List of manifest dictionaries for cached corpora
        """
        cached = []
        if not self.database_root.exists():
            return cached

        for item in self.database_root.iterdir():
            if item.is_dir():
                manifest_path = item / self.MANIFEST_FILE
                if manifest_path.exists():
                    try:
                        manifest = json.loads(manifest_path.read_text())
                        manifest["path"] = str(item)
                        cached.append(manifest)
                    except json.JSONDecodeError:
                        pass

        return cached

    def get_corpus_info(self, data_path: Path) -> dict:
        """Get information about a built corpus.

        Args:
            data_path: Path to the corpus data directory

        Returns:
            Dictionary with corpus statistics
        """
        import sqlite3

        info = {"data_path": str(data_path)}

        # Check if database exists
        toms_db = data_path / "toms.db"
        if toms_db.exists():
            conn = sqlite3.connect(str(toms_db))
            cursor = conn.cursor()

            # Get document count
            cursor.execute("SELECT COUNT(*) FROM toms WHERE philo_type='doc'")
            info["doc_count"] = cursor.fetchone()[0]

            # Get word count
            cursor.execute("SELECT COUNT(*) FROM toms WHERE philo_type='word'")
            info["word_count"] = cursor.fetchone()[0]

            conn.close()

        return info
