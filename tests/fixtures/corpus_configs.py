"""Predefined corpus configurations for PhiloLogic5 tests."""

from pathlib import Path

from .corpus_manager import CorpusConfig

TESTS_ROOT = Path(__file__).parent.parent
COLLECTIONS_DIR = TESTS_ROOT / "collections"
LOAD_CONFIGS_DIR = TESTS_ROOT / "load_configs"

# =============================================================================
# Shakespeare Corpus
# =============================================================================

SHAKESPEARE = CorpusConfig(
    name="shakespeare",
    source_dir=COLLECTIONS_DIR / "folger-shakespeare",
    load_config=LOAD_CONFIGS_DIR / "shakespeare_config.py",
    header="tei",
)

# =============================================================================
# ELTeC Corpus
# =============================================================================

ELTEC = CorpusConfig(
    name="eltec",
    source_dir=COLLECTIONS_DIR / "ELTeC-eng",
    load_config=LOAD_CONFIGS_DIR / "eltec_config.py",
    header="tei",
)
