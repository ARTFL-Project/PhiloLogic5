"""Test fixtures for PhiloLogic5 testing suite."""

from .corpus_manager import CorpusConfig, CorpusManager
from .corpus_configs import SHAKESPEARE, ELTEC

__all__ = [
    "CorpusConfig",
    "CorpusManager",
    "SHAKESPEARE",
    "ELTEC",
]
