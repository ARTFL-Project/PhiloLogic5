"""Unit tests for make_collocation_database: its files read in parallel, then combined, must give the arrays reading
them one after the other did."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import lz4.frame
import numpy as np
import pytest
from orjson import dumps, loads

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.loadtime.PostFilters import make_collocation_database

pytestmark = pytest.mark.unit

# Files of words_and_philo_ids, by name: (philo_type, token, sentence, attributes) for each line. File 3 continues
# the last sentence of file 2, file 4 has no words, and file 10 comes after file 4, as their names are numbers.
FILES = {
    "1": [
        ("word", "le", (1, 1, 1, 0, 1, 1), {"lemma": "le", "pos": "DET"}),
        ("word", "chat", (1, 1, 1, 0, 1, 1), {"lemma": "chat", "pos": "NOUN"}),
        ("punct", ".", (1, 1, 1, 0, 1, 1), {}),
        ("word", "les", (1, 1, 1, 0, 1, 2), {"lemma": "le"}),
        ("word", "chats", (1, 1, 1, 0, 1, 2), {"lemma": "chat", "pos": "NOUN"}),
    ],
    "2": [
        ("word", "un", (2, 1, 0, 0, 1, 1), {"lemma": "un", "pos": "DET"}),
        ("word", "chien", (2, 1, 0, 0, 1, 1), {"lemma": "chien", "pos": "NOUN"}),
        ("word", "le", (2, 1, 0, 0, 2, 1), {"lemma": "le", "pos": "DET"}),
    ],
    "3": [
        ("word", "chat", (2, 1, 0, 0, 2, 1), {"lemma": "chat", "pos": "NOUN"}),
        ("word", "dort", (2, 1, 0, 0, 2, 2), {"lemma": "dormir", "pos": "VERB"}),
    ],
    "4": [("punct", "!", (3, 1, 0, 0, 1, 1), {})],
    "10": [
        ("word", "chien", (4, 1, 0, 0, 1, 1), {"lemma": "chien", "pos": "NOUN"}),
        ("word", "aboie", (4, 1, 0, 0, 1, 1), {"pos": "VERB"}),
    ],
}
WORD_ATTRIBUTES = ["lemma", "pos"]


def write_files(destination):
    words_dir = destination / "words_and_philo_ids"
    words_dir.mkdir()
    for name, lines in FILES.items():
        records = []
        for position, (philo_type, token, sentence, attributes) in enumerate(lines, 1):
            philo_id = " ".join(map(str, sentence + (position, 0, 0)))
            records.append(dumps({"token": token, "position": philo_id, "philo_type": philo_type, **attributes}))
        with open(words_dir / f"{name}.lz4", "wb") as output:
            output.write(lz4.frame.compress(b"".join(record + b"\n" for record in records)))


def arrays_read_one_after_the_other(destination):
    """token ids, sentence offsets and keys, vocabulary and attribute ids and vocabularies, as the loader read them
    file after file before"""
    vocab, token_ids, sent_keys, sent_offsets = {}, [], [], [0]
    attr_vocabs = {attr_name: {} for attr_name in WORD_ATTRIBUTES}
    attr_ids = {attr_name: [] for attr_name in WORD_ATTRIBUTES}
    current_sentence = None
    word_count = 0
    files = sorted(os.scandir(destination / "words_and_philo_ids"), key=lambda entry: int(entry.name.split(".")[0]))
    for entry in files:
        with lz4.frame.open(entry.path) as input_file:
            for line in input_file:
                word_obj = loads(line)
                if word_obj["philo_type"] != "word":
                    continue
                sent_key = tuple(int(x) for x in word_obj["position"].split()[:6])
                if sent_key != current_sentence:
                    if current_sentence is not None:
                        sent_offsets.append(word_count)
                    current_sentence = sent_key
                    sent_keys.append(sent_key)
                token_ids.append(vocab.setdefault(word_obj["token"], len(vocab)))
                for attr_name in WORD_ATTRIBUTES:
                    value = word_obj.get(attr_name, "")
                    if attr_name == "lemma":
                        value = f"lemma:{value}"
                    attr_ids[attr_name].append(attr_vocabs[attr_name].setdefault(value, len(attr_vocabs[attr_name])))
                word_count += 1
    sent_offsets.append(word_count)
    return token_ids, sent_offsets, sent_keys, list(vocab), attr_ids, {k: list(v) for k, v in attr_vocabs.items()}


@pytest.mark.parametrize("cores", [1, 3])
def test_same_arrays_as_one_file_after_the_other(tmp_path, cores):
    write_files(tmp_path)
    loader = SimpleNamespace(
        destination=str(tmp_path), word_attributes=WORD_ATTRIBUTES, cores=cores, word_count=10, ascii_conversion=True
    )
    colloc_dir = tmp_path / "collocations"
    make_collocation_database(loader, str(colloc_dir))

    token_ids, sent_offsets, sent_keys, vocab, attr_ids, attr_vocabs = arrays_read_one_after_the_other(tmp_path)
    assert len(sent_keys) == 6  # the sentence continued in file 3 is one sentence
    assert np.load(colloc_dir / "token_ids.npy").tolist() == token_ids
    assert np.load(colloc_dir / "sent_offsets.npy").tolist() == sent_offsets
    assert [tuple(key) for key in np.load(colloc_dir / "sent_keys_native.npy").tolist()] == sent_keys
    assert np.load(colloc_dir / "vocab.npy", allow_pickle=True).tolist() == vocab
    for attr_name in WORD_ATTRIBUTES:
        assert np.load(colloc_dir / f"attr_{attr_name}_ids.npy").tolist() == attr_ids[attr_name]
        assert np.load(colloc_dir / f"attr_{attr_name}_vocab.npy", allow_pickle=True).tolist() == attr_vocabs[attr_name]
