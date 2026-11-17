"""Unit tests for parsing the files of a load whose spaCy model runs on the GPU, in the loading process: the parse
workers parse the files and the loading process tags their words (Loader.parse_files_around_tagger)."""

import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

SENTENCES = [
    "The king said that Paris was far.",
    "Marie went there, with the queen!",
    "Who rules the city",  # no full stop: the sentence ends with the paragraph
]


def tei(paragraphs):
    """A text of chapters of paragraphs: larger with more paragraphs"""
    chapters = []
    for chapter in range(1, paragraphs // 5 + 2):
        body = "".join(f"<p>{' '.join(SENTENCES[(chapter + p) % 3:])}</p>" for p in range(min(5, paragraphs)))
        chapters.append(f'<div type="chapter"><head>Chapter {chapter}</head>{body}</div>')
    return (
        "<TEI><teiHeader><fileDesc><titleStmt><title>A text</title><author>Someone</author></titleStmt></fileDesc>"
        f"</teiHeader><text><body>{''.join(chapters)}</body></text></TEI>"
    )


class Tagger:
    """A pipeline in the loading process, as when it runs on the GPU, tagging each word with what its text and position
    give, and no lemma for punctuation"""

    def __init__(self):
        import spacy

        self.vocab = spacy.blank("en").vocab
        self.words = 0

    def pipe(self, documents, batch_size):
        for document in documents:
            for position, token in enumerate(document):
                capitalized = token.text[:1].isupper()
                token.pos_ = "PROPN" if capitalized else ("NOUN" if position % 2 else "VERB")
                token.tag_ = token.text.upper()
                token.ent_type_ = "PER" if capitalized else ""
                token.lemma_ = token.text[::-1] if token.text.isalpha() else ""
                self.words += 1
            yield document


@pytest.fixture
def parse(tmp_path, monkeypatch):
    """A function loading texts of these numbers of paragraphs with the default load filters and the tagger, as
    philoload5 does with a spaCy model, up to the parse: it returns the files of the parse, by their path in the data directory, and the line
    counts and samples of the words and lemmas it found"""
    from philologic.loadtime import LoadFilters
    from philologic.loadtime.Loader import Loader
    from philologic.loadtime.LoadOptions import LoadOptions

    monkeypatch.chdir(tmp_path)  # parse_files works in the load's WORK directory

    def parse(paragraph_counts, serial):
        # The Loader keeps its state in class attributes, which other tests load corpora with
        for attribute in ("parser_config", "metadata_fields", "metadata_types", "metadata_hierarchy", "data_dicts",
                          "filequeue"):  # fmt: skip
            monkeypatch.setattr(Loader, attribute, type(getattr(Loader, attribute))())
        for attribute in ("post_filters", "debug", "words_to_index", "destination", "workdir", "textdir",
                          "web_app_dir", "default_object_level", "token_regex", "cores", "ascii_conversion",
                          "metadata_sql_types", "suppress_word_attributes", "nlp"):  # fmt: skip
            monkeypatch.setattr(Loader, attribute, getattr(Loader, attribute, None), raising=False)
        if serial:
            monkeypatch.setenv("PHILOLOGIC_SERIAL_GPU_PARSE", "1")
        else:
            monkeypatch.delenv("PHILOLOGIC_SERIAL_GPU_PARSE", raising=False)
        sources = tmp_path / "sources"
        sources.mkdir(exist_ok=True)
        files = []
        for number, paragraphs in enumerate(paragraph_counts):
            files.append(sources / f"text{number}.xml")
            files[-1].write_text(tei(paragraphs), encoding="utf8")
        load_options = LoadOptions()
        load_options.parse(["philoload5", "db", *map(str, files)])
        destination = tmp_path / ("serial" if serial else "split")
        load_options["db_destination"] = str(destination)
        load_options["data_destination"] = str(destination / "data")
        load_filters = LoadFilters.set_load_filters()
        load_filters.insert(-3, LoadFilters.spacy_tagger)  # where LoadOptions puts it
        load_options["load_filters"] = load_filters
        load_options["debug"] = True  # keeps the tagged records
        loader = Loader.set_class_attributes(load_options.values)
        monkeypatch.setattr(Loader, "nlp", Tagger())
        loader.add_files(load_options.files)
        loader.set_file_data(None, loader.textdir, loader.workdir)
        loader.parse_files(2, verbose=False)
        assert Loader.nlp.words > 0
        data = destination / "data"
        outputs = {str(path.relative_to(data)): path.read_bytes() for path in data.rglob("*") if path.is_file()}
        counts = (Loader.parsed_line_counts, sorted(Loader.word_sample), sorted(Loader.lemma_sample))
        return outputs, counts

    return parse


@pytest.mark.unit
def test_parsing_in_the_workers_gives_the_same_files(parse, monkeypatch):
    """The workers parsing the files and the loading process tagging them give the same files, word and lemma counts
    and samples as the loading process parsing and tagging them all, as it does with PHILOLOGIC_SERIAL_GPU_PARSE"""
    from philologic.loadtime.Loader import Loader

    split_parses = []
    parse_files_around_tagger = Loader.parse_files_around_tagger

    def counted(cls, *args):
        split_parses.append(len(cls.filequeue))
        return parse_files_around_tagger.__func__(cls, *args)

    monkeypatch.setattr(Loader, "parse_files_around_tagger", classmethod(counted))
    # Several files fit in the largest one, so several are parsed ahead of the tagging, two at once
    paragraph_counts = [12, 1, 30, 4, 2, 9, 1]
    serial_outputs, serial_counts = parse(paragraph_counts, serial=True)
    assert split_parses == []
    split_outputs, split_counts = parse(paragraph_counts, serial=False)
    assert split_parses == [len(paragraph_counts)]
    assert sorted(split_outputs) == sorted(serial_outputs)
    assert any(name.endswith(".raw") for name in split_outputs)
    assert any(name.endswith(".raw.lemma.lz4") for name in split_outputs)
    for name, content in serial_outputs.items():
        assert split_outputs[name] == content, name
    assert split_counts == serial_counts
    assert serial_counts[0][1] > 0  # lemmas were counted


@pytest.mark.unit
def test_without_the_tagger_files_are_parsed_in_the_loading_process(parse, monkeypatch):
    """A file whose load filters have no tagger is parsed in the loading process, and so are all the others"""
    from philologic.loadtime.Loader import Loader

    split_parses = []
    monkeypatch.setattr(Loader, "parse_files_around_tagger", classmethod(lambda cls, *args: split_parses.append(1)))
    monkeypatch.setattr(Loader, "text_load_filters", classmethod(lambda cls, text: []))
    parse([2, 1], serial=False)
    assert split_parses == []
