"""Unit tests for the loader's merges and index builds by ranges of words (and lemmas), which must give the same
output as without ranges: split_sorted, the sampling of sort keys, Loader.merge_files, the index parts and the
frequency files built from ranges of the sorted files."""

import bisect
import filecmp
import io
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import zlib
from collections import Counter
from itertools import groupby
from pathlib import Path

import lmdb
import lz4.frame
import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.loadtime import Loader as loader_module
from philologic.loadtime import split_sorted as split_sorted_module
from philologic.loadtime.Loader import (
    SORT_BY_ID,
    SORT_BY_WORD,
    WORD_SAMPLE_INTERVAL,
    Loader,
    count_and_sample_lines,
    index_lemmas,
    index_word_attributes,
    index_words,
    join_unique_lines,
    merge_indexes,
)
from philologic.loadtime.PostFilters import (
    count_lemma_runs,
    frequency_file_key,
    lemma_and_attribute_frequencies,
    open_lz4_lines,
    word_frequencies,
    write_lemma_counts,
    write_lemma_frequencies,
    write_unique_word_attributes,
    write_word_frequency_table,
    write_word_frequency_table_from_runs,
)
from philologic.loadtime.split_sorted import range_of, sort_key, split_sorted


def gnu_tools_available():
    """The loader's merges need GNU sort and the lz4 command"""
    if shutil.which("lz4") is None or shutil.which("lz4cat") is None:
        return False
    try:
        return "GNU" in subprocess.run(["sort", "--version"], capture_output=True, text=True).stdout
    except OSError:
        return False


pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not gnu_tools_available(), reason="needs GNU sort and lz4"),
]

C_LOCALE = {**os.environ, "LC_ALL": "C"}
SKIPPED_ATTRIBUTES = Loader.attributes_to_skip

# Words whose C locale order, and order among words of the same frequency, are easy to get wrong
TRICKY_WORDS = ["a", "A", "aa", "ab", "b", "B", "e", "é", "É", "ê", "œuvre", "ſ", "l'", "-", "--", "1789", "1790"]
TRICKY_WORDS += ["z", "zz", "ça", "Ça", "ß", "ﬁn"]


def gnu_sort(data, *options):
    """Output of LC_ALL=C sort on data (bytes)"""
    return subprocess.run(["sort", *options], input=data, capture_output=True, env=C_LOCALE, check=True).stdout


def lines_of(data):
    """Lines of data, each with its newline (bytes.splitlines would also split lines at \\r)"""
    return [line + b"\n" for line in data.split(b"\n")[:-1]]


def lmdb_items(path):
    env = lmdb.open(str(path), readonly=True, lock=False)
    with env.begin() as txn:
        items = list(txn.cursor())
    env.close()
    return items


def lz4_lines(path, byte_range=None):
    with open_lz4_lines(str(path), byte_range) as input_file:
        return list(input_file)


class Corpus:
    """Documents with the words and lemmas files a parse writes: word\\t{word}\\t{philo_id}\\t{attributes}"""

    def __init__(self, documents=7, words_per_document=700, seed=5):
        rng = random.Random(seed)
        letters = "abcdeéèfghijklmnopqrstuvwxyzœç'-"
        vocabulary = list(TRICKY_WORDS)
        while len(vocabulary) < 300:
            word = "".join(rng.choice(letters) for _ in range(rng.randrange(1, 9)))
            if word not in vocabulary:
                vocabulary.append(word)
        rng.shuffle(vocabulary)
        weights = [1 / (rank + 1) for rank in range(len(vocabulary))]  # a few frequent words, many rare ones
        self.documents = {}
        for doc in range(1, documents + 1):
            word_lines, lemma_lines = [], []
            for position in range(words_per_document):
                word = rng.choices(vocabulary, weights)[0]
                byte = 100 + 7 * position
                philo_id = (
                    f"{doc} {1 + position // 300} {1 + position // 100} 0 {1 + position // 40} {1 + position // 12}"
                    f" {position + 1} {1 + position // 250} {byte}"
                )
                attributes = {"pos": rng.choice(["NOUN", "VERB", "ADJ"]), "lemma": word.lower(), "start_byte": byte}
                if rng.random() < 0.3:
                    attributes["gender"] = rng.choice(["fem", "masc"])
                if rng.random() < 0.1:
                    attributes["n"] = rng.randrange(3)  # not a string
                attributes = json.dumps(attributes, ensure_ascii=False)
                word_lines.append(f"word\t{word}\t{philo_id}\t{attributes}\n")
                lemma_lines.append(f"lemma\t{word.lower()}\t{philo_id}\t{attributes}\n")
            self.documents[f"doc{doc}"] = (word_lines, lemma_lines)
        self.word_lines = [line for word_lines, _ in self.documents.values() for line in word_lines]
        self.lemma_lines = [line for _, lemma_lines in self.documents.values() for line in lemma_lines]
        sort_options = shlex.split(f"{SORT_BY_WORD} {SORT_BY_ID}")
        self.sort_options = sort_options
        # What the merges write: all lines sorted as each words file is (see LoadFilters.generate_words_sorted)
        self.sorted_words = gnu_sort("".join(self.word_lines).encode("utf-8"), *sort_options)
        self.sorted_lemmas = gnu_sort("".join(self.lemma_lines).encode("utf-8"), *sort_options)

    def write_files(self, workdir):
        """Write the sorted words file and the lemmas file of each document, as parse_text leaves them"""
        for name, (word_lines, lemma_lines) in self.documents.items():
            sorted_words = gnu_sort("".join(word_lines).encode("utf-8"), *self.sort_options)
            (workdir / f"{name}.words.sorted.lz4").write_bytes(lz4.frame.compress(sorted_words))
            (workdir / f"{name}.raw.lemma.lz4").write_bytes(lz4.frame.compress("".join(lemma_lines).encode("utf-8")))

    def sample(self, lines):
        """Sort keys of one line out of 37: more than count_and_sample_lines would take from such small files"""
        return [sort_key(line.encode("utf-8")) for line in lines[::37]]


def make_loader(workdir, destination):
    """Loader class with the attributes merge_files and build_inverted_index use (see Loader.set_class_attributes)"""

    class TestLoader(Loader):
        pass

    TestLoader.workdir, TestLoader.destination = str(workdir), str(destination)
    TestLoader.debug = True  # merges keep their input files
    TestLoader.cores = 8  # 4 ranges
    TestLoader.word_sample = TestLoader.lemma_sample = None
    TestLoader.overflow_words = set()
    TestLoader.precomputed_files = {}
    TestLoader.has_attributes = False
    return TestLoader


def build(root, corpus, ranges, file_num=1000, precomputed=True):
    """Merge the corpus files, build the index and write the frequency files, by ranges or not. Without precomputed,
    the frequency files are written by the post-filters from the merged files, as before the index wrote them."""
    workdir, destination = root / "workdir", root / "data"
    workdir.mkdir(parents=True)
    destination.mkdir()
    corpus.write_files(workdir)
    loader = make_loader(workdir, destination)
    if ranges:
        loader.word_sample, loader.lemma_sample = corpus.sample(corpus.word_lines), corpus.sample(corpus.lemma_lines)
    cwd = os.getcwd()
    os.chdir(workdir)  # the merges name their input files relative to it
    try:
        loader.__new__(loader).merge_files("words", file_num)
        loader.__new__(loader).merge_files("lemmas", file_num)
    finally:
        os.chdir(cwd)
    loader.word_count, loader.lemma_count = len(corpus.word_lines), len(corpus.lemma_lines)
    loader.build_inverted_index()
    loader.registered_files = dict(loader.precomputed_files)
    if not precomputed:
        loader.precomputed_files = {}
    word_frequencies(loader)
    lemma_and_attribute_frequencies(loader)
    return loader


@pytest.fixture(scope="module")
def corpus():
    return Corpus()


@pytest.fixture(scope="module")
def builds(corpus, tmp_path_factory):
    """reference: the loader without ranges or precomputed frequency files, which the others must give the same
    output as: without ranges; by ranges; by ranges of several batches of files merged first"""
    root = tmp_path_factory.mktemp("builds")
    return {
        "reference": build(root / "reference", corpus, ranges=False, precomputed=False),
        "whole": build(root / "whole", corpus, ranges=False),
        "ranges": build(root / "ranges", corpus, ranges=True),
        "ranges_batches": build(root / "ranges_batches", corpus, ranges=True, file_num=3),
    }


class TestSortKey:
    def test_same_order_as_gnu_sort(self):
        """Sorting on sort_key, stably, gives the order of sort -s -k 2,2: fields start with their blanks"""
        rng = random.Random(1)
        alphabet = [" ", "\t", "\r", "a", "b", "é", "-", "0"]
        lines = ["".join(rng.choice(alphabet) for _ in range(rng.randrange(12))) for _ in range(3000)]
        data = "".join(line + "\n" for line in lines).encode("utf-8")
        assert b"".join(sorted(lines_of(data), key=sort_key)) == gnu_sort(data, "-s", "-k", "2,2")

    def test_key_of_word_lines(self):
        assert sort_key("word\tété\t1 2 0 0 1 1 1 1 100\t{}\n".encode("utf-8")) == "\tété".encode("utf-8")
        assert sort_key(b"x\n") == b""
        assert sort_key(b"a\tb\nword\tc\t1\n", 4) == b"\tc"

    def test_range_of(self):
        boundaries = [b"\tb", b"\td", b"\tf"]
        for key in [b"", b"\ta", b"\tb", b"\tc", b"\td", b"\tdd", b"\tf", b"\tz"]:
            assert range_of(boundaries, key) == bisect.bisect_right(boundaries, key)


class TestSplitSorted:
    @pytest.fixture
    def boundaries(self, corpus):
        keys = sorted({sort_key(line) for line in lines_of(corpus.sorted_words)})
        return [
            b"\t",  # less than all keys: no lines in range 0
            keys[len(keys) // 4],  # a key: its lines start range 2
            keys[len(keys) // 2] + b"\x00",  # between two keys
            keys[3 * len(keys) // 4],
            b"\t\xff",  # more than all keys: no lines in the last range
        ]

    def check_ranges(self, data, boundaries, prefix):
        output = b""
        for index in range(len(boundaries) + 1):
            path = Path(f"{prefix}.{index}")
            lines = lz4_lines(path) if path.exists() else []
            assert path.exists() == (0 < index < len(boundaries))  # files for ranges with lines only
            for line in lines:
                assert index == 0 or boundaries[index - 1] <= sort_key(line)
                assert index == len(boundaries) or sort_key(line) < boundaries[index]
            output += b"".join(lines)
        assert output == data

    @pytest.mark.parametrize("chunk_size", [7, 1000, 1 << 22])
    def test_ranges_of_the_lines(self, corpus, boundaries, tmp_path, chunk_size):
        """Chunks smaller than a line, chunks with several ranges and a single chunk"""
        data = corpus.sorted_words
        split_sorted(io.BytesIO(data), boundaries, tmp_path / "part", chunk_size)
        self.check_ranges(data, boundaries, tmp_path / "part")

    def test_script(self, corpus, boundaries, tmp_path):
        """As merge_ranges runs it: at the end of a pipeline, boundaries in a file"""
        (tmp_path / "boundaries").write_bytes(b"".join(boundary + b"\n" for boundary in boundaries))
        subprocess.run(
            [sys.executable, split_sorted_module.__file__, tmp_path / "boundaries", tmp_path / "part"],
            input=corpus.sorted_words,
            check=True,
        )
        self.check_ranges(corpus.sorted_words, boundaries, tmp_path / "part")

    def test_unsorted_input(self, tmp_path):
        """Found where a chunk starts in an earlier range than the previous one: here, chunks of a line or less"""
        with pytest.raises(ValueError, match="sorted"):
            split_sorted(io.BytesIO(b"word\tc\t1\nword\ta\t1\n"), [b"\tb"], tmp_path / "part", chunk_size=4)

    def test_last_line_without_newline(self, tmp_path):
        with pytest.raises(ValueError, match="newline"):
            split_sorted(io.BytesIO(b"word\ta\t1\nword\tc\t1"), [b"\tb"], tmp_path / "part")


class TestOpenLz4Lines:
    def test_byte_ranges_of_frames(self, tmp_path):
        """A file of several lz4 frames, as merge_ranges joins them: each range of whole frames reads alone"""
        frames = [b"a\nb\n", b"c\n" * 5000, b"d\ne\n"]
        path, ranges, start = tmp_path / "joined.lz4", [], 0
        with open(path, "wb") as output:
            for frame in frames:
                output.write(lz4.frame.compress(frame))
                ranges.append((start, output.tell()))
                start = output.tell()
        assert lz4_lines(path) == lines_of(b"".join(frames))
        for frame, byte_range in zip(frames, ranges):
            assert lz4_lines(path, byte_range) == lines_of(frame)
        assert lz4_lines(path, (ranges[1][0], ranges[2][1])) == lines_of(frames[1] + frames[2])


class TestCountAndSampleLines:
    @pytest.mark.parametrize("compressed", [False, True])
    def test_count_and_sample(self, tmp_path, compressed):
        """Over 16 MB, read in several blocks, with lines across them"""
        rng = random.Random(2)
        lines = [f"word\t{'x' * rng.randrange(1, 90)}{number}\t1 2 3\t{{}}\n".encode() for number in range(300000)]
        data = b"".join(lines)
        assert len(data) > 1 << 24
        path = tmp_path / "doc.words.sorted"
        path.write_bytes(lz4.frame.compress(data) if compressed else data)
        count, sample = count_and_sample_lines(str(path), "doc.xml", compressed)
        assert count == len(lines)
        first = zlib.crc32(b"doc.xml") % WORD_SAMPLE_INTERVAL
        assert sample == [sort_key(line) for line in lines[first::WORD_SAMPLE_INTERVAL]]

    @pytest.mark.parametrize("data, lines", [(b"", 0), (b"a\tb\n", 1), (b"a\tb\na\tc", 1)])
    def test_count_as_wc(self, tmp_path, data, lines):
        (tmp_path / "file").write_bytes(data)
        assert count_and_sample_lines(str(tmp_path / "file"), "file")[0] == lines

    def test_sample_depends_on_the_name(self, tmp_path):
        """Files with fewer lines than the sampling interval are sampled too, some of them"""
        (tmp_path / "file").write_bytes(b"".join(f"word\tw{n}\t1\t{{}}\n".encode() for n in range(10)))
        samples = [count_and_sample_lines(str(tmp_path / "file"), f"doc{n}")[1] for n in range(1000)]
        assert 0 < sum(1 for sample in samples if sample) < 1000


class TestRangeBoundaries:
    def test_boundaries(self, tmp_path):
        loader = make_loader(tmp_path, tmp_path)
        assert loader.range_boundaries("words") is None
        loader.word_sample = [b"\t%03d" % n for n in range(1000)]
        loader.lemma_sample = [b"\tsame"] * 50
        assert loader.range_boundaries("words") == [b"\t250", b"\t500", b"\t750"]  # max(4, cores // 2) ranges
        assert loader.range_boundaries("lemmas") == [b"\tsame"]
        loader.cores = 32
        assert len(loader.range_boundaries("words")) == 15


class TestBuildByRanges:
    """Merges, index and frequency files by ranges, compared with the loader without them"""

    def test_ranges(self, builds):
        for name in ("reference", "whole"):
            assert builds[name].word_ranges is None and builds[name].lemma_ranges is None
        for name in ("ranges", "ranges_batches"):
            assert len(builds[name].word_ranges) == 4 and len(builds[name].lemma_ranges) == 4

    @pytest.mark.parametrize("name", ["whole", "ranges", "ranges_batches"])
    def test_merged_files(self, builds, corpus, name):
        """The merges write all lines sorted, whether by ranges or not"""
        for merged, expected in [
            ("all_words_sorted.lz4", corpus.sorted_words),
            ("all_lemmas_sorted.lz4", corpus.sorted_lemmas),
        ]:
            for loader in (builds["reference"], builds[name]):
                assert b"".join(lz4_lines(Path(loader.workdir) / merged)) == expected

    @pytest.mark.parametrize("name", ["ranges", "ranges_batches"])
    @pytest.mark.parametrize("file_type", ["words", "lemmas"])
    def test_ranges_are_whole_frames_of_whole_words(self, builds, name, file_type):
        loader = builds[name]
        path = Path(loader.workdir) / f"all_{file_type}_sorted.lz4"
        ranges = loader.word_ranges if file_type == "words" else loader.lemma_ranges
        assert ranges[0][0] == 0 and ranges[-1][1] == path.stat().st_size
        assert all(previous[1] == following[0] for previous, following in zip(ranges, ranges[1:]))
        range_lines = [lz4_lines(path, byte_range) for byte_range in ranges]
        assert sum(range_lines, []) == lz4_lines(path)
        for previous, following in zip(range_lines, range_lines[1:]):
            assert sort_key(previous[-1]) < sort_key(following[0])

    @pytest.mark.parametrize("name", ["whole", "ranges", "ranges_batches"])
    def test_index(self, builds, name):
        for database in ("words.lmdb", "lemmas.lmdb"):
            expected = lmdb_items(Path(builds["reference"].destination) / database)
            assert expected and lmdb_items(Path(builds[name].destination) / database) == expected
        assert builds[name].has_attributes is True
        assert builds[name].all_word_attribute_names[3] == builds["reference"].all_word_attribute_names[3]
        assert builds[name].overflow_words == builds["reference"].overflow_words

    @pytest.mark.parametrize("name", ["whole", "ranges", "ranges_batches"])
    def test_frequency_files(self, builds, name):
        reference, loader = builds["reference"], builds[name]
        files = ["word_frequencies", "lemmas", "word_attributes", "lemma_word_attributes"]
        for file_name in files:
            assert filecmp.cmp(
                f"{reference.destination}/frequencies/{file_name}",
                f"{loader.destination}/frequencies/{file_name}",
                shallow=False,
            ), file_name
        assert filecmp.cmp(f"{reference.workdir}/all_frequencies", f"{loader.workdir}/all_frequencies", shallow=False)

    @pytest.mark.parametrize("name", ["whole", "ranges", "ranges_batches"])
    def test_precomputed_files_used(self, builds, name):
        """The frequency files written while building the index were those the post-filters would write"""
        registered = builds[name].registered_files
        assert sorted(key[0] for key in registered) == [
            "write_lemma_frequencies",
            "write_unique_word_attributes",
            "write_unique_word_attributes",
            "write_word_frequency_table",
        ]
        assert not any(os.path.exists(path) for path in registered.values())  # moved in place by the post-filters


@pytest.fixture
def small_limits(monkeypatch):
    """Overflow files for words of more than 40 occurrences, philo_ids packed 7 at a time"""
    monkeypatch.setattr(loader_module, "OVERFLOW_LIMIT", 36 * 40)
    monkeypatch.setattr(loader_module, "PHILO_ID_PACK_CHUNK", 7)


def directory_files(path):
    return {entry.name: entry.read_bytes() for entry in Path(path).iterdir()}


@pytest.mark.usefixtures("small_limits")
class TestIndexPartsByRanges:
    """Index parts of each range merged by merge_indexes, compared with a single part, with overflow files"""

    def index_by_ranges(self, tmp_path, function, sorted_file, ranges, *args):
        whole = function(sorted_file, str(tmp_path / "whole.lmdb"), str(tmp_path / "whole_overflow"), *args)
        parts = [
            function(sorted_file, str(tmp_path / f"part.{i}.lmdb"), str(tmp_path / "parts_overflow"), *args, byte_range)
            for i, byte_range in enumerate(ranges)
        ]
        merge_indexes([str(tmp_path / f"part.{i}.lmdb") for i in range(len(ranges))], str(tmp_path / "merged.lmdb"))
        assert lmdb_items(tmp_path / "merged.lmdb") == lmdb_items(tmp_path / "whole.lmdb")
        assert directory_files(tmp_path / "parts_overflow") == directory_files(tmp_path / "whole_overflow")
        return whole, parts

    @pytest.fixture
    def overflow_dirs(self, tmp_path):
        (tmp_path / "whole_overflow").mkdir()
        (tmp_path / "parts_overflow").mkdir()

    @pytest.mark.usefixtures("overflow_dirs")
    def test_words(self, builds, tmp_path):
        loader = builds["ranges"]
        words_file = f"{loader.workdir}/all_words_sorted.lz4"
        whole, parts = self.index_by_ranges(
            tmp_path, index_words, words_file, loader.word_ranges, False, SKIPPED_ATTRIBUTES, 50
        )
        count, overflow_keys, has_attributes, runs = whole
        assert overflow_keys  # some words overflowed
        assert sum(part[0] for part in parts) == count
        assert sum((part[1] for part in parts), []) == overflow_keys
        assert has_attributes is True and all(part[2] for part in parts)
        assert sum((part[3] for part in parts), []) == runs
        assert runs == [(word, len(list(lines))) for word, lines in groupby(lz4_lines(words_file), word_of)]

    @pytest.mark.usefixtures("overflow_dirs")
    def test_lemmas(self, builds, tmp_path):
        loader = builds["ranges"]
        lemmas_file = f"{loader.workdir}/all_lemmas_sorted.lz4"
        (count, overflow_keys), parts = self.index_by_ranges(
            tmp_path, index_lemmas, lemmas_file, loader.lemma_ranges, 50
        )
        assert overflow_keys
        assert sum(part[0] for part in parts) == count
        assert sum((part[1] for part in parts), []) == overflow_keys

    @pytest.mark.usefixtures("overflow_dirs")
    @pytest.mark.parametrize("file_type, prefix", [("words", ""), ("lemmas", "lemma:")])
    def test_attributes(self, builds, tmp_path, file_type, prefix):
        loader = builds["ranges"]
        sorted_file = f"{loader.workdir}/all_{file_type}_sorted.lz4"
        ranges = loader.word_ranges if file_type == "words" else loader.lemma_ranges
        whole, parts = self.index_by_ranges(
            tmp_path, index_word_attributes, sorted_file, ranges, prefix, SKIPPED_ATTRIBUTES, 50, True
        )
        assert sum(part[0] for part in parts) == whole[0]
        assert sum((part[1] for part in parts), []) == whole[1]
        assert set().union(*(part[2] for part in parts)) == whole[2] == {"pos", "lemma", "start_byte", "gender", "n"}


def word_of(line):
    return line.split(b"\t", 2)[1]


class TestFrequencyFiles:
    def test_word_frequency_table(self, tmp_path):
        """Same table as uniq -c | sort -rn -k 1,1, whose last resort comparison orders words of the same count"""
        counts = {word: 3 for word in TRICKY_WORDS}
        counts.update({"a": 9, "b": 10, "B": 10, "c": 100, "é": 9, "zz": 1, "Z": 1})
        lines = [f"word\t{word}\t1 2 0 0 1 1 {n} 1 1\t{{}}\n" for word, count in counts.items() for n in range(count)]
        sorted_words = gnu_sort("".join(lines).encode("utf-8"), *shlex.split(f"{SORT_BY_WORD} {SORT_BY_ID}"))
        (tmp_path / "words.lz4").write_bytes(lz4.frame.compress(sorted_words))
        write_word_frequency_table(str(tmp_path / "words.lz4"), str(tmp_path / "shell_table"))
        runs = [(word, len(list(lines))) for word, lines in groupby(lines_of(sorted_words), word_of)]
        write_word_frequency_table_from_runs(runs, str(tmp_path / "table"))
        assert (tmp_path / "table").read_bytes() == (tmp_path / "shell_table").read_bytes()

    def test_lemma_counts_by_ranges(self, builds, tmp_path):
        loader = builds["ranges"]
        lemmas_file = f"{loader.workdir}/all_lemmas_sorted.lz4"
        write_lemma_frequencies(lemmas_file, str(tmp_path / "whole"))
        lemma_count = Counter()
        for byte_range in loader.lemma_ranges:
            lemma_count.update(count_lemma_runs(lemmas_file, byte_range))
        write_lemma_counts(lemma_count, str(tmp_path / "ranges"))
        assert (tmp_path / "ranges").read_bytes() == (tmp_path / "whole").read_bytes()

    @pytest.mark.parametrize("file_type, prefix", [("words", ""), ("lemmas", "lemma:")])
    def test_unique_word_attributes_by_ranges(self, builds, tmp_path, file_type, prefix):
        loader = builds["ranges"]
        sorted_file = f"{loader.workdir}/all_{file_type}_sorted.lz4"
        ranges = loader.word_ranges if file_type == "words" else loader.lemma_ranges
        write_unique_word_attributes(sorted_file, str(tmp_path / "whole"), prefix, SKIPPED_ATTRIBUTES)
        parts = [str(tmp_path / f"part.{i}") for i in range(len(ranges))]
        for part, byte_range in zip(parts, ranges):
            write_unique_word_attributes(sorted_file, part, prefix, SKIPPED_ATTRIBUTES, byte_range)
        join_unique_lines(parts, str(tmp_path / "joined"))
        assert (tmp_path / "joined").read_bytes() == (tmp_path / "whole").read_bytes()
        assert not any(os.path.exists(part) for part in parts)

    def test_frequency_file_key(self, tmp_path):
        """Keys tell whether a file was written from the same input, with the same arguments"""
        path = tmp_path / "input"
        path.write_bytes(b"a\n")
        key = frequency_file_key(write_unique_word_attributes, (str(path), "output", "", {"b", "a"}))
        assert key == frequency_file_key(write_unique_word_attributes, (str(path), "other output", "", {"a", "b"}))
        assert key != frequency_file_key(write_unique_word_attributes, (str(path), "output", "lemma:", {"a", "b"}))
        assert key != frequency_file_key(write_lemma_frequencies, (str(path), "output", "", {"a", "b"}))
        path.write_bytes(b"a\nb\n")
        assert key != frequency_file_key(write_unique_word_attributes, (str(path), "output", "", {"a", "b"}))


class TestMergeIndexes:
    def write_index(self, path, items):
        env = lmdb.open(str(path), map_size=1 << 30)
        with env.begin(write=True) as txn:
            for key, value in items:
                txn.put(key, value)
        env.close()

    def test_merge(self, tmp_path, monkeypatch):
        """Entries of all databases in key order, the value of the last database for keys in several of them"""
        monkeypatch.setattr(loader_module, "MERGE_COMMIT_BYTES", 100)  # several commits
        first = [(b"a", b"1" * 50), (b"c", b"first"), (b"e", b"3")]
        second = [(b"b", b"2" * 60), (b"c", b"second"), (b"d", b"4" * 70)]
        self.write_index(tmp_path / "first.lmdb", first)
        self.write_index(tmp_path / "second.lmdb", second)
        merge_indexes([str(tmp_path / "first.lmdb"), str(tmp_path / "second.lmdb")], str(tmp_path / "merged.lmdb"))
        assert lmdb_items(tmp_path / "merged.lmdb") == sorted(dict(first + second).items())
        # Written without writemap: the file has the size of its pages, not of the map (ls -lh shows the right size)
        assert (tmp_path / "merged.lmdb" / "data.mdb").stat().st_size < 1 << 20
