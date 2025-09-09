#!/var/lib/philologic5/philologic_env/bin/python3

import hashlib
import mmap
import os
import struct
import subprocess
import sys
from collections import deque
from itertools import product
from operator import eq, le
from pathlib import Path

import lmdb
import msgspec
import numba
import numpy as np
import regex as re
from philologic.runtime import HitList
from philologic.runtime.QuerySyntax import group_terms, parse_query
from unidecode import unidecode

numba.config.CACHE_DIR = "/tmp/numba"

OBJECT_LEVEL = {"para": 5, "sent": 6}


class WordData(msgspec.Struct):
    full_array: np.ndarray = None
    array: np.ndarray = None
    start: int = 0
    first_doc: int = 0


def query(
    db,
    terms,
    corpus_file=None,
    method=None,
    method_arg=None,
    filename="",
    query_debug=False,
    sort_order=None,
    raw_results=False,
    raw_bytes=False,
    ascii_conversion=True,
    object_level="sent",
):
    """Runs concordance queries"""
    sys.stdout.flush()
    parsed = parse_query(terms)
    grouped = group_terms(parsed)
    split = split_terms(grouped)
    words_per_hit = len(split)
    if not filename:
        hfile = str(os.getpid()) + ".hitlist"
    dir = db.path + "/hitlists/"
    filename = filename or (dir + hfile)
    if not os.path.exists(filename):
        Path(filename).touch()
    frequency_file = db.path + "/frequencies/normalized_word_frequencies"
    pid = os.fork()
    if pid == 0:  # In child process
        os.umask(0)
        os.setsid()
        pid = os.fork()  # double fork to detach completely from parent
        if pid > 0:
            os._exit(0)
        else:
            with open(f"{filename}.terms", "w") as terms_file:
                expand_query_not(
                    split, frequency_file, terms_file, db.locals.ascii_conversion, db.locals["lowercase_index"]
                )
            method_arg = int(method_arg) if method_arg else 0
            if method == "single_term":
                # Search for one term
                search_word(db.path, filename, db.locals.overflow_words, corpus_file=corpus_file)
            elif method == "phrase_ordered":
                # Phrase searching where words need to be in a specific order with no words in between
                search_phrase(db.path, filename, db.locals.overflow_words, corpus_file=corpus_file)
            elif method == "phrase_unordered":
                # Phrase searching where words need to be in a specific order with possible words in between
                search_within_word_span(db.path, filename, db.locals.overflow_words, method_arg or 1, False, False, corpus_file=corpus_file)
            elif method == "proxy_ordered":
                # Proximity searching with possible words in between
                search_within_word_span(db.path, filename, db.locals.overflow_words, method_arg or 1, True, False, corpus_file=corpus_file)
            elif method == "proxy_unordered":
                # Proximity searching with possible words in between unordered
                search_within_word_span(db.path, filename, db.locals.overflow_words, method_arg or 1, False, False, corpus_file=corpus_file)
            elif method == "exact_cooc_ordered":
                # Co-occurrence searching where words need to be within n words of each other
                search_within_word_span(db.path, filename, db.locals.overflow_words, method_arg or 1, True, True, corpus_file=corpus_file)
            elif method == "exact_cooc_unordered":
                # Co-occurrence searching where words need to be within n words of each othera and can be unordered
                search_within_word_span(db.path, filename, db.locals.overflow_words, method_arg or 1, False, True, corpus_file=corpus_file)
            elif method == "sentence_ordered":  # no support for para search for now
                # Co-occurrence searching where words need to be within an object irrespective of word order
                search_within_text_object(db.path, filename, db.locals.overflow_words, object_level, True, corpus_file=corpus_file)
            elif method == "sentence_unordered":
                # Co-occurrence searching where words need to be within an object irrespective of word order
                search_within_text_object(db.path, filename, db.locals.overflow_words, object_level, False, corpus_file=corpus_file)

            with open(filename + ".done", "w") as flag:  # do something to mark query as finished
                flag.write(" ".join(sys.argv) + "\n")
                flag.flush()  # make sure the file is written to disk. Otherwise we get an infinite loop with 0 hits
            os._exit(0)  # Exit child process
    else:
        hits = HitList.HitList(
            filename,
            words_per_hit,
            db,
            method=method,
            sort_order=sort_order,
            raw=raw_results,
            raw_bytes=raw_bytes,
            ascii_conversion=ascii_conversion,
        )
        return hits


def get_expanded_query(hitlist):
    fn = hitlist.filename + ".terms"
    query = []
    term = []
    try:
        grep_results = open(fn, "r", encoding="utf8")
    except:
        return []
    for line in grep_results:
        if line == "\n":
            query.append(term)
            term = []
        else:
            term.append('"' + line[:-1] + '"')
    if term:
        query.append(term)
    return query


def split_terms(grouped):
    split = []
    for group in grouped:
        if len(group) == 1:
            kind, token = group[0]
            if kind == "QUOTE" and token.find(" ") > 1:  # we can split quotes on spaces if there is no OR
                for split_tok in token[1:-1].split(" "):
                    split.append((("QUOTE", '"' + split_tok + '"'),))
            elif kind == "RANGE":
                split.append((("TERM", token),))
            else:
                split.append(group)
        else:
            split.append(group)
    return split


def get_word_array(txn, word, overflow_words, db_path):
    """Returns numpy array either from LMDB buffer or memmap"""
    if word not in overflow_words:
        buffer = txn.get(word.encode("utf8"))
        if buffer is None:
            return np.array([], dtype="u4").reshape(-1, 9)
        return np.frombuffer(buffer, dtype="u4").reshape(-1, 9)
    file_path = os.path.join(db_path, "overflow_words", f"{hashlib.sha256(word.encode('utf8')).hexdigest()}.bin")
    return np.memmap(file_path, dtype="u4", mode='r').reshape(-1, 9)


def __filter_philo_ids_on_void(corpus_philo_ids, philo_ids):
    """Filter philo_ids based on corpus metadata."""

    def __rows_as_void(rows):
        arr_contiguous = np.ascontiguousarray(rows) # ensure array is C-contiguous for a valid view
        # Create a dtype for a row; itemsize is element_size_in_bytes * num_columns
        void_dtype = np.dtype((np.void, arr_contiguous.dtype.itemsize * arr_contiguous.shape[1]))
        return arr_contiguous.view(void_dtype).ravel()

    corpus_philo_ids_void = __rows_as_void(corpus_philo_ids)
    philo_ids_void = __rows_as_void(philo_ids)
    matching_indices_void = np.isin(philo_ids_void, corpus_philo_ids_void)
    return matching_indices_void


def filter_philo_ids(corpus_file, philo_ids) -> np.ndarray:
    """Filter philo_ids based on corpus metadata."""
    with open(corpus_file, "rb") as corpus:
        buffer = corpus.read()
        corpus_philo_ids = np.frombuffer(buffer, dtype="u4").reshape(-1, 7)
    corpus_padded = np.pad(corpus_philo_ids, ((0, 0), (0, 1)), 'constant', constant_values=0)
    actual_corpus_lengths = np.argmax(corpus_padded == 0, axis=1)
    if np.all(actual_corpus_lengths == actual_corpus_lengths[0]): # check if all rows have the same length
        object_level = actual_corpus_lengths[0]
        matching_indices = __filter_philo_ids_on_void(corpus_philo_ids[:, :object_level], philo_ids[:, :object_level])
        return philo_ids[matching_indices]
    else:
        unique_lengths = np.unique(actual_corpus_lengths) # get unique lengths
        num_philo_rows = philo_ids.shape[0]
        overall_match_mask = np.zeros(num_philo_rows, dtype=bool)

        for current_len in unique_lengths:
            # Create a mask for the corpus_philo_ids that match the current length
            corpus_rows_for_this_len_mask = (actual_corpus_lengths == current_len)

            # Extract these actual corpus prefixes (all are of length current_len)
            relevant_corpus_prefixes = corpus_philo_ids[corpus_rows_for_this_len_mask, :current_len]
            philo_ids_prefixes = philo_ids[:, :current_len]
            current_matching_indices = __filter_philo_ids_on_void(relevant_corpus_prefixes, philo_ids_prefixes)
            overall_match_mask |= current_matching_indices
        return philo_ids[overall_match_mask]


def search_word(db_path, hitlist_filename, overflow_words, corpus_file=None):
    """Search for a single word in the database."""
    with open(f"{hitlist_filename}.terms", "r") as terms_file:
        words = terms_file.read().split()
    env = lmdb.open(f"{db_path}/words.lmdb", readonly=True, lock=False, readahead=False)
    if len(words) == 1:
        with env.begin(buffers=True) as txn, open(hitlist_filename, "wb") as output_file:
            word = words[0]
            if corpus_file is None:
                if word not in overflow_words:
                    buffer = txn.get(word.encode("utf8"))
                    output_file.write(buffer)
                else:
                    file_path = os.path.join(db_path, "overflow_words", f"{hashlib.sha256(word.encode('utf8')).hexdigest()}.bin")
                    with open(file_path, "rb") as overflow_file:
                        output_file.write(overflow_file.read())
            else:
                word_array = get_word_array(txn, word, overflow_words, db_path)
                filtered_philo_ids = filter_philo_ids(
                    corpus_file,
                    word_array,
                )
                output_file.write(filtered_philo_ids.tobytes())
    else:
        with env.begin(buffers=True) as txn, open(hitlist_filename, "wb") as output_file:
            for philo_ids in merge_word_group(db_path, txn, words, overflow_words):
                if corpus_file is None:
                    output_file.write(philo_ids.tobytes())
                else:
                    filtered_philo_ids = filter_philo_ids(
                        corpus_file,
                        philo_ids,
                    )
                    output_file.write(filtered_philo_ids.tobytes())
    env.close()


def search_phrase(db_path, hitlist_filename, overflow_words, corpus_file=None):
    """Phrase searches where words need to be in a specific order"""
    word_groups = get_word_groups(f"{hitlist_filename}.terms")
    common_object_ids = get_cooccurrence_groups(
        db_path, word_groups, overflow_words, corpus_file=corpus_file, cooc_order=True
    )
    mapping_order = next(common_object_ids)
    with open(hitlist_filename, "wb") as output_file:
        for philo_id_groups in common_object_ids:
            for group_combination in product(*philo_id_groups):
                raw_positions = [a[7] for a in group_combination]
                mapped_raw_positions = [p for _, p in sorted(zip(mapping_order, raw_positions))]
                positions = sorted(mapped_raw_positions)
                if positions != mapped_raw_positions:  # are positions in sorted order?
                    continue
                # we now need to check if the positions are within 1 word of each other
                if positions[0] + len(word_groups) - 1 == positions[-1]:
                    starting_id = group_combination[0].tobytes()
                    for group_num in range(1, len(word_groups)):
                        starting_id += group_combination[group_num][7:].tobytes()
                    output_file.write(starting_id)


def search_within_word_span(db_path, hitlist_filename, overflow_words, n, cooc_order, exact_distance, corpus_file=None):
    """Search for co-occurrences of multiple words within n words of each other in the database."""
    word_groups = get_word_groups(f"{hitlist_filename}.terms")
    common_object_ids = get_cooccurrence_groups(
        db_path, word_groups, overflow_words, corpus_file=corpus_file, cooc_order=cooc_order
    )

    if cooc_order is True:
        mapping_order = next(common_object_ids)

    if len(word_groups) > 1 and n == 1:
        n = len(word_groups) - 1

    if exact_distance is True:
        comp = eq  # distance between words equals n
    else:
        comp = le  # distance between words is less than or equal to n
    with open(hitlist_filename, "wb") as output_file:
        for philo_id_groups in common_object_ids:
            hit_cache = set()
            for group_combination in product(*philo_id_groups):
                if cooc_order is True:
                    raw_positions = [a[7] for a in group_combination]
                    mapped_raw_positions = [p for _, p in sorted(zip(mapping_order, raw_positions))]
                    positions = sorted(mapped_raw_positions)
                    if positions != mapped_raw_positions:  # are positions in sorted order?
                        continue
                else:
                    positions: list[int] = sorted({philo_id[7:8][0] for philo_id in group_combination})
                # we now need to check if the positions are within n words of each other
                if len(positions) != len(word_groups):  # we had duplicate words
                    continue
                if comp(positions[-1] - positions[0], n):
                    group_combination = sorted(group_combination, key=lambda x: x[-1])
                    starting_id = group_combination[0].tobytes()
                    for group_num in range(1, len(word_groups)):
                        starting_id += group_combination[group_num][7:].tobytes()
                    if starting_id not in hit_cache:
                        hit_cache.add(starting_id)
                        output_file.write(starting_id)


def search_within_text_object(db_path, hitlist_filename, overflow_words, level, cooc_order, corpus_file=None):
    """Search for co-occurrences of multiple words in the same sentence in the database."""
    word_groups = get_word_groups(f"{hitlist_filename}.terms")
    common_object_ids = get_cooccurrence_groups(
        db_path,
        word_groups,
        overflow_words,
        level=level,
        corpus_file=corpus_file,
        cooc_order=cooc_order,
    )

    if cooc_order is True:
        mapping_order = next(common_object_ids)

    with open(hitlist_filename, "wb") as output_file:
        for philo_id_groups in common_object_ids:
            hit_cache = set()
            for group_combination in product(*philo_id_groups):
                if cooc_order is True:
                    raw_positions = [a[7] for a in group_combination]
                    mapped_raw_positions = [p for _, p in sorted(zip(mapping_order, raw_positions))]
                    positions = sorted(mapped_raw_positions)
                    if positions != mapped_raw_positions:  # are positions in sorted order?
                        continue
                else:
                    positions: list[int] = sorted({philo_id[7] for philo_id in group_combination})
                if len(set(positions)) != len(word_groups):  # we had duplicate words
                    continue
                group_combination = sorted(group_combination, key=lambda x: x[-1])
                starting_id = group_combination[0].tobytes()
                for group_num in range(1, len(word_groups)):
                    starting_id += group_combination[group_num][7:].tobytes()
                if starting_id not in hit_cache:
                    hit_cache.add(starting_id)
                    output_file.write(starting_id)


def get_word_groups(terms_file):
    word_groups = []
    with open(terms_file, "r") as terms_file:
        word_group = []
        for line in terms_file:
            word = line.strip()
            if word:
                word_group.append(word)
            elif word_group:
                word_groups.append(word_group)
                word_group = []
        if word_group:
            word_groups.append(word_group)
    return word_groups


def get_cooccurrence_groups(
    db_path, word_groups, overflow_words, level="sent", corpus_file=None, cooc_order=False
):
    cooc_slice = 6
    if level == "para":
        cooc_slice = 5
    env = lmdb.open(f"{db_path}/words.lmdb", readonly=True, lock=False, readahead=False)
    with env.begin(buffers=True) as txn:
        # Determine which group has the smallest byte size
        byte_size_per_group = []
        for index, group in enumerate(word_groups):
            byte_size = 0
            for word in group:
                if word not in overflow_words:
                    byte_size += len(txn.get(word.encode("utf8")))
                else:
                    file_path = os.path.join(db_path, "overflow_words", f"{hashlib.sha256(word.encode('utf8')).hexdigest()}.bin")
                    byte_size += os.stat(file_path).st_size
            byte_size_per_group.append(byte_size)
        # Perform an argsort on the list to get the indices of the groups sorted by byte size
        sorted_indices = np.argsort(byte_size_per_group)
        if cooc_order is True:
            yield sorted_indices

        def one_word_generator(word):
            yield get_word_array(txn, word, overflow_words, db_path)

        # Process each word group
        first_group_data = np.array([])
        group_generators = []
        for index in sorted_indices:
            words = word_groups[index]
            if index == sorted_indices[0]:  # grab the entire first group
                if len(words) == 1:
                    first_group_data = get_word_array(txn, words[0], overflow_words, db_path)
                else:
                    first_group_data = np.concatenate([i for i in merge_word_group(db_path, txn, words, overflow_words)], dtype="u4")
            else:
                if len(words) == 1:
                    group_generators.append(one_word_generator(words[0]))
                else:
                    group_generators.append(merge_word_group(db_path, txn, words, overflow_words, chunk_size=36 * 1000))

        if corpus_file is not None:
            first_group_data = filter_philo_ids(corpus_file, first_group_data)

        group_data = [None for _ in range(len(word_groups) - 1)]  # Start with None for each group
        break_out = False
        previous_row = None
        match = True
        for index in first_group_data:
            philo_id_object = index[:cooc_slice]
            if previous_row is not None and compare_rows(philo_id_object, previous_row) == 0:
                if match is True:
                    results[0] = index.reshape(-1, 9)  # replace the previous row with the current row
                    yield results
                continue
            results = deque()
            match = True
            previous_row = philo_id_object
            for group_index, philo_id_group in enumerate(group_generators):
                if group_data[group_index] is None:
                    philo_id_array = next(philo_id_group)  # load the first chunk
                else:
                    philo_id_array = group_data[group_index]

                if philo_id_array.shape[0] == 0:  # type: ignore
                    break_out = True
                    break

                # Is the first row greater than the current philo_id_object?
                if compare_rows(philo_id_array[0, :cooc_slice], philo_id_object) == 1:
                    match = False
                    group_data[group_index] = philo_id_array
                    break

                # Is the last row less than the current philo_id_object?
                while compare_rows(philo_id_array[-1, :cooc_slice], philo_id_object) == -1:
                    try:
                        philo_id_array = next(philo_id_group)  # load the next chunk
                    except StopIteration:  # no more philo_ids in this group, we are done
                        break_out = True
                        break

                if break_out is True:
                    break

                # Find matching rows
                matching_indices = find_matching_indices_sorted(philo_id_array, philo_id_object, cooc_slice)
                matching_rows = philo_id_array[matching_indices]
                group_data[group_index] = philo_id_array
                if matching_rows.shape[0] == 0:  # no match found
                    match = False
                    break
                if matching_indices.shape[0] > 0:
                    if matching_indices[-1] + 1 == philo_id_array.shape[0]:
                        try:
                            group_data[group_index] = next(philo_id_group)  # load the next chunk
                        except StopIteration:
                            break_out = True
                    else:
                        group_data[group_index] = philo_id_array[matching_indices[-1] + 1 :]  # slice off matching rows

                results.append(matching_rows)  # We only keep the first instance of a hit in the first group

            if break_out is True:
                break
            elif match is True:
                results.appendleft(index.reshape(-1, 9))  # We only keep the first instance of a hit in the first group
                yield results

    env.close()


@numba.jit(nopython=True, cache=True)
def compare_rows(row1, row2):
    for i in range(len(row1)):
        if row1[i] != row2[i]:
            return 1 if row1[i] > row2[i] else -1
    return 0


@numba.jit(nopython=True, cache=True)
def find_matching_indices_sorted(philo_id_array, philo_id_object, cooc_slice):
    matching_indices = []
    low = 0
    high = len(philo_id_array) - 1

    while low <= high:
        mid = (low + high) // 2
        current_row = philo_id_array[mid][:cooc_slice]

        comparison = compare_rows(current_row, philo_id_object)

        if comparison == 0:
            # Match found, append index
            matching_indices.append(mid)
            # Roll back as long as consecutive matches exist
            rollback = mid - 1
            while rollback >= low and compare_rows(philo_id_array[rollback][:cooc_slice], philo_id_object) == 0:
                matching_indices.append(rollback)
                rollback -= 1
            low = mid + 1  # Continue searching in the higher half
        elif comparison < 0:
            low = mid + 1  # Search in the higher half
        else:
            high = mid - 1  # Search in the lower half
    matching_indices.sort()
    return np.array(matching_indices)


def merge_word_group(db_path: str, txn, words: list[str], overflow_words: set[str], chunk_size=None):
    # Initialize data structures for each word
    word_data = {}
    if chunk_size is None:
        chunk_size = 10000  # 10000 hits: happy median between performance and memory usage, potentially reevaluate.

    # Load initial chunks
    for word in words:
        if word not in overflow_words:
            buffer = txn.get(word.encode("utf8"))
            array = np.frombuffer(buffer[:3600], dtype="u4").reshape(-1, 9)
            first_doc = array[0, 0]
            word_data[word] = WordData(full_array=np.frombuffer(buffer, dtype="u4").reshape(-1, 9), array=array, start=0, first_doc=first_doc)
        else:
            file_path = os.path.join(db_path, "overflow_words", f"{hashlib.sha256(word.encode('utf8')).hexdigest()}.bin")
            mmap_array = np.memmap(file_path, dtype="u4", mode='r').reshape(-1, 9)
            array = mmap_array[:100] # 100 36 byte rows
            first_doc = array[0, 0]
            word_data[word] = WordData(full_array=mmap_array, array=array, start=0, first_doc=first_doc)

    def build_first_last_rows():
        first_finishing_doc = np.iinfo(np.uint32).max
        first_finishing_byte = np.iinfo(np.uint32).max
        result = []

        for word, data in word_data.items():
            if data.first_doc > first_finishing_doc or (
                data.first_doc == first_finishing_doc and data.array[0, -1] > first_finishing_byte
            ):  # row starts after finishing row
                continue
            if data.array[-1, 0] < first_finishing_doc or (
                data.array[-1, 0] == first_finishing_doc and data.array[-1, -1] < first_finishing_byte
            ):  # row ends before finishing row
                first_finishing_doc = data.array[-1, 0]
                first_finishing_byte = data.array[-1, -1]
                first_word = word
            result.append((word, data.array[0, ::8], data.array[-1, ::8]))

        return result, np.array([first_finishing_doc, first_finishing_byte], dtype="u4"), first_word

    # Merge sort and write loop
    while word_data:
        # Determine which words start before the first finishing word ends
        # Save index of first row that exceeds the first finishing word
        words_first_last_row, first_finishing_row, first_word_to_finish = build_first_last_rows()

        words_to_keep = []
        for other_word, other_first_row, _ in words_first_last_row:
            if other_word == first_word_to_finish:
                words_to_keep.append((other_word, None))
                continue
            elif other_first_row[0] > first_finishing_row[0] or (
                other_first_row[0] == first_finishing_row[0] and other_first_row[1] > first_finishing_row[1]
            ):  # dismiss words that start before the first finishing word ends
                continue
            else:
                first_exceeding_index = np.searchsorted(
                    word_data[other_word].array[:, 0], first_finishing_row[0], side="right"
                )
                if first_exceeding_index < word_data[other_word].array.shape[0]:
                    remaining_array = word_data[other_word].array[:first_exceeding_index]
                else:
                    remaining_array = word_data[other_word].array
                if np.all(remaining_array[:, 0] < first_finishing_row[0]):  # all doc_ids are less than finishing doc id
                    words_to_keep.append((other_word, remaining_array.shape[0]))
                    continue
                # Are there equal doc_ids? If so, we need to break the tie by comparing byte offsets
                equal_doc_rows = np.where(remaining_array[:, 0] == first_finishing_row[0])
                last_equal_index = equal_doc_rows[0][-1] + 1  # +1 to include the last equal row
                remaining_array = word_data[other_word].array[:last_equal_index]
                exceeding_rows_mask = (remaining_array[:, 0] == first_finishing_row[0]) & (
                    remaining_array[:, -1] > first_finishing_row[1]
                )
                potential_exceeding_indices = np.where(exceeding_rows_mask)
                if potential_exceeding_indices[0].size != 0:
                    first_exceeding_index = potential_exceeding_indices[0][0]
                    words_to_keep.append((other_word, first_exceeding_index))
                else:
                    words_to_keep.append((other_word, last_equal_index))

        # Merge sort partial philo_id arrays
        combined_arrays = np.concatenate(
            [word_data[word].array[:index] for word, index in words_to_keep],
            dtype="u4",
        )
        # Sort by doc id and byte offset, 3x faster than np.lexsort((combined_arrays[:, -1], combined_arrays[:, 0]))
        composite_key = combined_arrays[:, 0].astype(np.uint64) << 32 | combined_arrays[:, -1]
        yield combined_arrays[np.argsort(composite_key, kind="stable")]

        # Load next chunks for all words based on the indices we saved
        for word, index in words_to_keep:
            if index is None:
                index = word_data[word].array.shape[0]  # no need to slice, we have the full array
            word_data[word].start += index
            end = word_data[word].start + chunk_size
            word_data[word].array = word_data[word].full_array[word_data[word].start:end]
            if word_data[word].array.size > 0:
                word_data[word].first_doc = word_data[word].array[0, 0]
            else:
                del word_data[word]


def expand_query_not(split, freq_file, dest_fh, ascii_conversion, lowercase=True):
    """Expand search term"""
    first = True
    grep_proc = None
    for group in split:
        if first == True:
            first = False
        else:  # bare newline starts a new group, except the first
            try:
                dest_fh.write("\n")
            except TypeError:
                dest_fh.write(b"\n")
            dest_fh.flush()

        # find all the NOT terms and separate them out by type
        exclude = []
        for i, g in enumerate(group):
            kind, token = g
            if kind == "NOT":
                exclude = group[i + 1 :]
                group = group[:i]
                break
        cut_proc = subprocess.Popen("cut -f 2 | sort | uniq", stdin=subprocess.PIPE, stdout=dest_fh, shell=True)
        filter_inputs = [cut_proc.stdin]
        filter_procs = [cut_proc]

        # We will chain all NOT operators backward from the main filter.
        for kind, token in exclude:
            if kind == "TERM" and ascii_conversion is True:
                proc = invert_grep(token, subprocess.PIPE, filter_inputs[0], lowercase)
            if kind == "TERM" and ascii_conversion is True:
                proc = invert_grep_exact(token, subprocess.PIPE, filter_inputs[0])
            if kind == "QUOTE":
                token = token[1:-1]
                proc = invert_grep_exact(token, subprocess.PIPE, filter_inputs[0])
            filter_inputs = [proc.stdin] + filter_inputs
            filter_procs = [proc] + filter_procs

        # then we append output from all the greps into the front of that filter chain.
        for kind, token in group:  # or, splits, and ranges should have been taken care of by now.
            if (kind == "TERM" and ascii_conversion is True) or kind == "RANGE":
                grep_proc = grep_word(token, freq_file, filter_inputs[0], lowercase)
                grep_proc.wait()
            elif kind == "TERM" and ascii_conversion is False:
                grep_proc = grep_exact(token, freq_file, filter_inputs[0])
                grep_proc.wait()
            elif kind == "QUOTE":
                token = token[1:-1]
                grep_proc = grep_exact(token, freq_file, filter_inputs[0])
                grep_proc.wait()
            elif kind == "LEMMA":
                grep_proc = grep_word_attributes(token, freq_file, filter_inputs[0], "lemmas")
                grep_proc.wait()
            elif kind == "LEMMA_ATTR":
                grep_proc = grep_word_attributes(token, freq_file, filter_inputs[0], "lemma_word_attributes")
                grep_proc.wait()
            elif kind == "ATTR":
                grep_proc = grep_word_attributes(token, freq_file, filter_inputs[0], "word_attributes")
                grep_proc.wait()
        # close all the pipes and wait for procs to finish.
        for pipe, proc in zip(filter_inputs, filter_procs):
            pipe.close()
            proc.wait()


def grep_word_attributes(token, freq_file, dest_fh, token_type):
    """Grep on lemmas or word attributes"""
    forms_file = os.path.join(os.path.dirname(freq_file), token_type)
    try:
        grep_proc = subprocess.Popen(["rg", "-a", b"^%s$" % token, forms_file], stdout=dest_fh)
    except (UnicodeEncodeError, TypeError):
        grep_proc = subprocess.Popen(["rg", "-a", b"^%s$" % token.encode("utf8"), forms_file], stdout=dest_fh)
    return grep_proc


def grep_word(token, freq_file, dest_fh, lowercase=True):
    """Grep on normalized words"""
    if lowercase:
        token = token.lower()
    norm_tok_uni_chars = unidecode(token)
    norm_tok = "".join(norm_tok_uni_chars)
    try:
        grep_command = ["rg", "-a", "^%s[[:blank:]]" % norm_tok, freq_file]
        grep_proc = subprocess.Popen(grep_command, stdout=dest_fh)
    except (UnicodeEncodeError, TypeError):
        grep_command = ["rg", "-a", b"^%s[[:blank:]]" % norm_tok.encode("utf8"), freq_file]
        grep_proc = subprocess.Popen(grep_command, stdout=dest_fh)
    return grep_proc


def invert_grep(token, in_fh, dest_fh, lowercase=True):
    """NOT grep"""
    if lowercase:
        token = token.lower()
    norm_tok_uni_chars = unidecode(token)
    norm_tok = "".join(norm_tok_uni_chars)
    try:
        grep_command = ["rg", "-a", "-v", "^%s[[:blank:]]" % norm_tok]
        grep_proc = subprocess.Popen(grep_command, stdin=in_fh, stdout=dest_fh)
    except (UnicodeEncodeError, TypeError):
        grep_command = ["rg", "-a", "-v", b"^%s[[:blank:]]" % norm_tok.encode("utf8")]
        grep_proc = subprocess.Popen(grep_command, stdin=in_fh, stdout=dest_fh)
    return grep_proc


def grep_exact(token, freq_file, dest_fh):
    """Exact grep"""
    try:
        grep_proc = subprocess.Popen(["rg", "-a", b"[[:blank:]]%s$" % token, freq_file], stdout=dest_fh)
    except (UnicodeEncodeError, TypeError):
        grep_proc = subprocess.Popen(["rg", "-a", b"[[:blank:]]%s$" % token.encode("utf8"), freq_file], stdout=dest_fh)
    return grep_proc


def invert_grep_exact(token, in_fh, dest_fh):
    """NOT exact grep"""
    # don't strip accent or case, exact match only.
    try:
        grep_proc = subprocess.Popen(["rg", "-a", "-v", b"[[:blank:]]%s$" % token], stdin=in_fh, stdout=dest_fh)
    except (UnicodeEncodeError, TypeError):
        grep_proc = subprocess.Popen(
            ["rg", "-a", "-v", b"[[:blank:]]%s$" % token.encode("utf8")], stdin=in_fh, stdout=dest_fh
        )
    # can't wait because input isn't ready yet.
    return grep_proc


def query_parse(query_terms, config):
    """Parse query function."""
    for pattern, replacement in config.query_parser_regex:
        query_terms = re.sub(rf"{pattern}", rf"{replacement}", query_terms, re.U)
    return query_terms


if __name__ == "__main__":
    path = sys.argv[1]
    terms = sys.argv[2:]
    parsed = parse_query(" ".join(terms))
    print("PARSED:", parsed, file=sys.stderr)
    grouped = group_terms(parsed)
    print("GROUPED:", grouped, file=sys.stderr)
    split = split_terms(grouped)
    print("parsed %d terms:" % len(split), split, file=sys.stderr)

    env = lmdb.open(f"{path}/data/words.lmdb", readonly=True, readahead=False)
    with env.begin(buffers=True) as txn:
        raw_bytes = txn.get(terms[0].encode("utf8"))
        # print(len(raw_bytes))
        # hits = np.frombuffer(txn.get(terms[0].encode("utf8")), dtype="u4").reshape(-1, 9)



    # class Fake_DB:
    #     pass

    # fake_db = Fake_DB()
    # from philologic.Config import DB_LOCALS_DEFAULTS, DB_LOCALS_HEADER, Config

    # fake_db.path = path + "/data/"
    # fake_db.locals = Config(fake_db.path + "/db.locals.py", DB_LOCALS_DEFAULTS, DB_LOCALS_HEADER)
    # fake_db.encoding = "utf-8"
    # freq_file = path + "/data/frequencies/normalized_word_frequencies"
    # # expand_query_not(split, freq_file, sys.stdout)
    # hits = query(fake_db, " ".join(terms), query_debug=True, raw_results=True)
    # hits.finish()
    # print(len(hits))
    # for hit in hits:
    #     print(hit)
