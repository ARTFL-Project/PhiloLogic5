"""Split lines sorted by sort -k 2,2 (LC_ALL=C) into ranges of their key, compressing each range with lz4.
Run as a script at the end of a sort pipeline: split_sorted.py boundaries_file output_prefix
boundaries_file has one key per line, in increasing order: range i gets the lines whose key is at least boundary i-1
and less than boundary i, written to output_prefix.i (a range without lines gets no file)."""

import re
import sys

import lz4.frame

# The key of sort -k 2,2 (without -b, as in the merges split): fields are made of blanks (space or tab, in the C
# locale) then other characters, and the second field is empty if there isn't one
SORT_KEY = re.compile(rb"[ \t]*[^ \t\n]*([ \t]*[^ \t\n]*)")


def sort_key(data, start=0):
    """Key sort -k 2,2 compares for the line starting at start in data"""
    return SORT_KEY.match(data, start).group(1)


def range_of(boundaries, key):
    """Position of the range of key: number of boundaries less than or equal to it"""
    low, high = 0, len(boundaries)
    while low < high:
        middle = (low + high) // 2
        if boundaries[middle] <= key:
            low = middle + 1
        else:
            high = middle
    return low


class RangeFile:
    """lz4 frame file, compressed in 64 KB blocks"""

    def __init__(self, path):
        self.file = open(path, "wb")
        self.compressor = lz4.frame.LZ4FrameCompressor(block_size=lz4.frame.BLOCKSIZE_MAX64KB)
        self.file.write(self.compressor.begin())

    def write(self, data):
        self.file.write(self.compressor.compress(data))

    def close(self):
        self.file.write(self.compressor.flush())
        self.file.close()


def next_line(data, offset, start):
    """Start of the first line at or after offset, lines starting at start"""
    if offset == start or data[offset - 1] == 10:
        return offset
    return data.find(b"\n", offset) + 1 or len(data)


def split_sorted(input_file, boundaries, prefix, chunk_size=1 << 22):
    """Write the sorted lines of input_file to prefix.{range}, splitting chunks at the boundaries: since lines are
    sorted, a chunk is only searched (by bisection) when its last line is in a later range than its first."""
    current, output = -1, None
    rest = b""
    while chunk := input_file.read(chunk_size):
        data = rest + chunk if rest else chunk
        end = data.rfind(b"\n") + 1
        rest = data[end:]
        view = memoryview(data)
        position = 0
        while position < end:
            index = range_of(boundaries, sort_key(data, position))
            if index != current:
                if index < current:
                    raise ValueError("input lines aren't sorted")
                if output is not None:
                    output.close()
                current, output = index, RangeFile(f"{prefix}.{index}")
            last = data.rfind(b"\n", position, end - 1) + 1
            if index == len(boundaries) or sort_key(data, last) < boundaries[index]:  # rest of the chunk in range
                output.write(view[position:end])
                break
            low, high = position, end  # first line whose key is not less than the boundary
            while low < high:
                middle = (low + high) // 2
                line = next_line(data, middle, position)
                if line < end and sort_key(data, line) < boundaries[index]:
                    low = line + 1
                else:
                    high = middle
            split = next_line(data, low, position)
            output.write(view[position:split])
            position = split
    if rest:
        raise ValueError("input doesn't end with a newline")
    if output is not None:
        output.close()


if __name__ == "__main__":
    with open(sys.argv[1], "rb") as boundaries_file:
        split_boundaries = [line.rstrip(b"\n") for line in boundaries_file]
    split_sorted(sys.stdin.buffer, split_boundaries, sys.argv[2])
