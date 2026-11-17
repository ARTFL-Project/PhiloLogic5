"""Unit tests for PostFilters.make_sql_table: its batched inserts must fill tables as inserting rows one by one did."""

import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from orjson import loads

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.loadtime import PostFilters
from philologic.loadtime.PostFilters import make_sql_table

pytestmark = pytest.mark.unit

METADATA_SQL_TYPES = {"date": "int"}


def load_rows_one_by_one(table, file_in, db_destination, depth):
    """The rows of make_sql_table as it used to insert them, one by one, adding columns when an insert failed"""
    conn = sqlite3.connect(db_destination)
    cursor = conn.cursor()
    if table == "toms":
        query = f"create table if not exists {table} (philo_type text, philo_name text, philo_id text, philo_seq text, year int)"
    else:
        query = f"create table if not exists {table} (philo_type, philo_name, philo_id, philo_seq)"
    cursor.execute(query)
    with open(file_in, encoding="utf8") as input_file:
        for sequence, line in enumerate(input_file):
            philo_type, philo_name, philo_id, attrib = line.split("\t", 3)
            fields = philo_id.split(None, 8)
            if len(fields) == 9:
                row = loads(attrib)
                row["philo_type"] = philo_type
                row["philo_name"] = philo_name
                row["philo_id"] = " ".join(fields[:depth])
                row["philo_seq"] = sequence
                insert = f"INSERT INTO {table} ({','.join(list(row.keys()))}) values ({','.join('?' for i in range(len(row)))});"
                try:
                    cursor.execute(insert, list(row.values()))
                except sqlite3.OperationalError:
                    cursor.execute(f"PRAGMA table_info({table})")
                    column_list = [i[1] for i in cursor]
                    for column in row:
                        if column not in column_list:
                            if column not in METADATA_SQL_TYPES:
                                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} text;")
                            else:
                                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {METADATA_SQL_TYPES[column]};")
                    cursor.execute(insert, list(row.values()))
    conn.commit()
    conn.close()


def write_objects(path, rows):
    """Object lines as merge_objects writes them: type, name, philo_id (9 numbers), attributes"""
    with open(path, "w", encoding="utf8") as output:
        for n, (philo_type, attributes) in enumerate(rows, 1):
            philo_id = f"1 {n} 0 0 0 0 0 0 {n}" if philo_type != "bad" else "1 2 3"
            output.write(f"{philo_type}\tname{n}\t{philo_id}\t{json.dumps(attributes, ensure_ascii=False)}\n")


def table_contents(db_path, table):
    conn = sqlite3.connect(db_path)
    columns = [(name, declared_type) for _, name, declared_type, *_ in conn.execute(f"PRAGMA table_info({table})")]
    rows = conn.execute(f"select * from {table} order by rowid").fetchall()
    indexes = sorted(row[0] for row in conn.execute("select name from sqlite_master where type='index'"))
    conn.close()
    return columns, rows, indexes


ROWS = [
    ("doc", {"author": "Zola", "title": "Nana", "year": 1880}),
    ("div1", {"head": "I"}),
    ("div1", {"head": "II"}),
    ("div1", {"head": "III", "n": 3}),  # new column
    ("bad", {"head": "skipped"}),  # not a 9 number philo_id: skipped, but counted in philo_seq
    ("div1", {"head": "IV", "n": 4}),
    ("div2", {"Head": "same column", "n": "5"}),  # SQLite column names ignore the case of ASCII letters
    ("div2", {"date": 1881, "Éd": "é", "éd": "not the same column"}),  # typed column; non-ASCII case is kept
    ("para", {}),
] + [("para", {"n": n}) for n in range(20)]


@pytest.mark.parametrize("batch_size", [1, 3, 10000])
@pytest.mark.parametrize("table, depth", [("toms", 7), ("pages", 9)])
def test_same_table_as_row_by_row(tmp_path, monkeypatch, batch_size, table, depth):
    monkeypatch.setattr(PostFilters, "SQL_INSERT_BATCH", batch_size)
    write_objects(tmp_path / "objects", ROWS)
    loader = SimpleNamespace(
        destination=str(tmp_path), debug=True, parser_config={"metadata_sql_types": METADATA_SQL_TYPES}
    )
    make_sql_table(table, str(tmp_path / "objects"), indices=[("head",), ("n",)], depth=depth, verbose=False)(loader)
    load_rows_one_by_one(table, str(tmp_path / "objects"), str(tmp_path / "reference.db"), depth)
    columns, rows, indexes = table_contents(tmp_path / "toms.db", table)
    reference_columns, reference_rows, _ = table_contents(tmp_path / "reference.db", table)
    assert columns == reference_columns and rows == reference_rows
    assert ("date", "INT") in columns and ("Éd", "TEXT") in columns and ("éd", "TEXT") in columns
    assert len(rows) == len(ROWS) - 1
    expected_indexes = [f"{table}_head_index", f"{table}_n_index"]
    if table == "toms":
        expected_indexes += ["head_null_index", "n_null_index"]
    assert indexes == sorted(expected_indexes)


def test_same_failure_as_row_by_row(tmp_path):
    """A row with a new column and another differing from an existing one by case only failed: it still does"""
    rows = [("doc", {"author": "Zola"}), ("doc", {"Author": "Hugo", "title": "new column"})]
    write_objects(tmp_path / "objects", rows)
    with pytest.raises(sqlite3.OperationalError, match="duplicate column"):
        load_rows_one_by_one("toms", str(tmp_path / "objects"), str(tmp_path / "reference.db"), 7)
    loader = SimpleNamespace(destination=str(tmp_path), debug=True, parser_config={"metadata_sql_types": {}})
    with pytest.raises(sqlite3.OperationalError, match="duplicate column"):
        make_sql_table("toms", str(tmp_path / "objects"), verbose=False)(loader)
