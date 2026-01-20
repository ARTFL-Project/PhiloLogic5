#!/var/lib/philologic5/philologic_env/bin/python3
"""SQL validation utilities to prevent SQL injection attacks.

This module provides centralized validation for column names, table names,
and other SQL identifiers that cannot be parameterized in SQLite.
"""

# Core system columns that are always valid in SQL queries
SYSTEM_COLUMNS = frozenset({
    # Core identification columns
    "philo_id", "philo_type", "philo_name", "philo_seq",
    # Hierarchical ID columns
    "philo_doc_id", "philo_div1_id", "philo_div2_id", "philo_div3_id",
    # Byte/position columns
    "start_byte", "end_byte", "position",
    # Ancestor columns
    "doc_ancestor", "div1_ancestor", "div2_ancestor", "div3_ancestor", "para_ancestor",
    # Document/file columns
    "filename", "doc_id",
    # Structural columns
    "parent", "target", "type", "page", "n",
    # Aggregate/computed columns
    "word_count", "rowid",
    # Reference/graphic columns
    "id", "facs", "img",
    # Word-specific
    "token", "lemma",
    # Common metadata fields
    "head", "who", "resp", "speaker",
    # Year field (commonly used)
    "year",
})

# Valid philo_type values
VALID_PHILO_TYPES = frozenset({
    "doc", "div", "div1", "div2", "div3", "para", "sent", "word", "page", "ref", "graphic", "line"
})

# Valid object levels for queries
VALID_OBJECT_LEVELS = frozenset({
    "doc", "div1", "div2", "div3", "para", "sent", "word"
})


def validate_column(column, db):
    """Validate that column name is a valid SQL column to prevent SQL injection.

    Args:
        column: The column name to validate
        db: Database object with locals containing metadata_fields

    Returns:
        The validated column name

    Raises:
        ValueError: If column is not a valid column name
    """
    if column in SYSTEM_COLUMNS:
        return column
    # Check against database-specific metadata fields
    if hasattr(db, 'locals') and column in db.locals.metadata_fields:
        return column
    # Check word attributes
    if hasattr(db, 'locals') and column in db.locals.word_attributes:
        return column
    raise ValueError(f"Invalid column name: {column}")


def validate_columns(columns, db):
    """Validate a list of column names.

    Args:
        columns: List of column names to validate
        db: Database object with locals containing metadata_fields

    Returns:
        List of validated column names
    """
    return [validate_column(col, db) for col in columns]


def validate_sort_order(sort_order, db):
    """Validate all columns in sort_order list.

    Args:
        sort_order: List of column names for ORDER BY clause
        db: Database object with locals containing metadata_fields

    Returns:
        List of validated column names, or None if sort_order is None
    """
    if sort_order is None:
        return None
    return validate_columns(sort_order, db)


def validate_philo_type(philo_type):
    """Validate philo_type against whitelist.

    Args:
        philo_type: The philo_type value to validate

    Returns:
        The validated philo_type

    Raises:
        ValueError: If philo_type is not valid
    """
    if philo_type not in VALID_PHILO_TYPES:
        raise ValueError(f"Invalid philo_type: {philo_type}")
    return philo_type


def validate_object_level(obj_level):
    """Validate object level for queries.

    Args:
        obj_level: The object level to validate (doc, div1, etc.)

    Returns:
        The validated object level

    Raises:
        ValueError: If obj_level is not valid
    """
    if obj_level not in VALID_OBJECT_LEVELS:
        raise ValueError(f"Invalid object level: {obj_level}")
    return obj_level


def validate_metadata_field(field, db):
    """Validate a metadata field name.

    This is an alias for validate_column but with clearer intent.

    Args:
        field: The field name to validate
        db: Database object with locals containing metadata_fields

    Returns:
        The validated field name
    """
    return validate_column(field, db)
