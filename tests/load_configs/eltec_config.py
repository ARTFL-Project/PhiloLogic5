"""
Load configuration for ELTeC (European Literary Text Collection) English corpus.

The ELTeC texts use TEI encoding with:
- POS tagging and lemmatization on <w> elements
- Standardized metadata in the TEI header
- div elements for chapters/sections
- s elements for sentences
"""

# Document-level metadata XPaths
doc_xpaths = {
    "author": [
        ".//titleStmt/author",
    ],
    "title": [
        ".//titleStmt/title",
    ],
    "pub_date": [
        ".//sourceDesc/bibl[@type='firstEdition']/date",
        ".//publicationStmt/date",
    ],
    "publisher": [
        ".//sourceDesc/bibl[@type='firstEdition']/publisher",
    ],
    "word_count": [
        ".//extent/measure[@unit='words']",
    ],
    "page_count": [
        ".//extent/measure[@unit='pages']",
    ],
    "language": [
        ".//profileDesc/langUsage/language/@ident",
    ],
    "idno": [
        ".//TEI/@xml:id",
    ],
}

# Use default tag_to_obj_map and metadata_to_parse from Parser.py
# Note: Do not override these as custom mappings can cause issues
# with the metadata hierarchy structure

# Sort order for documents
sort_order = ["year", "author", "title", "filename"]

# The ELTeC corpus has pre-tokenized words with POS and lemma attributes
# These are automatically extracted by the parser from <w pos="..." lemma="..."> tags
