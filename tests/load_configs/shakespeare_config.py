"""
Load configuration for Folger Shakespeare collection.

The Folger Shakespeare Library texts use a specific TEI encoding with:
- div1 for acts, div2 for scenes
- sp tags for speeches with speaker information
- stage tags for stage directions
- Detailed character encoding via listPerson
"""

# Document-level metadata XPaths
doc_xpaths = {
    "author": [
        ".//titleStmt/author",
        ".//sourceDesc/biblStruct/monogr/author/name",
    ],
    "title": [
        ".//titleStmt/title",
        ".//sourceDesc/biblStruct/monogr/title",
    ],
    "editor": [
        ".//titleStmt/editor",
    ],
    "publisher": [
        ".//publicationStmt/publisher",
    ],
    "pub_date": [
        ".//publicationStmt/date",
        ".//sourceDesc/biblStruct/monogr/imprint/date",
    ],
    "idno": [
        ".//publicationStmt/idno",
    ],
    "collection": [
        ".//seriesStmt/title",
    ],
}

# Use default tag_to_obj_map and metadata_to_parse from Parser.py
# Note: Do not override these as custom mappings can cause issues
# with the metadata hierarchy structure

# Sort order for documents
sort_order = ["title", "author", "filename"]
