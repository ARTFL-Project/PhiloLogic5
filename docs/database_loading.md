
Loading PhiloLogic databases is very straight forward, and most of the time, you shouldn't need to specify any specialized load option.

A few important notes:

-   Before loading any databases, you should first make sure the global configuration file located in `/etc/philologic/philologic5.cfg` has been edited appropriately. For more info, see [here](installation.md#global-config)
-   The PhiloLogic5 Parser's behavior is configurable from an external load config file, though only to a certain extent. You can also supply a replacement Parser class if you need to.
-   The loading process is designed to be short, and easy to understand and configure.

### Text format support
PhiloLogic supports the following text files formats
- TEI-XML: see [here](#configuring-the-xml-parser) for configuring XML parser
- Plain text files (available with PhiloLogic 4.7.3 and up): see [here](#plain-text-parser) for using the plain text parser.

To load databases from a web page instead, with previews of the load and its options explained, see [philologic5-webui-loader](webui_loader.md).

### Executing the load command

In order for PhiloLogic to index your files, you need to execute the `philoload5` command. The basic command is run as so:

`philoload5 [database_name] [path_to_files]`

The `philoload5` command requires the following required arguments::

1.  the name of the database to create, which will be the subdirectory
    into your web space directory, i.e. `/var/www/html/mydatabase`,
2.  the paths to each of the files you wish to load,
    i.e. `mycorpus/xml/*.xml`.

`philoload5` also accepts a number of optional command line arguments::

`-h`, `--help` show this help message and exit

`-b BIBLIOGRAPHY`, `--bibliography=BIBLIOGRAPHY` Defines a file containing the document-level bibliography of the texts. One of the fields needs to be the filename

`-c CORES`, `--cores=CORES` define the number of cores used for parsing

`-d`, `--debug` add debugging at parse time

`-D`, `--force_delete` overwrite database without confirmation

`-F`, `--file-list` Defines whether the file argument is a file containing fullpaths to the files to load

`-H HEADER`, `--header=HEADER` define header type (tei or dc) of files to parse

`-l LOAD_CONFIG`, `--load_config=LOAD_CONFIG` load external config for specialized load

`-t FILE_TYPE`, `--file-type=FILE_TYPE` Define file type for parsing: plain_text or xml

`-w WEB_CONFIG`, `--use-webconfig=WEB_CONFIG` use a predefined web_config.cfg file for the database

So our command for loading texts could be::

`philoload5 -c 8 -d my_database files/*xml`

### Database Load Configuration

In the event you need to customize the behavior of the parser, you can pass the `-l` option along with a `load_config.py` file to be found in `PhiloLogic5/extras/` directory. In this file, you can configure a number of different parameters:

```python
# Define default object level
default_object_level = 'doc'

# Define navigable objects
navigable_objects = ('doc', 'div1', 'div2', 'div3', 'para')

```

`default_object_level` defines the type of object returned for the purpose of most navigation reports--for most database, this will be "doc", but you might want to use "div1" for dictionary or encyclopedia databases.

`navigable_objects` is a list of the object types stored in the database and available for searching, reporting, and navigation--("doc","div1","div2","div3") is the default, but you might want to append "para" if you are parsing interesting metadata on paragraphs, like in drama. Pages are handled separately, and don't need to be included here.

`filters` and `post_filters` are lists of loader functions--their behavior and design will be documented separately, but they are basically lists of modular loader functions to be executed in order, and so shouldn't be modified carelessly.



### Plain text Parser
To use the plain text parser, you will need to specify the `t plain_text` command-line argument and also provide a bibliography file (with the `-b` argument) in CSV or TSV format, such as in the following example:
`philoload5 -t plain_text -b metadata.csv database_name /path/to/files`

Note that the plain text parser is fairly rudimentary and does not detect structure within the files themselves, with the sole exception being paragraphs when there is an empty line between blocks of text. In order for better structure detection, you will need to convert your files to TEI.


### Configuring the XML Parser

The next section of the load script is setup for the XML Parser:

```python
## Set-up database load ###
###########################

# These are doc level XPATHS used to parse a standard TEI header.
# These XPATHS need to be inside a <teiHeader> and strictly apply to an entire document..
# Only useful if you parse a TEI header.
doc_xpaths = {
    "author": [
        ".//sourceDesc/bibl/author[@type='marc100']",
        ".//sourceDesc/bibl/author[@type='artfl']",
        ".//sourceDesc/bibl/author",
        ".//titleStmt/author",
        ".//sourceDesc/biblStruct/monogr/author/name",
        ".//sourceDesc/biblFull/titleStmt/author",
        ".//sourceDesc/biblFull/titleStmt/respStmt/name",
        ".//sourceDesc/biblFull/titleStmt/author",
        ".//sourceDesc/bibl/titleStmt/author",
    ],
    "title": [
        ".//sourceDesc/bibl/title[@type='marc245']",
        ".//sourceDesc/bibl/title[@type='artfl']",
        ".//sourceDesc/bibl/title",
        ".//titleStmt/title",
        ".//sourceDesc/bibl/titleStmt/title",
        ".//sourceDesc/biblStruct/monogr/title",
        ".//sourceDesc/biblFull/titleStmt/title",
    ],
    "author_dates": [".//sourceDesc/bibl/author/date", ".//titlestmt/author/date"],
    "create_date": [
        ".//profileDesc/creation/date",
        ".//fileDesc/sourceDesc/bibl/imprint/date",
        ".//sourceDesc/biblFull/publicationStmt/date",
        ".//profileDesc/dummy/creation/date",
        ".//fileDesc/sourceDesc/bibl/creation/date",
    ],
    "publisher": [
        ".//sourceDesc/bibl/imprint[@type='artfl']",
        ".//sourceDesc/bibl/imprint[@type='marc534']",
        ".//sourceDesc/bibl/imprint/publisher",
        ".//sourceDesc/biblStruct/monogr/imprint/publisher/name",
        ".//sourceDesc/biblFull/publicationStmt/publisher",
        ".//sourceDesc/bibl/publicationStmt/publisher",
        ".//sourceDesc/bibl/publisher",
        ".//publicationStmt/publisher",
        ".//publicationStmp",
    ],
    "pub_place": [
        ".//sourceDesc/bibl/imprint/pubPlace",
        ".//sourceDesc/biblFull/publicationStmt/pubPlace",
        ".//sourceDesc/biblStruct/monog/imprint/pubPlace",
        ".//sourceDesc/bibl/pubPlace",
        ".//sourceDesc/bibl/publicationStmt/pubPlace",
    ],
    "pub_date": [
        ".//sourceDesc/bibl/imprint/date",
        ".//sourceDesc/biblStruct/monog/imprint/date",
        ".//sourceDesc/biblFull/publicationStmt/date",
        ".//sourceDesc/bibFull/imprint/date",
        ".//sourceDesc/bibl/date",
        ".//text/front/docImprint/acheveImprime",
    ],
    "extent": [".//sourceDesc/bibl/extent", ".//sourceDesc/biblStruct/monog//extent", ".//sourceDesc/biblFull/extent"],
    "editor": [
        ".//sourceDesc/bibl/editor",
        ".//sourceDesc/biblFull/titleStmt/editor",
        ".//sourceDesc/bibl/title/Stmt/editor",
    ],
    "identifiers": [".//publicationStmt/idno"],
    "text_genre": [".//profileDesc/textClass/keywords[@scheme='genre']/term", ".//SourceDesc/genre"],
    "keywords": [".//profileDesc/textClass/keywords/list/item"],
    "language": [".//profileDesc/language/language"],
    "notes": [".//fileDesc/notesStmt/note", ".//publicationStmt/notesStmt/note"],
    "auth_gender": [".//publicationStmt/notesStmt/note"],
    "collection": [".//seriesStmt/title"],
    "period": [
        ".//profileDesc/textClass/keywords[@scheme='period']/list/item",
        ".//SourceDesc/period",
        ".//sourceDesc/period",
    ],
    "text_form": [".//profileDesc/textClass/keywords[@scheme='form']/term"],
    "structure": [".//SourceDesc/structure", ".//sourceDesc/structure"],
    "idno": [".//fileDesc/publicationStmt/idno/"],
}

# Maps any given tag to one of PhiloLogic's types. Available types are div, para, page, and ref.
# Below is the default mapping.
tag_to_obj_map = {
    "div": "div",
    "div1": "div",
    "div2": "div",
    "div3": "div",
    "hyperdiv": "div",
    "front": "div",
    "note": "para",
    "p": "para",
    "sp": "para",
    "lg": "para",
    "epigraph": "para",
    "argument": "para",
    "postscript": "para",
    "opener": "para",
    "closer": "para",
    "stage": "para",
    "castlist": "para",
    "list": "para",
    "q": "para",
    "add": "para",
    "pb": "page",
    "ref": "ref",
    "graphic": "graphic",
}

# Defines which metadata to parse out for each object. All metadata defined here are attributes of a tag,
# with the exception of head and div_date which are their own tags. Below are defaults.
metadata_to_parse = {
    "div": ["head", "type", "n", "id", "vol", "div_date"],
    "para": ["who", "speaker", "resp", "id"],
    "page": ["n", "id", "facs"],
    "ref": ["target", "n", "type"],
    "graphic": ["facs"],
    "line": ["n", "id"],
}

# Define how your metadata fields are stored. This is defines as a dictionary with the field as key.
# Types possible are text, int, date, which should be defined as strings.
# If not defined, the metadata field will be stored and queried as text.
metadata_sql_types = {}

# Define a file (with full path) containing words to index. Must be one word per line.
# Useful for filtering out dirty OCR.
words_to_index = ""

# This regex defines how to tokenize words and punctuation
# For Asian script, try using this token_regex: r"[\p{L}\p{M}\p{N}\p{Po}]+|[&\p{L};]+"
token_regex = r"[\p{L}\p{M}\p{N}]+|[&\p{L};]+"

# This defines whether you want to convert your text and metadata to an ASCII representation for
# search and autocomplete. Turn off if your language does not translate well to ascii (non-European languages in general)
ascii_conversion = True


# Define the order in which files are sorted. This will affect the order in which
# results are displayed. Supply a list of metadata strings, e.g.:
# ["date", "author", "title"]
sort_order = ["year", "author", "title", "filename"]

# A list of tags to ignore: contents will not be indexed
# This should be a list of tag names, such as ["desc", "gap"]
suppress_tags = []

# --------------------- Set Apostrophe Break ------------------------
# Set to True to break words on apostrophe.  Probably False for
# English, True for French.  Your milage may vary.
break_apost = True

# ------------- Define Characters to Exclude from Index words -------
# Leading to a second list, characters which can be in words
# but you don't want to index.
chars_not_to_index = r"[\[\{\]\}]"

# ---------------------- Treat Lines as Sentences --------------------
# In linegroups, break sentence objects on </l> and turns off
# automatic sentence recognition.  Normally off.
break_sent_in_line_group = False

# ------------------ Skip in word tags -------------------------------
# Tags normally break words.  There may be exceptions.  To run the
# exception, turn on the exception and list them as patterns.
# Tags will not be indexed and will not break words. An empty list turns off the feature
tag_exceptions = [
    r"<hi[^>]*>",
    r"<emph[^>]*>",
    r"<\/hi>",
    r"<\/emph>",
    r"<orig[^>]*>",
    r"<\/orig>",
    r"<sic[^>]*>",
    r"<\/sic>",
    r"<abbr[^>]*>",
    r"<\/abbr>",
    r"<i>",
    r"</i>",
    r"<sup>",
    r"</sup>",
]

# Define a list of word attributes to ingore at parse time. These will not be stored.
# This should be a list of attribute names, such as ["type", "id"]
suppress_word_attributes = []

#  ----------------- Set Long Word Limit  -------------------
#  Words greater than 235 characters (bytes) cause an indexing
#  error.  This sets a limit.  Longer words are left out of the index.
long_word_limit = 200

# ------------------ Hyphenated Word Joiner ----------------------------
# Softhypen word joiner.  At this time, I'm trying to join
# words broken by &shy;\n and possibly some additional
# selected tags.  Could be extended.
join_hyphen_in_words = True

# ------------------ Abbreviation Expander for Indexing. ---------------
# This is to handle abbreviation tags.  I have seen two types:
#       <abbr expan="en">&emacr;</abbr>
#       <abbr expan="Valerius Maximus">Val. Max.</abbr>
# For now, lets's try the first.
abbrev_expand = True

# ---------------------- Flatten Ligatures for Indexing --------------
# Convert SGML ligatures to base characters for indexing.
# &oelig; = oe.  Leave this on.  At one point we should think
# Unicode, but who knows if this is important.
flatten_ligatures = True

# Define a list of strings which mark the end of a sentence.
# Note that this list will be added to the current one which is [".", "?", "!"]
sentence_breakers = []

# Define which punctuation should be flagged as such. This should NOT include
# any punctuation which mark sentence breaks. Use regex to match characters.
punctuation = ""

# Defines whether words should be stored in lowercase form in the index.
lowercase_index = True

# Path to a file containing a mapping of words to their lemmatized form.
# Format is one word per line, separated by a tab from its lemma.
lemma_file = ""

# Define a SpaCy model to use for lemmatization, part-of-speech tagging, and named entity recognition.
# Unless the model runs on the GPU, each parsing process (see cores) loads its own copy of the model, and the
# threads the model would use in a single process are shared out among them: memory use grows with cores.
# On the GPU, files are tagged in a single process.
# Only official SpaCy models are supported.
spacy_model = ""
```

The basic layout is this:

`doc_xpaths` is a dictionary that maps philologic document-level object contained in the TEI header to absolute XPaths--that is, XPaths evaluated where `.` refers to the TEI document root element. You can define multiple XPaths for the same type of object, but you will get much better and more consistent results if you do not.

`tag_to_obj_map` is a dictionary that maps XML tags to the standard types PhiloLogic stores in its index.

`metadata_to_parse` is a dictionary that maps one or more non-document-level object types to a list of metadata (usually attributes) to retrieve.

`suppress_tags` is a list of tags in which you do not want to perform tokenization at all--that is, no words in them will be searchable via full-text search. It does not prohibit extracting metadata from the content of those tags. `<gap>` is always suppressed, whether or not it is listed here.

`token_regex` is a regular expression used to drive our tokenizer.

`words_to_index` is a file containing all words that should be indexed. You'd want to define this in the event you're dealing with dirty OCR and would end up with way too many unique words, which would blow up the index, or just kill search performance. Leaving this empty means that all words will be indexed.

`sort_order` is a list of metadata fields which defines the order in which the parser will load and store files in the database. This affects the default order in which search results are returned.

The remaining options are self-explanatory given the comments...

So to use a load config file as an argument, you would run the following:

`philoload5 -l load_config.py db_name path_to_files`

In a load config, an empty value (`""`, `[]`, `{}`, `None`) leaves the option's default, except for `tag_exceptions` and `sort_order`, where an empty list turns the feature off.

### Loading a database again

Every database keeps a copy of the load config it was loaded with in `data/load_config.py`, completed with the values of all the other options. To load a database again with the same options, possibly with other files, pass this copy with `-l`:

`philoload5 -l /var/www/html/philologic5/my_database/data/load_config.py my_new_database path_to_files`

What to load and how to run the load are only taken from the command line: the files, `-b`, `-c`, `-D`, `-d` and `-F`, as well as the database location of the global config. A load config can set the header type (`header`) and the file type (`file_type`), but `-H` and `-t` override it.




