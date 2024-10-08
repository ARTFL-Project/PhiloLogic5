### 5.0 ###
* Rewriten core search routines from the old C to Python, move word index to LMDB
* Count all co-occurrences (old behavior only counted one co-occurrence per sentence)
* Integrate SpaCy in parser for part-of-speech tagging, lemmatization, and ner. 
* Lemma searching in all reports enabled if present in TEI or if lemma mapping file provided
* Word attribute searching
* Completely rewritten collocations with better performance (1.5x) and a LOT of new functionality:
   * Search within n words in collocations
   * Search for lemmas
   * Filter by word attribute (e.g. by part-of-speech)
   * Compare collocation distributions based on count and z-score difference.
   * Find most similar collocation distribution based on metadata (e.g. finding authors with most similar collocation distribution)
   * Collocations over time
* Speed-up time series (5x or more)

### 4.7 ###
- New aggregation report
- New metadata stats in search results
- Results bibliography in concordance and KWIC results.
- Database size should be between 50% and 80% (or more) smaller
- Significant speed-ups for:
    * Collocations: in some cases 3-4X
    * Sorted KWICs: between 6X and 25X (or more) depending on use case, with no more limits on the size of the sort as a result.
    * Faceted browsing (frequencies): anywhere from 3X to 100X (or more)
    * Landing page browsing: 10X faster or more on large corpora
- Export results to CSV
- Web config has been simplified with the use of global variables for citations
- Some breaking changes to web config: you should not use a 4.6 config
- Revamped Web UI: move to VueJS and Bootstrap 5.
- Cleaner URLS for queries
- Faster database loads
- New generic dictionary lookup code
- Support for date and integer types for metadata fields.

### 4.6 ###
- Port PhiloLogic4 codebase to Python3
- Switch load time compression from Gzip to LZ4: big speed-up in loading large databases
- Lib reorganization

#### 4.0 => 4.5 ####
- Completely rewritten parser: can now parse broken XML
- Massive lib reorg
- A new system wide config
- Loading process completely revamped: use philoload4 command
- Completely rewritten collocations: faster and accurate
- Added relative frequencies to frequencies in facets
- Added sorted KWIC
- Added support for regexes in quoted term searches (aka exact matches)
- Added ability to filter out words in query expansion through a popup using the NOT syntax
- Added configurable citations for all reports
- Added concordance results sorting by metadata
- Added approximate word searches using Levenshtein distance
- Redesign facets and time series
- Bug fixes and optimizations everywhere...
