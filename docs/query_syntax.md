PhiloLogic4's query syntax has 5 basic operators:

1. the plain token, essentially, any word at all, split on space, e.g. `token`
2. the quoted token--a string in double quotes, which may contain a space, e.g. `"token"`
3. the range--two tokens separated by a dash, e.g. `1700-1750`
4. boolean OR, represented grep-style as `|`, e.g. `token | word`
5. boolean NOT, represented SQL-style as `NOT`, e.g. `token.* NOT tokens`

This syntax is the same, but interpreted slightly differently, for the two different types of text query fields: word search and metadata search.

### Word Searches

Full-text word search is unique in having the concept of a "term", which is either a single plain/quoted term,
or a group of plain/quoted terms joined by `|`, optionally followed by `NOT` and another term-like filter expression.
When specifying a query, one can select a query method to constrain the relation between terms, such as `within k words` or `in the same sentence`

1. plain terms are evaluated without regard to accent or to case. Regexes are permitted.
2. quoted terms are case and accent sensitive. Regexes are permitted.
3. a quoted term with spaces is a phrase: its words are found next to each other, in order. With a query method such as
   `within k words` or `in the same sentence`, the phrase is one term: its words stay together, and the phrase counts as
   one word for the distance, so `"peuple français" souverain` within 3 words finds `souverain` up to 3 words before
   `peuple` or after `français`.
4. the range is not operational. In the future, stub this out to make hyphenated search terms less of a pain to escape.
5. `OR` can conjoin plain and quoted tokens, and precedes evaluation of phrase distance.
6. `NOT` is a filter on a preceding term, but cannot stand alone: `a.* NOT abalone` is legal, `NOT a.*` is illegal

#### Lemma and word attribute Searches
If you text collection contains lemma and/or word attribute information (usually in <w> tags), then PhiloLogic allows you to query words based on lemma and/or word attribute value
1. For simple lemma searching, just preprend the lemma with `lemma:` such as in `lemma:have`. Regexes are permitted on the token portion of the search, e.g. `lemma:constitut.*`.
   - Multi-word lemmas (e.g. the French compound `parce que`) are stored with whitespace collapsed to underscores. Query them as `lemma:parce_que`, not `lemma:parce que`.
2. For word attribute searching, use the `word:attribute:attribute` syntax such as in `love:pos:NOUN`. Regexes are permitted on the token portion of the search, e.g. `lov.*:pos:NOUN`.
3. You can combine lemma searching with word attribute filtering. Just preprend your token with `lemma:` such as in `lemma:love:pos:NOUN`. Regexes are permitted on the token portion of the search, e.g. `lemma:lov.*:pos:NOUN`.
4. Note that you cannot combine multiple word attributes filters on one token, such as in `charles:pos:PROPN:ner:PERS`.

### Metadata Searches

Metadata values have a syntax of their own: the database's word-search settings (`query_parser_regex`) don't apply to them.

1. plain words separated by spaces must all be words of the value, in any order, regardless of case and (with `ascii_conversion`) accents.
   A word with hyphens or apostrophes, as `jean-jacques`, needs its parts side by side. Regexes are permitted, and match whole words.
2. quoted text must match the ENTIRE value exactly, including spaces and punctuation. It is not a regex.
   A quote inside it is written twice: `"Les Révoltés de la ""Bounty"""`.
3. ranges, as `1700-1750`, `-1750` or `1750-`, work in numeric fields only. In text fields, `-` is part of a word.
4. `OR` (or `|`) joins alternatives, preceding the implied Boolean AND.
5. `NOT` excludes what follows, and may stand alone: `contrat NOT social` is legal, so is `NOT rousseau`.
   Objects with no value are not excluded: `NOT rousseau NOT NULL` leaves them out too.
6. `NULL` matches the objects with no value, `NOT NULL` those with one.

Metadata objects also have the unique property of recursion, which creates some unusual consequences for search semantics.
Searching for a div that has property `NOT x` does not guarantee that the result does not contain a child with property x,
or a parent with property x. This can often be handled at the database level by normalizing metadata to a single fine-grained layer,
but is tricky. Likewise, searches for `NULL` values in recursive objects will often return "virtual" philologic objects,
which don't exist in the XML but are necessary for balanced tree arithmetic.

### Regexp syntax

Basic regexp syntax, adapted from the [**egrep regular expression syntax**](http://www.gnu.org/software/findutils/manual/html_node/find_html/egrep-regular-expression-syntax.html#egrep-regular-expression-syntax).

-   The character `.` matches any single character except newline.
-   Bracket expressions can match sets or ranges of characters: `[aeiou]` or `[a-z]`, but will only match a single character unless followed by one of the quantifiers below.
-   `*` indicates that the regular expression should match zero or more occurrences of the previous character or bracketed group.
-   `+` indicates that the regular expression should match one or more occurrences of the previous character or bracketed group.
-   `?` indicates that the regular expression should match zero or one occurrence of the previous character or bracketed group.
    Thus, `.*` is an approximate "match anything" wildcard operator, rather than the more traditional (but less precise) `*` in many other search engines.
