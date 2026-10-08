# Collocation report review (October 2026)

Scope: the collocation report end to end. That covers collocate counting; usage-pattern ("thread") detection
and its word map and timeline; the distinctive-groups, comparison and similar-usage statistics; and the
representative-passages modal. The design goal it is measured against is **transparency**: every rule should be
statable, and every result checkable by the user.

Findings only: nothing has been fixed yet. Every number below was measured on `main` as of 2026-10-06, offline,
against `frantext`, `rousseau_complete_works` and `balzac`, using a private hitlist directory (see the
[appendix](#appendix-how-the-checks-were-run)). Unless stated otherwise, runs use the production default filter
(drop words in more than 1% of sentences) and the automatic number of patterns.

Sections 1–2 cover the thread-detection algorithm, which is the priority. Sections 3–7 cover everything else.

---

## Summary

- The **collocation table** is solid and transparent. It shows raw counts that match the concordances they link
  to (covered by integration tests), plus the exact list of filtered words.
- **Thread detection** (HyperLex) works as designed on large whole-corpus queries. It has three algorithmic
  problems:
  1. On thin data (small corpora, metadata filters) it builds its graph from links resting on a single sentence.
     It then reports confident-looking grab-bags covering 88–100% of the vocabulary.
  2. Nothing tests whether a pattern is about the query word. Corpus-wide topics, such as a "sun, water, fire,
     sky" group, appear for unrelated queries and can rank first.
  3. There is no honest "these collocates don't form distinct groups" outcome, and the numbers that would reveal
     weak structure are never shown.
- The **word map and timeline** don't show what was computed:
  - 64% of map nodes are not pattern members but are drawn as if they were.
  - 78% of drawn links are weaker than the method's own threshold.
  - The timeline plots a normalized share and never the computed usage rate, while its tooltip says otherwise.
- Five **confirmed bugs**: comparison and similar-usage ignore the n-word window; mixed lemma/surface queries count
  their own words; representative passages break on multi-word queries; two word-map menu items crash;
  similar-usage hides groups by a similarity threshold.
- The two existing design docs are out of date (§8).

---

## 1. How thread detection works today

Written down here because neither existing doc describes the current code
([usage_patterns.py](python/philologic/runtime/reports/usage_patterns.py),
[get_usage_patterns.py](www/scripts/get_usage_patterns.py)).

1. **Hits.** The collocation search for the query: each occurrence of a single term or phrase, or each
   co-occurrence of several terms within n words or in a sentence. The search's metadata filters apply.
2. **Bags.** Each hit becomes the bag of tokens of its **whole sentence**, from the lemma or the surface
   vocabulary. Query words are removed, and with a part-of-speech filter, tokens of other attributes are too.
   - The bag is the whole sentence even when the collocation table counts within n words.
   - Hits without a year are dropped, so the non-temporal word map also needs a year field.
   - Two hits in one sentence give two identical bags.
3. **Candidates.** Words in at least max(3, 0.5% of bags) and at most 30% of bags, at least 2 characters long, and
   not in the active filter list. At most 1,000 are kept, the most frequent first.
4. **Graph.** NPMI between every pair of candidates, computed over the bags. Distance = 1 − NPMI when NPMI > 0,
   else 1.
5. **Edge threshold.** Chosen automatically ([`_auto_floor`](python/philologic/runtime/reports/usage_patterns.py#L932)):
   - Start at 0.22 and rise in steps of 0.02, up to at most 0.34.
   - Stop at the first threshold where the largest hub group has at most max(40, 8% of candidates) words, with at
     least 4 groups surviving. This sizing uses up to 8 hubs.
6. **Hubs.** ([`_hyperlex`](python/philologic/runtime/reports/usage_patterns.py#L854))
   - Candidates are taken in order of degree, counting only links ≥ threshold. A hub needs at least 4 such
     neighbours, and claims them: they can't become hubs.
   - How many hubs: the user's choice, or else the knee of the hub-degree curve, clamped to at least 3
     ([`_default_pattern_count`](python/philologic/runtime/reports/usage_patterns.py#L967)).
   - The dropdown offers 2 up to the number of hubs found with a cap of 12.
7. **Assignment.** Each word joins the hub it has its strongest direct link to (≥ threshold). Then one round of
   growth: each unplaced word joins the pattern containing its strongest link (≥ threshold) to any member.
   Everything else stays unplaced.
8. **Labels.** Each pattern's words are ranked by centrality, the sum of their NPMI to the other members. The label
   is the top 5.
9. **Order and colours.** Patterns are sorted by the summed yearly share of hits whose sentence contains any
   pattern word.
10. **Time projection.** Two series are computed per pattern and year:
    - A **usage rate**: hits per million corpus words, rolling over a window of max(3, min(20, span/25)) years, and
      zeroed where the window holds fewer than 8 corpus documents. It is computed and sent but **never displayed**.
      It is only used to drop patterns whose rate is zero everywhere.
    - A **composition**: each hit is assigned to the single pattern whose words in its sentence have the largest
      summed centrality, and the counts are smoothed. The frontend normalizes these to 100% per year. **Both**
      timeline charts show this.
11. **Word map** ([`_build_graph`](python/philologic/runtime/reports/usage_patterns.py#L604)):
    - Nodes: the pattern members, padded up to 500 nodes with "context" candidates. Each context word is attached
      to the pattern with the largest summed NPMI to its members, counting any positive NPMI.
    - Links: for each node, up to 6 same-pattern and 2 cross-pattern nearest neighbours by NPMI, again any positive
      value.

**Knobs.**

| Kind | Knobs |
|---|---|
| Interpretable | Threshold base 0.22 (the standard NPMI cutoff for a real association); hub ≥ 4 neighbours |
| Fitted | Over-grab cap max(40, 8%·K); minimum 4 surviving groups; threshold ceiling 0.34; knee clamp ≥ 3 |
| Arbitrary caps | 8 hubs (threshold sizing) vs 12 (dropdown); 1,000 candidates; 500 map nodes; 6/2 neighbours per node |
| Shared with the table | Candidate band 0.5–30% of bags; the stopword or common-word filter |

---

## 2. Algorithm problems

### 2.1 Thin data: links resting on a single sentence (critical)

On large whole-corpus queries the method behaves as the May findings describe. It doesn't on small corpora or
metadata-filtered queries, which are a core PhiloLogic use (single-author databases):

| Query | Hits | Candidates | Words placed | Patterns | First pattern's label |
|---|---|---|---|---|---|
| frantext `lemma:esprit` | 171,875 | 598 | 20% | 6 | cerveau, mouvement, animal, faculté, partie |
| frantext `lemma:nature` | 96,448 | 635 | 28% | 4 | organisation, organe, génération, faculté |
| frantext `lemma:raison` | 120,287 | 538 | 22% | 3 | nuit, soleil, eau, feu, ciel (see §2.2) |
| frantext `lemma:nature`, author=Rousseau | 1,250 | 803 | **100%** | 9 | plupart, remarquer, degré, tandis, égal |
| rousseau_complete_works `nature` | 1,355 | 876 | **100%** | 3 | auquel, mesure, progrès, aucune, concevoir |
| balzac `amour` | 4,247 | 432 | **88%** | 3 | livre, présence, loin, espoir, aurait |

The thin cases are blobs of 70–330 words with cohesion 0.04–0.10 and unreadable labels. The UI gives no sign
that anything is wrong.

**Cause.** A link needs only NPMI ≥ threshold, with no minimum number of shared sentences. NPMI is unreliable for
small counts. With about 1,300 bags, two words that each occur in 6 bags and share **one** sentence score NPMI ≈ 0.5,
more than twice the 0.22 threshold. Measured support of the links above 0.22:

| Query | Links ≥ 0.22 | On 1 sentence | On ≤ 2 | On ≤ 5 | Automatic threshold |
|---|---|---|---|---|---|
| frantext `lemma:nature` | 824 | 0% | 0% | 0% | 0.22 |
| frantext `lemma:raison` | 587 | 0% | 0% | 0% | 0.22 |
| frantext `lemma:esprit` | 441 | 0% | 0% | 0% | 0.22 |
| frantext `lemma:nature`, author=Rousseau | 33,431 | 64% | 92% | 100% | 0.32 |
| rousseau_complete_works `nature` | 35,398 | 76% | 96% | 100% | 0.34 (the ceiling) |
| balzac `amour` | 7,015 | 28% | 69% | 97% | 0.28 |

The thin graphs have 8–80× more links than the whole-corpus ones, almost all of them noise. Raising the
threshold can't fix this: the scan stops at 0.34, and links resting on one sentence score as high as real ones.

The [`_build_distance`](python/philologic/runtime/reports/usage_patterns.py#L554) docstring records that a
chi²/FDR significance gate was removed after an A/B found byte-identical clusterings. That A/B could only have used
large queries, where low-support links don't exist; the gate would have mattered here.

**Tested fix: require at least N shared sentences per link** (N = 5). The rule stays one sentence: "two words are
linked if they share at least 5 sentences and their NPMI is at least 0.22."

| Query | Before | With N = 5 |
|---|---|---|
| frantext `lemma:raison` (also nature, esprit) | — | **Identical** (no link above 0.22 has ≤ 5 sentences) |
| frantext `lemma:nature`, author=Rousseau | 9 patterns, 100% placed | 5 patterns, 19% placed: *vrai, sentiment, cœur, passion, factice* · *droit, naturel, liberté, inégalité, social, civil* · *peuple, gouvernement, politique, corps* · *force, vice, vertu, qualité* · *loi, dessus, au, élever, société, citoyen* |
| rousseau_complete_works `nature` | 3, 100% | 2, 3%: *droit, naturel, naturelle, enfant, liberté, loi, inégalité* · *celles, loix, objet, qualités, constitution, uns* |
| balzac `amour` | 3, 88% | 3, 23%: *plaisirs, pensées, jouissances, âmes, sentiment, créature, aimée* · *goût, passion, véritable, vanité* · *choses, ciel, jeunes, font, détails* |

The fix is necessary but not sufficient:
- Thin-data patterns now read as real uses (natural right and inequality, nature against artifice, sensual love).
- Junk still rides along (*dessus, au, font, celles, uns*).
- Cohesion stays low (0.03–0.06, against 0.10–0.15 on the whole corpus).
- Open: whether N should be fixed (easiest to state) or scale with the number of hits, and whether the knee default
  should also see the thinner graph.

**Alternative without a fixed N: a significance test per link.** For each pair, compute the probability of seeing at
least that many shared sentences if the two words were unrelated (one-sided hypergeometric test over the query's
sentences). Keep the links that pass a Benjamini–Hochberg false discovery rate of 5%, together with NPMI ≥ 0.22.
- **It adapts.** The required number of shared sentences comes out of the data: the weakest surviving link rests on
  22–32 sentences on whole-corpus frantext and on 3 on thin data. Whole-corpus patterns are unchanged. The one
  parameter, the 5% false discovery rate, means the same thing on every corpus and can be stated to the user.
- **It exposes a second problem: links coming from a single work.** The test assumes sentences are independent,
  but words cluster within a passage or a work.
  - Before any gate, 80% (frantext Rousseau) and 90% (rousseau_complete_works) of the links above 0.22 come from
    one document; 54% for Balzac.
  - The test removes most of these (8–21% remain), and also requiring the shared sentences to come from at least
    two works removes the rest.
- **Thin data stays thin.**

  | Query | Gate | Patterns | Placed | Patterns' words |
  |---|---|---|---|---|
  | frantext `lemma:nature`, author=Rousseau | N = 5 | 5 | 19% | (see above) |
  | same | 5% FDR + two works | 3 | 6% | *charme, attachement, enivrer, délicieux, vrai* · *vertueux, celle, éclairé, réduire, question, ci* · *remarquer, peuple, également, degré, nation* |
  | rousseau_complete_works `nature` | 5% FDR + two works | 3 | 3% | *émotion, passent, delà, oreille, forte, douce* · *présence, insensible, excite, image, objet, art* · *dis, délicieux, attachement, charme, haine* |
  | balzac `amour` | 5% FDR + two works | 6 | 22% | *goût, véritable, passion, manières* · *propre, orgueil, vanité, intérêt* · … · *six, jeunesse, vingt, mois* |

  The test's patterns are small and tight (cohesion up to 0.36). Some may be real Rousseau themes (sound and
  melody, imitation in the arts), but filler words lead their labels. No gate creates structure that the data
  doesn't hold. Which gate to use should be decided on principle (statable, adapting to the data), and the UI
  should report weak structure (§2.3) rather than rely on the gate to produce good patterns.

### 2.2 Nothing tests whether a pattern is about the query

The graph measures how candidates co-occur **with each other** inside the query's sentences. Candidates are
selected by frequency (the 0.5–30% band), not by association with the query. So any cluster of words that
co-occur everywhere in the corpus surfaces as a "use" of any frequent query word.

**Lift** below = (share of the query's sentences containing the word) ÷ (share of all corpus sentences containing
it). Median lift of each pattern's words, frantext:

| Query | Typical candidate | Patterns (median lift) |
|---|---|---|
| `lemma:raison` | 2.4 | **nuit, soleil, eau, feu, ciel (1.3), ranked 1st** · souverain, prince, peuple, loi (2.6) · phénomène, physique, principe, univers (3.4) |
| `lemma:esprit` | 2.6 | cerveau, faculté (3.5) · christ, église (3.2) · science, philosophie (4.2) · ciel, soleil, feu, nuit (2.0) · europe, france (2.0) · politique, gouvernement (3.0) |
| `lemma:nature` | 3.0 | organisation, organe (4.5) · vent, bois, nuit, mer (2.0) · politique, nation (3.0) · peindre, tableau, art (4.2) |

- The same "elements" group appears for all three words. For `raison` it is barely more frequent in *raison*
  sentences than anywhere else (lift 1.3).
- It ranks **first** there, because ordering by hit mass favours patterns made of common words.
- For `nature` the landscape sense is genuine (lift 2.0), so a hard lift cutoff would need care.

**Options, roughly in order of how much they change:**
- Show lift per pattern.
- Order patterns by lift (or lift × mass) instead of mass.
- Select candidates by association with the query (lift or log-likelihood against corpus sentence counts, which
  are already cached as `sentence_df_*.npy`) instead of by frequency band.

The last option changes which words exist in the graph at all and should be evaluated on the full battery (§2.7).

### 2.3 No honest "weak structure" outcome

Whenever hubs exist, the method returns at least 2 patterns. Cohesion and coverage are computed but used for
nothing and shown nowhere. The May findings already recommended telling the user when no stable structure exists;
that is still undone.

Proposal: show coverage (share of candidates placed, and share of hits touching any pattern) and per-pattern
cohesion. Define a stated weak-structure condition (for example "fewer than 2 patterns with support", or
"coverage below X%") that shows a plain message plus the flat collocate list instead of forced patterns.

Guardrail from the May findings (§6.10): don't tune anything by maximizing an internal metric. They all reward
over-tightening.

### 2.4 Inconsistent rules and caps

- **Two attachment rules.** Pattern growth attaches a word through its single strongest link ≥ threshold
  ([usage_patterns.py:909](python/philologic/runtime/reports/usage_patterns.py#L909)). The word map attaches
  "context" words by their *summed* NPMI to members, counting links of any strength
  ([usage_patterns.py:664](python/philologic/runtime/reports/usage_patterns.py#L664)). A sum favours large patterns
  and admits sub-threshold links.
- **Two hub caps.** The threshold is sized with 8 hubs
  ([usage_patterns.py:1150](python/philologic/runtime/reports/usage_patterns.py#L1150)); the dropdown goes up to 12
  ([usage_patterns.py:1036](python/philologic/runtime/reports/usage_patterns.py#L1036)). With a single hub, the
  dropdown still offers 2.
- **Threshold ceiling.** The scan stops at 0.34, and thin data hits it (§2.1).
- **Fitted knobs.** The over-grab cap max(40, 8%·K), the minimum of 4 surviving groups, and the knee's floor of 3.
  Each is reasonable but none is stated to the user.

### 2.5 Context and bags

- **Window mismatch.** Bags are whole sentences even when the collocation table and representative passages use
  "within n words". The three views of one query therefore rest on different contexts.
- **Duplicate bags.** Two hits in one sentence give two identical bags, double-weighting that sentence's
  co-occurrences in the NPMI and the timeline.
- **Year requirement.** Hits without a year are dropped before the graph is built. A corpus without years gets no
  word map, though the map has no time dimension.

### 2.6 Time projection

- **What's shown.** Both charts show a pattern's share among hits assigned to *some* pattern. Unassigned hits
  vanish from the 100% stack, so coverage over time is invisible.
- **The rate.** The per-million rate is computed but never displayed
  ([UsagePatternsTimeline.vue:183](app/src/components/collocation/UsagePatternsTimeline.vue#L183),
  [:244](app/src/components/collocation/UsagePatternsTimeline.vue#L244)). It still decides which patterns exist
  ([usage_patterns.py:1274](python/philologic/runtime/reports/usage_patterns.py#L1274)).
- **Evidence floor.** The "≥ 8 documents" test ([usage_patterns.py:1204-1217](python/philologic/runtime/reports/usage_patterns.py#L1204-L1217))
  counts **corpus-wide** documents, so with author=X the bins look well supported even when they rest on one of
  X's books.
  - It also multiplies the window *mean* by the nominal window size.
  - At the ends of the range the real window is up to half as wide, so it can pass with about 5 documents.
  - For even window sizes the real window is one year wider than stated.
- **Year parsing.** Usage patterns parse years as the first four characters
  ([usage_patterns.py:221](python/philologic/runtime/reports/usage_patterns.py#L221)). The corpus totals use
  SQLite `CAST` ([time_series.py:36](python/philologic/runtime/reports/time_series.py#L36)). They disagree on
  values such as "c. 1650".
- **Decision needed:** should the per-pattern charts show *how often* (the rate) or *which use dominates* (the
  share)? Then label them accordingly, and either use the rate or stop computing it.

### 2.7 Suggested plan for the algorithm work

1. **Evaluation harness first.** Run a fixed battery and print, per query: candidates, coverage, threshold and hub
   count; per pattern: size, cohesion, lift, link support and label. Judge mainly by reading the labels and
   checking for an honest unplaced tail.
   - Whole-corpus frantext: the 14-word battery from the May findings.
   - Thin cases: frantext with author filters (Rousseau, Voltaire, Diderot), small single-author databases
     (`rousseau_complete_works`, `tout-voltaire`, `condorcet_complete_works`, `balzac`), and one English corpus
     (`eebo_ecco_combo`, surface forms).
   - The probe used for §2.1–2.2 is a starting point (see the appendix).
2. **Minimum link support** (§2.1): decide N, and whether it is fixed or scaled.
3. **Query specificity** (§2.2): try ordering by lift, then candidate selection by association.
4. **Weak-structure outcome** (§2.3) and display of coverage and cohesion.
5. **Unify the rules and caps** (§2.4), and settle the bag and window questions (§2.5).
6. **Time projection** decisions (§2.6).
7. Then the visualization changes (§3), which mostly follow from the decisions above.

Steps 2–6 change output by design, so each should be compared before and after on the battery.

---

## 3. Visualization

### 3.1 Word map ([WordMap.vue](app/src/components/collocation/WordMap.vue))

Measured on frantext `lemma:nature`:
- **Context nodes look like members.** 320 of the 500 nodes (64%) belong to no pattern, but are drawn in a
  pattern's colour, inside its hull, at nearly the same size (median 8.1 against 8.9 px for members). 20% of them
  are larger than the median member. The pattern on the map is therefore larger than, and shaped differently from,
  the one the timeline counts.
- **Links contradict the tooltip.** 78% of the 3,012 drawn links are below the 0.22 threshold, yet the info tip
  says "links connect words that co-occur strongly" ([en.json:249](app/src/locales/en.json#L249)).
- **No numbers.** Node frequency, centrality, link NPMI and pattern cohesion are all in the response; none is
  shown (the `weight` attribute is "kept for tooltips" that don't exist).
- **Not reproducible.** The layout seeds with `Math.random()`
  ([WordMap.vue:534](app/src/components/collocation/WordMap.vue#L534)), so the same query draws differently each
  time, which is awkward for screenshots in publications.
- **Accessibility.**
  - The map is canvas-only, with no keyboard path or text alternative.
  - Its actions need hover or right-click.
  - The legend popup has no Escape or focus handling.
  - The search box has only a placeholder ([WordMap.vue:41](app/src/components/collocation/WordMap.vue#L41)).

  A per-pattern details table (word, count, centrality, link to the hub, lift) would serve transparency and
  accessibility at once.

### 3.2 Timeline ([UsagePatternsTimeline.vue](app/src/components/collocation/UsagePatternsTimeline.vue))

- **The tooltip is wrong.** It says the per-pattern charts "show how often each is used"
  ([en.json:248](app/src/locales/en.json#L248)). They plot the share, each scaled to its own peak (§2.6).
- **No axes or values.** Neither chart has a y-axis, values or hover readouts.
- **Accessibility.** Streamgraph bands are clickable SVG paths with no focus or role.
- **Stale message.** The "no patterns" message cites "insufficient distributional change"
  ([en.json:253](app/src/locales/en.json#L253)), which the current method doesn't measure.

---

## 4. Distinctiveness statistics

- **Distinctive groups**
  ([get_outlier_groups.py](www/scripts/get_outlier_groups.py)). The score is the sum of a group's 20 largest
  positive fightin' words z-scores against the rest of the corpus.
  - z grows with the amount of data, so the score tracks group size: Spearman 0.78 (`lemma:nature`, 688 authors
    with ≥ 10 hits) and 0.79 (`lemma:amour`, 757 authors).
  - It isn't purely size: Buffon (288 hits) and Apollinaire (185) reach the top 10.
  - But the bare score badge ("115.3") invites reading it as "how different". It needs an explanation, or a
    size-normalized variant.
  - The explainer words under each group are a good transparency device.
- **Prior strength.** Both fightin' words variants set the prior strength α₀ to the full count (`n_a + n_b`),
  which shrinks heavily. Defensible, but undocumented for users; worth a sentence in the help text.
- **Similar usage** ([get_similar_collocate_distributions.py](www/scripts/get_similar_collocate_distributions.py)).
  "Excess over median" cosine similarity. See bug 5 below.

---

## 5. Confirmed bugs

1. **Comparison and similar-usage ignore "within n words".**
   - Their requests omit `method_arg` ([Compare.vue:239](app/src/components/collocation/Compare.vue#L239),
     [Similar.vue:97](app/src/components/collocation/Similar.vue#L97)). So the comparison side is counted over whole
     sentences while the primary side uses n words.
   - Clicking a collocate in the comparison cloud ([Compare.vue:218](app/src/components/collocation/Compare.vue#L218))
     sends `method=proxy` with no argument, which `resolve_method` turns into an adjacent-phrase search.
2. **Mixed lemma/surface queries count their own words.**
   - `"lemma:bon sens"` lists `lemma:sens` as its top collocate: 6,214 occurrences in 5,929 hits. The query's
     surface terms are filtered by surface form, which never matches the lemma vocabulary being counted
     ([collocation.py:644-661](python/philologic/runtime/reports/collocation.py#L644-L661)).
   - `lemma:aller chez` escapes only because `chez` is on the common-word list.
   - The endpoints also disagree on when to count lemmas: `"lemma:" in q`
     ([collocation.py:631](python/philologic/runtime/reports/collocation.py#L631)) against `q.startswith("lemma:")`
     ([get_usage_patterns.py:48](www/scripts/get_usage_patterns.py#L48),
     [get_representative_passages.py:51](www/scripts/get_representative_passages.py#L51)). So for a quoted lemma
     phrase the table counts lemmas while the word map counts surface forms.
3. **Representative passages assume a one-word query.** The hit rows are read as two words
   ([get_representative_passages.py:125-128](www/scripts/get_representative_passages.py#L125-L128)).
   - A quoted phrase returns 0 passages.
   - `lemma:aller chez` with signature *dîner, soir* scores 1,439 of the 2,159 matching sentences and never
     highlights *chez*.
4. **Word map right-click "Solo this pattern" and "Hide this pattern" throw a ReferenceError.** They call
   `onLegendClick`, which no longer exists ([WordMap.vue:397](app/src/components/collocation/WordMap.vue#L397),
   [:404](app/src/components/collocation/WordMap.vue#L404)).
5. **Similar usage has two problems:**
   - It hides the reference group by similarity < 0.99 rather than by name, so a genuinely near-identical group
     silently disappears ([get_similar_collocate_distributions.py:162](www/scripts/get_similar_collocate_distributions.py#L162)).
   - Its score is announced to screen readers as "Count" ([Similar.vue:43](app/src/components/collocation/Similar.vue#L43)).

---

## 6. Smaller issues

- **Stale comments:**
  - "truncate to the top-N senses by mass" ([get_usage_patterns.py:72](www/scripts/get_usage_patterns.py#L72),
    [usage_patterns.py:1355](python/philologic/runtime/reports/usage_patterns.py#L1355)): the count is the hub
    count now.
  - The module docstring's step 5, and "the displayed curve is the rate": it is the share.
- **Double fetch.** The word map fetches twice when you return to the tab: its own `onMounted`
  ([WordMap.vue:1015](app/src/components/collocation/WordMap.vue#L1015)) plus the parent's call
  ([Collocation.vue:379](app/src/components/Collocation.vue#L379)).
- **Leftover "T#".** The legend popup still shows the pattern number
  ([WordMap.vue:131](app/src/components/collocation/WordMap.vue#L131)), which the May findings say was removed.
- **Unused code.** [collocation_time_series.py](www/scripts/collocation_time_series.py) has no frontend caller;
  `hdbscan` is still a dependency ([pyproject.toml:33](python/pyproject.toml#L33)) with no production import (only
  the untracked `tests/tools` scripts use it).
- **Rare race.** `time_series_year_data.npz` is written non-atomically
  ([time_series.py:68](python/philologic/runtime/reports/time_series.py#L68)) and is deleted by the 10-minute
  hitlist cleanup. A read that overlaps a rewrite fails, and usage patterns then silently fall back to the share
  view.
- **Unexplained modal ordering.** The representative-passages modal doesn't show each passage's score or matched
  words, though the endpoint returns both.

---

## 7. Tests

- **Missing unit tests:** there are none for `_hyperlex`, `_auto_floor`, the NPMI matrix, `_build_graph`, the time
  projection, fightin' words, outlier ranking, representative passages or similar usage.
- **Existing coverage:** one smoke call ([test_hitlist_dir.py:312](tests/integration/test_hitlist_dir.py#L312)).
  There are no frontend tests for WordMap, UsagePatternsTimeline, DistinctivePassagesModal or GroupRanking.
- **Untracked prototype:** `tests/tools/hyperlex_threads.py` is not under version control.
- **Suggested first targets:**
  - `_hyperlex` on small hand-built graphs (hub selection, claiming, growth, the unplaced tail).
  - The NPMI formula against a direct computation.
  - The multi-word cases of bugs 2 and 3.

---

## 8. Status of the older docs

- [collocation-threads-findings.md](collocation-threads-findings.md) stops in May 2026.
  - It describes `threads.py`, `get_threads.py`, `ThreadTimeline.vue` and HDBSCAN (Part 2), and §6.7 says
    "production is still HDBSCAN".
  - Not in it: HyperLex replaced HDBSCAN (May 29); the June move to `usage_patterns.py`; the knee default; the
    composition timeline with one sense per hit; the per-million rate and its evidence floor; the sigma.js word map.
  - Parts 3–4 are superseded by Part 6 within the same file.
- [distinctive-uses-design.md](distinctive-uses-design.md) says "ready to implement" and specifies z-weighted scoring
  through a per-hit `_uses_input.npz` cache, with a score shown per passage. The shipped endpoint runs a
  co-occurrence search instead, scores without weights (no caller sends them), and shows no score.

This document supersedes both for the current state. Section 1 is the reference description of the algorithm.

---

## 9. The replacement under test: the "Word senses (beta)" tab

Status: implemented beside the current tabs for comparison. Uncommitted, and not deployed to production. A private
test server (127.0.0.1:8765, frantext, Balzac, Rousseau) runs it. This section describes the method as of
7 October 2026.

**Files**
- Backend: [word_senses.py](python/philologic/runtime/reports/word_senses.py), with the endpoints
  [get_word_senses.py](www/scripts/get_word_senses.py) and
  [get_sense_passages.py](www/scripts/get_sense_passages.py).
- Frontend: [WordSenses.vue](app/src/components/collocation/WordSenses.vue), plus the `orderingNote` prop and
  per-passage notes in [DistinctivePassagesModal.vue](app/src/components/DistinctivePassagesModal.vue).
- Tests: [unit](tests/unit/test_word_senses.py) (24, made-up sentences),
  [integration](tests/integration/test_word_senses.py) (5, ELTeC and Shakespeare),
  [WordSenses.test.js](app/tests/WordSenses.test.js) (12).

**The method**
1. **Units.** One unit per sentence containing the query; identical sentences count once. Each sentence gives three
   groups of words: its words, the word just before the query, and the word just after.
2. **Filter and weights.** The collocation view's filter (its stopwords, or its most common words) is for
   **display only**: its words are never shown in a sense's words, rarer words, groups or deciding words, and the
   response lists them as the other tabs do. The analysis reads every word, each word's evidence weighted by its
   inverse document frequency: log(N sentences / df) for sentence words, log(tokens / tf) for neighbours.
   - **Why** (pseudoword test, item 11): filtering the analysis removes the words that mark a rare meaning (*dieu*
     for *prière*). The best group for the rare word drops from 0.85 to 0.19 (*argent / fleur*), and the mean ARI
     from 0.15 to 0.11.
   - **Default views read better too.** The first splits are balanced (*bank* 60% money / 40% river; *light* 57%
     lighting / 43% natural light), where before they peeled a small group off a large remainder described only
     by what it lacked. Fitting counts stay unweighted. Only the query's collocates may
   mark a sense. Collocates are words markedly more frequent around the query than in the corpus, by BIC:
   G² > log n.
3. **Root.** Katz backoff with absolute discounting against the corpus, with D = n1/(n1+2n2) per group of words.
4. **Splits are sparse.** The words that mark either half (markedly more or less frequent than in the parent, BIC)
   get their own rates in both halves. Every other word keeps the parent's rate, scaled. This is the
   "feature saliency" form of a mixture (Law, Figueiredo & Jain 2004). Own rates use interpolated absolute
   discounting: the discount is taken over every word the half has, and shared out as the parent's rates are. So a
   word marking the other half keeps a small rate, not its parent's, and never zero.
5. **Search.** Up to 5 seeds per node: collocates scored by how much better their sentences and the others predict
   their other collocates (leave-one-out, idf-weighted). From each seed, EM runs twice:
   - first with every collocate getting its own rates;
   - then with only the marked words, as in step 4.

   A seed covers only part of its sense, so a gated first fit would mark only the seed itself. If no word tells the
   two halves apart, there is no split.
6. **Test.** 5 folds of whole works held out. Each node's unsplit model is refitted from its parent. The gain is the
   mixture log-likelihood minus the unsplit log-likelihood, per held-out sentence. A split is accepted when
   mean / SE over sentences > 2 (`EVIDENCE`) and its importance is at least the floor.
7. **Importance.** Node share × net gain per sentence, shown relative to the main (largest) distinction. The search
   stops below `FINEST` = 10% of the root's importance.
8. **Display.** The slider adds distinctions in order of importance. By default it stops at the **natural break**:
   the largest drop in importance from one distinction to the next, as a ratio (like the eigengap rule for a number
   of clusters), with the 10% floor counting as the value after the last. This replaces the fixed 25% default and
   gives 2–3 senses: Rousseau *nature* 2, *bank* 2 (money / river), *light* 2, Balzac *amour* 2 (*l'amour* /
   *amour-propre*), *révolution* 3 (two political / planets and change), *raison* 3. Each sense is described in
   three ways:
   - by words markedly more frequent in it than in its parent (×), ranked by weighted G²;
   - by words markedly rarer in it (÷), which name "the rest" senses ("without …");
   - by its works, flagged if one work makes up more than half of it.
9. **Time.** Soft sense counts per year, a Whittaker smoother with second differences and λ chosen by GCV, years
   weighted by works, and bands from 200 resamplings of the works. Shown as a share or as uses per million words.
11. **Pseudoword test** (two unrelated words queried as one, scored on how well the groups match the two words; scratch
    `pseudowords.py`). Over 23 pairs in 4 corpora, mean ARI: senses 0.11 with the filter in the analysis, 0.15
    without; usage patterns 0.02 either way. Usage patterns never separates the words: its patterns mix them in their
    overall proportions. Senses separates large balanced pairs (frantext *navire / prière* 0.64, *étoile / vin*
    0.46–0.59). It fails on lopsided pairs (the rare word under ~5%) and on thin corpora.
    - **Diagnosis.** The frequent word's sub-topics differ more among themselves than the rare word differs from it.
      So EM from any start splits those first, and the rare word appears, if at all, deeper in the tree.
    - **Tried, no effect:** diverse seeds, 20 seeds, the principal direction, starts from the worst-fitting
      sentences, and less weight on the sentence's words against its neighbours.
10. **Word groups (HyperLex inside a sense).** The marker words of each sense are linked where they go together in
    its sentences, by the Topics tab's rule: replicated across works, 2 SE (see §10). Then hub grouping without
    thresholds: the most linked word is a hub and takes the words linked to it, then the next hub among the rest. A
    word goes to the hub it is most linked to. Cards show hub-led rows, then "Other words". *révolution*'s social
    sense groups as *classe*: ouvrier, bourgeoisie, prolétariat · *abolir*: féodal, privilège · *social*:
    hiérarchie, ordre, communiste. Thin data (Rousseau) has too few marker words for groups: 1 sense of 35 has one.
    Run time is unchanged.

**What it gives** (no filter, idf). Times are for the current code; the first figure in parentheses is the same
method before optimisation, the second the previous version (one-sided markers, no search phase).

| Query | Sentences | Time | Senses at 25% / all | Main distinction |
|---|---|---|---|---|
| `rousseau_complete_works` nature | 1,315 | 1.9 s (6.6 s; 8 s) | 8 / 18 | *la nature* against *état, loi, droit*, z = 21 |
| frantext `lemma:nature`, Rousseau | 1,149 | 1.3 s (3.2 s; 8 s) | 6 / 16 | *état, loi, droit, société* against the rest, z = 21 |
| ELTeC `lemma:bank` | 1,048 | 1.0 s (2.9 s; 8 s) | 6 / 15 | money against river, z = 18 (before: 10% "pound" against the rest) |
| ELTeC `lemma:light` | 6,731 | 1.5 s (8.0 s; 19 s) | 4 / 9 | natural light against lighting, z = 49; then candle/lamp/gas against pipe/cigar |
| Balzac `amour` | 3,980 | 1.7 s (7.2 s; 13 s) | 7 / 19 | *amour-propre* against *l'amour*, z = 31 |
| frantext `lemma:révolution` | 35,527 | 4.1 s (25.7 s; 11 s) | 6 / 12 | political against astronomical/change, z = 115; then counter-revolution, history, social revolution |
| frantext `lemma:raison` | 112,267 | 10.3 s (84.6 s; 45 s) | 5 / 10 | abstract reason against *avoir raison / tort*, z = 203; then *en raison inverse / suffisante*, *perdre la raison*, Kant |

**Speed.** The optimisations leave the output the same to the last bit: the JSON of all 7 queries is identical to the
unoptimised code's.
- **Compiled EM.** Each iteration's counting, marker test and rate fitting, and the likelihoods, run in numba
  kernels without the GIL. They reproduce numpy's pairwise sums exactly. `log` and `exp` stay in numpy, whose
  AVX-512 versions differ in the last bit from the C library's (and so from numba's) for 0.2–8% of values.
- **Exact `logsumexp`.** scipy's 2-column `logsumexp` is replaced by the same arithmetic without its overhead.
- **Threads.** A sense's 5 seed fits run at once, then all their fold fits. Below the first split, the senses
  grow at once too: they are independent once the root sets the floor, and numbering comes later from the tree.
- **Fold data once per sense.** The training matrices, unsplit fits and held-out likelihoods are built once per
  sense, not once per seed.
- **Tests.** Unit tests check the compiled EM against plain numpy, and the sums against `np.sum`, bit for bit.

**Single-threaded speed** (Oct 8, for a busy server). The output is still the same to the last bit on 10 reference
queries and on eebo_ecco_combo *liberty* and *reason*. The 10 are the 7 above, Shakespeare `"my lord"`, frantext
`lemma:grâce` within 5 words, and Balzac `amour` without a filter. Warm, on one thread, their total went from
77.6 s to 31.7 s:

| Query | Before | After |
|---|---|---|
| frantext `lemma:raison` | 38.2 s | 15.2 s |
| frantext `lemma:grâce`, within 5 | 18.2 s | 6.5 s |
| frantext `lemma:révolution` | 10.7 s | 4.4 s |
| eebo_ecco_combo *liberty* | 121 s | 48 s |
| the 7 small queries | 0.7–2.5 s | 1.4–2.0× faster |

- **Only the words the sentences have.** A sense's EM iterations follow only the words its sentences have. Below the
  root that is about a third of the query's words. Sums over a group's words still run over all of them, with the
  missing ones as zero. They use a pairwise sum that skips the parts with no entry, which is exact because adding
  zero changes no sum.
- **Sliced ELLPACK.** Sentence sums take 4 sentences of about the same length at a time. Each is padded with an entry
  of zero evidence. Word counts take 4 words at a time, each word's sentences in order. Every sum keeps its order,
  and the four independent sums keep the processor busy.
- **Unsigned indices.** numba checks every signed index for being negative. `uint32` indices made both kernels
  1.5× faster.
- **numpy's slow paths.**
  - `r.sum(axis=0)` on an n × 2 array: numpy sums it down each column in order, now compiled.
  - `np.nanpercentile` loops over places in Python when any value is NaN. The undrawn resamples are NaN as a whole,
    so `np.percentile` over the drawn ones gives the same result.
  - `np.unique(axis=0)` on the 1-D works sorted them as records.
- **Seed scores** skip terms that cannot be positive: log x ≤ x − 1, with a margin for rounding. A word none of a
  seed's sentences have is skipped on its count alone. Co-occurrence counts add up as integers, exact in any order.
- **The marker test** skips its two logarithms where a bound on G² is under the threshold: G² ≤ 2t(c − e)²/(e(t − e)),
  since KL ≤ χ².
- **The kernels' loops** use unsigned counters and flat arrays: no sign check and no multiply by a stride per entry.
- **Vocabulary of the sentences:** `np.unique` over all the words' entries became marking the ids. On eebo *reason*,
  reading the sentences went from 10.6 s to 2.8 s.
- **Threads** leave the output identical. They are capped at numba's thread count, `NUMBA_NUM_THREADS`, as for the
  rest of the app. The web server sets it to 2 per worker. All the work runs on that pool, and the threads that grow
  senses only hand work to it. One thread runs without any pool.
  - With 2 threads, *raison* takes 9.6 s (15.2 s on one thread). eebo_ecco_combo *liberty* (191k sentences) takes
    32 s (48 s), and *reason* (785k sentences) 169 s (301 s), under the server's 300 s timeout.
  - Tiny queries lose a little: Shakespeare `"my lord"` takes 0.59 s against 0.43 s on one thread.
  - Uncapped (64 threads), small queries were slower still (0.55 s to 1.0 s), their Python-heavy tasks contending
    for the GIL.
- **What is left.** The two kernels are half the time, at about 2 cycles per entry. With all their data in L1 they
  are only 20% faster, so the processor itself is the limit. Tried, no real gain: both senses as one complex number,
  and 5 EM runs in one pass (at most 1.4× on that part; numba's LLVM does not vectorize either). Exact output fixes the
  rest: up to 35 EM runs per sense, up to 200 iterations each. On *raison*, the candidates' fold fits are 63% of the EM
  time, and half of all iterations are in the 37 runs that reach the cap.
- **Bad dates.** eebo_ecco_combo has years 1 and 16461, so its curves span 16,461 years: 2.4 s and a 12.5 MB report
  for *liberty*. That is unchanged.

On made-up sentences it finds two balanced senses, a single work's two senses, unequal shares (1/3 and 2/3) and a
three-sense tree. It finds nothing in sentences alike. The previous version missed the balanced cases.

**Lessons.** Don't repeat these.
- **A bag without a stopword list or weighting** splits by person (*je/vous*). idf weighting plus collocate-only
  markers replaces the list, generically.
- **A single Dirichlet precision** kills departures from the parent.
- **One-sided sparse senses are weak.** If a sense may only raise its markers, and all other words are scaled
  uniformly, then in sense *b* the words of sense *a* keep 85% of their parent rate. Balanced senses then give about
  0.1 nat per sentence instead of 1.4, and EM collapses on thin data. Both halves need own rates for the words
  marking either.
- **Words absent from a half need positive rates.** Otherwise held-out sentences become impossible, and their folds
  are silently skipped.
- **Degenerate fits fake evidence.** When both halves equal the parent, gains of 10⁻¹⁵ with an SE of 10⁻¹⁶ gave
  z = 12. Splits that no word tells apart are now no splits.
- **Interpolated backoff from a sense to its parent** compounds the novelty discount for words new to both. Use
  Katz at the root. Interpolation is used only for split words.
- **The CV can leak.** The unsplit model must refit from the node's parent.
- **CA's first axis** isolates outliers.
- **"Both sides must gain"** rejects sense-against-rest splits.
- **Sentence-level folds** let passages pass as senses. Hold out whole works.
- **Candidates from node against parent** are just the sibling's complement. Seed from the query's collocates.
- **Words drawn without replacement** (the made-up sentences) are a real pattern that splits exploit within a pure
  sense. Those splits rank at 11–22% of the main one, under the default.
- **Merging "closely related" senses** (after "too many senses, many closely related"). None of these tells a sense
  from a topical facet:
  - *Split-and-merge* (refit the finest senses together, join them by least information lost, test each join on
    unseen works): 0 merges, because every join passes, and more senses at 25%. Reverted. The tree does hold
    near-duplicates in different branches (*révolution*'s bourgeoisie and archives senses under both sides of one
    split), but they are distinguishable on unseen works.
  - *One sense per discourse* (merge senses used in the same works): almost no merges, and it wrongly merged the
    planetary and political senses.
  - *How clearly sentences are told apart*: 0.83–1.00 for every split.
  - *Information per word next to the query against the rest of the sentence*: no clean separation. *bank*'s
    money/river split has the lowest ratio.
  - Conclusion: how many senses to show is a granularity question, so the default became a natural break in the
    importances.

**Remaining fixed values**
- Conventions: 5 folds, EVIDENCE = 2 SE, 200 resamplings and a 95% band.
- Display: the default detail of 25% and the search floor of 10% (the slider covers everything in between).
- Search breadth: 5 seeds per node.
- Payload caps: 40 words and 50 examples per sense.
- Numerics: EM tolerance 10⁻⁶ and 200 iterations, and the λ grid.
- Distance 1 for the neighbour words.

**Open points**
- **Speed on the largest queries.** `raison` (112k sentences) still takes 10 s. Within a sense, each phase waits
  for its slowest fit.
  - Further levers that keep the output: parallel loops inside the kernels for the largest senses.
  - Levers that change it slightly: SQUAREM, numba transcendental functions, testing fewer seeds.
- **Threads per request.** A request uses up to 2 × the core count of threads. That is fine on the private server;
  for production with 8 workers, cap it.
- **Payload.** About 450 KB for Rousseau and 840 KB for large queries, mostly the curves of every node.
- **Function words in descriptions.** *par, leur, des, l', d'* still appear, though ranked lower. The word before the
  query is also in the sentence's bag, so it counts twice.
- **One-work senses** cannot be confirmed on unseen works. This is by design, and the tab flags senses mostly from
  one work.
- **Hidden deciding words** fall outside the concordance snippet.
- **Rate bands** resample numerators only.

## 10. HyperLex without thresholds: the "Topics (beta)" tab (dropped)

Status: **dropped** on 7 October 2026: "word senses and usage patterns just work better". Its tab, endpoint, module
and tests are removed. Its link rule (replicated across works) and hub grouping moved into
[word_senses.py](python/philologic/runtime/reports/word_senses.py), for the word groups inside each sense (§9,
item 10). Their tests are in [test_word_groups.py](tests/unit/test_word_groups.py). What follows records what was
tried.

**Files**
- Backend: [word_topics.py](python/philologic/runtime/reports/word_topics.py), endpoint
  [get_word_topics.py](www/scripts/get_word_topics.py). Examples render through `get_sense_passages.py`.
- Frontend: [WordTopics.vue](app/src/components/collocation/WordTopics.vue).
- Tests: [unit](tests/unit/test_word_topics.py) (11), [integration](tests/integration/test_word_topics.py) (5),
  [WordTopics.test.js](app/tests/WordTopics.test.js) (8).

**The method.** It keeps HyperLex's structure, a network of the query's collocates, and replaces each threshold:
1. **Nodes.** The query's collocates: words markedly more frequent in its sentences than in the corpus (BIC). The
   collocation view's filter applies.
2. **Links.** Two collocates are linked when their PMI is above zero (independence) even at the low end of its
   uncertainty: the exact Poisson bound on the co-occurrence count, at 2 standard errors. A pair seen in a single
   sentence is not a strong link. The weight is that lower bound, normalised (NPMI).
3. **Grouping, by one of two methods** (a switch in the tab):
   - **Link communities** (Ahn, Bagrow & Lehmann 2010). Links are grouped, not words: two links of a word are
     alike as the words at their other ends have alike weighted neighbourhoods (Tanimoto). Then single linkage,
     with the cut at maximum partition density. Each word only needs the maximum spanning tree of its own links,
     because two links share at most one word.
   - **Watset, simplified** (Ustalov et al. 2019). Chinese Whispers (Biemann 2006) groups each word's neighbours
     into its senses. Each link joins the senses of its two words that hold each other, and Chinese Whispers then
     groups the senses. Nodes are processed strongest first, so the result is the same on every run.
4. **Topics.** A topic is a group with more links than a tree of its words, so its words are tied by more than a
   chain. A lone link or a chain has no density, so no size cutoff is needed.
5. **Sentences.** A word belongs to each of its topics in proportion to its link weight there, so a word can be in
   several topics. A sentence goes to the topic its words belong to most, which lets its other words decide where a
   shared word counts. A sentence with no topic word is unplaced.
6. **Display.** Curves over time for the 16 largest topics (the chart's colours), with the rest shown together.
   Shared words are dashed, with "mainly in topic k" when another topic holds most of their links.

**What it gives** (default filter)

| Query | Link communities | Watset |
|---|---|---|
| Rousseau `nature` | 9 topics, 64% unplaced | 19 topics, 63% unplaced |
| ELTeC `light` | 48 topics, 41% unplaced: hair, blossom/sky, stained glass, fireside, sea | 139 topics, 19% unplaced: one natural-light topic (sky, cloud, sun, sea) at 38%, then hair, river, stained glass, fireside glow, candles |
| ELTeC `bank` | 20 topics: money, hills, river scenes, banking house | 46 topics: money 13%, banking house, several river scenes |
| frantext `révolution` | 94 topics: bourgeoisie, European powers, revolutionaries, planets, Louis XVI, archives | One political topic at 90%, then planets, revolutionaries, archives |
| frantext `raison` | One topic at 85% (nature, principe, loi, corps, cause), then *tort*, *folie*, *libre arbitre* | One topic at 90% |

Run times are 0.5–3 s.

**Findings**
- **Topical by nature.** Grammatical uses disappear: *avoir raison / tort* is 42% of uses in the senses tab and
  under 1% here, because the filter removes *avoir*.
- **Dense cores stay whole.** *raison*'s abstract vocabulary is mutually associated, and neither method splits it.
  These failed too:
  - re-clustering inside the giant topic;
  - keeping only links stronger than their word's average;
  - linking only words together at least twice as often as chance;
  - the disparity-filter backbone (Serrano et al. 2009), which removes almost every link in thin data.

  This is acceptable when the main topic makes sense on its own, as it does here.
- **The number of topics grows with data**, at about one per 12 collocates.
- **Watset tends to form one giant topic on dense graphs** (*révolution*), a known behaviour of Chinese Whispers.
  Link communities split those better. Watset places more sentences and gives coarser groups on medium data
  (*light*).
- **Filter bug (fixed, both tabs).** With a filter, the collocate test compared against corpus rates over all
  words, so every remaining word looked over-represented. Rates are now taken among the words left in.

**Repair: works as the unit of evidence** (after "link communities work worse than usage patterns; an explosion of
topics means spurious signals"):
- **Diagnosis.**
  - 22–49% of links rested on co-occurrences within a single work.
  - The topic explosion came mainly from partition density, which gives any small clique the top score (a triangle
    scores 1).
- **Links** now also need their excess of co-occurrence over chance to hold across works: the mean excess per work
  must exceed 2 SE over works.
- **Topics** need the same test on their words' within-sentence pairs.
- **Results** (link communities):
  - *révolution* has 40 topics instead of 94, all coherent: the European powers; rights, liberty and equality;
    England and the coalition; the feudal regime; the Committee of Public Safety; religion.
  - *light* has 14 instead of 48.
  - Balzac *amour* keeps one, *amour-propre*.
  - Rousseau *nature* and ELTeC *bank* get none, because few pairs recur in enough works.
- **"Co-occurs in at least two works"** for links rescued medium data (*bank* 4 topics, *amour* 4) but not Rousseau.
  It is a fixed number and wasn't kept.
- **Watset still forms giants** (*révolution* 91%).
- **Verdict.** Spurious topics are gone on large data, but thin data gets none. So option 3, the word groups inside
  senses (§9, item 10), became the HyperLex layer.

**Remaining fixed values:** the 2-standard-error convention, BIC for collocates, the sentence as context, and display
caps (40 words, 40 pairs and 50 examples per topic; 16 curves).

## 11. Ego network + Leiden: the "Ego network (beta)" tab (dropped)

Status: **dropped** on 7 October 2026. Its tab, endpoint, module and tests are removed; what follows records it. Given
all the words, it did worse on the pseudoword test (mean ARI 0.04 against 0.06 with the filter). Common words become
hubs, because a network has no weighting, whereas the senses tab gains from them (0.11 → 0.15).

**Files**
- Backend: [ego_communities.py](python/philologic/runtime/reports/ego_communities.py), endpoint
  [get_ego_communities.py](www/scripts/get_ego_communities.py).
- Frontend: [EgoCommunities.vue](app/src/components/collocation/EgoCommunities.vue).
- Tests: [unit](tests/unit/test_ego_communities.py) (4), [integration](tests/integration/test_ego_communities.py) (5),
  [EgoCommunities.test.js](app/tests/EgoCommunities.test.js) (7).

**The method**
1. **Nodes:** the query's collocates (BIC), with the collocation filter applied as in usage patterns.
2. **Links:** two collocates are linked if the lower bound of their PMI is above zero (2 SE), tested sentence by
   sentence. Requiring links to replicate across works left Rousseau with 1 link and *bank* without its money
   community.
3. **Communities:** Leiden with modularity at resolution 1, from networkx 3.7 (no new dependency). Note that networkx
   defaults to the Constant Potts Model, which gives near-singletons; modularity must be asked for explicitly.
4. **Sentences:** each goes to the community its words weigh most in, a word weighing its links within its community.

**What it gives**
- *révolution*: 9 communities, 6% of sentences unplaced: the Convention and Girondins, religion and principles, the
  European powers, the proletariat, planets, Louis XVI, the archives, the revolutionaries.
- *bank*: 13: river and reeds, money, hills and towers, the banking house.
- *light*: 9: windows and stained glass, sky and sun, blossom, hair, fireside and gas, incense.
- Rousseau *nature* and Balzac *amour*: 17 each, with 27–33% of sentences unplaced.
- Run time is 0.2–6 s; Leiden in networkx is pure Python.

**Pseudoword test** (23 pairs; mean ARI): senses 0.15, ego network 0.06, usage patterns 0.02. The ego network finds
many communities (8 to 38 per pair), each fairly pure, and leaves 4–56% of sentences unplaced. A sentence can join a
community through a single word linked to it by chance (a weakness usage patterns shares).

## 12. Senses 2: the "Word senses 2 (beta)" tab (Oct 8, under test)

The same method, made cheaper where that keeps its quality. It is a tab of its own, next to the first, for comparison
(`get_word_senses.py?senses_version=2`, its own cache tag). The code is `fit_split_screened` and the `version`
argument of `induce_senses`.

**What changed.** The words with rates of their own in a split are chosen once, then held. The first method chooses
them again at every step of EM. That switch kept fits from settling: on *raison*, half of all EM steps were in 37 fits
that stopped at the 200-step cap. Senses 2 does the same search first, letting every collocate tell the senses
apart. It then chooses the words that mark either sense, by the same BIC test, and fits the split with them held. A
fold's fit chooses its words from its own training sentences. The fold fits of *raison* took 2,991 steps instead of
9,692, and none reached the cap.

**Speed** (2 threads):

| Query | Senses 1 | Senses 2 |
|---|---|---|
| frantext `lemma:raison` | 9.0 s | 6.7 s |
| frantext `lemma:grâce`, within 5 | 4.2 s | 2.5 s |
| eebo_ecco_combo *liberty* | 31.7 s | 19.7 s |
| eebo_ecco_combo *reason* | 168.5 s | 64.1 s |
| the small queries | about the same | about the same |

**Quality.**
- Pseudoword test, scored as the report analyses: every word read, the filter only hiding. The mean ARI is 0.162
  for Senses 2 against 0.148, purity 0.79 against 0.78, and both show 2.9 senses. 5 pairs gain, 2 lose a little.
  - The test script had been left filtering the analysis, as before the filter became display-only. It now reads
    every word, as the report does. Filtered, Senses 2 had looked worse, by showing more senses.
- Reference queries: the same main distinctions. *light* is sky against lighting, *bank* money against river,
  *raison* abstract reason against *avoir raison*, *révolution* political against astronomical, *grâce* and *my lord*
  as before. Balzac *amour* shows 7 senses at its default detail instead of 2: the top splits are alike, but the
  natural break falls lower.

**First view, then finer distinctions on request.** Senses 2 grows only the senses its first view needs, then
stops. That view is the splits down to the largest drop in importance (the natural break). A split found later
changes it only if it is at least as important as the split just below that drop: one landing in a smaller drop
makes no drop larger.
- The senses are tried largest first, in turn. Growth stops once none left could produce a split that important.
  An untried sense is taken to gain at most 1.5 times the most bits per sentence any split found so far gained
  (`_MARGIN`).
  - 1.5 is the least margin under which the first view was the whole tree's in all 35 trees: the reference
    queries, the pseudoword pairs, and eebo *liberty* and *reason*. With 1, 2 of the 35 differed.
- The tree so far and its sentences are kept in the hitlist directory (`.senses-state.npz` and `.json`, no pickle),
  with the cache's lifetime: the cleanup removes them after 10 minutes.
  - The button "Finer distinctions" (`senses_finer=1`) takes the tree up again and tries the rest.
  - Untried senses show as "not tried yet".
  - Without the kept state, the whole tree is grown from the start.
- Checked on the 35 trees: the tree taken up again is the whole tree grown at once, the same JSON to the byte, and
  the first view's default view is the whole tree's every time.

| | First view | Whole tree at once | The rest, on request |
|---|---|---|---|
| frantext *raison* | 5.1 s | 6.6 s | 2.5 s |
| eebo *liberty* | 17.0 s | 19.5 s | 7.4 s |
| eebo *reason* | 46.8 s | 62.7 s | 25.5 s |
| all 35 trees | 99.5 s | 126.2 s | 57.0 s |

- The first request is 20 to 25% faster. Asking for the rest costs about a quarter more than the whole tree at once.
- The state takes 44 MB for *raison* and about 350 MB for eebo *reason*: the sentence matrices (counts as bytes),
  and each sense's rates and counts. Floats are kept exact, so the rest is the same.
- The first view grows its senses one at a time (each sense's fits still use both threads), for the same view
  whatever the number of threads.

**Tried, dropped:**
- **Testing the candidates in turn**, in order of their gain on their own sentences, and keeping the first that
  passes. The in-sample gain picked the candidate the test picks only half the time, at nodes that split. The first
  that passes is not always the most important, so *bank* and *my lord* came out in more, smaller senses. Combined
  with held words, *révolution* lost its astronomical sense at the top. Every candidate is tested, as in Senses 1.
- **EM over the entries of the held words alone.** The other words differ between the senses only by a scale, so a
  sentence's sum of them is two numbers, made once. The E-step is the same, and the first step matched to 1e-14.
  - The discount freed by all the words needs every word's count, so the others' share was held and made again when
    EM settled.
  - Each step cost a third less, but the refreshes took 40% more steps: no gain. Fits of held words take only 6 to 9
    steps, too few to pay for making their entries apart.
  - With a stale share, a sense's probabilities summed past 1; it was then held as a share of the others' counts.
- **Skipping the test of candidates that fall short on their own sentences** (in-sample importance below the floor,
  or z within 2), taking in-sample gain as optimistic. It gained only 1 to 7%: the in-sample measure costs about what
  it saves. Once, on ELTeC *violin | plough*, the held-out z beat the in-sample z, and the skip dropped the root's best
  candidate, changing the main distinction.
- **Leaving out candidates alike to an earlier one.** A fifth of later candidates are identical on small queries,
  none on *raison*, *révolution* or *light*. Not done.

## What's solid

- **The collocation table:** counts that match the concordances they link to (integration-tested), the exact filter
  list shown to the user, and a common-word filter whose rule is one sentence.
- **HyperLex's rule** really is statable in one sentence, and on large corpora its senses are good: *esprit* gives
  Christ and church, science and philosophy, mind and brain.
- **Distinctive groups** show their evidence words.
- **Engineering:** the numba kernels, flat CSR bags and per-query intermediate cache are careful work.

---

## Appendix: how the checks were run

All runs were offline Python calls into the source tree. They used the installed interpreter, with a scratch
global config so every hitlist and cache was written to a private directory rather than the production one:

```sh
# scratch philologic5.cfg: database_root = "/var/www/html/philologic5"; hitlist_dir = "<scratch>/hitlists"
export PHILOLOGIC_CONFIG=<scratch>/philologic5.cfg
export NUMBA_CACHE_DIR=<scratch>/numba
export PYTHONPATH=/disk1/shared/PhiloLogic5/python
/var/lib/philologic5/philologic_env/bin/python probe.py frantext lemma:nature '{"author": "Rousseau"}' 5
```

What `probe.py` does:
- Calls `detect_usage_patterns` with the production default filter (`_df_filter_list(path, count_lemmas, 1.0)`,
  `lemma:` prefix stripped).
- Disables the intermediate cache, by patching `_load_intermediates` and `_save_intermediates`.
- Optionally wraps `_build_distance` to set distance 1.0 wherever two candidates share fewer than N bags. The shared
  counts come from the same sparse bag × candidate matrix `_build_distance` builds.
- Computes lift from `_accumulate_df` over the bags and `_load_sentence_df` for the corpus.

Endpoint checks (bugs 2–3, distinctive scores) built requests with
`WSGIHandler({"QUERY_STRING": ..., "PHILOLOGIC_DBPATH": root}, WebConfig(root))`, as
[tests/integration/test_collocation.py](tests/integration/test_collocation.py) does. Bugs 1, 4 and 5 were confirmed
by reading the code.
