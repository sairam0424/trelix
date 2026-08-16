# Evaluation fixtures

Golden sets consumed by `trelix eval` and `trelix eval-synthesis`.

## `golden.jsonl` — retrieval quality

One JSON object per line:

```json
{"query": "how are files excluded from indexing", "relevant_files": ["src/trelix/indexing/walker.py"]}
```

| Field | Meaning |
|---|---|
| `query` | Natural-language question, phrased the way a developer would ask it |
| `relevant_files` | Repo-relative POSIX paths whose contents answer the query |

Run it with:

```bash
trelix eval . --golden eval/golden.jsonl
```

Reports nDCG@10, Recall@10 and MRR over the top-10 **distinct files**.

### Two ways to get a silently wrong score

**Paths are compared by exact string equality** against each result's `rel_path`
(`src/trelix/eval/harness.py`). A typo, a `./` prefix or an absolute path matches
nothing, and the query scores 0 — identical to a genuine retrieval failure, with no
warning to tell the two apart. Verify paths exist before adding an entry.

**An entry with an empty or missing `relevant_files` is skipped entirely**, so it
silently shrinks the denominator. Check `Queries evaluated` in the output against the
line count of this file:

```bash
wc -l eval/golden.jsonl        # must equal "Queries evaluated"
```

### Metrics count files, not chunks

Retrieval ranks *chunks*, and one file supplies many of them — this repository averages
roughly 77 chunks per file. Ground truth here is file-level, so the metric functions
collapse repeats to each file's best rank before scoring. `@10` therefore means ten
distinct files.

Before v3.1.2 they did not: a relevant file appearing five times in the top ten scored
`recall@10 = 5.0` and `nDCG@10 = 2.52`, against a documented range of `[0, 1]`. **Scores
produced before v3.1.2 are not comparable with scores produced after it.**

### Scores move between runs

Retrieval is not deterministic when the LLM-dependent legs are on. With
`TRELIX_RETRIEVAL_HYDE_FALLBACK` and `TRELIX_RETRIEVAL_FLARE` enabled, two consecutive
runs over an identical index measured mean precision@10 of 94.0% and 95.0%. Treat small
differences as noise, and disable those flags when you need a repeatable number.

## `golden_synthesis_sample.jsonl` — synthesis quality

Input to `trelix eval-synthesis`, which scores answer faithfulness and completeness
GroUSE-style rather than measuring retrieval.

## Related tooling

`scripts/measure_index_hygiene.py` answers a different question from `trelix eval`: not
"did we rank the right files" but "how much of what we ranked was never source code to
begin with". It needs no golden set, so it works on any index:

```bash
python scripts/measure_index_hygiene.py . --json > docs/reports/index-hygiene-after.json
```
