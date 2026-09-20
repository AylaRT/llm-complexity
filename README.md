# Syntactic complexity annotation by large language models

Measures of syntactic complexity in L2 writing research (mean length of sentence,
mean length of T-unit, clauses per T-unit, and so on) all rest on the same five
counts: words, sentences, T-units, clauses and dependent clauses. Those counts are
produced either by hand, which is slow, or by a parser such as L2SCA or TAASSC,
which was built for a different kind of text than beginner learner writing.

This repository tests a third option. Each model is given the same annotation
guidelines the human annotators used, and asked to identify the five unit types in
one learner text. Its answer is parsed into counts by code, and those counts are
compared against the human gold standard and against both parser baselines. No
measure is ever computed by a model.

Two output formats are run for every model and text. In the lists format the model
writes out the units themselves, one per line (words comma-separated), which makes
its segmentation inspectable. In the counts format it reports five integers. Each
model is run with reasoning on and off, and every cell is repeated three times, so
that instability within one condition can be told apart from disagreement between
conditions.

## The corpus is not in this repository

The data are 80 texts by Dutch-speaking learners of English, beginner to
lower-intermediate. Neither the texts nor the gold standard are public, and
everything under `data/`, `outputs/`, `results/` and `logs/` is gitignored. The
annotation guidelines themselves are public, as part of the two prompt templates
in `prompts/`.

The code therefore does not run as cloned. To run it on your own material you need
`data/texts.tsv`, a tab-separated UTF-8 file with columns `text_id`, `level` and
`text`, one text per row and no newlines inside a text. Scoring against a gold
standard additionally needs `data/results_new.xlsx` in the layout `src/score.py`
documents.

## Install

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

API keys are read from the environment and never logged: `OPENAI_API_KEY`,
`DASHSCOPE_API_KEY`, `MISTRAL_API_KEY`. All three providers are called through the
OpenAI SDK, since DashScope and Mistral expose OpenAI-compatible endpoints.

## Running the pipeline

The stages are deliberately separate, and each one writes to disk before the next
one reads. Parsing and scoring never call a model, so they can be rerun freely.

Build the corpus. `prep_corpus.py` reads .docx files holding one text per table
row, identifying the id, level and text cell by content rather than by column
position, and stops on any structural anomaly instead of writing a partial file.
The source filenames default to `CORPUS_SOURCE_FILES` in `config/settings.py`,
which are placeholders; point `--sources` at your own.

```bash
PYTHONPATH=. python src/prep_corpus.py --sources data/first.docx data/second.docx
```

The validation is specific to this corpus (exactly 80 texts, ids 1 to 80
complete and unique) and will need adjusting for another one.

Run a phase. Model conditions, corpus and repetition count come from
`config/settings.py`; `--dry-run` prints the cells without calling anything.

```bash
PYTHONPATH=. python src/run_experiment.py --phase phase1
```

A cell that already has a valid saved response is skipped, so an interrupted run
resumes where it stopped. To split a phase across terminals, give each process a
different `--models`.

Turn the saved responses into tables:

```bash
PYTHONPATH=. python src/extract_results.py --phase all
```

This writes `results/all_results.tsv`, one row per cell with the five counts, the
gold and baseline counts, and for the lists format the words and spans the model
omitted or invented; and `results/all_lists.xlsx`, one row per unit with the items
themselves. Add `--overrides data/manual_overrides.json` to apply the manual
corrections described below, `--per-condition` for one file per condition, or
`--all-attempts` to keep the retries.

Review the responses that came back in an unusual shape:

```bash
PYTHONPATH=. python src/nonstandard_responses.py
```

## How model output is turned into counts

`src/parse.py` splits a response by its five headers and counts the items in each
block. It validates structure only, meaning whether the blocks are present and
non-empty, and never judges whether the content is right.

The tolerance built into the parser is limited to formatting variation any model
could produce given the prompt, and it is applied to every response identically:
blank lines between items are ignored; a bare "none" in the dependent clause block
is dropped, because the prompt asks for that word and models sometimes write it
alongside real clauses; numbered or bulleted prefixes are stripped when every item
in a block has one; and a comma between two digits is not read as a word separator,
so that a thousands separator survives the comma-separated word list.

Six cells needed a correction that could only be found by looking at the data, such
as a model echoing a prompt instruction as a dependent clause or leaking a reasoning
trace into a block. These are not in the parser. They are listed one by one in
`data/manual_overrides.json`, each with the parsed count, the corrected count, the
reason and how the correct value was established, and they are applied by
`src/overrides.py` only when `--overrides` is passed. Results are extractable both
ways, and the results file flags every affected cell whether or not the correction
was applied.

## Reproducibility

Every call attempt is written to its own JSON file before anything is parsed,
including the model identifier, the exact request parameters, a hash of the prompt
template, the unmodified response, and the snapshot identifier the API returned.
Counts and scores are recomputed from those files, so a change to the parser can be
checked against the original responses without spending another call.

A response that fails validation is retried once with identical settings, and both
attempts are saved, so first-pass success, retry recovery and outright failure can
each be counted from the saved records. A network error is retried separately with
backoff, before any response exists to save. Nothing beyond that is retried, and no
input is skipped or approximated because it was hard to handle. Every text goes
through the same code path.

Scoring uses the texts present in both the predicted and the reference set
and reports how many that was, because a condition that fails on the hard texts is
otherwise scored on an easier subset.

Model identifiers are pinned in `config/settings.py` and are the only place they
appear.

## Layout

- `config/` model identifiers, conditions, phases, paths
- `prompts/` the two prompt templates, fixed text
- `src/prep_corpus.py` builds the corpus TSVs from the .docx sources
- `src/prompt.py` inserts a text into a template
- `src/runner.py`, `src/run_experiment.py` call the models and save the responses
- `src/parse.py` turns a response into five counts
- `src/compare.py` finds omitted and invented words and spans
- `src/overrides.py` applies the manual corrections
- `src/score.py` reads the gold standard and baselines, scores per unit type
- `src/extract_results.py` writes the result tables
- `src/nonstandard_responses.py` scans the saved responses for unusual output

## License

MIT, see [LICENSE](LICENSE).
