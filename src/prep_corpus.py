"""Build the corpus TSVs from the raw .docx sources.

Each learner text sits in a one-row table. The id, level and text cells are
identified by their content, because the column order is not consistent across
tables. Two files are written, both UTF-8 with columns text_id, level, text:

  texts_original.tsv  purely extracted, unmodified
  texts.tsv           encoding artefacts fixed, names replaced

texts.tsv is the only text source for the pipeline and is treated as read-only
once written. Everything else in it is verbatim, including learner spelling and
punctuation. Any structural anomaly stops the run rather than writing a partial
file.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from docx import Document

from config.settings import (
    CORPUS_ORIGINAL_PATH,
    CORPUS_PATH,
    CORPUS_SOURCE_FILES,
    NAME_MAP_PATH,
)

EXPECTED_TEXTS = 80

# The sources hold UTF-8 byte sequences that were decoded as Windows-1252. Each
# key is the mojibake as python-docx reports it.
ENCODING_MAP = {
    "â€™": "'",
    "â€¦": "…",
    "â€“": "–",
    "â‚¬": "€",
    " ": " ",  # non-breaking space
}

# Curly quotes left over after the map. Straight is the dominant form in the
# corpus (127 against 52), so everything is normalised to it.
_CURLY_QUOTES = "’‘"


def _fatal(message: str) -> None:
    print(f"FATAL: {message}", file=sys.stderr)
    sys.exit(1)


def _identify_cells(cells: list[str]) -> tuple[str, str, str]:
    """Return (id, level, text) by content: an id names a text, a level is a
    single digit, and whatever is left is the text."""
    identified: dict[str, str] = {}
    for raw in cells:
        stripped = raw.strip()
        if re.search(r"[Tt]ext\s+\d+", stripped) and len(stripped) < 50:
            kind, value = "id", stripped
        elif re.match(r"^\d$", stripped):
            kind, value = "level", stripped
        else:
            kind, value = "text", raw  # keep whitespace; stripped later
        if kind in identified:
            raise ValueError(
                f"two cells look like {kind}: "
                f"{identified[kind][:40]!r}, {value[:40]!r}"
            )
        identified[kind] = value

    missing = [k for k in ("id", "level", "text") if k not in identified]
    if missing:
        raise ValueError(f"no cell identified as {', '.join(missing)}")
    return identified["id"], identified["level"], identified["text"]


def extract_records(sources: list[Path]) -> list[tuple[int, int, str]]:
    """Return (text_id, level, text) for every row in the sources, sorted."""
    records: list[tuple[int, int, str]] = []

    for source in sources:
        if not source.exists():
            _fatal(f"source file not found: {source}")
        for table_index, table in enumerate(Document(source).tables):
            for row in table.rows:
                try:
                    id_cell, level_cell, text_cell = _identify_cells(
                        [cell.text for cell in row.cells])
                except ValueError as exc:
                    _fatal(f"{source.name} table {table_index}: {exc}")
                text_id = int(re.search(r"[Tt]ext\s+(\d+)", id_cell).group(1))
                records.append((text_id, int(level_cell), text_cell.strip()))

    records.sort()
    return records


def validate(records: list[tuple[int, int, str]]) -> None:
    """Stop on any anomaly: the corpus must be complete and well formed."""
    errors: list[str] = []

    if len(records) != EXPECTED_TEXTS:
        errors.append(f"expected {EXPECTED_TEXTS} rows, got {len(records)}")

    ids = [r[0] for r in records]
    expected = list(range(1, EXPECTED_TEXTS + 1))
    if sorted(ids) != expected:
        if missing := set(expected) - set(ids):
            errors.append(f"missing text_ids: {sorted(missing)}")
        if extra := set(ids) - set(expected):
            errors.append(f"unexpected text_ids: {sorted(extra)}")
        if duplicates := {i for i in ids if ids.count(i) > 1}:
            errors.append(f"duplicate text_ids: {sorted(duplicates)}")

    for text_id, level, text in records:
        if not 1 <= text_id <= EXPECTED_TEXTS:
            errors.append(f"text_id out of range: {text_id}")
        if level < 1:
            errors.append(f"text {text_id}: level out of range: {level}")
        if not text:
            errors.append(f"text {text_id}: empty text")
        if "\t" in text:
            errors.append(f"text {text_id}: contains a tab")
        if "\n" in text:
            errors.append(f"text {text_id}: contains a newline")

    if errors:
        print("FATAL: validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)


def _apply_encoding_map(text: str) -> str:
    for mojibake, intended in ENCODING_MAP.items():
        text = text.replace(mojibake, intended)
    for quote in _CURLY_QUOTES:
        text = text.replace(quote, "'")
    return text


def _apply_name_map(text: str, name_map: dict[str, str]) -> str:
    for name in sorted(name_map, key=len, reverse=True):
        text = text.replace(name, name_map[name])
    return text


def _load_name_map() -> dict[str, str]:
    if not NAME_MAP_PATH.exists():
        return {}
    with open(NAME_MAP_PATH, encoding="utf-8") as f:
        return json.load(f)


def _write_tsv(path: Path, records: list[tuple[int, int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("text_id\tlevel\ttext\n")
        for text_id, level, text in records:
            f.write(f"{text_id}\t{level}\t{text}\n")
    print(f"Wrote {len(records)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the corpus TSVs from the raw .docx sources."
    )
    parser.add_argument(
        "--sources", type=Path, nargs="+", default=CORPUS_SOURCE_FILES,
        help="raw .docx files (default: CORPUS_SOURCE_FILES in config)",
    )
    args = parser.parse_args()

    records = extract_records(args.sources)
    validate(records)
    _write_tsv(CORPUS_ORIGINAL_PATH, records)

    name_map = _load_name_map()
    if not name_map:
        print(
            f"WARNING: {NAME_MAP_PATH} not found or empty; "
            "names will not be replaced.",
            file=sys.stderr,
        )

    _write_tsv(CORPUS_PATH, [
        (text_id, level, _apply_name_map(_apply_encoding_map(text), name_map))
        for text_id, level, text in records
    ])


if __name__ == "__main__":
    main()
