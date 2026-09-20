"""Apply the manual count corrections listed in data/manual_overrides.json.

These are the corrections that needed data inspection: cells where a model
behaved in a way the format-agnostic parsing rules cannot handle. Each entry in
the file records the cell, the parsed count, the corrected count, the reason
and how the correct value was established. Applying them is a separate step
from parsing, so results can be produced with and without.
"""

import json
from pathlib import Path

from config.settings import DATA_DIR

OVERRIDES_PATH = DATA_DIR / "manual_overrides.json"


def cell_key(entry: dict) -> tuple:
    """Identify one (phase, condition, format, text, rep, attempt) cell.

    Indexes rather than gets: an entry missing a field would otherwise match
    nothing and be dropped without a word.
    """
    return (
        entry["phase"],
        entry["condition"],
        entry["format"],
        str(entry["text_id"]),
        int(entry["rep"]),
        int(entry["attempt"]),
    )


def load_overrides(path: Path) -> dict[tuple, dict[str, int]]:
    """Return {cell key: {unit: corrected count}} from the overrides file."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        cell_key(entry): entry["overrides"]
        for entry in data.get("corrections", [])
    }


def apply_override(row: dict, overrides: dict[tuple, dict[str, int]]) -> bool:
    """Correct a result row in place. Returns whether an override applied."""
    corrections = overrides.get(cell_key(row))
    if corrections is None:
        return False
    row.update(corrections)
    return True
