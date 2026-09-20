"""Turn a raw model response into five unit counts.

The lists format is split into blocks by header and the items per block are
counted; the counts format is read as five integers. Validation checks
structure only: whether the blocks are there, not whether the content is right.

Formatting tolerance here is limited to variation any model could produce given
the prompt instructions, and is applied to every response. Corrections that
needed data inspection live in src/overrides.py instead.
"""

import re

UNIT_KEYS = ["words", "sentences", "t_units", "clauses", "dependent_clauses"]

HEADERS = {
    "words": "WORDS:",
    "sentences": "SENTENCES:",
    "t_units": "T-UNITS:",
    "clauses": "CLAUSES:",
    "dependent_clauses": "DEPENDENT CLAUSES:",
}

# DEPENDENT CLAUSES must be located before CLAUSES, which is a substring of it.
HEADER_ORDER = ["words", "sentences", "t_units", "dependent_clauses", "clauses"]

# Lines models write in the DC block when a text has no dependent clauses.
# The prompt asks for "none", so these appear alongside real clauses too.
# No learner text has a DC starting with any of them.
NONE_VALUES = {"none", "n/a", "-", "0"}

_NUMBERED_PREFIX_RE = re.compile(r"^\s*\d+[.)]\s+")
_BULLETED_PREFIX_RE = re.compile(r"^\s*[-*]\s+")

# Comma as word separator, except inside numeric tokens: thousands separators
# ("10,000") and price notation ("€1000,-").
_WORDS_COMMA_RE = re.compile(r"(?<!\d),|,(?![\d\-])")

_COUNTS_RE = {
    key: re.compile(rf"^{re.escape(header)}\s*(\d+)", re.MULTILINE)
    for key, header in HEADERS.items()
}


def _strip_uniform_prefixes(lines: list[str]) -> list[str]:
    """Strip numbered or bulleted prefixes, but only if every line has one.

    Partial prefixing is left alone: a lone leading "- " may be content.
    """
    if len(lines) < 2:
        return lines
    for pattern in (_NUMBERED_PREFIX_RE, _BULLETED_PREFIX_RE):
        if all(pattern.match(ln) for ln in lines):
            return [pattern.sub("", ln, count=1).strip() for ln in lines]
    return lines


def split_blocks(text: str) -> dict[str, str] | None:
    """Split a lists response into {unit key: text between its header and the
    next}. Returns None if any header is missing."""
    if not text:
        return None

    found: dict[str, int] = {}
    for key in HEADER_ORDER:
        idx = text.find(HEADERS[key])
        if idx == -1:
            return None
        if key == "clauses" and "dependent_clauses" in found:
            # The match may have fallen inside the DEPENDENT CLAUSES header.
            dc_start = found["dependent_clauses"]
            dc_end = dc_start + len(HEADERS["dependent_clauses"])
            if dc_start <= idx < dc_end:
                idx = text.find(HEADERS[key], dc_end)
                if idx == -1:
                    return None
        found[key] = idx

    positions = sorted((idx, key) for key, idx in found.items())
    blocks: dict[str, str] = {}
    for i, (pos, key) in enumerate(positions):
        start = pos + len(HEADERS[key])
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        blocks[key] = text[start:end].strip()
    return blocks


def split_words(block: str) -> list[str]:
    """Split the WORDS block on commas, keeping numeric commas intact."""
    return [w.strip() for w in _WORDS_COMMA_RE.split(block) if w.strip()]


def split_lines(block: str) -> list[str]:
    """Split a one-item-per-line block, ignoring blank lines."""
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    return _strip_uniform_prefixes(lines)


def parse_dc_lines(block: str) -> list[str]:
    """Split the DEPENDENT CLAUSES block, dropping bare none-equivalents."""
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    kept = [ln for ln in lines if ln.lower().rstrip("., ") not in NONE_VALUES]
    return _strip_uniform_prefixes(kept)


def extract_items(response: str) -> dict[str, list[str]] | None:
    """Return {unit key: items} from a lists response, or None if unparseable."""
    blocks = split_blocks(response)
    if blocks is None:
        return None
    items = {"words": split_words(blocks["words"])}
    for key in ["sentences", "t_units", "clauses"]:
        items[key] = split_lines(blocks[key])
    items["dependent_clauses"] = parse_dc_lines(blocks["dependent_clauses"])
    return items


def parse_lists(response: str) -> dict[str, int] | None:
    items = extract_items(response)
    if items is None:
        return None
    return {key: len(values) for key, values in items.items()}


def parse_counts(response: str) -> dict[str, int] | None:
    counts: dict[str, int] = {}
    for key, pattern in _COUNTS_RE.items():
        match = pattern.search(response)
        if match is None:
            return None
        counts[key] = int(match.group(1))
    return counts


def parse(response: str, output_format: str) -> dict[str, int] | None:
    """Parse a response in the given format. Returns None if it cannot be
    parsed."""
    if not response:
        return None
    if output_format == "lists":
        return parse_lists(response)
    if output_format == "counts":
        return parse_counts(response)
    raise ValueError(f"unknown output format: {output_format!r}")


def validate(response: str, output_format: str) -> bool:
    """Check that all five blocks are present and non-empty. Dependent clauses
    may legitimately be 0."""
    counts = parse(response, output_format)
    if counts is None:
        return False
    if any(counts[key] < 1 for key in UNIT_KEYS if key != "dependent_clauses"):
        return False
    return counts["dependent_clauses"] >= 0
