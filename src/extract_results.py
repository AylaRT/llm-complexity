"""Turn the saved model responses into result tables.

Reads the JSON records under outputs/ and writes:

  {batch}_results.tsv   one row per (condition, format, text, repetition) cell,
                        with the five counts, the gold and baseline counts, and
                        for the lists format the omission and hallucination
                        counts against the source text
  {batch}_lists.xlsx    one row per unit per lists-format cell, with the items
                        themselves and the differences flagged

Nothing here calls a model: it reads only what is already on disk.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from config.settings import GOLD_PATH, MODELS, OUTPUTS_DIR, PHASES, PRICING, RESULTS_DIR
from src.compare import compare_spans, compare_words
from src.overrides import OVERRIDES_PATH, apply_override, cell_key, load_overrides
from src.parse import UNIT_KEYS, extract_items, parse
from src.run_experiment import load_corpus
from src.score import load_baselines, load_gold, load_levels

_GOLD_COLUMNS = [f"gold_{unit}" for unit in UNIT_KEYS]
_BASELINE_COLUMNS = [f"{source}_{unit}"
                     for source in ("taassc", "l2sca") for unit in UNIT_KEYS]

# Prefix used in the comparison columns for each unit.
_SPAN_UNITS = {"sentences": "sent", "t_units": "tunit",
               "clauses": "clause", "dependent_clauses": "dc"}

# Omissions are only meaningful where the units are meant to cover the whole
# text, so they are reported for words and sentences alone.
_COMPARISON_COLUMNS = [
    "words_omit", "words_omit_detail",
    "words_halluc", "words_halluc_detail",
    "sent_omit", "sent_omit_detail",
    "sent_halluc", "sent_halluc_detail",
    "tunit_halluc", "tunit_halluc_detail",
    "clause_halluc", "clause_halluc_detail",
    "dc_halluc", "dc_halluc_detail",
]

COLUMNS = [
    "text_id", "level", "phase", "model", "model_snapshot", "model_size",
    "reasoning", "condition", "format", "rep", "attempt", "valid",
    *UNIT_KEYS,
    "flag", "flag_units",
    *_GOLD_COLUMNS,
    *_BASELINE_COLUMNS,
    *_COMPARISON_COLUMNS,
    "elapsed_seconds", "prompt_tokens", "completion_tokens",
    "estimated_cost_usd", "timestamp", "prompt_version",
]

ITEMS_COLUMNS = [
    "text_id", "phase", "model", "model_snapshot", "model_size", "reasoning",
    "condition", "rep", "unit", "count", "items",
    "omitted", "hallucinated", "omitted_items", "hallucinated_items",
]

# Item and difference columns in the xlsx, which hold one item per line.
_WRAP_COLUMNS = ["K", "N", "O"]

_PROVIDER_LABELS = {"openai": "gpt", "dashscope": "qwen", "mistral": "mistral"}

# Phase 2 models are not all the same tier, but each is the larger model of its
# provider, so "large" is used as the relative label.
_PHASE_SIZES = {"pilot": "small", "phase1": "small", "phase2": "large"}

# A response that echoes the whole source text as one item rather than
# segmenting it. Only checked on texts long enough for this to be unambiguous.
_MIN_TEXT_LENGTH = 100
_NO_SEGMENTATION_SHARE = 0.7


def condition_meta() -> dict[str, dict[str, str]]:
    """Decompose each condition name into the metadata columns."""
    meta = {
        m["name"]: {
            "model": _PROVIDER_LABELS.get(m["provider"], m["provider"]),
            "model_snapshot": m["model_id"],
            "reasoning": "on" if m["name"].endswith("_reasoning-on") else "off",
            "model_size": "",
            "phase": "",
        }
        for m in MODELS
    }
    for phase_name, phase_config in PHASES.items():
        for name in phase_config.get("models", []):
            if name in meta:
                meta[name]["model_size"] = _PHASE_SIZES.get(phase_name, "")
                meta[name]["phase"] = phase_name
    return meta


def _estimate_cost(record: dict) -> str:
    """USD for one call, from its token counts. Blank if either is unknown."""
    rates = PRICING.get(record.get("model_id", ""))
    if rates is None:
        return ""
    usage = record.get("response_metadata", {}).get("usage", {})
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens is None or completion_tokens is None:
        return ""
    cost = (prompt_tokens * rates["input"]
            + completion_tokens * rates["output"]) / 1_000_000
    return f"{cost:.6f}"


def _format_span_halluc(items: list[dict]) -> list[str]:
    """Render span hallucinations. A span whose words are all in the source has
    a boundary problem, so it is shown on its own; otherwise the new words are
    named first."""
    return [
        f"{', '.join(item['diff'])} (in: {item['span']})" if item["diff"]
        else item["span"]
        for item in items
    ]


def _detect_no_segmentation(items: dict[str, list[str]],
                            source_text: str) -> list[str]:
    """Units where the model returned the source text as a single item."""
    if len(source_text) <= _MIN_TEXT_LENGTH:
        return []
    return [
        unit for unit in ["words", "sentences", "t_units", "clauses"]
        if len(items[unit]) == 1
        and len(items[unit][0]) > _NO_SEGMENTATION_SHARE * len(source_text)
    ]


def _add_comparisons(row: dict, items: dict[str, list[str]],
                     source_text: str) -> None:
    """Fill the omission and hallucination columns for a lists-format row."""
    words = compare_words(source_text, items["words"])
    row["words_omit"] = words["omitted_count"]
    row["words_omit_detail"] = ", ".join(words["omitted_items"])
    row["words_halluc"] = words["hallucinated_count"]
    row["words_halluc_detail"] = ", ".join(words["hallucinated_items"])

    for unit, prefix in _SPAN_UNITS.items():
        spans = compare_spans(source_text, items[unit])
        row[f"{prefix}_halluc"] = spans["hallucinated_count"]
        row[f"{prefix}_halluc_detail"] = " | ".join(
            _format_span_halluc(spans["hallucinated_items"]))
        if unit == "sentences":
            row["sent_omit"] = spans["omitted_count"]
            row["sent_omit_detail"] = ", ".join(spans["omitted_items"])


def extract_row(record: dict, source_text: str | None,
                meta: dict) -> dict:
    """Flatten one saved record into a result row."""
    usage = record.get("response_metadata", {}).get("usage", {})
    row = {
        "text_id": record["text_id"],
        "level": "",
        "phase": record.get("_phase", meta.get("phase", "")),
        "model": meta.get("model", ""),
        "model_snapshot": meta.get("model_snapshot", ""),
        "model_size": meta.get("model_size", ""),
        "reasoning": meta.get("reasoning", ""),
        "condition": record["condition"],
        "format": record["output_format"],
        "rep": record["repetition"],
        "attempt": record["attempt"],
        "valid": record["valid"],
        "flag": "",
        "flag_units": "",
        "elapsed_seconds": record.get("elapsed_seconds", ""),
        "prompt_tokens": usage.get("prompt_tokens", ""),
        "completion_tokens": usage.get("completion_tokens", ""),
        "estimated_cost_usd": _estimate_cost(record),
        "timestamp": record["timestamp"],
        "prompt_version": record.get("prompt_version", ""),
    }
    for column in _BASELINE_COLUMNS + _COMPARISON_COLUMNS:
        row[column] = ""

    counts = (parse(record["raw_response"], record["output_format"])
              if record["valid"] else None)
    for unit in UNIT_KEYS:
        row[unit] = counts[unit] if counts else ""

    if (record["valid"] and record["output_format"] == "lists"
            and source_text is not None):
        items = extract_items(record["raw_response"])
        if items is not None:
            _add_comparisons(row, items, source_text)
            unsegmented = _detect_no_segmentation(items, source_text)
            if unsegmented:
                row["flag"] = "no_segmentation"
                row["flag_units"] = ",".join(unsegmented)

    return row


def extract_unit_rows(record: dict, source_text: str, meta: dict) -> list[dict]:
    """One row per unit for a lists-format record, with the items themselves.

    Words go on one line, comma-separated; the other units use real newlines so
    each item gets its own line in the Excel cell.
    """
    if not record["valid"] or record["output_format"] != "lists":
        return []
    items = extract_items(record["raw_response"])
    if items is None:
        return []

    rows = []
    for unit in UNIT_KEYS:
        unit_items = items[unit]

        if unit == "words":
            difference = compare_words(source_text, unit_items)
            joined = ", ".join(unit_items)
            hallucinated = ", ".join(difference["hallucinated_items"])
        else:
            difference = compare_spans(source_text, unit_items)
            joined = "\n".join(unit_items)
            hallucinated = "\n".join(
                _format_span_halluc(difference["hallucinated_items"]))
            if unit != "sentences":
                difference["omitted_count"] = 0
                difference["omitted_items"] = []

        rows.append({
            "text_id": record["text_id"],
            "phase": meta.get("phase", ""),
            "model": meta.get("model", ""),
            "model_snapshot": meta.get("model_snapshot", ""),
            "model_size": meta.get("model_size", ""),
            "reasoning": meta.get("reasoning", ""),
            "condition": record["condition"],
            "rep": record["repetition"],
            "unit": unit,
            "count": len(unit_items),
            "items": joined or "none",
            "omitted": difference["omitted_count"],
            "hallucinated": difference["hallucinated_count"],
            "omitted_items": ", ".join(difference["omitted_items"]),
            "hallucinated_items": hallucinated,
        })
    return rows


def collect_records(output_dir: Path) -> list[dict]:
    """Load every saved response under a directory."""
    records = []
    for json_path in sorted(output_dir.rglob("*.json")):
        with open(json_path, encoding="utf-8") as f:
            records.append(json.load(f))
    return records


def group_by_cell(records: list[dict]) -> dict[tuple, list[dict]]:
    """Group records by cell, each group ordered by attempt."""
    groups: dict[tuple, list[dict]] = {}
    for record in records:
        key = (record["condition"], record["output_format"],
               record["text_id"], record["repetition"])
        groups.setdefault(key, []).append(record)
    for group in groups.values():
        group.sort(key=lambda r: r["attempt"])
    return groups


def best_attempt(records: list[dict]) -> dict:
    """The valid attempt for a cell, or the last one if none was valid."""
    return next((r for r in records if r.get("valid")), records[-1])


def _text_number(text_id: str) -> int:
    """Sortable number for a text id. Pilot ids are prefixed with P."""
    return int(text_id[1:]) if text_id.startswith("P") else int(text_id)


def write_tsv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def write_items_xlsx(rows: list[dict], path: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(ITEMS_COLUMNS)
    for row in rows:
        sheet.append([row[column] for column in ITEMS_COLUMNS])

    wrap = Alignment(wrap_text=True, vertical="top")
    for letter in _WRAP_COLUMNS:
        for cell in sheet[letter]:
            cell.alignment = wrap

    workbook.save(path)
    print(f"Wrote {len(rows)} rows to {path}")


def _flag_overrides(rows: list[dict],
                    overrides: dict[tuple, dict[str, int]]) -> None:
    """Mark the rows a manual correction exists for, whether or not it is
    applied, so a corrected cell is visible in either extraction."""
    for row in rows:
        corrections = overrides.get(cell_key(row))
        if corrections is None:
            continue
        existing = row["flag_units"].split(",") if row["flag_units"] else []
        row["flag"] = f"{row['flag']},override" if row["flag"] else "override"
        row["flag_units"] = ",".join(
            dict.fromkeys(existing + list(corrections)))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract experiment results into TSV and xlsx."
    )
    parser.add_argument(
        "--phase", type=str, choices=[*PHASES, "all"],
        help="experiment phase, or 'all' for phase1 and phase2 combined",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="directory of saved JSON outputs (overrides phase default)",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help="directory for the result files (default: results/)",
    )
    parser.add_argument(
        "--batch-name", type=str, default=None,
        help="name for this batch (overrides phase default)",
    )
    parser.add_argument(
        "--per-condition", action="store_true",
        help="write one TSV per condition instead of one combined file",
    )
    parser.add_argument(
        "--all-attempts", action="store_true",
        help="include every attempt (default: the best attempt per cell)",
    )
    parser.add_argument(
        "--overrides", type=Path, default=None,
        help="path to manual_overrides.json, applying the manual corrections",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.phase == "all":
        phases = [p for p in PHASES if p != "pilot"]
    elif args.phase:
        phases = [args.phase]
    else:
        phases = []

    if args.output_dir:
        records = collect_records(args.output_dir)
    elif phases:
        records = []
        for phase in phases:
            for record in collect_records(OUTPUTS_DIR / phase):
                record["_phase"] = phase
                records.append(record)
    else:
        records = collect_records(OUTPUTS_DIR)

    if not records:
        print("No JSON files found", file=sys.stderr)
        sys.exit(1)

    # All phases currently share one corpus, so the first phase's is the
    # source text for every record. Extend this if that stops being true.
    corpus: dict[str, str] = {}
    if phases:
        corpus_path = PHASES[phases[0]]["corpus"]
        corpus = {t["id"]: t["text"] for t in load_corpus(corpus_path)}

    meta = condition_meta()
    gold = load_gold(GOLD_PATH) if GOLD_PATH.exists() else {}
    levels = load_levels(GOLD_PATH) if GOLD_PATH.exists() else {}
    baselines = load_baselines(GOLD_PATH) if GOLD_PATH.exists() else {}

    groups = group_by_cell(records)

    rows = []
    for group in groups.values():
        chosen = group if args.all_attempts else [best_attempt(group)]
        for record in chosen:
            rows.append(extract_row(
                record,
                corpus.get(record["text_id"]),
                meta.get(record["condition"], {}),
            ))

    for row in rows:
        text_id = row["text_id"]
        number = None if text_id.startswith("P") else int(text_id)

        row["level"] = str(levels.get(number, "")) if number is not None else ""
        gold_counts = gold.get(number, {})
        for unit, column in zip(UNIT_KEYS, _GOLD_COLUMNS):
            row[column] = gold_counts.get(unit, "")
        for source in ("taassc", "l2sca"):
            counts = baselines.get(source, {}).get(number, {})
            for unit in UNIT_KEYS:
                row[f"{source}_{unit}"] = counts.get(unit, "")

    overrides_path = args.overrides or OVERRIDES_PATH
    overrides = load_overrides(overrides_path)
    _flag_overrides(rows, overrides)
    if args.overrides:
        applied = sum(apply_override(row, overrides) for row in rows)
        print(f"Applied {applied} manual overrides from {args.overrides}")

    rows.sort(key=lambda r: (
        r["phase"], r["model"], r["model_size"], r["reasoning"],
        r["format"], _text_number(r["text_id"]), r["rep"],
    ))

    batch_name = args.batch_name or args.phase or "all"
    if args.per_condition:
        by_condition: dict[str, list[dict]] = {}
        for row in rows:
            by_condition.setdefault(row["condition"], []).append(row)
        for condition, condition_rows in sorted(by_condition.items()):
            write_tsv(condition_rows,
                      args.results_dir / f"{batch_name}_{condition}.tsv")
    else:
        write_tsv(rows, args.results_dir / f"{batch_name}_results.tsv")

    if corpus:
        item_rows = []
        for group in groups.values():
            record = best_attempt(group)
            source_text = corpus.get(record["text_id"])
            if source_text is None:
                continue
            item_rows.extend(extract_unit_rows(
                record, source_text, meta.get(record["condition"], {})))
        item_rows.sort(key=lambda r: (
            r["phase"], r["condition"], _text_number(r["text_id"]),
            r["rep"], UNIT_KEYS.index(r["unit"]),
        ))
        write_items_xlsx(item_rows, args.results_dir / f"{batch_name}_lists.xlsx")

    _print_summary(rows)


def _print_summary(rows: list[dict]) -> None:
    valid = sum(1 for row in rows if row["valid"])
    print(f"\nSummary: {len(rows)} cells, {valid} valid, {len(rows) - valid} failed.")

    times = [float(row["elapsed_seconds"]) for row in rows
             if row.get("elapsed_seconds")]
    if times:
        print(f"Elapsed: min={min(times):.1f}s, max={max(times):.1f}s, "
              f"mean={sum(times) / len(times):.1f}s")

    costs = [float(row["estimated_cost_usd"]) for row in rows
             if row.get("estimated_cost_usd")]
    if costs:
        print(f"Estimated cost: ${sum(costs):.4f} total, "
              f"${sum(costs) / len(costs):.6f} mean per call")


if __name__ == "__main__":
    main()
