"""Run the annotation experiment over the grid of condition, format, text and
repetition.

A cell that already has a valid saved response is skipped, so a run can be
stopped and resumed. Phase defaults come from config/settings.py and every one
of them can be overridden on the command line.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from config.settings import (
    CORPUS_PATH,
    MODELS,
    OUTPUT_FORMATS,
    OUTPUTS_DIR,
    PHASES,
    REPETITIONS,
)
from src.prompt import build_prompt
from src.runner import output_path, run_text


def _fatal(message: str) -> None:
    print(f"FATAL: {message}", file=sys.stderr)
    sys.exit(1)


def load_corpus(path: Path) -> list[dict]:
    """Read a corpus TSV as [{id, text}]. The id column is text_id or, for the
    pilot, pilot_id."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    if not rows:
        _fatal(f"empty corpus file: {path}")

    id_column = next((c for c in ("text_id", "pilot_id") if c in rows[0]), None)
    if id_column is None:
        _fatal(f"no text_id or pilot_id column in {path}")

    return [{"id": row[id_column], "text": row["text"]} for row in rows]


def _already_done(
    output_dir: Path,
    condition_name: str,
    output_format: str,
    text_id: str,
    repetition: int,
) -> bool:
    """Whether either attempt for this cell is saved and valid."""
    for attempt in (1, 2):
        path = output_path(
            output_dir, condition_name, output_format,
            text_id, repetition, attempt,
        )
        if path.exists():
            with open(path, encoding="utf-8") as f:
                if json.load(f).get("valid"):
                    return True
    return False


def _resolve_models(names: list[str]) -> list[dict]:
    if names == ["all"]:
        return MODELS

    by_name = {m["name"]: m for m in MODELS}
    for name in names:
        if name not in by_name:
            _fatal(
                f"unknown model condition {name!r}. "
                f"Available: {', '.join(by_name)}"
            )
    return [by_name[name] for name in names]


def _select_range(texts: list[dict], text_range: str) -> list[dict]:
    low, high = (int(part) for part in text_range.split("-"))
    selected = [t for t in texts if low <= int(t["id"]) <= high]
    if not selected:
        _fatal(f"no texts in range {low}-{high}")
    return selected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LLM annotation experiment."
    )
    parser.add_argument(
        "--phase", type=str, choices=list(PHASES),
        help="experiment phase (sets corpus, models, reps, output dir)",
    )
    parser.add_argument(
        "--corpus", type=Path, default=None,
        help="path to corpus TSV (overrides phase default)",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        help="model condition names, or 'all' (overrides phase default)",
    )
    parser.add_argument(
        "--formats", type=str, nargs="+", default=OUTPUT_FORMATS,
        choices=OUTPUT_FORMATS,
        help="output formats to run (default: all)",
    )
    parser.add_argument(
        "--reps", type=int, default=None,
        help="repetitions per cell (overrides phase default)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="output directory (overrides phase default)",
    )
    parser.add_argument(
        "--text-range", type=str, default=None,
        help="run only text_ids in this range, e.g. '1-40'",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print what would run without calling APIs",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    phase_config = PHASES.get(args.phase, {}) if args.phase else {}

    corpus_path = args.corpus or phase_config.get("corpus", CORPUS_PATH)
    reps = args.reps if args.reps is not None else phase_config.get("reps", REPETITIONS)
    output_dir = args.output_dir or (
        OUTPUTS_DIR / args.phase if args.phase else OUTPUTS_DIR
    )

    models = _resolve_models(args.models or phase_config.get("models", ["all"]))
    texts = load_corpus(corpus_path)
    if args.text_range:
        texts = _select_range(texts, args.text_range)

    total = len(models) * len(args.formats) * len(texts) * reps
    done = skipped = failures = 0

    phase_label = f" (phase: {args.phase})" if args.phase else ""
    print(f"Output dir: {output_dir}{phase_label}")
    print(f"Models: {', '.join(m['name'] for m in models)}")
    print(f"Formats: {', '.join(args.formats)}, reps: {reps}, texts: {len(texts)}")
    print(f"Total cells: {total}\n")

    for model_config in models:
        condition = model_config["name"]
        for output_format in args.formats:
            for text in texts:
                for rep in range(1, reps + 1):
                    done += 1

                    if _already_done(
                        output_dir, condition, output_format, text["id"], rep
                    ):
                        skipped += 1
                        continue

                    label = (f"[{done}/{total}] {condition}/{output_format}/"
                             f"{text['id']}/rep{rep}")

                    if args.dry_run:
                        print(f"  DRY RUN: {label}")
                        continue

                    print(f"  {label} ...", end=" ", flush=True)
                    record = run_text(
                        model_config,
                        build_prompt(output_format, text["text"]),
                        output_format, text["id"], rep, output_dir,
                        output_format,
                    )

                    if record["valid"]:
                        print(f"ok (attempt {record['attempt']})")
                    else:
                        failures += 1
                        print("FAILED")

    print(f"\nDone. {done} cells, {skipped} skipped, {failures} failures.")


if __name__ == "__main__":
    main()
