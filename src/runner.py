"""Call a model and save the raw response.

Every attempt is written to its own JSON file before anything is parsed, so
counts and scores can be recomputed without calling the models again. On an
invalid response the call is retried once with identical settings; both
attempts are saved, so first-pass success, retry recovery and outright failure
are all recoverable from the saved records.

All providers are called through the OpenAI SDK; DashScope and Mistral expose
OpenAI-compatible endpoints.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI

from src.parse import validate
from src.prompt import prompt_version

_NETWORK_ERRORS = (APIConnectionError, APITimeoutError, InternalServerError)
_MAX_NETWORK_RETRIES = 5
_BASE_DELAY = 10


def _build_client(model_config: dict) -> OpenAI:
    api_key = os.environ.get(model_config["api_key_env"])
    if not api_key:
        print(
            f"FATAL: environment variable {model_config['api_key_env']} not set",
            file=sys.stderr,
        )
        sys.exit(1)
    kwargs = {"api_key": api_key}
    if model_config["base_url"]:
        kwargs["base_url"] = model_config["base_url"]
    return OpenAI(**kwargs)


def output_path(
    output_dir: Path,
    condition_name: str,
    output_format: str,
    text_id: str,
    repetition: int,
    attempt: int,
) -> Path:
    return (
        output_dir
        / condition_name
        / output_format
        / str(text_id)
        / f"rep{repetition}_attempt{attempt}.json"
    )


def _field(block, name: str, default=""):
    """Read a field from an SDK content block, which may be a dict or object."""
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def _split_content(content) -> tuple[str, str | None]:
    """Separate answer text from reasoning text.

    Most providers return a plain string. Mistral reasoning responses return a
    list of typed blocks, where a thinking block holds either a string or a
    further list of blocks.
    """
    if content is None:
        return "", None
    if not isinstance(content, list):
        return content, None

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    for block in content:
        block_type = _field(block, "type", None)
        if block_type == "text":
            text_parts.append(_field(block, "text"))
        elif block_type == "thinking":
            thinking = _field(block, "thinking")
            if isinstance(thinking, list):
                thinking_parts.extend(_field(sub, "text") for sub in thinking)
            else:
                thinking_parts.append(thinking)

    thinking = "\n".join(thinking_parts) if thinking_parts else None
    return "\n".join(text_parts), thinking


def _call_model(client: OpenAI, model_config: dict, prompt: str) -> dict:
    """Make one API call and return the response with its metadata.

    No max_tokens is set; the provider default applies. A network error is
    retried with exponential backoff, and raises once the retries run out.
    """
    kwargs = {
        "model": model_config["model_id"],
        "messages": [{"role": "user", "content": prompt}],
        **model_config.get("params", {}),
    }
    if model_config.get("extra_body"):
        kwargs["extra_body"] = model_config["extra_body"]

    timestamp = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    for network_attempt in range(1, _MAX_NETWORK_RETRIES + 1):
        try:
            completion = client.chat.completions.create(**kwargs)
            break
        except _NETWORK_ERRORS as exc:
            if network_attempt == _MAX_NETWORK_RETRIES:
                raise
            delay = _BASE_DELAY * (2 ** (network_attempt - 1))
            print(
                f"  network error ({type(exc).__name__}), "
                f"retry {network_attempt}/{_MAX_NETWORK_RETRIES} in {delay}s...",
                file=sys.stderr,
            )
            time.sleep(delay)
    elapsed = round(time.monotonic() - started, 2)

    choice = completion.choices[0]
    usage = completion.usage
    raw_response, thinking = _split_content(choice.message.content)

    return {
        "raw_response": raw_response,
        "thinking": thinking,
        "finish_reason": choice.finish_reason,
        "timestamp": timestamp,
        "elapsed_seconds": elapsed,
        "usage": {
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
        },
        "system_fingerprint": getattr(completion, "system_fingerprint", None),
        # The snapshot the API actually served, which an undated alias hides.
        "resolved_model": getattr(completion, "model", None),
    }


def _save_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def run_text(
    model_config: dict,
    prompt: str,
    output_format: str,
    text_id: str,
    repetition: int,
    output_dir: Path,
    template_name: str,
) -> dict:
    """Run one (condition, format, text, repetition) cell and return its record.

    An attempt already on disk is loaded rather than called again, so a run can
    be resumed without losing or overwriting a saved response.
    """
    client = _build_client(model_config)
    condition_name = model_config["name"]

    request_params = {
        "model_id": model_config["model_id"],
        **model_config.get("params", {}),
    }
    if model_config.get("extra_body"):
        request_params["extra_body"] = model_config["extra_body"]

    for attempt in (1, 2):
        path = output_path(
            output_dir, condition_name, output_format,
            text_id, repetition, attempt,
        )

        if path.exists():
            with open(path, encoding="utf-8") as f:
                record = json.load(f)
            if record.get("valid"):
                return record
            continue

        result = _call_model(client, model_config, prompt)
        is_valid = validate(result["raw_response"], output_format)

        record = {
            "model_id": model_config["model_id"],
            "condition": condition_name,
            "output_format": output_format,
            "text_id": text_id,
            "repetition": repetition,
            "attempt": attempt,
            "prompt_version": prompt_version(template_name),
            "prompt": prompt,
            "raw_response": result["raw_response"],
            "thinking": result["thinking"],
            "valid": is_valid,
            "timestamp": result["timestamp"],
            "elapsed_seconds": result["elapsed_seconds"],
            "request_params": request_params,
            "response_metadata": {
                "finish_reason": result["finish_reason"],
                "usage": result["usage"],
                "system_fingerprint": result["system_fingerprint"],
                "resolved_model": result["resolved_model"],
            },
        }
        _save_record(path, record)

        if is_valid:
            return record
        if attempt == 1:
            print(
                f"  invalid response for {condition_name}/{output_format}/"
                f"{text_id}/rep{repetition}, retrying...",
                file=sys.stderr,
            )

    return record
