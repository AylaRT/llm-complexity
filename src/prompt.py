"""Build a prompt by inserting a text into a fixed template."""

import hashlib

from config.settings import PROMPTS_DIR

PLACEHOLDER = "[TEXT]"


def _template_path(template_name: str):
    return PROMPTS_DIR / f"msd_{template_name}.txt"


def build_prompt(template_name: str, text: str) -> str:
    """Return the template with the text placeholder substituted."""
    path = _template_path(template_name)
    if not path.exists():
        raise FileNotFoundError(f"prompt template not found: {path}")
    template = path.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise ValueError(f"placeholder {PLACEHOLDER!r} not found in {path}")
    return template.replace(PLACEHOLDER, text)


def prompt_version(template_name: str) -> str:
    """Return the template name and a hash of its contents."""
    content = _template_path(template_name).read_text(encoding="utf-8")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return f"msd_{template_name}:{digest}"
