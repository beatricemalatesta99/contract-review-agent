"""
Thin wrapper around the Anthropic API.

Why have this layer at all?
    1. Retries on transient errors (so the pipeline doesn't die after 50
       contracts because of one timeout).
    2. Enforces JSON-only output (we parse JSON downstream).
    3. Logs all calls to disk — useful both for debugging and for the
       paper's reproducibility ("all LLM responses are saved in /results").
    4. Centralises model + temperature, so ablations are honest.
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv

import config

# Load the API key from .env into the environment.
load_dotenv()

# The SDK picks up ANTHROPIC_API_KEY automatically.
_client = Anthropic()

# Directory for raw LLM responses (one file per call, for traceability).
LOG_DIR = config.RESULTS_DIR / "llm_calls"
LOG_DIR.mkdir(exist_ok=True)


def call_llm(
    prompt: str,
    system: Optional[str] = None,
    tag: str = "generic",
) -> str:
    """
    Send a prompt to Claude and return the text response.

    Args:
        prompt: the user message.
        system: optional system prompt (role / persona).
        tag: short string used in the log filename so you can find calls
             by stage ("extractor", "classifier", etc).

    Returns:
        The model's text output, stripped of any surrounding whitespace.
    """
    messages = [{"role": "user", "content": prompt}]

    # Retry loop — the SDK raises on network errors, rate limits, etc.
    last_error = None
    for attempt in range(config.MAX_RETRIES):
        try:
            kwargs = {
                "model": config.MODEL,
                "max_tokens": config.MAX_OUTPUT_TOKENS,
                "temperature": config.TEMPERATURE,
                "messages": messages,
            }
            if system is not None:
                kwargs["system"] = system

            resp = _client.messages.create(**kwargs)
            text = resp.content[0].text.strip()

            # Log the raw response for later inspection.
            _log_call(tag, prompt, text)
            return text

        except Exception as e:
            last_error = e
            print(f"  [call_llm] attempt {attempt+1} failed: {e}")
            time.sleep(config.RETRY_SLEEP_SECONDS)

    # All retries failed.
    raise RuntimeError(
        f"LLM call failed after {config.MAX_RETRIES} retries: {last_error}"
    )


def call_llm_json(
    prompt: str,
    system: Optional[str] = None,
    tag: str = "generic",
) -> object:
    """
    Same as call_llm, but expects and parses a JSON response.

    LLMs sometimes wrap JSON in ```json ... ``` fences or add
    preamble/postamble text. We extract the first valid JSON object
    or array and ignore everything else.

    Returns:
        Parsed Python object (dict or list).

    Raises:
        ValueError if no valid JSON can be found.
    """
    raw = call_llm(prompt, system=system, tag=tag)
    cleaned = _strip_json_fences(raw)

    # Try strict parsing first (fastest, most common case).
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: find the first balanced JSON array or object in the
    # raw text. This handles cases where the model adds prose before
    # or after the JSON.
    parsed = _extract_first_json(raw)
    if parsed is not None:
        return parsed

    # If we get here, nothing worked. Save the bad output for
    # debugging and raise a clear error.
    bad_path = LOG_DIR / f"bad_json_{tag}_{int(time.time())}.txt"
    bad_path.write_text(raw, encoding="utf-8")
    raise ValueError(
        f"Could not parse JSON from {tag}. "
        f"Raw output saved to {bad_path}."
    )


def _extract_first_json(text: str) -> object:
    """
    Scan `text` and return the first balanced JSON array or object
    that parses successfully. Returns None if nothing is found.
    """
    # Find the earliest opening bracket.
    for i, ch in enumerate(text):
        if ch not in "[{":
            continue
        closing = "]" if ch == "[" else "}"
        # Walk forward, tracking nesting depth and string state.
        depth = 0
        in_string = False
        escape = False
        for j in range(i, len(text)):
            c = text[j]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c in "[{":
                depth += 1
            elif c in "]}":
                depth -= 1
                if depth == 0:
                    # Found a candidate; try to parse it.
                    candidate = text[i : j + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break  # try next opening bracket
    return None


def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences around JSON if present."""
    text = text.strip()
    if text.startswith("```"):
        # Drop first line (```json or ```) and last line (```)
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _log_call(tag: str, prompt: str, response: str) -> None:
    """Save one LLM call to disk for later inspection."""
    # Use a hash of the prompt so repeated identical calls overwrite
    # each other (keeps the directory small).
    h = hashlib.md5(prompt.encode()).hexdigest()[:10]
    path = LOG_DIR / f"{tag}_{h}.json"
    path.write_text(json.dumps({
        "tag": tag,
        "prompt": prompt,
        "response": response,
    }, indent=2))
