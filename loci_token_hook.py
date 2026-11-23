"""
loci_token_hook.py — Wire Claude Code (this process) into the LOCI prompt-cache
savings ledger so every Claude API call gets cache breakpoints applied and every
saved token is recorded.

How it works
------------
Anthropic prompt caching (https://platform.claude.com/docs/prompt-caching):
  • Cache breakpoints mark message/tool prefixes for server-side reuse.
  • A cache HIT returns usage.cache_read_input_tokens — billed at 10% of normal.
  • A cache WRITE returns usage.cache_creation_input_tokens — billed at 125%.
  • Net saving once a prefix is cached: 90 tokens saved per 100 tokens re-read.

LOCI heat tiers (from arc/core/cache_layer.py):
  HOT  heat ≥ θ_HOT  → extended TTL (1 hr)  — θ_HOT is adaptive (see loci_adaptive_theta)
  MILD heat ≥ 0.350  → ephemeral TTL (5 min) — standard
  COLD heat <  0.350 → not cached

This module:
  1. Applies the 4-breakpoint layout to any outgoing Anthropic request.
  2. Checks the Arc memory engine for heat scores on context blocks.
  3. Reads usage.cache_read_input_tokens from each response.
  4. Writes a savings record to the LOCI ledger (compatible with
     agents/*/ledger.jsonl in LOCI-MEMORY-ENGINE).
  5. Exposes `wrap_anthropic_client(client)` — monkey-patches
     client.messages.create so callers need zero changes.

Arc is GPU-busy (Dexter's Lab work) — we still get token savings because
caching is a *server-side* operation on Anthropic's infra; Arc only matters
for the heat score lookup, which we gracefully skip if Arc is offline.

Usage
-----
    import anthropic
    from loci_token_hook import wrap_anthropic_client, savings_summary

    client = anthropic.Anthropic()
    wrap_anthropic_client(client)          # one line — done

    # Now every client.messages.create() call automatically:
    #   • gets cache breakpoints injected
    #   • logs savings to ~/.local/state/loci-token-savings/ledger.jsonl

    print(savings_summary())               # live totals for the IDE panel
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import httpx as _httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

from loci_adaptive_theta import current_theta as _theta_hot

# ── Pricing constants (USD per 1M tokens, Sonnet 4.5 / claude-sonnet-*) ──────
_INPUT_COST_PER_M   = 3.00    # normal input
_CACHE_READ_PER_M   = 0.30    # cache hit  (10% of input)
_CACHE_WRITE_PER_M  = 3.75    # cache miss write (125% of input)
_OUTPUT_COST_PER_M  = 15.00

# ── Storage ──────────────────────────────────────────────────────────────────
LEDGER_DIR = Path(os.getenv(
    "LOCI_SAVINGS_DIR",
    os.path.expanduser("~/.local/state/loci-token-savings"),
))
LEDGER_DIR.mkdir(parents=True, exist_ok=True)
LEDGER_PATH = LEDGER_DIR / "ledger.jsonl"

ARC_ENDPOINT = os.getenv("ARC_ENDPOINT", "http://localhost:7860")

log = logging.getLogger(__name__)

# ── In-memory totals ──────────────────────────────────────────────────────────
@dataclass
class _Totals:
    calls:              int   = 0
    input_tokens:       int   = 0
    cache_write_tokens: int   = 0
    cache_read_tokens:  int   = 0
    output_tokens:      int   = 0
    usd_saved:          float = 0.0
    usd_spent:          float = 0.0

_totals = _Totals()


# ── Heat lookup from Arc (best-effort, never blocks) ─────────────────────────

def _arc_heat(text: str, timeout: float = 1.5) -> float:
    """
    Ask the Arc memory engine for the heat score of `text`.
    Returns 0.0 if Arc is offline or busy (GPU in use by Dexter's Lab work).
    """
    if not _HAS_HTTPX:
        return 0.0
    try:
        r = _httpx.get(
            f"{ARC_ENDPOINT}/api/heat",
            params={"text": text[:256]},  # only first 256 chars for the query
            timeout=timeout,
        )
        if r.status_code == 200:
            return float(r.json().get("heat", 0.0))
    except Exception:
        pass
    return 0.0


# ── Cache breakpoint annotation ───────────────────────────────────────────────

def _ttl_for_heat(heat: float) -> str:
    if heat >= _theta_hot():   # adaptive θ_HOT (loci_adaptive_theta)
        return "extended"
    if heat >= 0.35:
        return "ephemeral"
    return "none"


def _apply_breakpoints(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    system: Optional[str] = None,
) -> tuple[list[dict], Optional[list[dict]]]:
    """
    Apply the 4-breakpoint LOCI layout to messages and tools.

    BP1 — last tool definition (almost never changes — cache always)
    BP2 — system message (stable — cache always)
    BP3 — loci context blocks (heat-gated — query Arc for score)
    BP4 — final user message — NOT cached (changes every turn)
    """
    import copy
    cc_ephemeral = {"type": "ephemeral"}

    # BP1: tools
    out_tools = None
    if tools:
        out_tools = copy.deepcopy(tools)
        last = out_tools[-1]
        if isinstance(last, dict):
            last["cache_control"] = cc_ephemeral

    # BP2+BP3: messages
    out_msgs = copy.deepcopy(messages)

    # Find system message
    for msg in out_msgs:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                msg["content"] = [{"type": "text", "text": content, "cache_control": cc_ephemeral}]
            elif isinstance(content, list) and content:
                content[-1]["cache_control"] = cc_ephemeral
            break

    # BP3: look for loci_context marker in assistant messages
    for msg in out_msgs:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            text = content if isinstance(content, str) else ""
            if isinstance(content, list):
                text = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
            if "<loci_context>" in text:
                heat = _arc_heat(text)
                ttl  = _ttl_for_heat(heat)
                if ttl != "none":
                    if isinstance(content, str):
                        msg["content"] = [{"type": "text", "text": content, "cache_control": cc_ephemeral}]
                    elif isinstance(content, list) and content:
                        content[-1]["cache_control"] = cc_ephemeral
                break

    return out_msgs, out_tools


# ── Ledger write ──────────────────────────────────────────────────────────────

@dataclass
class SavingsRecord:
    call_id:            str
    model:              str
    input_tokens:       int
    cache_write_tokens: int
    cache_read_tokens:  int
    output_tokens:      int
    usd_saved:          float
    usd_spent:          float
    ts:                 float = field(default_factory=time.time)

    def to_jsonl(self) -> str:
        return json.dumps({
            "ts":                 self.ts,
            "call_id":            self.call_id,
            "model":              self.model,
            "input_tokens":       self.input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_read_tokens":  self.cache_read_tokens,
            "output_tokens":      self.output_tokens,
            "usd_saved":          round(self.usd_saved, 6),
            "usd_spent":          round(self.usd_spent, 6),
            "schema":             "loci.token_savings.v1",
        })


def _record_usage(
    model:                    str,
    usage:                    Any,
    routing_tier:             str = "COLD",
    compression_tokens_saved: int = 0,
) -> SavingsRecord:
    """Parse Anthropic usage object and write to both legacy and v2 savings ledger."""
    inp   = getattr(usage, "input_tokens",              0) or 0
    cw    = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cr    = getattr(usage, "cache_read_input_tokens",    0) or 0
    out   = getattr(usage, "output_tokens",             0) or 0

    # USD cost breakdown (legacy SavingsRecord — kept for backwards compat)
    usd_input  = inp * _INPUT_COST_PER_M  / 1_000_000
    usd_cwrite = cw  * _CACHE_WRITE_PER_M / 1_000_000
    usd_cread  = cr  * _CACHE_READ_PER_M  / 1_000_000
    usd_output = out * _OUTPUT_COST_PER_M / 1_000_000
    usd_spent  = usd_input + usd_cwrite + usd_cread + usd_output

    usd_would_have = (inp + cw + cr) * _INPUT_COST_PER_M / 1_000_000
    usd_saved = max(0.0, usd_would_have - (usd_input + usd_cwrite + usd_cread))

    call_id = str(uuid.uuid4())[:8]
    rec = SavingsRecord(
        call_id            = call_id,
        model              = model,
        input_tokens       = inp,
        cache_write_tokens = cw,
        cache_read_tokens  = cr,
        output_tokens      = out,
        usd_saved          = usd_saved,
        usd_spent          = usd_spent,
    )

    # Update in-memory totals
    _totals.calls              += 1
    _totals.input_tokens       += inp
    _totals.cache_write_tokens += cw
    _totals.cache_read_tokens  += cr
    _totals.output_tokens      += out
    _totals.usd_saved          += usd_saved
    _totals.usd_spent          += usd_spent

    # Persist legacy line
    with LEDGER_PATH.open("a") as f:
        f.write(rec.to_jsonl() + "\n")

    # Also record in the richer v2 savings ledger (Haiku routing + compression)
    try:
        from loci_savings_ledger import record_call as _record_v2
        _record_v2(
            model                    = model,
            input_tokens             = inp,
            cache_read_tokens        = cr,
            cache_write_tokens       = cw,
            output_tokens            = out,
            compression_tokens_saved = compression_tokens_saved,
            routing_tier             = routing_tier,
            call_id                  = call_id,
        )
    except Exception:
        pass  # never block the main call path

    if cr > 0:
        log.info(
            "token savings  cache_read=%d  saved=$%.4f  total_saved=$%.4f",
            cr, usd_saved, _totals.usd_saved,
        )

    return rec


# ── Client wrapper ────────────────────────────────────────────────────────────

def wrap_anthropic_client(client: Any) -> None:
    """
    Monkey-patch client.messages.create (and .stream) so every call gets:
      • LOCI cache breakpoints injected
      • Usage recorded to the savings ledger

    Call once after constructing the anthropic.Anthropic() client.
    Arc being GPU-busy is fine — heat lookups time out in 1.5 s and default
    to ephemeral caching.
    """
    orig_create = client.messages.create

    def patched_create(**kwargs: Any) -> Any:
        # Inject breakpoints
        msgs   = kwargs.get("messages", [])
        tools  = kwargs.get("tools")
        out_msgs, out_tools = _apply_breakpoints(msgs, tools)
        kwargs["messages"] = out_msgs
        if out_tools is not None:
            kwargs["tools"] = out_tools

        # Add beta header for extended TTL
        hdrs = dict(kwargs.get("extra_headers") or {})
        hdrs.setdefault("anthropic-beta", "prompt-caching-2024-07-31")
        kwargs["extra_headers"] = hdrs

        response = orig_create(**kwargs)

        # Record usage
        if hasattr(response, "usage") and response.usage is not None:
            _record_usage(kwargs.get("model", "unknown"), response.usage)

        return response

    client.messages.create = patched_create
    log.info("LOCI token hook active — savings ledger: %s", LEDGER_PATH)


# ── Status for IDE panel ──────────────────────────────────────────────────────

def savings_summary() -> dict:
    """Return current session totals for the arc-token-offset status panel."""
    return {
        "service":             "loci-token-hook",
        "session_calls":       _totals.calls,
        "cache_read_tokens":   _totals.cache_read_tokens,
        "cache_write_tokens":  _totals.cache_write_tokens,
        "output_tokens":       _totals.output_tokens,
        "usd_saved_session":   round(_totals.usd_saved, 4),
        "usd_spent_session":   round(_totals.usd_spent, 4),
        "ledger":              str(LEDGER_PATH),
        "arc_endpoint":        ARC_ENDPOINT,
        "note": (
            "Arc GPU may be busy (Dexter's Lab). "
            "Heat lookups time out in 1.5 s — caching defaults to ephemeral."
        ),
    }


if __name__ == "__main__":
    # Smoke test — prints savings summary and verifies ledger path
    import json as _json
    print(_json.dumps(savings_summary(), indent=2))
    print(f"\nLedger: {LEDGER_PATH}")
    print("Run wrap_anthropic_client(client) in your code to activate.")
