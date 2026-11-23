"""
loci_savings_ledger.py — Empirical savings tracker for LOCI optimisations.

Tracks two orthogonal sources of cost reduction:

  1. Haiku routing
     ─────────────
     Claude Code routes background tasks (subagents, inline completions,
     auto-mode classifications) to claude-haiku-* (ANTHROPIC_DEFAULT_HAIKU_MODEL).
     The InferenceRouter (model_fusion) sends COLD-tier queries to Claude API
     and MILD code queries to Codex — both are cheaper than Sonnet-class.
     Saving = (what it would have cost at Sonnet price) − (what we paid at Haiku price).

  2. LOCI prompt compression
     ───────────────────────
     The ContextFuser drops COLD blocks, truncates MILD blocks, and injects only
     HOT/MILD context into the prompt. The token difference vs. a naive
     "dump everything" baseline is the compression saving.
     The token hook's BP1-BP4 breakpoints further reduce billed input tokens via
     Anthropic prompt-caching (90% discount on cache-read tokens).

Empirical evidence strategy
───────────────────────────
Every real API call writes one JSONL line.  The report aggregates over ALL lines
(cross-session, lifetime) so savings accumulate with use rather than resetting.
No synthetic data is mixed in — this is a pure empirical ledger.

Usage
─────
    from loci_savings_ledger import LedgerEntry, SavingsLedger, record_call

    # Minimal usage (token hook calls this automatically):
    record_call(
        model="claude-haiku-4-5",
        input_tokens=1200,
        cache_read_tokens=800,
        cache_write_tokens=0,
        output_tokens=150,
        compression_tokens_saved=3400,  # tokens NOT sent due to LOCI compression
    )

    ledger = SavingsLedger()
    print(ledger.report())              # human-readable summary

CLI
───
    python loci_savings_ledger.py --report
    python loci_savings_ledger.py --report --json
    python loci_savings_ledger.py --report --days 7
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


# ── Pricing (USD per 1M tokens) ───────────────────────────────────────────────

PRICING: dict[str, dict[str, float]] = {
    # Claude Sonnet 4.5 / 4.6 baseline (what we compare against)
    "sonnet": {
        "input": 3.00, "cache_read": 0.30, "cache_write": 3.75, "output": 15.00,
    },
    # Claude Haiku 4.5 (cheap fast routing)
    "haiku": {
        "input": 0.80, "cache_read": 0.08, "cache_write": 1.00, "output": 4.00,
    },
    # Codex mini (MILD-code routing)
    "codex": {
        "input": 1.50, "cache_read": 0.15, "cache_write": 1.88, "output": 6.00,
    },
    # Fallback for unrecognised model IDs
    "unknown": {
        "input": 3.00, "cache_read": 0.30, "cache_write": 3.75, "output": 15.00,
    },
}

# Token savings from compression are priced at Sonnet input rate (the avoided cost)
_SONNET_INPUT_PER_M = PRICING["sonnet"]["input"]


def _pricing_for(model: str) -> dict[str, float]:
    m = model.lower()
    if "haiku" in m:
        return PRICING["haiku"]
    if "codex" in m or "gpt" in m or "openai" in m:
        return PRICING["codex"]
    if "sonnet" in m or "claude" in m or "opus" in m:
        return PRICING["sonnet"]
    return PRICING["unknown"]


# ── Storage ───────────────────────────────────────────────────────────────────

LEDGER_DIR = Path(os.getenv(
    "LOCI_SAVINGS_DIR",
    os.path.expanduser("~/.local/state/loci-token-savings"),
))
LEDGER_DIR.mkdir(parents=True, exist_ok=True)
LEDGER_PATH = LEDGER_DIR / "savings_ledger.jsonl"


# ── Entry dataclass ───────────────────────────────────────────────────────────

@dataclass
class LedgerEntry:
    """One real API call with full empirical cost accounting."""

    ts:                       float   # Unix timestamp
    model:                    str     # actual model used (e.g. "claude-haiku-4-5")
    routing_tier:             str     # HOT / MILD / COLD / SOVEREIGN (InferenceRouter tier)

    # Raw token counts from Anthropic usage object
    input_tokens:             int
    cache_read_tokens:        int     # billed at 10% — cache HIT
    cache_write_tokens:       int     # billed at 125% — cache WRITE
    output_tokens:            int

    # LOCI compression saving: tokens NOT sent because ContextFuser dropped/truncated
    compression_tokens_saved: int

    # Computed costs (USD)
    usd_actual:               float   # what we paid
    usd_baseline_no_cache:    float   # what we'd pay with no caching (all tokens at input rate)
    usd_baseline_no_routing:  float   # what we'd pay if Sonnet was used for everything
    usd_cache_saving:         float   # from prompt caching
    usd_routing_saving:       float   # from Haiku/Codex routing instead of Sonnet
    usd_compression_saving:   float   # from sending fewer tokens (LOCI context compression)
    usd_total_saving:         float   # sum of all three

    schema: str = "loci.savings_ledger.v2"
    call_id: str = ""


def _compute_costs(
    model:                    str,
    input_tokens:             int,
    cache_read_tokens:        int,
    cache_write_tokens:       int,
    output_tokens:            int,
    compression_tokens_saved: int,
) -> dict[str, float]:
    p = _pricing_for(model)
    ps = PRICING["sonnet"]

    total_input_billed = input_tokens + cache_write_tokens + cache_read_tokens

    usd_input  = input_tokens        * p["input"]        / 1_000_000
    usd_cwrite = cache_write_tokens  * p["cache_write"]  / 1_000_000
    usd_cread  = cache_read_tokens   * p["cache_read"]   / 1_000_000
    usd_output = output_tokens       * p["output"]       / 1_000_000
    usd_actual = usd_input + usd_cwrite + usd_cread + usd_output

    # Baseline: same token counts, Sonnet price, no caching (all tokens billed at full input)
    usd_baseline_no_cache = (
        total_input_billed * ps["input"] / 1_000_000
        + output_tokens    * ps["output"]/ 1_000_000
    )

    # Baseline routing: Sonnet price with current caching
    usd_baseline_routing = (
        input_tokens        * ps["input"]        / 1_000_000
        + cache_write_tokens * ps["cache_write"]  / 1_000_000
        + cache_read_tokens  * ps["cache_read"]   / 1_000_000
        + output_tokens      * ps["output"]       / 1_000_000
    )

    # Cache saving = no-cache baseline at actual model price minus actual
    usd_actual_no_cache = (
        total_input_billed * p["input"] / 1_000_000
        + output_tokens    * p["output"]/ 1_000_000
    )
    usd_cache_saving = max(0.0, usd_actual_no_cache - usd_actual)

    # Routing saving = Sonnet (with current caching) minus actual model cost
    usd_routing_saving = max(0.0, usd_baseline_routing - usd_actual)

    # Compression saving = tokens NOT sent × Sonnet input rate
    usd_compression_saving = compression_tokens_saved * _SONNET_INPUT_PER_M / 1_000_000

    usd_total_saving = usd_cache_saving + usd_routing_saving + usd_compression_saving

    return {
        "usd_actual":              round(usd_actual, 7),
        "usd_baseline_no_cache":   round(usd_baseline_no_cache, 7),
        "usd_baseline_no_routing": round(usd_baseline_routing, 7),
        "usd_cache_saving":        round(usd_cache_saving, 7),
        "usd_routing_saving":      round(usd_routing_saving, 7),
        "usd_compression_saving":  round(usd_compression_saving, 7),
        "usd_total_saving":        round(usd_total_saving, 7),
    }


def record_call(
    model:                    str,
    input_tokens:             int,
    cache_read_tokens:        int         = 0,
    cache_write_tokens:       int         = 0,
    output_tokens:            int         = 0,
    compression_tokens_saved: int         = 0,
    routing_tier:             str         = "COLD",
    call_id:                  str         = "",
    ts:                       Optional[float] = None,
) -> LedgerEntry:
    """
    Record one real API call.  Called automatically by loci_token_hook.wrap_anthropic_client.
    Can also be called manually from InferenceRouter to track routing decisions.
    """
    import uuid
    costs = _compute_costs(
        model, input_tokens, cache_read_tokens, cache_write_tokens,
        output_tokens, compression_tokens_saved,
    )
    entry = LedgerEntry(
        ts                       = ts or time.time(),
        model                    = model,
        routing_tier             = routing_tier,
        input_tokens             = input_tokens,
        cache_read_tokens        = cache_read_tokens,
        cache_write_tokens       = cache_write_tokens,
        output_tokens            = output_tokens,
        compression_tokens_saved = compression_tokens_saved,
        call_id                  = call_id or str(uuid.uuid4())[:8],
        **costs,
    )
    with LEDGER_PATH.open("a") as fh:
        fh.write(json.dumps(asdict(entry)) + "\n")
    return entry


# ── Ledger reader + aggregator ────────────────────────────────────────────────

class SavingsLedger:
    """
    Read the empirical ledger and produce human-readable savings reports.
    Aggregates ALL lines (lifetime, cross-session) unless filtered by date.
    """

    def __init__(self, path: Path = LEDGER_PATH) -> None:
        self.path = path
        self._entries: list[LedgerEntry] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    # Support both v1 (from token_hook) and v2 (this module)
                    if d.get("schema") in ("loci.token_savings.v1", "loci.savings_ledger.v2"):
                        self._entries.append(_entry_from_dict(d))
                except (json.JSONDecodeError, KeyError):
                    continue

    def filter_since(self, days: float) -> "SavingsLedger":
        cutoff = time.time() - days * 86400
        sl = SavingsLedger.__new__(SavingsLedger)
        sl.path = self.path
        sl._entries = [e for e in self._entries if e.ts >= cutoff]
        return sl

    # ── Aggregate stats ───────────────────────────────────────────────────────

    def total_calls(self) -> int:
        return len(self._entries)

    def total_input_tokens(self) -> int:
        return sum(e.input_tokens for e in self._entries)

    def total_cache_read_tokens(self) -> int:
        return sum(e.cache_read_tokens for e in self._entries)

    def total_compression_tokens_saved(self) -> int:
        return sum(e.compression_tokens_saved for e in self._entries)

    def total_usd_actual(self) -> float:
        return sum(e.usd_actual for e in self._entries)

    def total_usd_cache_saving(self) -> float:
        return sum(e.usd_cache_saving for e in self._entries)

    def total_usd_routing_saving(self) -> float:
        return sum(e.usd_routing_saving for e in self._entries)

    def total_usd_compression_saving(self) -> float:
        return sum(e.usd_compression_saving for e in self._entries)

    def total_usd_saving(self) -> float:
        return sum(e.usd_total_saving for e in self._entries)

    def total_usd_baseline_no_cache(self) -> float:
        return sum(e.usd_baseline_no_cache for e in self._entries)

    def total_usd_baseline_no_routing(self) -> float:
        return sum(e.usd_baseline_no_routing for e in self._entries)

    def haiku_calls(self) -> int:
        return sum(1 for e in self._entries if "haiku" in e.model.lower())

    def cache_hit_rate(self) -> float:
        total_in = self.total_input_tokens()
        total_cr = self.total_cache_read_tokens()
        if total_in + total_cr == 0:
            return 0.0
        return total_cr / (total_in + total_cr)

    def routing_model_breakdown(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self._entries:
            key = e.model.split("-")[2] if e.model.count("-") >= 2 else e.model
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    # ── Human-readable report ─────────────────────────────────────────────────

    def report(self, title: str = "LOCI Empirical Savings Report") -> str:
        n = self.total_calls()
        if n == 0:
            return f"{title}\n{'─'*50}\nNo calls recorded yet.\n"

        actual        = self.total_usd_actual()
        baseline_nc   = self.total_usd_baseline_no_cache()
        baseline_nr   = self.total_usd_baseline_no_routing()
        full_baseline = baseline_nc + self.total_usd_routing_saving()

        cache_save   = self.total_usd_cache_saving()
        route_save   = self.total_usd_routing_saving()
        compr_save   = self.total_usd_compression_saving()
        total_save   = cache_save + route_save + compr_save

        pct = total_save / (actual + total_save) * 100 if (actual + total_save) > 0 else 0.0

        lines = [
            f"\n{'═'*58}",
            f"  {title}",
            f"{'═'*58}",
            f"  Calls recorded (lifetime):  {n:,}",
            f"  Haiku-routed calls:          {self.haiku_calls():,} "
            f"({self.haiku_calls()/n*100:.1f}%)",
            f"  Cache hit rate:              {self.cache_hit_rate()*100:.1f}%",
            f"  Compressed tokens saved:     {self.total_compression_tokens_saved():,}",
            f"",
            f"  {'Cost breakdown':─<46}",
            f"  What you paid (actual):       ${actual:.4f}",
            f"  Baseline (no cache, Sonnet):  ${baseline_nc:.4f}",
            f"",
            f"  {'Savings breakdown':─<46}",
            f"  Prompt-cache savings:         ${cache_save:.4f}",
            f"  Haiku/Codex routing savings:  ${route_save:.4f}",
            f"  LOCI compression savings:     ${compr_save:.4f}",
            f"  ─────────────────────────────────────────",
            f"  Total savings:                ${total_save:.4f}  ({pct:.1f}% reduction)",
            f"",
            f"  {'Model routing breakdown':─<46}",
        ]
        for model, count in self.routing_model_breakdown().items():
            lines.append(f"    {model:<30} {count:>6} calls")
        lines.append(f"{'═'*58}\n")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Machine-readable summary for the IDE status panel."""
        n = max(self.total_calls(), 1)
        return {
            "service":                    "loci-savings-ledger",
            "schema":                     "loci.savings_report.v2",
            "calls_total":                self.total_calls(),
            "haiku_calls":                self.haiku_calls(),
            "haiku_routing_pct":          round(self.haiku_calls() / n * 100, 1),
            "cache_hit_rate_pct":         round(self.cache_hit_rate() * 100, 1),
            "compression_tokens_saved":   self.total_compression_tokens_saved(),
            "usd_actual":                 round(self.total_usd_actual(), 4),
            "usd_baseline_no_cache":      round(self.total_usd_baseline_no_cache(), 4),
            "usd_cache_saving":           round(self.total_usd_cache_saving(), 4),
            "usd_routing_saving":         round(self.total_usd_routing_saving(), 4),
            "usd_compression_saving":     round(self.total_usd_compression_saving(), 4),
            "usd_total_saving":           round(self.total_usd_saving(), 4),
            "total_saving_pct":           round(
                self.total_usd_saving()
                / max(self.total_usd_actual() + self.total_usd_saving(), 1e-9)
                * 100, 1
            ),
            "routing_model_breakdown":    self.routing_model_breakdown(),
            "ledger_path":                str(self.path),
        }


def _entry_from_dict(d: dict) -> LedgerEntry:
    """Reconstruct a LedgerEntry from a stored dict (handles v1 and v2)."""
    # v1 had no routing/compression fields — use safe defaults
    return LedgerEntry(
        ts                       = d.get("ts", 0.0),
        model                    = d.get("model", "unknown"),
        routing_tier             = d.get("routing_tier", "COLD"),
        input_tokens             = d.get("input_tokens", 0),
        cache_read_tokens        = d.get("cache_read_tokens",  d.get("cache_read_input_tokens", 0)),
        cache_write_tokens       = d.get("cache_write_tokens", d.get("cache_creation_input_tokens", 0)),
        output_tokens            = d.get("output_tokens", 0),
        compression_tokens_saved = d.get("compression_tokens_saved", 0),
        usd_actual               = d.get("usd_actual",               d.get("usd_spent", 0.0)),
        usd_baseline_no_cache    = d.get("usd_baseline_no_cache",    d.get("usd_spent", 0.0)),
        usd_baseline_no_routing  = d.get("usd_baseline_no_routing",  d.get("usd_spent", 0.0)),
        usd_cache_saving         = d.get("usd_cache_saving",         d.get("usd_saved", 0.0)),
        usd_routing_saving       = d.get("usd_routing_saving", 0.0),
        usd_compression_saving   = d.get("usd_compression_saving", 0.0),
        usd_total_saving         = d.get("usd_total_saving",         d.get("usd_saved", 0.0)),
        schema                   = d.get("schema", "loci.token_savings.v1"),
        call_id                  = d.get("call_id", ""),
    )


# ── Convenience: seed a demo entry for testing ────────────────────────────────

def seed_demo() -> None:
    """Write one synthetic HOT-teacher entry to show non-empty report on first run."""
    record_call(
        model                    = "claude-haiku-4-5",
        input_tokens             = 1_200,
        cache_read_tokens        = 4_800,  # 80% cache hit
        cache_write_tokens       = 0,
        output_tokens            = 180,
        compression_tokens_saved = 6_200,  # LOCI dropped COLD context
        routing_tier             = "COLD",
        call_id                  = "demo_01",
    )
    record_call(
        model                    = "claude-sonnet-4-6",
        input_tokens             = 800,
        cache_read_tokens        = 2_400,
        cache_write_tokens       = 400,
        output_tokens            = 650,
        compression_tokens_saved = 1_800,
        routing_tier             = "HOT",
        call_id                  = "demo_02",
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LOCI empirical savings report")
    ap.add_argument("--report", action="store_true", help="Print savings report")
    ap.add_argument("--json",   action="store_true", help="Output JSON instead of text")
    ap.add_argument("--days",   type=float, default=0, help="Only show last N days (0=all)")
    ap.add_argument("--seed-demo", action="store_true", help="Write two demo entries then report")
    args = ap.parse_args()

    if args.seed_demo:
        seed_demo()
        print("Demo entries written.")

    ledger = SavingsLedger()
    if args.days > 0:
        ledger = ledger.filter_since(args.days)

    if args.json:
        print(json.dumps(ledger.to_dict(), indent=2))
    else:
        print(ledger.report())
        print(f"Ledger: {LEDGER_PATH}")
