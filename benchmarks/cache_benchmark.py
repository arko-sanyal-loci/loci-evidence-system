"""
benchmarks/cache_benchmark.py — LOCI prompt-cache benchmark.

Answers the question: "how much do you carry, hot teachers cache"

Measures for Claude Sonnet 4.6 (and optionally GPT-4o via nexos.ai):
  1. Token carry — fraction of per-turn input tokens served from cache
  2. Hot teacher cache hit rate — % of tokens from extended (1hr) cache
  3. USD savings over a 20-turn multi-agent IDE coding session
  4. Comparison: LOCI-cached vs no-cache baseline vs GPT-4o (nexos.ai)

The benchmark has two modes:
  LIVE  — makes real API calls if ANTHROPIC_API_KEY is set; reads actual
           cache_read_input_tokens / cache_creation_input_tokens from responses.
  ANALYTICAL — pure simulation from LOCI pricing constants + realistic token
           profiles. Runs without any API key and produces exact same metrics.

Usage:
  python3 benchmarks/cache_benchmark.py             # analytical
  ANTHROPIC_API_KEY=sk-… python3 benchmarks/cache_benchmark.py --live
  python3 benchmarks/cache_benchmark.py --vis       # write HTML panel too
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ── Pricing (Anthropic docs, Nov 2026) ───────────────────────────────────────

MODELS = {
    "claude-sonnet-4-6": {
        "input_per_m":        3.00,
        "cache_read_per_m":   0.30,   # 10 % of input
        "cache_write_per_m":  3.75,   # 125 % of input (extended) or input (ephemeral)
        "output_per_m":      15.00,
        "ctx_window":       200_000,
        "extended_ttl_min":      60,
        "ephemeral_ttl_min":      5,
        "label": "Claude Sonnet 4.6",
    },
    "gpt-4o": {
        "input_per_m":        2.50,
        "cache_read_per_m":   1.25,   # OpenAI prompt caching: 50 % of input
        "cache_write_per_m":  2.50,   # same as input (implicit)
        "output_per_m":      10.00,
        "ctx_window":       128_000,
        "extended_ttl_min":    None,  # no extended TTL
        "ephemeral_ttl_min":      5,
        "label": "GPT-4o",
    },
    "claude-sonnet-4-6-nocache": {
        "input_per_m":        3.00,
        "cache_read_per_m":   3.00,   # no cache: reads billed at full input rate
        "cache_write_per_m":  3.00,
        "output_per_m":      15.00,
        "ctx_window":       200_000,
        "extended_ttl_min":    None,
        "ephemeral_ttl_min":    None,
        "label": "Claude Sonnet 4.6 (no cache)",
    },
}

# LOCI 4-breakpoint layout token profile for a typical multi-agent IDE session:
#
#  BP1 — tools (MCP + native tool schemas)   ~8 000 tokens  HOT / extended
#  BP2 — system prompt + AGENTS.md           ~3 200 tokens  HOT / extended
#  BP3 — loci_context (agent fusion output)  ~2 400 tokens  varies by heat
#  BP4 — user turn (this turn only)          grows each turn
#
# Heat tier for BP3 blocks (from ContextFuser):
#   arc memory:   heat=0.91 → extended
#   claude block: heat=0.65 → ephemeral
#   codex block:  heat=0.78 → extended (≥0.80 boundary check: just below)
#   copilot:      heat=0.42 → ephemeral

BP1_TOKENS        = 8_000   # tools — extended, written once per hour
BP2_TOKENS        = 3_200   # system — extended
BP3_HOT_TOKENS    = 1_200   # arc memory + codex code blocks (heat ≥ 0.80) — extended
BP3_MILD_TOKENS   = 1_200   # claude + copilot blocks (0.35 ≤ heat < 0.80) — ephemeral
OUTPUT_PER_TURN   =   800   # typical assistant output tokens/turn
USER_PER_TURN     =   300   # typical user message tokens/turn
HISTORY_PER_TURN  = OUTPUT_PER_TURN + USER_PER_TURN  # tokens added to sliding history


# ── Token profile per turn ────────────────────────────────────────────────────

@dataclass
class TurnProfile:
    turn:             int
    fresh_input:      int    # BP4 user + recent history (not cached)
    cached_extended:  int    # BP1 + BP2 + BP3-HOT (extended cache)
    cached_ephemeral: int    # BP3-MILD (ephemeral cache) — may miss after 5 min
    cache_write:      int    # tokens written to cache this turn (first turn only for extended)
    output_tokens:    int


def build_turn_profiles(
    n_turns: int = 20,
    session_duration_min: float = 45.0,
    hits_per_session: int = 1,         # extended cache write count per session
) -> list[TurnProfile]:
    """
    Build realistic token profiles for a multi-turn IDE session.

    Assumptions:
    - Extended (HOT) cache written once at turn 1; all later turns are reads.
    - Ephemeral (MILD) cache: each turn re-writes (worst case); in practice
      many turns within 5 min re-read — we model a 70 % hit rate.
    - BP4 (user + history): sliding window of ~8 most recent turns.
    """
    profiles: list[TurnProfile] = []
    HISTORY_WINDOW = 8

    for t in range(1, n_turns + 1):
        # Sliding history: accumulates up to HISTORY_WINDOW turns
        history_tokens = min(t - 1, HISTORY_WINDOW) * HISTORY_PER_TURN
        fresh_input = USER_PER_TURN + history_tokens

        # Extended cache: write only on turn 1
        if t == 1:
            cached_ext = 0
            write_ext  = BP1_TOKENS + BP2_TOKENS + BP3_HOT_TOKENS
        else:
            cached_ext = BP1_TOKENS + BP2_TOKENS + BP3_HOT_TOKENS
            write_ext  = 0

        # Ephemeral cache: 70 % hit rate within 5-min window
        # Turn 1 always writes; subsequent turns: 70 % read, 30 % re-write
        if t == 1:
            cached_eph = 0
            write_eph  = BP3_MILD_TOKENS
        elif (t - 1) % 3 == 0:         # every 3rd turn past window → re-write
            cached_eph = 0
            write_eph  = BP3_MILD_TOKENS
        else:
            cached_eph = BP3_MILD_TOKENS
            write_eph  = 0

        profiles.append(TurnProfile(
            turn             = t,
            fresh_input      = fresh_input,
            cached_extended  = cached_ext,
            cached_ephemeral = cached_eph,
            cache_write      = write_ext + write_eph,
            output_tokens    = OUTPUT_PER_TURN,
        ))

    return profiles


# ── Cost calculator ───────────────────────────────────────────────────────────

@dataclass
class TurnCost:
    turn:         int
    model:        str
    input_cost:   float
    read_cost:    float
    write_cost:   float
    output_cost:  float
    total:        float
    carry_ratio:  float    # cached_tokens / total_input_tokens


@dataclass
class SessionCost:
    model:         str
    turns:         list[TurnCost] = field(default_factory=list)

    @property
    def total_input_cost(self) -> float:
        return sum(t.input_cost for t in self.turns)

    @property
    def total_read_cost(self) -> float:
        return sum(t.read_cost for t in self.turns)

    @property
    def total_write_cost(self) -> float:
        return sum(t.write_cost for t in self.turns)

    @property
    def total_output_cost(self) -> float:
        return sum(t.output_cost for t in self.turns)

    @property
    def total(self) -> float:
        return (self.total_input_cost + self.total_read_cost +
                self.total_write_cost + self.total_output_cost)

    @property
    def avg_carry(self) -> float:
        if not self.turns:
            return 0.0
        return sum(t.carry_ratio for t in self.turns) / len(self.turns)

    @property
    def cache_hit_tokens(self) -> int:
        # Total tokens served from cache (read, not fresh input)
        return sum(
            int((t.read_cost / MODELS[self.model]["cache_read_per_m"]) * 1_000_000)
            for t in self.turns
            if MODELS[self.model]["cache_read_per_m"] > 0
        )


def calc_session(
    model_key: str,
    profiles: list[TurnProfile],
    has_cache: bool = True,
) -> SessionCost:
    m = MODELS[model_key]
    session = SessionCost(model=model_key)

    for p in profiles:
        total_input = p.fresh_input + p.cached_extended + p.cached_ephemeral + p.cache_write

        if has_cache:
            inp_cost   = p.fresh_input * m["input_per_m"] / 1_000_000
            read_cost  = (p.cached_extended + p.cached_ephemeral) * m["cache_read_per_m"] / 1_000_000
            write_cost = p.cache_write * m["cache_write_per_m"] / 1_000_000
        else:
            # No cache: everything billed as fresh input
            inp_cost   = total_input * m["input_per_m"] / 1_000_000
            read_cost  = 0.0
            write_cost = 0.0

        out_cost = p.output_tokens * m["output_per_m"] / 1_000_000

        cached_tokens = p.cached_extended + p.cached_ephemeral if has_cache else 0
        carry = cached_tokens / max(total_input, 1)

        session.turns.append(TurnCost(
            turn        = p.turn,
            model       = model_key,
            input_cost  = inp_cost,
            read_cost   = read_cost,
            write_cost  = write_cost,
            output_cost = out_cost,
            total       = inp_cost + read_cost + write_cost + out_cost,
            carry_ratio = carry,
        ))

    return session


# ── Live API mode ─────────────────────────────────────────────────────────────

def _api_call(messages: list[dict], system: str, tools_text: str, api_key: str,
              base_url: str, model: str = "claude-sonnet-4-6") -> dict:
    """Make one Anthropic API call with LOCI 4-breakpoint caching layout."""
    # BP1: tools as first system block with cache_control
    # BP2: main system block
    # BP3: loci_context (embedded in system)
    system_blocks = [
        {
            "type": "text",
            "text": f"<tools_definitions>\n{tools_text}\n</tools_definitions>",
            "cache_control": {"type": "ephemeral"},   # BP1
        },
        {
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},   # BP2
        },
    ]

    payload = json.dumps({
        "model": model,
        "max_tokens": 256,
        "system": system_blocks,
        "messages": messages,
    }).encode()

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/messages",
        data=payload,
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "prompt-caching-2024-07-31",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def run_live(n_turns: int = 6) -> Optional[list[dict]]:
    """Run a live benchmark session. Returns list of raw API usage dicts."""
    api_key  = os.getenv("ANTHROPIC_API_KEY", "")
    base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")

    if not api_key:
        return None

    print(f"  Live mode: {base_url} — {n_turns} turns")

    tools_text = (
        "# IDE Agent Tools\n"
        "arc_chat(query: str) — query Arc local reasoning\n"
        "arc_memory_search(query: str) — search LOCI memory\n"
        "codex_generate(prompt: str) — OpenAI Codex code gen\n"
        "copilot_complete(context: str) — inline completion\n"
        "arc_heat(text: str) -> float — LOCI heat score\n"
        "token_savings_summary() -> dict — LOCI savings ledger\n"
    )

    system = (
        "You are the Anti-gravity IDE orchestrator. You route queries across "
        "Arc (local 70B), Codex, Copilot, and Claude via the LOCI heat law. "
        "HOT queries (heat≥0.80) go to Arc VRAM; MILD to Codex; COLD to Claude API. "
        "Always prefer the most token-efficient path.\n\n"
        "<loci_context>\n"
        "[ARC · memory · heat=0.91] Active files: src/model_fusion/, agents/orchestrator.py\n"
        "[CLAUDE · message · heat=0.65] Last turn: reviewed distillation.py SAD-LoRA\n"
        "[CODEX · code · heat=0.78] Last gen: FusionDistiller.fused_loss() \n"
        "</loci_context>"
    )

    conversation: list[dict] = []
    prompts = [
        "What heat tier would a query about optimizing LoRA rank budgets fall into?",
        "Show me the token savings summary for the current session.",
        "Route this code-completion request: def fused_loss(self, student",
        "How does gaussian conduction promote COLD candidates to MILD?",
        "What's the VRAM admission threshold and which GPU handles prefill?",
        "Summarize the speculative pipeline: Arc B70 prefill + RTX decode.",
    ]

    results = []
    for i, prompt in enumerate(prompts[:n_turns]):
        conversation.append({"role": "user", "content": prompt})
        try:
            resp = _api_call(conversation, system, tools_text, api_key, base_url)
            usage = resp.get("usage", {})
            results.append({
                "turn": i + 1,
                "prompt_tokens": usage.get("input_tokens", 0),
                "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
            })
            assistant_text = ""
            for block in resp.get("content", []):
                if block.get("type") == "text":
                    assistant_text = block["text"]
                    break
            conversation.append({"role": "assistant", "content": assistant_text})
            print(f"    turn {i+1}: read={usage.get('cache_read_input_tokens',0):,} "
                  f"write={usage.get('cache_creation_input_tokens',0):,} "
                  f"fresh={usage.get('input_tokens',0):,}")
        except Exception as e:
            print(f"    turn {i+1}: API error — {e}")
            break

    return results if results else None


# ── Ledger ────────────────────────────────────────────────────────────────────

def _write_ledger(records: list[dict]) -> Path:
    ledger_dir = Path.home() / ".local/state/loci-token-savings"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger = ledger_dir / "ledger.jsonl"
    with ledger.open("a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return ledger


# ── HTML visual panel ─────────────────────────────────────────────────────────

def _render_html(sessions: dict[str, SessionCost], live_data: Optional[list[dict]]) -> str:
    claude  = sessions["claude-sonnet-4-6"]
    nocache = sessions["claude-sonnet-4-6-nocache"]
    gpt4o   = sessions["gpt-4o"]

    savings_vs_nocache = nocache.total - claude.total
    savings_pct        = 100 * savings_vs_nocache / nocache.total if nocache.total else 0

    carry_pct = 100 * claude.avg_carry

    # Per-turn carry data for sparkline
    carry_data = [round(100 * t.carry_ratio, 1) for t in claude.turns]
    carry_json = json.dumps(carry_data)

    # Per-turn cost breakdown for Claude
    turn_labels = json.dumps([t.turn for t in claude.turns])
    turn_input  = json.dumps([round(t.input_cost * 1000, 4) for t in claude.turns])
    turn_read   = json.dumps([round(t.read_cost * 1000, 4) for t in claude.turns])
    turn_write  = json.dumps([round(t.write_cost * 1000, 4) for t in claude.turns])

    live_section = ""
    if live_data:
        rows = ""
        for r in live_data:
            total = r["prompt_tokens"] + r["cache_creation_input_tokens"] + r["cache_read_input_tokens"]
            cr = 100 * r["cache_read_input_tokens"] / max(total, 1)
            rows += (
                f"<tr><td>{r['turn']}</td>"
                f"<td>{r['prompt_tokens']:,}</td>"
                f"<td>{r['cache_creation_input_tokens']:,}</td>"
                f"<td>{r['cache_read_input_tokens']:,}</td>"
                f"<td>{cr:.0f}%</td>"
                f"<td>{r['completion_tokens']:,}</td></tr>\n"
            )
        live_section = f"""
        <div class="section">
          <h2>Live API turns</h2>
          <table>
            <thead><tr>
              <th>Turn</th><th>Fresh input</th><th>Cache write</th>
              <th>Cache read</th><th>Carry %</th><th>Output</th>
            </tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LOCI Cache Benchmark — Claude Sonnet 4.6</title>
<style>
  :root {{
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --hot: #ff6b35; --mild: #ffd166; --cold: #06d6a0;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font: 14px/1.6 'JetBrains Mono', monospace;
          padding: 24px; }}
  h1 {{ font-size: 20px; color: var(--accent); margin-bottom: 4px; }}
  .subtitle {{ color: var(--muted); font-size: 12px; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
           gap: 16px; margin-bottom: 24px; }}
  .kpi {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px;
          padding: 16px; }}
  .kpi .label {{ font-size: 11px; color: var(--muted); text-transform: uppercase;
                 letter-spacing: 0.08em; margin-bottom: 4px; }}
  .kpi .value {{ font-size: 28px; font-weight: 700; }}
  .kpi .sub   {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}
  .hot  {{ color: var(--hot); }}
  .mild {{ color: var(--mild); }}
  .cold {{ color: var(--cold); }}
  .blue {{ color: var(--accent); }}
  .section {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px;
              padding: 16px; margin-bottom: 20px; }}
  .section h2 {{ font-size: 14px; color: var(--accent); margin-bottom: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th {{ text-align: left; color: var(--muted); border-bottom: 1px solid var(--border);
        padding: 6px 8px; }}
  td {{ padding: 5px 8px; border-bottom: 1px solid #21262d; }}
  .bar-wrap {{ position: relative; background: #21262d; border-radius: 4px;
               height: 8px; width: 100%; }}
  .bar {{ position: absolute; left: 0; top: 0; height: 100%; border-radius: 4px; }}
  canvas {{ display: block; width: 100%; }}
  .tier-row td:nth-child(2) {{ color: var(--hot); }}
  .tier-row td:nth-child(3) {{ color: var(--mild); }}
  .tier-row td:nth-child(4) {{ color: var(--cold); }}
</style>
</head>
<body>
<h1>LOCI Cache Benchmark · Anti-gravity IDE</h1>
<p class="subtitle">Claude Sonnet 4.6 token carry &amp; hot teacher cache · 20-turn IDE session · {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>

<div class="grid">
  <div class="kpi">
    <div class="label">Avg carry ratio</div>
    <div class="value hot">{carry_pct:.0f}%</div>
    <div class="sub">tokens served from cache / total input</div>
  </div>
  <div class="kpi">
    <div class="label">USD saved vs no-cache</div>
    <div class="value blue">${savings_vs_nocache:.4f}</div>
    <div class="sub">{savings_pct:.0f}% reduction over 20 turns</div>
  </div>
  <div class="kpi">
    <div class="label">Total (Claude + LOCI)</div>
    <div class="value">${claude.total:.4f}</div>
    <div class="sub">vs ${nocache.total:.4f} no-cache · ${gpt4o.total:.4f} GPT-4o</div>
  </div>
  <div class="kpi">
    <div class="label">HOT cache (extended)</div>
    <div class="value hot">{BP1_TOKENS + BP2_TOKENS + BP3_HOT_TOKENS:,}</div>
    <div class="sub">tokens in 1-hr cache (BP1+BP2+BP3-HOT)</div>
  </div>
  <div class="kpi">
    <div class="label">MILD cache (ephemeral)</div>
    <div class="value mild">{BP3_MILD_TOKENS:,}</div>
    <div class="sub">tokens in 5-min cache (BP3-MILD)</div>
  </div>
  <div class="kpi">
    <div class="label">Context window</div>
    <div class="value blue">200K</div>
    <div class="sub">Claude Sonnet 4.6 max context</div>
  </div>
</div>

<div class="section">
  <h2>Carry ratio per turn (% of input tokens from cache)</h2>
  <canvas id="carryChart" height="80"></canvas>
</div>

<div class="section">
  <h2>Cost breakdown per turn (milli-USD)</h2>
  <canvas id="costChart" height="100"></canvas>
</div>

<div class="section">
  <h2>Cache layout — 4 breakpoints (LOCI Heat Law)</h2>
  <table>
    <thead><tr>
      <th>Breakpoint</th><th>Tier</th><th>Tokens</th><th>TTL</th>
      <th>Cache $/M write</th><th>Cache $/M read</th><th>Savings/turn</th>
    </tr></thead>
    <tbody>
      <tr>
        <td>BP1 — Tool schemas</td>
        <td class="hot">HOT</td>
        <td>{BP1_TOKENS:,}</td>
        <td>60 min (extended)</td>
        <td>$3.75</td><td>$0.30</td>
        <td>${(BP1_TOKENS * (3.00 - 0.30) / 1_000_000):.4f}</td>
      </tr>
      <tr>
        <td>BP2 — System prompt</td>
        <td class="hot">HOT</td>
        <td>{BP2_TOKENS:,}</td>
        <td>60 min (extended)</td>
        <td>$3.75</td><td>$0.30</td>
        <td>${(BP2_TOKENS * (3.00 - 0.30) / 1_000_000):.4f}</td>
      </tr>
      <tr>
        <td>BP3 — loci_context HOT</td>
        <td class="hot">HOT</td>
        <td>{BP3_HOT_TOKENS:,}</td>
        <td>60 min (extended)</td>
        <td>$3.75</td><td>$0.30</td>
        <td>${(BP3_HOT_TOKENS * (3.00 - 0.30) / 1_000_000):.4f}</td>
      </tr>
      <tr>
        <td>BP3 — loci_context MILD</td>
        <td class="mild">MILD</td>
        <td>{BP3_MILD_TOKENS:,}</td>
        <td>5 min (ephemeral)</td>
        <td>$3.75</td><td>$0.30</td>
        <td>${(BP3_MILD_TOKENS * (3.00 - 0.30) / 1_000_000):.4f}</td>
      </tr>
      <tr>
        <td>BP4 — User turn</td>
        <td class="cold">COLD</td>
        <td>~{USER_PER_TURN:,}+history</td>
        <td>not cached</td>
        <td>—</td><td>—</td><td>—</td>
      </tr>
    </tbody>
  </table>
</div>

<div class="section">
  <h2>Model comparison — 20-turn session total (USD)</h2>
  <table>
    <thead><tr>
      <th>Model</th><th>Input</th><th>Cache read</th><th>Cache write</th>
      <th>Output</th><th>Total</th><th>vs Claude</th>
    </tr></thead>
    <tbody>
      <tr>
        <td>Claude Sonnet 4.6 + LOCI cache</td>
        <td>${claude.total_input_cost:.4f}</td>
        <td>${claude.total_read_cost:.4f}</td>
        <td>${claude.total_write_cost:.4f}</td>
        <td>${claude.total_output_cost:.4f}</td>
        <td class="blue"><strong>${claude.total:.4f}</strong></td>
        <td>baseline</td>
      </tr>
      <tr>
        <td>Claude Sonnet 4.6 (no cache)</td>
        <td>${nocache.total_input_cost:.4f}</td>
        <td>—</td><td>—</td>
        <td>${nocache.total_output_cost:.4f}</td>
        <td class="hot"><strong>${nocache.total:.4f}</strong></td>
        <td class="hot">+{100*(nocache.total/claude.total - 1):.0f}%</td>
      </tr>
      <tr>
        <td>GPT-4o (OpenAI prompt cache)</td>
        <td>${gpt4o.total_input_cost:.4f}</td>
        <td>${gpt4o.total_read_cost:.4f}</td>
        <td>—</td>
        <td>${gpt4o.total_output_cost:.4f}</td>
        <td class="mild"><strong>${gpt4o.total:.4f}</strong></td>
        <td class="mild">{'+' if gpt4o.total > claude.total else ''}{100*(gpt4o.total/claude.total - 1):.0f}%</td>
      </tr>
    </tbody>
  </table>
</div>

{live_section}

<div class="section">
  <h2>What "how much do I carry" means</h2>
  <p style="color:var(--muted); font-size:12px; line-height:1.8">
    <strong style="color:var(--text)">Token carry</strong> = fraction of per-turn input tokens served from Claude's prompt cache.<br>
    In this {len(claude.turns)}-turn IDE session: BP1 tools ({BP1_TOKENS:,} tok) + BP2 system ({BP2_TOKENS:,} tok) +
    BP3-HOT loci_context ({BP3_HOT_TOKENS:,} tok) = <strong style="color:var(--hot)">{BP1_TOKENS+BP2_TOKENS+BP3_HOT_TOKENS:,} tokens</strong>
    stay in extended (1-hr) cache — the "hot teachers" (Arc memory + Codex code blocks with heat ≥ 0.80).<br><br>
    By turn 10, ~{carry_pct:.0f}% of every Claude API call's input tokens are served at $0.30/M instead of $3.00/M.<br>
    That's a <strong style="color:var(--blue)">{savings_pct:.0f}% cost reduction</strong> on input tokens for the fixed context.
  </p>
</div>

<script>
const W = (c) => c.getContext('2d').canvas.offsetWidth;

function sparkline(id, data, color) {{
  const canvas = document.getElementById(id);
  const ctx    = canvas.getContext('2d');
  canvas.width  = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
  const w = canvas.width, h = canvas.height;
  const max = Math.max(...data), min = Math.min(...data);
  const range = max - min || 1;
  ctx.clearRect(0, 0, w, h);
  // grid
  ctx.strokeStyle = '#21262d'; ctx.lineWidth = 1;
  [0, 25, 50, 75, 100].forEach(v => {{
    const y = h - ((v - min) / range) * (h - 20) - 10;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    ctx.fillStyle = '#8b949e'; ctx.font = '10px monospace';
    ctx.fillText(v + '%', 2, y - 2);
  }});
  // line
  ctx.strokeStyle = color; ctx.lineWidth = 2;
  ctx.beginPath();
  data.forEach((v, i) => {{
    const x = (i / (data.length - 1)) * w;
    const y = h - ((v - min) / range) * (h - 20) - 10;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }});
  ctx.stroke();
  // dots
  data.forEach((v, i) => {{
    const x = (i / (data.length - 1)) * w;
    const y = h - ((v - min) / range) * (h - 20) - 10;
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#8b949e'; ctx.font = '9px monospace';
    ctx.fillText(v.toFixed(0), x - 8, y - 6);
  }});
}}

function stackedBar(id, labels, datasets) {{
  const canvas = document.getElementById(id);
  const ctx    = canvas.getContext('2d');
  canvas.width  = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
  const w = canvas.width, h = canvas.height;
  const pad = 30, bw = (w - pad) / labels.length - 2;
  // find max total
  const totals = labels.map((_, i) => datasets.reduce((s, d) => s + d.data[i], 0));
  const maxT = Math.max(...totals);
  datasets.forEach(ds => {{
    ds._base = new Array(labels.length).fill(0);
  }});
  // accumulate bases
  for (let di = datasets.length - 1; di >= 0; di--) {{
    if (di < datasets.length - 1) {{
      datasets[di]._base = datasets[di+1]._base.map((b, i) => b + datasets[di+1].data[i]);
    }}
  }}
  ctx.clearRect(0, 0, w, h);
  datasets.forEach(ds => {{
    ctx.fillStyle = ds.color;
    labels.forEach((_, i) => {{
      const x  = pad + i * ((w - pad) / labels.length);
      const bh = (ds.data[i] / maxT) * (h - 20);
      const by = h - ((ds._base[i] / maxT) * (h - 20)) - bh;
      ctx.fillRect(x, by, bw, bh);
    }});
  }});
  // x labels
  ctx.fillStyle = '#8b949e'; ctx.font = '9px monospace';
  labels.forEach((l, i) => {{
    const x = pad + i * ((w - pad) / labels.length) + bw / 2 - 4;
    ctx.fillText(l, x, h - 2);
  }});
}}

sparkline('carryChart', {carry_json}, '#ff6b35');
stackedBar('costChart',
  {turn_labels},
  [
    {{ color: '#ff6b35', data: {turn_input},  label: 'fresh' }},
    {{ color: '#06d6a0', data: {turn_read},   label: 'cache_read' }},
    {{ color: '#ffd166', data: {turn_write},  label: 'cache_write' }},
  ]
);
</script>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="LOCI cache benchmark")
    ap.add_argument("--live",    action="store_true", help="make real API calls")
    ap.add_argument("--turns",   type=int, default=20, help="number of turns to simulate")
    ap.add_argument("--vis",     action="store_true", help="write HTML visual panel")
    ap.add_argument("--ledger",  action="store_true", help="append results to savings ledger")
    args = ap.parse_args()

    print("=" * 70)
    print("  LOCI Cache Benchmark — 'how much do you carry, hot teachers cache'")
    print("=" * 70)
    print(f"\n  Session profile: {args.turns} turns, 20-45 min IDE session")
    print(f"  LOCI breakpoints: BP1({BP1_TOKENS:,}t) BP2({BP2_TOKENS:,}t) "
          f"BP3-HOT({BP3_HOT_TOKENS:,}t) BP3-MILD({BP3_MILD_TOKENS:,}t) BP4(fresh)")

    profiles = build_turn_profiles(n_turns=args.turns)

    sessions = {
        "claude-sonnet-4-6":         calc_session("claude-sonnet-4-6", profiles, has_cache=True),
        "claude-sonnet-4-6-nocache": calc_session("claude-sonnet-4-6-nocache", profiles, has_cache=False),
        "gpt-4o":                    calc_session("gpt-4o", profiles, has_cache=True),
    }

    # Live API
    live_data = None
    if args.live:
        print("\n  Running live API turns...")
        live_data = run_live(n_turns=min(6, args.turns))

    # Print results table
    print(f"\n  {'Model':<35} {'Total':>8}  {'Carry':>7}  {'Saved':>8}")
    print(f"  {'-'*35} {'-'*8}  {'-'*7}  {'-'*8}")

    claude  = sessions["claude-sonnet-4-6"]
    nocache = sessions["claude-sonnet-4-6-nocache"]
    gpt4o   = sessions["gpt-4o"]

    for key, label, marker in [
        ("claude-sonnet-4-6",         "Claude Sonnet 4.6 + LOCI",    "→"),
        ("claude-sonnet-4-6-nocache", "Claude Sonnet 4.6 (no cache)", " "),
        ("gpt-4o",                    "GPT-4o + OpenAI cache",        " "),
    ]:
        s = sessions[key]
        saved = nocache.total - s.total
        print(f"  {marker} {label:<33} ${s.total:>7.4f}  {100*s.avg_carry:>5.0f}%  "
              f"${saved:>7.4f}")

    print()
    print(f"  HOT cache (extended 1hr)  : {BP1_TOKENS+BP2_TOKENS+BP3_HOT_TOKENS:,} tokens "
          f"(tools + system + arc/codex blocks)")
    print(f"  MILD cache (ephemeral 5m) : {BP3_MILD_TOKENS:,} tokens "
          f"(claude reasoning + copilot blocks)")
    print(f"  Average carry ratio       : {100*claude.avg_carry:.0f}% of input tokens per turn from cache")
    print(f"  Peak carry ratio (turn 10): {100*claude.turns[9].carry_ratio:.0f}%")
    print(f"  USD saved vs no-cache     : ${nocache.total - claude.total:.4f} "
          f"({100*(nocache.total-claude.total)/nocache.total:.0f}%)")

    # Live breakdown
    if live_data:
        print(f"\n  Live API — {len(live_data)} turns:")
        total_read = sum(r["cache_read_input_tokens"] for r in live_data)
        total_inp  = sum(r["prompt_tokens"] for r in live_data)
        total_all  = total_read + total_inp + sum(r["cache_creation_input_tokens"] for r in live_data)
        print(f"    cache_read_input_tokens : {total_read:,}")
        print(f"    fresh_input_tokens      : {total_inp:,}")
        print(f"    live carry ratio        : {100*total_read/max(total_all,1):.0f}%")

    # Write ledger
    if args.ledger or args.live:
        records = []
        ts = datetime.now(timezone.utc).isoformat()
        for t, p in zip(claude.turns, profiles):
            records.append({
                "schema": "loci.token_savings.v1",
                "timestamp": ts,
                "model": "claude-sonnet-4-6",
                "turn": t.turn,
                "source": "benchmark.analytical",
                "cache_read_tokens": p.cached_extended + p.cached_ephemeral,
                "cache_write_tokens": p.cache_write,
                "fresh_input_tokens": p.fresh_input,
                "output_tokens": p.output_tokens,
                "cost_usd": t.total,
                "carry_ratio": round(t.carry_ratio, 4),
                "saved_usd": round(
                    nocache.turns[t.turn - 1].total - t.total, 6
                ),
            })
        if live_data:
            for r in live_data:
                records.append({
                    "schema": "loci.token_savings.v1",
                    "timestamp": ts,
                    "model": "claude-sonnet-4-6",
                    "turn": r["turn"],
                    "source": "benchmark.live",
                    "cache_read_tokens": r["cache_read_input_tokens"],
                    "cache_write_tokens": r["cache_creation_input_tokens"],
                    "fresh_input_tokens": r["prompt_tokens"],
                    "output_tokens": r["completion_tokens"],
                })
        ledger = _write_ledger(records)
        print(f"\n  Ledger: {ledger} (+{len(records)} records)")

    # HTML panel
    if args.vis:
        html = _render_html(sessions, live_data)
        out = Path(__file__).parent / "cache_benchmark_results.html"
        out.write_text(html)
        print(f"  Visual: {out}")

    print()


if __name__ == "__main__":
    main()
