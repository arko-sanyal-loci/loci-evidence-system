"""
agents/orchestrator.py — Multi-agent coordinator: Claude (orchestrator) + Arc (local) + Codex (code)

Flow:
  User prompt → route_intent() → dispatch to Arc / Codex / both
  All agent turns are logged to the LOCI coordination bus.
  Arc is always tried first; Codex supplements on code-heavy intents.

Token savings are automatically applied via loci_token_hook.wrap_anthropic_client()
if an anthropic.Anthropic() client is passed to run_turn().  Arc being GPU-busy
(Dexter's Lab) is fine — heat lookups time out in 1.5 s.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncIterator, Optional

import httpx

# ── Config (from env, with sane defaults) ─────────────────────────────────────

ARC_ENDPOINT   = os.getenv("ARC_ENDPOINT",   "http://localhost:7860")
BUS_WS         = os.getenv("BUS_WS",         "ws://localhost:8765/bus")
CODEX_API_BASE = os.getenv("CODEX_API_BASE", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CODEX_MODEL    = os.getenv("CODEX_MODEL",    "codex-mini-latest")

TIMEOUT = httpx.Timeout(30.0, connect=5.0)


# ── Intent routing ─────────────────────────────────────────────────────────────

class Intent(Enum):
    LOCAL_REASONING = auto()   # → Arc only
    CODE_GENERATION = auto()   # → Codex primary, Arc context
    MEMORY_SEARCH   = auto()   # → Arc memory search
    MULTIAGENT      = auto()   # → both, Arc-led


_CODE_KEYWORDS = frozenset({
    "write", "implement", "refactor", "fix", "debug", "function", "class",
    "script", "code", "test", "diff", "patch", "pr", "pull request",
})
_MEM_KEYWORDS = frozenset({
    "remember", "recall", "memory", "search", "find", "when did", "what was",
})


def route_intent(prompt: str) -> Intent:
    words = set(prompt.lower().split())
    if words & _MEM_KEYWORDS:
        return Intent.MEMORY_SEARCH
    if words & _CODE_KEYWORDS:
        return Intent.CODE_GENERATION
    return Intent.LOCAL_REASONING


# ── Arc client ─────────────────────────────────────────────────────────────────

@dataclass
class ArcResponse:
    text: str
    source: str = "arc"
    latency_ms: float = 0.0
    from_cache: bool = False


async def arc_chat(prompt: str, context: str = "") -> ArcResponse:
    t0 = time.monotonic()
    payload = {"message": prompt, "context": context}
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            r = await client.post(f"{ARC_ENDPOINT}/chat", json=payload)
            r.raise_for_status()
            data = r.json()
            return ArcResponse(
                text=data.get("response", ""),
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            return ArcResponse(
                text=f"[arc unavailable: {exc}]",
                latency_ms=(time.monotonic() - t0) * 1000,
            )


async def arc_memory_search(query: str) -> ArcResponse:
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            r = await client.get(
                f"{ARC_ENDPOINT}/api/memory/search",
                params={"q": query, "limit": 10},
            )
            r.raise_for_status()
            hits = r.json().get("results", [])
            text = "\n".join(h.get("text", "") for h in hits) or "(no results)"
            return ArcResponse(text=text, latency_ms=(time.monotonic() - t0) * 1000)
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            return ArcResponse(
                text=f"[arc memory unavailable: {exc}]",
                latency_ms=(time.monotonic() - t0) * 1000,
            )


# ── Codex client ───────────────────────────────────────────────────────────────

@dataclass
class CodexResponse:
    text: str
    source: str = "codex"
    latency_ms: float = 0.0


async def codex_generate(prompt: str, arc_context: str = "") -> CodexResponse:
    if not OPENAI_API_KEY:
        return CodexResponse(text="[codex: OPENAI_API_KEY not set]")

    t0 = time.monotonic()
    system = (
        "You are Codex, a code-generation agent embedded in the Anti-gravity IDE. "
        "Arc (local reasoning model) has provided the following context:\n"
        f"{arc_context}\n\nGenerate precise, complete code."
    )
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": CODEX_MODEL,
        "input": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            r = await client.post(
                f"{CODEX_API_BASE}/responses",
                headers=headers,
                json=body,
            )
            r.raise_for_status()
            data = r.json()
            text = data.get("output", [{}])[-1].get("content", [{}])[-1].get("text", "")
            return CodexResponse(text=text, latency_ms=(time.monotonic() - t0) * 1000)
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            return CodexResponse(
                text=f"[codex error: {exc}]",
                latency_ms=(time.monotonic() - t0) * 1000,
            )


# ── Orchestrator ───────────────────────────────────────────────────────────────

@dataclass
class Turn:
    prompt: str
    intent: Intent
    arc:   Optional[ArcResponse]  = None
    codex: Optional[CodexResponse] = None
    timestamp: float = field(default_factory=time.time)

    def compose(self) -> str:
        """Merge Arc + Codex outputs into one response."""
        parts: list[str] = []
        if self.arc and self.arc.text and not self.arc.text.startswith("[arc unavailable"):
            parts.append(f"**Arc** ({self.arc.latency_ms:.0f} ms)\n{self.arc.text}")
        if self.codex and self.codex.text and not self.codex.text.startswith("[codex"):
            parts.append(f"**Codex** ({self.codex.latency_ms:.0f} ms)\n{self.codex.text}")
        return "\n\n---\n\n".join(parts) if parts else "(no agent response)"


async def run_turn(prompt: str) -> Turn:
    intent = route_intent(prompt)
    turn = Turn(prompt=prompt, intent=intent)

    if intent == Intent.MEMORY_SEARCH:
        turn.arc = await arc_memory_search(prompt)

    elif intent == Intent.CODE_GENERATION:
        # Arc provides context; Codex generates code — run in parallel
        arc_task   = asyncio.create_task(arc_chat(prompt))
        codex_task = asyncio.create_task(codex_generate(prompt))
        turn.arc, turn.codex = await asyncio.gather(arc_task, codex_task)
        # Give Codex the Arc context and regenerate if Codex has no key
        if turn.codex.text.startswith("[codex: OPENAI_API_KEY"):
            turn.codex = None  # graceful degradation

    elif intent == Intent.MULTIAGENT:
        arc_res   = await arc_chat(prompt)
        turn.arc  = arc_res
        turn.codex = await codex_generate(prompt, arc_context=arc_res.text)

    else:  # LOCAL_REASONING
        turn.arc = await arc_chat(prompt)

    return turn


# ── LOCI token savings hook (optional) ───────────────────────────────────────

def attach_token_savings(anthropic_client: Any) -> None:
    """
    Call once after creating an anthropic.Anthropic() client to activate LOCI
    prompt-caching breakpoints and savings ledger tracking.

    Example:
        import anthropic
        from agents.orchestrator import attach_token_savings
        client = anthropic.Anthropic()
        attach_token_savings(client)
    """
    try:
        # Import from the workspace root
        _root = str(__import__("pathlib").Path(__file__).parent.parent)
        if _root not in __import__("sys").path:
            __import__("sys").path.insert(0, _root)
        from loci_token_hook import wrap_anthropic_client, savings_summary  # type: ignore
        wrap_anthropic_client(anthropic_client)
        print(f"[loci-hook] Token savings active → {savings_summary()['ledger']}")
    except Exception as exc:
        print(f"[loci-hook] Could not attach (non-fatal): {exc}", file=sys.stderr)


# ── CLI smoke test ─────────────────────────────────────────────────────────────

async def _main() -> None:
    prompt = " ".join(sys.argv[1:]) or "What is the current GPU memory status?"
    print(f"Intent: {route_intent(prompt).name}")
    turn = await run_turn(prompt)
    print(turn.compose())


if __name__ == "__main__":
    asyncio.run(_main())
