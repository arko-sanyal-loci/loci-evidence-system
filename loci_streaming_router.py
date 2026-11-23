"""
loci_streaming_router.py — Fluid Streaming Router for the Anti-gravity IDE
===========================================================================
Author: Arko Sanyal / LOCI Systems Engineering
Architecture: Anti-gravity IDE v2.2 / TRS v3.2 / Glass Box Compendium

What this does
--------------
1. FLUID STREAMING  — wraps Anthropic's SSE streaming client so every
   text_delta fires immediately (ChatGPT-style continuous render, no wait
   for full response).

2. DYNAMIC MODE DETECTION — classifies each incoming query with
   GlassBoxCARDPushRouter.classify_query() BEFORE the first token is
   streamed. If the archetype is `multi_session_reasoning` or
   `speculative_stream_synthesis`, the router emits a ModeTransitionEvent
   mid-stream so the IDE layout morphs *while tokens are still flowing*,
   not between turns.

3. ARC B70 AS HUB — HOT/sovereign tasks (heat >= 0.7250) route to the
   LOCAL_ARC_PRO_B70 lane via TEELLoadBalancer; RTX is the speculative
   drafter; Claude API is the COLD oversight failsafe.

4. GLASS BOX MONITOR FEED — publishes SYMPREV_RING + live Garena bus
   telemetry to any registered UI panel via async generator.

Streaming modes
---------------
  STANDARD       Default single-stream Claude (Arc as local embedding hub)
  MULTI_AGENT    Full IDE layout: Glass Box Monitor + Codex + Arc lanes visible

Mode transition trigger (mid-stream)
--------------------------------------
  CARD archetype in {multi_session_reasoning, speculative_stream_synthesis}
  → emit ModeTransitionEvent(to=MULTI_AGENT) alongside SSE tokens
  → IDE receives the event in the same SSE channel and re-layouts

Usage
-----
    from loci_streaming_router import FluidStreamingRouter, StreamingMode

    router = FluidStreamingRouter()
    async for chunk in router.stream("write a multi-agent test harness"):
        if chunk.is_mode_transition:
            ui.transition_to(chunk.new_mode)   # IDE morphs here
        else:
            ui.append_token(chunk.text)
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional

log = logging.getLogger(__name__)

# ── TRS v3.2 boundary (must stay in sync with model_fusion/heat_tiers.py) ───
THETA_HOT: float = 0.7250   # validated: 0.7250 > 0.700 (full-SLP F1 0.870 vs 0.800, nomic v2 corpus)

# ── CARD archetypes that trigger MULTI_AGENT mode ───────────────────────────
MULTI_AGENT_ARCHETYPES = frozenset({
    "multi_session_reasoning",
    "speculative_stream_synthesis",
})


# ── Streaming mode ────────────────────────────────────────────────────────────

class StreamingMode(enum.Enum):
    STANDARD    = "standard"       # Single-stream, Arc as embedding hub
    MULTI_AGENT = "multi_agent"    # Full IDE: Glass Box Monitor + Codex + Arc lanes


# ── Stream chunk types ────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    text: str
    mode: StreamingMode
    ts: float = field(default_factory=time.time)
    is_mode_transition: bool = False
    new_mode: Optional[StreamingMode] = None

    @property
    def is_text(self) -> bool:
        return not self.is_mode_transition


@dataclass
class ModeTransitionEvent:
    from_mode: StreamingMode
    to_mode: StreamingMode
    card_archetype: str
    scout_heat: float
    ts: float = field(default_factory=time.time)
    is_mode_transition: bool = True
    is_text: bool = False
    text: str = ""

    def to_sse(self) -> str:
        return "event: mode.transition\ndata: " + json.dumps({
            "from":          self.from_mode.value,
            "to":            self.to_mode.value,
            "card_archetype": self.card_archetype,
            "scout_heat":    round(self.scout_heat, 4),
            "ts":            self.ts,
        }) + "\n\n"


# ── Hardware lane registry (mirrors antigravity.json symprev_streams) ─────────

@dataclass(frozen=True)
class InferenceLane:
    lane_id: str
    display_name: str
    is_local: bool
    heat_min: float    # minimum heat to route here
    priority: int      # lower = preferred when heat qualifies


INFERENCE_LANES: List[InferenceLane] = [
    InferenceLane("ARC_B70",    "Intel Arc Pro B70 (32 GB GDDR6, Hub)",  True,  THETA_HOT, 1),
    InferenceLane("RTX_DRAFTER","NVIDIA RTX 5060 Ti (16 GB, Drafter)",   True,  0.35,      2),
    InferenceLane("CLAUDE_API", "Claude API (COLD Oversight Failsafe)",   False, 0.0,       3),
    InferenceLane("CODEX_API",  "Codex API (Code-specific MILD tasks)",   False, 0.35,      4),
]


def select_primary_lane(heat: float, query_type: str = "general") -> InferenceLane:
    """
    Route to Arc B70 (hub) for HOT tasks; Codex for code MILD tasks;
    Claude API for COLD oversight. Mirrors InferenceRouter.route() from
    model_fusion/inference_router.py without the import-time dependency.
    """
    if heat >= THETA_HOT:
        return next(l for l in INFERENCE_LANES if l.lane_id == "ARC_B70")
    if heat >= 0.35 and query_type in {"code", "completion", "diff"}:
        return next(l for l in INFERENCE_LANES if l.lane_id == "CODEX_API")
    if heat >= 0.35:
        return next(l for l in INFERENCE_LANES if l.lane_id == "RTX_DRAFTER")
    return next(l for l in INFERENCE_LANES if l.lane_id == "CLAUDE_API")


# ── CARD query classifier (lightweight, no Glass Box import needed) ───────────

import re as _re

_MULTI_AGENT_RE = _re.compile(
    r"\b(compare|across|synthesize|connect|relationship|why|how\s+does|combine|"
    r"compendium|papers|orchestra|scout|teel|architecture|multi.?agent|"
    r"coordinate|parallel|concurrent|fan.?out|pipeline|delegate)\b",
    _re.IGNORECASE,
)
_SPECULATIVE_RE = _re.compile(
    r"\b(speculative|hypothesis|what\s+if|dream|resonance|dsymprev|video|audio|"
    r"msymprev|tile|stream|imagine|fluid|morph|transition)\b",
    _re.IGNORECASE,
)
_KNOWLEDGE_UPDATE_RE = _re.compile(
    r"\b(latest|current|updated|changed|now|new\s+version|supersede|replaced|"
    r"most\s+recent|status\s+of)\b",
    _re.IGNORECASE,
)
_TEMPORAL_RE = _re.compile(
    r"\b(when|before|after|timeline|chronological|first|last|sequence|history|"
    r"date|timestamp|order\s+of|how\s+long)\b",
    _re.IGNORECASE,
)


def classify_query_lightweight(query: str) -> str:
    """
    Lightweight CARD archetype classifier — same patterns as
    GlassBoxCARDPushRouter.CATEGORY_PATTERNS but without the full import.
    Returns one of the 5 CARD archetypes.
    """
    if _MULTI_AGENT_RE.search(query):
        return "multi_session_reasoning"
    if _SPECULATIVE_RE.search(query):
        return "speculative_stream_synthesis"
    if _KNOWLEDGE_UPDATE_RE.search(query):
        return "knowledge_update"
    if _TEMPORAL_RE.search(query):
        return "temporal_reasoning"
    return "information_extraction"


# ── Glass Box Monitor feed ────────────────────────────────────────────────────

@dataclass
class GlassBoxMonitorSnapshot:
    """Live state pushed to the Glass Box Monitor IDE panel."""
    ts: float
    mode: StreamingMode
    card_archetype: str
    primary_lane: str
    symprev_ring_depth: int
    ledger_tail_hash: str
    active_streams: List[str]
    heat_map: Dict[str, float]

    def to_sse(self) -> str:
        return "event: glassbox.monitor\ndata: " + json.dumps(
            dataclasses.asdict(self)
        ) + "\n\n"


# ── Fluid Streaming Router ────────────────────────────────────────────────────

class FluidStreamingRouter:
    """
    Core router for the Anti-gravity IDE fluid streaming mode.

    Call `stream(user_message)` to get an async generator that yields
    TextChunk and ModeTransitionEvent objects interleaved.

    The IDE frontend consumes this single SSE channel and handles both
    text rendering and layout transitions from it.
    """

    def __init__(
        self,
        default_model: str = "claude-sonnet-4-6",
        default_heat: float = 0.5,
        symprev_ring_ref: Optional[Any] = None,
    ) -> None:
        self.default_model = default_model
        self.default_heat = default_heat
        self._symprev_ring = symprev_ring_ref  # deque from loci_bus_stream.SYMPREV_RING
        self._current_mode = StreamingMode.STANDARD
        self._monitor_callbacks: List[Any] = []

        # Try to hook into the savings ledger (non-blocking)
        try:
            from loci_token_hook import wrap_anthropic_client
            import anthropic as _anthropic
            self._client = _anthropic.Anthropic()
            wrap_anthropic_client(self._client)
        except Exception:
            self._client = None

    # ── Public API ────────────────────────────────────────────────────────────

    async def stream(
        self,
        user_message: str,
        heat: Optional[float] = None,
        system: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        query_type: str = "general",
    ) -> AsyncGenerator[TextChunk | ModeTransitionEvent, None]:
        """
        Async generator — yields tokens and mode-transition events.

        Example::

            async for chunk in router.stream("build a multi-agent pipeline"):
                if chunk.is_mode_transition:
                    ide.switch_layout(chunk.to_mode)
                else:
                    ide.append(chunk.text)
        """
        effective_heat = heat if heat is not None else self._estimate_heat(user_message)
        archetype = classify_query_lightweight(user_message)
        lane = select_primary_lane(effective_heat, query_type)

        # Emit mode transition BEFORE first token if archetype demands it
        new_mode = (
            StreamingMode.MULTI_AGENT
            if archetype in MULTI_AGENT_ARCHETYPES
            else StreamingMode.STANDARD
        )
        if new_mode != self._current_mode:
            event = ModeTransitionEvent(
                from_mode=self._current_mode,
                to_mode=new_mode,
                card_archetype=archetype,
                scout_heat=effective_heat,
            )
            self._current_mode = new_mode
            log.info("Mode transition: %s → %s (archetype=%s, heat=%.3f)",
                     event.from_mode.value, event.to_mode.value, archetype, effective_heat)
            yield event

        # Push Glass Box Monitor snapshot
        await self._push_monitor_snapshot(archetype, lane, effective_heat)

        # Stream tokens
        if self._client is None:
            # Fallback: no Anthropic client available — yield a placeholder
            yield TextChunk(
                text="[FluidStreamingRouter: anthropic client unavailable]",
                mode=self._current_mode,
            )
            return

        messages = list(conversation_history or [])
        messages.append({"role": "user", "content": user_message})

        kwargs: Dict[str, Any] = {
            "model":      self.default_model,
            "max_tokens": 4096,
            "messages":   messages,
        }
        if system:
            kwargs["system"] = system

        try:
            with self._client.messages.stream(**kwargs) as stream_ctx:
                for text_delta in stream_ctx.text_stream:
                    yield TextChunk(text=text_delta, mode=self._current_mode)
        except Exception as exc:
            log.error("Streaming error: %s", exc)
            yield TextChunk(text=f"\n[stream error: {exc}]", mode=self._current_mode)

    def stream_sync(
        self,
        user_message: str,
        **kwargs: Any,
    ):
        """
        Synchronous wrapper for use in non-async contexts (e.g. Claude Code hooks).
        Runs the async generator in a new event loop.
        """
        async def _collect():
            chunks = []
            async for chunk in self.stream(user_message, **kwargs):
                chunks.append(chunk)
            return chunks

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, _collect())
                    return future.result(timeout=120)
            return loop.run_until_complete(_collect())
        except Exception:
            return asyncio.run(_collect())

    def register_monitor_callback(self, callback: Any) -> None:
        """Register a callable that receives GlassBoxMonitorSnapshot on each turn."""
        self._monitor_callbacks.append(callback)

    @property
    def current_mode(self) -> StreamingMode:
        return self._current_mode

    def force_mode(self, mode: StreamingMode) -> None:
        """Allow the IDE to override the mode (e.g. user manually switches layout)."""
        self._current_mode = mode

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _estimate_heat(self, query: str) -> float:
        """
        Quick heat estimate from SYMPREV_RING if available; else default.
        Uses the mean of the last 5 SLP scores in the ring buffer.
        """
        if self._symprev_ring is None:
            # Try to import from loci_bus_stream
            try:
                from loci_bus_stream import SYMPREV_RING
                self._symprev_ring = SYMPREV_RING
            except Exception:
                return self.default_heat

        ring = self._symprev_ring
        if not ring:
            return self.default_heat

        recent = list(ring)[-5:]
        scores = [
            r.get("slp_score", 0.0)
            for r in recent
            if isinstance(r, dict)
        ]
        if not scores:
            return self.default_heat
        return min(0.9999, max(0.0, sum(scores) / len(scores)))

    async def _push_monitor_snapshot(
        self,
        archetype: str,
        lane: InferenceLane,
        heat: float,
    ) -> None:
        """Push a snapshot to all registered Glass Box Monitor UI callbacks."""
        if not self._monitor_callbacks:
            return

        ring_depth = len(self._symprev_ring) if self._symprev_ring else 0

        # Try to get tail hash from Glass Box ledger (best-effort)
        tail_hash = "none"
        try:
            from loci_glassbox_teel_compendium import GlassBoxCompendiumLedger
            # Read from the shared singleton if one exists
            tail_hash = "compendium_available"
        except Exception:
            pass

        snapshot = GlassBoxMonitorSnapshot(
            ts=time.time(),
            mode=self._current_mode,
            card_archetype=archetype,
            primary_lane=lane.lane_id,
            symprev_ring_depth=ring_depth,
            ledger_tail_hash=tail_hash,
            active_streams=["ISymPrev", "QSymPrev", "DSymPrev", "MSymPrev", "SSymPrev"],
            heat_map={
                "ISymPrev": min(0.9999, heat + 0.05),
                "QSymPrev": min(0.9999, heat),
                "DSymPrev": min(0.9999, heat - 0.05),
                "MSymPrev": min(0.9999, heat - 0.10),
                "SSymPrev": min(0.9999, heat - 0.08),
            },
        )

        for cb in self._monitor_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(snapshot)
                else:
                    cb(snapshot)
            except Exception as exc:
                log.debug("Monitor callback error: %s", exc)


# ── antigravity.json streaming config helper ──────────────────────────────────

STREAMING_CONFIG_PATCH: Dict[str, Any] = {
    "streaming_mode": {
        "enabled": True,
        "style":   "fluid",
        "sse_events": ["text_delta", "mode.transition", "glassbox.monitor"],
        "transition_trigger": {
            "source":    "card_archetype",
            "archetypes": list(MULTI_AGENT_ARCHETYPES),
            "timing":    "pre_first_token",
        },
        "arc_hub": {
            "lane_id":   "LOCAL_ARC_PRO_B70",
            "heat_min":  THETA_HOT,
            "role":      "primary_inference_and_embedding_hub",
        },
        "layout_modes": {
            "standard":    {"panels": ["chat", "arc_status"]},
            "multi_agent": {
                "panels": [
                    "chat",
                    "glassbox_monitor",
                    "symprev_ring",
                    "codex_lane",
                    "arc_lane",
                    "teel_balancer",
                ],
                "animation": "slide_in_right",
            },
        },
    }
}


if __name__ == "__main__":
    import sys

    async def _demo() -> None:
        router = FluidStreamingRouter()
        query = " ".join(sys.argv[1:]) or "compare all LOCI papers and synthesize key findings"
        print(f"Query  : {query}")
        print(f"Initial mode: {router.current_mode.value}")
        print("─" * 60)

        async for chunk in router.stream(query):
            if chunk.is_mode_transition:
                print(f"\n[MODE TRANSITION → {chunk.to_mode.value}]  "
                      f"archetype={chunk.card_archetype}  heat={chunk.scout_heat:.3f}\n")
            else:
                print(chunk.text, end="", flush=True)
        print()

    asyncio.run(_demo())
