"""
model_fusion/inference_router.py — Heat-tiered inference weight routing.

Extends arc/core/speculative_pipeline.py from two GPUs to five model
endpoints (Arc-70B · RTX-local · Claude-API · Codex-API · Copilot-LSP).

Heat tier → route decision
──────────────────────────
  SOVEREIGN  H → 1.0   Arc-70B local only, sovereignty lock, k=1 draft token
  HOT        H ≥ 0.80  Arc-B70 (deep lane) + RTX drafter (fast lane), k=8
  MILD       H ≥ 0.35  Parallel: Codex + Arc-small speculative,         k=4
  COLD       H <  0.35 Claude API (cheapest, no GPU required),           k=2

LoRA adapter rank is chosen per-layer by the same heat budget:
  HOT r=16 · MILD r=8 · COLD r=1

The router is a pure data-flow object; it does NOT make HTTP calls itself.
It returns an InferenceRoute that the orchestrator dispatches.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from model_fusion.heat_tiers import (
    HeatTier, classify, lora_rank, draft_budget,
    VRAM_ADMISSION_THRESHOLD,
)


# ── Endpoint constants ────────────────────────────────────────────────────────

class Endpoint:
    ARC_70B       = "arc-70b-local"       # vLLM on Arc B70, full 70B weights
    ARC_SMALL     = "arc-small-local"     # vLLM on Arc B70, 7B/13B weights
    RTX_DRAFTER   = "rtx-5060ti-draft"    # RTX 5060 Ti speculative drafter
    CLAUDE_API    = "claude-api"          # Anthropic API (token savings hook)
    CODEX_API     = "codex-api"           # OpenAI Codex via Responses API
    COPILOT_LSP   = "copilot-lsp"         # GitHub Copilot LSP (completions only)
    LOCAL_METAL   = "local-metal-arc"     # generic vLLM on Intel Arc B70


# ── Route result ──────────────────────────────────────────────────────────────

@dataclass
class InferenceRoute:
    """
    Decision output of InferenceRouter.route().

    primary_endpoint   : model that generates the final response
    drafter_endpoint   : speculative drafter (None if no speculation)
    draft_tokens       : Dovetail draft-token budget for this query
    lora_ranks         : {layer_name: rank} for active LoRA adapters
    tier               : HeatTier classification of this query
    heat               : raw heat score
    sovereign_lock     : True → no network calls, Arc-70B only
    rationale          : human-readable routing explanation
    routed_at          : unix timestamp
    """
    primary_endpoint:  str
    drafter_endpoint:  Optional[str]
    draft_tokens:      int
    lora_ranks:        dict[str, int]
    tier:              HeatTier
    heat:              float
    sovereign_lock:    bool = False
    rationale:         str  = ""
    routed_at:         float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "primary":    self.primary_endpoint,
            "drafter":    self.drafter_endpoint,
            "k":          self.draft_tokens,
            "tier":       self.tier.name,
            "heat":       round(self.heat, 4),
            "sovereign":  self.sovereign_lock,
            "rationale":  self.rationale,
        }


# ── Router ────────────────────────────────────────────────────────────────────

class InferenceRouter:
    """
    Maps a query's heat score + intent to the optimal inference route.

    layer_heats: dict of {layer_name: heat_score} for the current query,
                 used to set per-layer LoRA ranks.  Can be empty; defaults
                 to tier-uniform ranks.

    Usage:
        router = InferenceRouter()
        route  = router.route(heat=0.85, query_type="code")
        # → primary=arc-70b-local, drafter=rtx-5060ti-draft, k=8, r=16
    """

    # Default layer set (matches arc/core/tiered_memory.py naming)
    _DEFAULT_LAYERS = [
        "layers.attn", "layers.mlp", "layers.ff1", "layers.ff2",
        "embed", "lm_head",
    ]

    def __init__(
        self,
        prefer_local: bool = True,
        copilot_for_completions: bool = True,
    ) -> None:
        self.prefer_local = prefer_local
        self.copilot_for_completions = copilot_for_completions

    def route(
        self,
        heat:              float,
        query_type:        str              = "general",
        layer_heats:       Optional[dict[str, float]] = None,
        sovereign_flag:    bool             = False,
    ) -> InferenceRoute:
        """
        Return the optimal inference route for a query with the given heat.

        query_type: "general" | "code" | "completion" | "memory"
        """
        tier = classify(heat, sovereign_flag=sovereign_flag)

        # Build per-layer LoRA ranks
        lh      = layer_heats or {l: heat for l in self._DEFAULT_LAYERS}
        ranks   = {name: lora_rank(h) for name, h in lh.items()}
        k       = draft_budget(heat, sovereign=sovereign_flag)

        # ── SOVEREIGN ─────────────────────────────────────────────────────────
        if tier == HeatTier.SOVEREIGN:
            return InferenceRoute(
                primary_endpoint = Endpoint.ARC_70B,
                drafter_endpoint = None,
                draft_tokens     = k,
                lora_ranks       = ranks,
                tier             = tier,
                heat             = heat,
                sovereign_lock   = True,
                rationale        = (
                    "Sovereignty lock: H≈1.0 → Arc-70B local only. "
                    "No cloud calls permitted."
                ),
            )

        # ── HOT ───────────────────────────────────────────────────────────────
        if tier == HeatTier.HOT:
            # Code queries → primary is Arc B70 large; Copilot supplements completions
            if query_type == "completion" and self.copilot_for_completions:
                return InferenceRoute(
                    primary_endpoint = Endpoint.COPILOT_LSP,
                    drafter_endpoint = Endpoint.ARC_SMALL,
                    draft_tokens     = k,
                    lora_ranks       = ranks,
                    tier             = tier,
                    heat             = heat,
                    rationale        = "HOT completion: Copilot LSP primary, Arc-small drafter",
                )
            # General / code: Arc-70B primary, RTX drafter (Intel E/PD disaggregation)
            return InferenceRoute(
                primary_endpoint = Endpoint.ARC_70B,
                drafter_endpoint = Endpoint.RTX_DRAFTER,
                draft_tokens     = k,
                lora_ranks       = ranks,
                tier             = tier,
                heat             = heat,
                rationale        = (
                    f"HOT H={heat:.2f}: Arc-70B (deep lane) + RTX drafter "
                    f"(fast lane), k={k}, LoRA r=16"
                ),
            )

        # ── MILD ──────────────────────────────────────────────────────────────
        if tier == HeatTier.MILD:
            # Code → Codex primary, Arc-small parallel speculative
            if query_type in ("code", "completion"):
                return InferenceRoute(
                    primary_endpoint = Endpoint.CODEX_API,
                    drafter_endpoint = Endpoint.ARC_SMALL,
                    draft_tokens     = k,
                    lora_ranks       = ranks,
                    tier             = tier,
                    heat             = heat,
                    rationale        = (
                        f"MILD code H={heat:.2f}: Codex API + Arc-small speculative "
                        f"k={k}, LoRA r=8"
                    ),
                )
            # General reasoning → Arc-small + Claude-API in parallel
            return InferenceRoute(
                primary_endpoint = Endpoint.ARC_SMALL,
                drafter_endpoint = Endpoint.CLAUDE_API,
                draft_tokens     = k,
                lora_ranks       = ranks,
                tier             = tier,
                heat             = heat,
                rationale        = (
                    f"MILD H={heat:.2f}: Arc-small primary, Claude-API fallback "
                    f"k={k}, LoRA r=8"
                ),
            )

        # ── COLD ──────────────────────────────────────────────────────────────
        # No GPU needed; cheapest path, LOCI cache saves most tokens here
        return InferenceRoute(
            primary_endpoint = Endpoint.CLAUDE_API,
            drafter_endpoint = None,
            draft_tokens     = k,
            lora_ranks       = ranks,
            tier             = HeatTier.COLD,
            heat             = heat,
            rationale        = (
                f"COLD H={heat:.2f}: Claude API (no GPU), "
                f"prompt cache saves ~90 % of repeated tokens"
            ),
        )

    def explain(self, heat: float, query_type: str = "general") -> str:
        """Human-readable routing explanation for the IDE status panel."""
        route = self.route(heat, query_type)
        lines = [
            f"Heat:     {route.heat:.3f}  ({route.tier.name})",
            f"Primary:  {route.primary_endpoint}",
            f"Drafter:  {route.drafter_endpoint or '—'}  k={route.draft_tokens}",
            f"LoRA:     {route.lora_ranks}",
            f"Reason:   {route.rationale}",
        ]
        return "\n".join(lines)
