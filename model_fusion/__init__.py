"""
model_fusion — Heat-tiered multi-model fusion for the Anti-gravity IDE.

Three layers, each adapted from the LOCI/Arc heat architecture:

  Layer 1 · distillation.py
    Multi-teacher KD where every teacher output is heat-scored before it
    enters the loss.  Extends arc/core/lora_kd_trainer.py's budgeted
    heat-KD from one teacher to N (Codex / Claude / Arc70B / local-metal).
    HOT teacher outputs drive distillation; COLD are filtered.
    Budgeted rank: LoRA adapters sized r ∝ per-layer heat across all teachers.

  Layer 2 · inference_router.py
    Selects which model weights / endpoints fire at inference time.
    Extends arc/core/speculative_pipeline.py's heat routing:
      SOVEREIGN → Arc 70B local only (sovereignty lock)
      HOT       → Arc B70 large + RTX drafter (speculative)
      MILD      → parallel Codex + Arc small
      COLD      → Claude API  (no GPU needed)
    Within each tier the LoRA adapter rank is chosen by the same heat budget.

  Layer 3 · context_fusion.py
    Merges context blocks from all 5 agents into one <loci_context> block.
    Extends arc/core/cache_layer.py's LociItem:
      HOT blocks  → extended TTL (1 hr), injected first in context window
      MILD blocks → ephemeral TTL (5 min)
      COLD blocks → dropped (not injected)
    Output is a heat-weighted attention merge across agent contexts,
    ready to drop into PromptCacheLayer's BP3 breakpoint.
"""
from __future__ import annotations

from model_fusion.distillation   import FusionDistiller,   TeacherOutput
from model_fusion.inference_router import InferenceRouter, InferenceRoute
from model_fusion.context_fusion import ContextFuser,     AgentContext

__all__ = [
    "FusionDistiller", "TeacherOutput",
    "InferenceRouter", "InferenceRoute",
    "ContextFuser",    "AgentContext",
]
