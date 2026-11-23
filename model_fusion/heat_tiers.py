"""
model_fusion/heat_tiers.py — Shared heat tier constants for the fusion stack.

Mirrors arc/core/heat.py tier boundaries so model_fusion is standalone
(no import-time dependency on the arc repo).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


# ── Tier boundaries (must stay in sync with arc/core/heat.py) ────────────────

TIER_HOT_BOUNDARY:        float = 0.80   # H ≥ 0.80 → HOT / VRAM
TIER_MILD_BOUNDARY:       float = 0.35   # H ≥ 0.35 → MILD / DDR5
VRAM_ADMISSION_THRESHOLD: float = 0.70   # crossing this promotes to VRAM

# LoRA rank budget: HOT layers get full rank; COLD layers get minimum
LORA_RANK_HOT:  int = 16
LORA_RANK_MILD: int = 8
LORA_RANK_COLD: int = 1

# Speculative draft-token budget (Dovetail-style)
DRAFT_TOKENS_HOT:      int = 8
DRAFT_TOKENS_MILD:     int = 4
DRAFT_TOKENS_COLD:     int = 2
DRAFT_TOKENS_SOVEREIGN: int = 1


class HeatTier(Enum):
    SOVEREIGN = auto()   # H → 1.0 (flag)  Arc-only sovereignty lock
    HOT       = auto()   # H ≥ 0.80
    MILD      = auto()   # H ≥ 0.35
    COLD      = auto()   # H <  0.35


def classify(heat: float, *, sovereign_flag: bool = False) -> HeatTier:
    if sovereign_flag or heat >= 0.9999:
        return HeatTier.SOVEREIGN
    if heat >= TIER_HOT_BOUNDARY:
        return HeatTier.HOT
    if heat >= TIER_MILD_BOUNDARY:
        return HeatTier.MILD
    return HeatTier.COLD


def lora_rank(heat: float) -> int:
    """Budget-aware LoRA rank for a layer with given heat score."""
    tier = classify(heat)
    if tier in (HeatTier.SOVEREIGN, HeatTier.HOT):
        return LORA_RANK_HOT
    if tier == HeatTier.MILD:
        return LORA_RANK_MILD
    return LORA_RANK_COLD


def draft_budget(heat: float, sovereign: bool = False) -> int:
    """Dovetail-style draft-token count for speculative decoding."""
    tier = classify(heat, sovereign_flag=sovereign)
    if tier == HeatTier.SOVEREIGN:
        return DRAFT_TOKENS_SOVEREIGN
    if tier == HeatTier.HOT:
        return DRAFT_TOKENS_HOT
    if tier == HeatTier.MILD:
        return DRAFT_TOKENS_MILD
    return DRAFT_TOKENS_COLD


@dataclass(frozen=True)
class TierWeights:
    """
    Per-tier KD loss weights for multi-teacher distillation.
    Mirrors the formula from arc/core/lora_kd_trainer.py:
      L_total = H · L_KD + (1−H) · L_CE + λ_sad · L_spectral
    """
    kd_weight:       float   # weight on soft teacher labels (L_KD)
    ce_weight:       float   # weight on hard ground-truth labels (L_CE)
    spectral_lambda: float   # SAD-LoRA spectral alignment coefficient


TIER_WEIGHTS: dict[HeatTier, TierWeights] = {
    HeatTier.SOVEREIGN: TierWeights(kd_weight=1.00, ce_weight=0.00, spectral_lambda=0.10),
    HeatTier.HOT:       TierWeights(kd_weight=0.90, ce_weight=0.10, spectral_lambda=0.05),
    HeatTier.MILD:      TierWeights(kd_weight=0.50, ce_weight=0.50, spectral_lambda=0.02),
    HeatTier.COLD:      TierWeights(kd_weight=0.00, ce_weight=1.00, spectral_lambda=0.00),
}
