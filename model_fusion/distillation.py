"""
model_fusion/distillation.py — Heat-tiered multi-teacher knowledge distillation.

Extends arc/core/lora_kd_trainer.py from one teacher to N teachers
(Codex · Claude · Arc-70B · local-metal).

Key ideas
─────────
1. Each teacher produces a TeacherOutput with an attached heat score.
   The heat score gates whether the output enters the loss at all
   (COLD → filtered) and how much weight it carries (HOT → full weight).

2. Fused KD loss across N teachers:
     L_fused = Σ_t w_t · [ H_t·L_KD(s, t) + (1−H_t)·L_CE(s, y) + λ·L_spectral ]
               ─────────────────────────────────────────────────────────────────
                              Σ_t w_t  (normalisation)

   where w_t = heat_t for HOT/MILD teachers, 0 for COLD.

3. SAD-LoRA spectral alignment (arXiv:2504.xxxxx):
   The fused gradient is decomposed into:
     L_subspace  — aligns student LoRA subspace with teacher's data-weighted SVD
     L_coeff     — matches coefficients within that subspace
     L_residual  — covers rank-deficient residual
   This is computed per-teacher, then heat-weighted before summation.

4. Budgeted rank per layer:
   Each transformer layer carries a heat score (from forward-pass gradient norm
   averaged over the teacher ensemble).  rank(layer) = lora_rank(layer_heat).
   HOT layers r=16, MILD r=8, COLD r=1 (from heat_tiers.py).

5. Gaussian conduction (from arc/core/lora_kd_trainer.py):
   When a HOT teacher output is processed its embedding radiates to
   semantically near MILD/COLD candidates, promoting them one tier.
   This prevents isolated learning islands.

Usage
─────
    from model_fusion.distillation import FusionDistiller, TeacherOutput

    distiller = FusionDistiller(student_model)

    teacher_outputs = [
        TeacherOutput(name="arc-70b",   logits=arc_logits,   heat=0.92),
        TeacherOutput(name="claude",    logits=claude_logits, heat=0.65),
        TeacherOutput(name="codex",     logits=codex_logits,  heat=0.78),
        TeacherOutput(name="local-70b", logits=local_logits,  heat=0.41),
    ]
    loss = distiller.fused_loss(student_logits, teacher_outputs, ground_truth)
    loss.backward()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from model_fusion.heat_tiers import (
    HeatTier, TierWeights, TIER_WEIGHTS,
    classify, lora_rank,
    TIER_MILD_BOUNDARY,
)


# ── Teacher output container ──────────────────────────────────────────────────

@dataclass
class TeacherOutput:
    """
    One teacher model's output for a single training example.

    name        : teacher identifier ("arc-70b", "claude", "codex", "local-metal")
    logits      : raw output logits (or log-probs) — shape [vocab_size] or [seq, vocab]
    heat        : LOCI heat score for this example (0 → COLD, 1 → SOVEREIGN)
    embedding   : optional hidden-state vector for spectral alignment (SAD-LoRA)
    sovereign   : if True, this teacher always gets full weight regardless of heat
    """
    name:      str
    logits:    "list[float]"          # accept plain lists; real use passes tensors
    heat:      float
    embedding: Optional[list[float]] = None
    sovereign: bool                   = False

    @property
    def tier(self) -> HeatTier:
        return classify(self.heat, sovereign_flag=self.sovereign)

    @property
    def weights(self) -> TierWeights:
        return TIER_WEIGHTS[self.tier]

    def effective_weight(self) -> float:
        """
        The scalar multiplier this teacher's loss contribution gets.
        COLD teachers are filtered (weight=0).
        """
        if self.tier == HeatTier.COLD:
            return 0.0
        return self.heat


# ── Spectral alignment (SAD-LoRA) ─────────────────────────────────────────────

@dataclass
class SpectralAlignment:
    """
    Lightweight stand-in for the full SAD-LoRA spectral decomposition.

    In a real training loop this would work on actual weight matrices.
    Here it carries the structure so the fusion pipeline is architecturally
    correct and can be completed by substituting tensor operations.

    decompose() returns three scalar loss components:
      subspace  — alignment of student LoRA col-space with teacher SVD basis
      coeff     — coefficient matching within that subspace
      residual  — rank-deficient tail coverage
    """
    student_emb: list[float]
    teacher_emb: list[float]

    def _cosine(self) -> float:
        if not self.student_emb or not self.teacher_emb:
            return 0.0
        dot = sum(a * b for a, b in zip(self.student_emb, self.teacher_emb))
        ns  = math.sqrt(sum(a * a for a in self.student_emb))
        nt  = math.sqrt(sum(b * b for b in self.teacher_emb))
        return dot / max(ns * nt, 1e-9)

    def decompose(self) -> tuple[float, float, float]:
        """Returns (L_subspace, L_coeff, L_residual) scalar losses."""
        cos_sim    = self._cosine()
        L_subspace = 1.0 - cos_sim            # 0 when perfectly aligned
        L_coeff    = (1.0 - cos_sim) ** 2     # squared penalty
        L_residual = max(0.0, 0.5 - cos_sim)  # only penalises near-orthogonal
        return L_subspace, L_coeff, L_residual


# ── KL-divergence / soft-label KD ─────────────────────────────────────────────

def _softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    scaled = [l / temperature for l in logits]
    m = max(scaled)
    exp_ = [math.exp(v - m) for v in scaled]
    s = sum(exp_)
    return [v / s for v in exp_]


def _kl_divergence(p: list[float], q: list[float]) -> float:
    """KL(p ‖ q) — soft label distillation loss."""
    return sum(
        pi * (math.log(pi) - math.log(max(qi, 1e-12)))
        for pi, qi in zip(p, q)
        if pi > 1e-12
    )


def _cross_entropy(logits: list[float], target_idx: int) -> float:
    """Hard-label CE loss — log-prob of the ground-truth token."""
    probs = _softmax(logits)
    return -math.log(max(probs[target_idx], 1e-12))


# ── Temperature schedule by tier ──────────────────────────────────────────────
#
# Soft labels with higher temperature spread probability more evenly,
# helping the student learn from a noisy/MILD teacher without over-fitting
# to its specific logit peaks.
#
_TIER_TEMPERATURE: dict[HeatTier, float] = {
    HeatTier.SOVEREIGN: 1.0,   # minimal smoothing — teacher is the oracle
    HeatTier.HOT:       1.5,
    HeatTier.MILD:      3.0,   # more smoothing — teacher may be uncertain
    HeatTier.COLD:      1.0,   # never used (COLD filtered before this)
}


# ── Gaussian conduction ───────────────────────────────────────────────────────

@dataclass
class ConductionNeighbour:
    """
    A COLD/MILD candidate that has been promoted one tier by heat conduction
    from a neighbouring HOT teacher output.
    """
    name:         str
    original_heat: float
    promoted_heat: float
    cosine_dist:   float    # distance from the HOT source that promoted it

    @property
    def new_tier(self) -> HeatTier:
        return classify(self.promoted_heat)


def gaussian_conduction(
    hot_embedding:   list[float],
    candidates:      list[tuple[str, float, list[float]]],  # (name, heat, emb)
    sigma:           float = 0.3,
) -> list[ConductionNeighbour]:
    """
    Radiate heat from a HOT embedding to semantically nearby MILD/COLD candidates.
    Candidates within cosine distance < 2σ are promoted one tier.

    Returns promoted neighbours (cosine distance < 2σ).
    """
    promoted: list[ConductionNeighbour] = []
    for name, cand_heat, cand_emb in candidates:
        if classify(cand_heat) == HeatTier.HOT:
            continue   # already HOT, skip
        # cosine distance = 1 − cosine_similarity
        dot = sum(a * b for a, b in zip(hot_embedding, cand_emb))
        nh  = math.sqrt(sum(a * a for a in hot_embedding))
        nc  = math.sqrt(sum(b * b for b in cand_emb))
        cos_sim  = dot / max(nh * nc, 1e-9)
        cos_dist = 1.0 - cos_sim
        if cos_dist < 2 * sigma:
            # Gaussian promotion: Δheat ∝ exp(−d²/2σ²)
            delta        = math.exp(-(cos_dist ** 2) / (2 * sigma ** 2))
            new_heat     = min(cand_heat + delta * (1.0 - cand_heat), 0.9999)
            promoted.append(ConductionNeighbour(
                name=name, original_heat=cand_heat,
                promoted_heat=new_heat, cosine_dist=cos_dist,
            ))
    return promoted


# ── FusionDistiller ───────────────────────────────────────────────────────────

@dataclass
class FusedLoss:
    """
    Per-component loss breakdown from a fused distillation step.

    total        : scalar to call .backward() on (or use directly in an
                   optimiser step if you're working with plain floats in tests)
    kd_component : heat-weighted sum of KD terms across active teachers
    ce_component : heat-weighted CE contribution
    spectral_component : SAD-LoRA spectral alignment term
    per_teacher  : {name: effective_weight} for the training log
    conducted    : list of neighbours promoted by Gaussian conduction
    """
    total:              float
    kd_component:       float
    ce_component:       float
    spectral_component: float
    per_teacher:        dict[str, float] = field(default_factory=dict)
    conducted:          list[ConductionNeighbour] = field(default_factory=list)


class FusionDistiller:
    """
    Multi-teacher heat-tiered knowledge distillation.

    student_logits_fn: callable(input) → list[float]  (or any tensor-compatible)
    Use without a real model for unit tests by passing None; fused_loss() then
    accepts student_logits directly.
    """

    def __init__(
        self,
        student_logits_fn: object = None,
        *,
        conduction_sigma: float = 0.3,
    ) -> None:
        self.student_fn      = student_logits_fn
        self.conduction_sigma = conduction_sigma

    def fused_loss(
        self,
        student_logits:  list[float],
        teachers:        Sequence[TeacherOutput],
        ground_truth_idx: int,
    ) -> FusedLoss:
        """
        Compute heat-fused distillation loss across all teachers.

        COLD teachers are filtered out.
        HOT teachers get full KD weight (temperature=1.5).
        MILD teachers get soft smoothing (temperature=3.0).

        Spectral alignment (SAD-LoRA) fires for teachers that provide embeddings.
        Gaussian conduction promotes near-COLD neighbours of HOT teachers.
        """
        active_teachers = [t for t in teachers if t.tier != HeatTier.COLD]
        if not active_teachers:
            # No signal — fall back to pure CE on student
            ce = _cross_entropy(student_logits, ground_truth_idx)
            return FusedLoss(total=ce, kd_component=0.0,
                             ce_component=ce, spectral_component=0.0)

        total_weight = sum(t.effective_weight() for t in active_teachers)
        if total_weight < 1e-9:
            total_weight = 1.0

        kd_sum = ce_sum = spec_sum = 0.0
        per_teacher: dict[str, float] = {}

        for t in active_teachers:
            w   = t.effective_weight() / total_weight
            tw  = TIER_WEIGHTS[t.tier]
            temp = _TIER_TEMPERATURE[t.tier]

            # Soft KD (distillation) term
            teacher_probs = _softmax(t.logits, temperature=temp)
            student_probs = _softmax(student_logits, temperature=temp)
            L_kd  = _kl_divergence(teacher_probs, student_probs)

            # Hard CE term
            L_ce  = _cross_entropy(student_logits, ground_truth_idx)

            # Spectral alignment (SAD-LoRA)
            L_spec = 0.0
            if t.embedding is not None:
                # student embedding: use first len(t.embedding) dims of student_logits as proxy
                stud_emb = student_logits[: len(t.embedding)]
                sa = SpectralAlignment(stud_emb, t.embedding)
                sub, coef, res = sa.decompose()
                L_spec = sub + coef + res

            teacher_loss = w * (
                tw.kd_weight       * L_kd
                + tw.ce_weight     * L_ce
                + tw.spectral_lambda * L_spec
            )
            kd_sum   += w * tw.kd_weight   * L_kd
            ce_sum   += w * tw.ce_weight   * L_ce
            spec_sum += w * tw.spectral_lambda * L_spec
            per_teacher[t.name] = round(teacher_loss, 6)

        total_loss = kd_sum + ce_sum + spec_sum

        # Gaussian conduction: radiate from HOT teachers to MILD/COLD neighbours
        conducted: list[ConductionNeighbour] = []
        cold_pool = [
            (t.name, t.heat, t.embedding or [])
            for t in teachers
            if t.tier == HeatTier.COLD and t.embedding
        ]
        for hot_t in active_teachers:
            if hot_t.tier in (HeatTier.HOT, HeatTier.SOVEREIGN) and hot_t.embedding:
                promoted = gaussian_conduction(
                    hot_t.embedding, cold_pool, sigma=self.conduction_sigma
                )
                conducted.extend(promoted)

        return FusedLoss(
            total              = total_loss,
            kd_component       = kd_sum,
            ce_component       = ce_sum,
            spectral_component = spec_sum,
            per_teacher        = per_teacher,
            conducted          = conducted,
        )

    def layer_rank_budget(
        self,
        layer_heat_scores: dict[str, float],
    ) -> dict[str, int]:
        """
        Return the budgeted LoRA rank for each named transformer layer.

        layer_heat_scores: {"layers.0.attn": 0.82, "layers.0.mlp": 0.31, ...}
        """
        return {name: lora_rank(h) for name, h in layer_heat_scores.items()}
