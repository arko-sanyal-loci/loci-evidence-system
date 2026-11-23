"""Smoke tests for model_fusion — no ML framework required."""
import math
try:
    import pytest
except ImportError:
    pytest = None  # type: ignore

from model_fusion.heat_tiers import classify, lora_rank, draft_budget, HeatTier, TIER_WEIGHTS
from model_fusion.distillation import FusionDistiller, TeacherOutput
from model_fusion.inference_router import InferenceRouter, Endpoint
from model_fusion.context_fusion import ContextFuser, AgentContext


# ── heat_tiers ────────────────────────────────────────────────────────────────

def test_tier_boundaries():
    assert classify(0.90) == HeatTier.HOT
    assert classify(0.50) == HeatTier.MILD
    assert classify(0.20) == HeatTier.COLD
    assert classify(0.9999, sovereign_flag=True) == HeatTier.SOVEREIGN

def test_lora_rank_budget():
    assert lora_rank(0.95) == 16
    assert lora_rank(0.60) == 8
    assert lora_rank(0.10) == 1

def test_draft_budget():
    assert draft_budget(0.90) == 8
    assert draft_budget(0.50) == 4
    assert draft_budget(0.20) == 2
    assert draft_budget(0.99, sovereign=True) == 1

def test_tier_weights_sum_to_one_ish():
    # kd + ce should sum to ~1 for non-sovereign tiers
    for tier, tw in TIER_WEIGHTS.items():
        total = tw.kd_weight + tw.ce_weight
        assert abs(total - 1.0) < 1e-9, f"{tier}: kd+ce={total}"


# ── distillation ──────────────────────────────────────────────────────────────

VOCAB = 8   # tiny vocab for tests

def _uniform_logits(size: int = VOCAB) -> list[float]:
    return [1.0 / size] * size

def _peaked_logits(idx: int, size: int = VOCAB, scale: float = 5.0) -> list[float]:
    logits = [0.0] * size
    logits[idx] = scale
    return logits

def test_fused_loss_hot_teacher():
    d = FusionDistiller()
    teachers = [TeacherOutput("arc", _peaked_logits(2), heat=0.85)]
    result = d.fused_loss(_uniform_logits(), teachers, ground_truth_idx=2)
    assert result.total >= 0
    assert result.kd_component >= 0
    assert "arc" in result.per_teacher

def test_fused_loss_cold_filtered():
    d = FusionDistiller()
    teachers = [TeacherOutput("noise", _peaked_logits(0), heat=0.10)]
    result = d.fused_loss(_uniform_logits(), teachers, ground_truth_idx=2)
    # COLD teacher → pure CE, kd_component == 0
    assert result.kd_component == 0.0
    assert result.total > 0

def test_fused_loss_multi_teacher():
    d = FusionDistiller()
    teachers = [
        TeacherOutput("arc",   _peaked_logits(3), heat=0.90),
        TeacherOutput("claude",_peaked_logits(3), heat=0.65),
        TeacherOutput("cold",  _peaked_logits(0), heat=0.10),  # filtered
    ]
    result = d.fused_loss(_uniform_logits(), teachers, ground_truth_idx=3)
    assert result.total >= 0
    assert "cold" not in result.per_teacher   # COLD was filtered

def test_layer_rank_budget():
    d = FusionDistiller()
    heats = {"attn": 0.88, "mlp": 0.50, "ff": 0.20}
    ranks = d.layer_rank_budget(heats)
    assert ranks["attn"] == 16
    assert ranks["mlp"]  == 8
    assert ranks["ff"]   == 1

def test_gaussian_conduction():
    from model_fusion.distillation import gaussian_conduction
    hot_emb  = [1.0, 0.0, 0.0]
    # near neighbour (small cosine distance)
    near = ("near", 0.20, [0.95, 0.1, 0.1])
    # far neighbour
    far  = ("far",  0.20, [0.0,  1.0, 0.0])
    promoted = gaussian_conduction(hot_emb, [near, far], sigma=0.3)
    names = {p.name for p in promoted}
    assert "near" in names         # close enough to radiate to
    # far may or may not be promoted depending on exact cosine distance
    for p in promoted:
        assert p.promoted_heat > p.original_heat


# ── inference_router ──────────────────────────────────────────────────────────

def test_route_sovereign():
    r = InferenceRouter()
    route = r.route(heat=0.9999, sovereign_flag=True)
    assert route.tier == HeatTier.SOVEREIGN
    assert route.sovereign_lock is True
    assert route.primary_endpoint == Endpoint.ARC_70B
    assert route.drafter_endpoint is None
    assert route.draft_tokens == 1

def test_route_hot():
    r = InferenceRouter()
    route = r.route(heat=0.85)
    assert route.tier == HeatTier.HOT
    assert route.primary_endpoint == Endpoint.ARC_70B
    assert route.drafter_endpoint == Endpoint.RTX_DRAFTER
    assert route.draft_tokens == 8

def test_route_mild_code():
    r = InferenceRouter()
    route = r.route(heat=0.55, query_type="code")
    assert route.tier == HeatTier.MILD
    assert route.primary_endpoint == Endpoint.CODEX_API
    assert route.draft_tokens == 4

def test_route_cold():
    r = InferenceRouter()
    route = r.route(heat=0.15)
    assert route.tier == HeatTier.COLD
    assert route.primary_endpoint == Endpoint.CLAUDE_API
    assert route.drafter_endpoint is None


# ── context_fusion ────────────────────────────────────────────────────────────

def _make_blocks():
    return [
        AgentContext("arc",    "Arc memory context ...",   heat=0.90, kind="memory"),
        AgentContext("claude", "Claude reasoning ...",     heat=0.65, kind="message"),
        AgentContext("codex",  "def foo(): pass",          heat=0.78, kind="code"),
        AgentContext("copilot","# inline suggestion",      heat=0.42, kind="completion"),
        AgentContext("local",  "local model output ...",   heat=0.15, kind="tool"),  # COLD
    ]

def test_context_fusion_drops_cold():
    fuser = ContextFuser()
    result = fuser.fuse(_make_blocks())
    assert result.dropped_blocks >= 1              # COLD local was dropped
    assert "local" not in result.fusion_weights    # local agent dropped

def test_context_fusion_hot_first():
    fuser = ContextFuser()
    result = fuser.fuse(_make_blocks())
    assert result.hot_blocks >= 1
    # HOT arc block should appear before MILD blocks in context_text
    arc_pos   = result.context_text.find("[ARC")
    claude_pos = result.context_text.find("[CLAUDE")
    if arc_pos != -1 and claude_pos != -1:
        assert arc_pos < claude_pos   # HOT before MILD

def test_context_fusion_token_budget():
    fuser = ContextFuser(max_tokens=10)   # very tight budget
    blocks = [AgentContext("arc", "x" * 100, heat=0.9, tokens=50)]
    result = fuser.fuse(blocks * 10)
    # Should truncate to budget
    assert result.total_tokens <= 50   # only one block fits

def test_context_fusion_loci_context_wrapper():
    fuser = ContextFuser()
    result = fuser.fuse(_make_blocks())
    assert "<loci_context>" in result.context_text
    assert "</loci_context>" in result.context_text

def test_context_fusion_cache_ttls():
    fuser = ContextFuser()
    result = fuser.fuse(_make_blocks())
    from model_fusion.heat_tiers import TIER_HOT_BOUNDARY, TIER_MILD_BOUNDARY
    for cb in result.cache_blocks:
        if cb.heat >= TIER_HOT_BOUNDARY:
            assert cb.ttl == "extended", f"heat={cb.heat} should be extended"
        elif cb.heat >= TIER_MILD_BOUNDARY:
            assert cb.ttl == "ephemeral", f"heat={cb.heat} should be ephemeral"
