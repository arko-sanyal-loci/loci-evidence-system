"""
loci_theta_hot_validation.py — Empirical θ_HOT boundary validation using
real local Nomic embeddings (nomic-embed-text:latest @ localhost:8000).

Builds 5-stream SLP scores for 30 curated LOCI-domain pair sets across three
expected heat bands (HOT / MILD / COLD), then sweeps candidate θ boundaries
from 0.60 to 0.80 and reports precision, recall, and F1 for HOT classification.

Usage
─────
    python3 loci_theta_hot_validation.py              # table output
    python3 loci_theta_hot_validation.py --json       # machine-readable
    python3 loci_theta_hot_validation.py --endpoint http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import urllib.request
import urllib.error

# ── Nomic stream config (mirrors loci_symprev_streams.py) ────────────────────

STREAM_WEIGHTS = {"ISymPrev": 0.30, "QSymPrev": 0.25,
                  "DSymPrev": 0.20, "MSymPrev": 0.15, "SSymPrev": 0.10}

STREAM_DIM = {"ISymPrev": 768, "QSymPrev": 384,
              "DSymPrev": 768, "MSymPrev": 128, "SSymPrev": 384}

STREAM_PREFIX = {
    "ISymPrev": "search_document",
    "QSymPrev": "classification",
    "DSymPrev": "clustering",
    "MSymPrev": "search_query",
    "SSymPrev": "search_document",
}

NOMIC_MODEL   = "nomic-embed-text:latest"
ALPHA_HOPE    = 0.050
SLP_COSINE_W  = 0.75
SLP_HEAT_W    = 0.15
SLP_CROSS_W   = 0.10


# ── 30-pair test corpus — 10 HOT, 10 MILD, 10 COLD ───────────────────────────
# HOT  — exact same knowledge unit, phrased differently (same Arc memory node)
# MILD — related concept, different specificity / adjacent sub-domain
# COLD — unrelated domain entirely
#
# Methodology fix v2: MILD pairs now have genuinely adjacent (not reworded) entities
# so the band does not inflate into HOT cosine territory.

TEST_PAIRS: list[tuple[str, str, str]] = [
    # ── HOT — same fact unit, different surface form ──────────────────────────
    ("HOT",  "CFMOTO 300SR OBD-II coolant PID 0105 fan threshold 95C",
             "PID 0105 returns coolant temp; fan cut-in 95 degrees; CFMOTO 300SR OBD"),
    ("HOT",  "Intel Arc B70 HOT-tier VRAM admission threshold heat score",
             "Arc B70 VRAM lane heat admission score HOT boundary GPU inference"),
    ("HOT",  "theta_hot 0.700 empirically validated Table 4 300 trials",
             "θ_HOT = 0.700 Table 4 benchmark 300 multi-turn controlled trials optimal"),
    ("HOT",  "nomic-embed-text 768-dim MRL slice ISymPrev DSymPrev",
             "MRL matryoshka slice dims 768 ISymPrev DSymPrev nomic embed text local"),
    ("HOT",  "SLP weighted cosine ISymPrev 0.30 QSymPrev 0.25 DSymPrev 0.20",
             "SLP score weights ISymPrev=0.30 QSymPrev=0.25 DSymPrev=0.20 MSymPrev=0.15 SSymPrev=0.10"),
    ("HOT",  "Brembo Corsa Corta 60-0 braking 14.9 m 1.79 s Road 6",
             "60-0 km/h braking 14.9 metres 1.79 seconds Brembo Corsa Corta Michelin Road 6"),
    ("HOT",  "LOCI 4-breakpoint caching BP3 loci_context extended TTL",
             "4-breakpoint prompt cache BP3 loci_context tag heat-dependent extended TTL"),
    ("HOT",  "2-hop Gaussian conductance sigma 0.3 amplification 1000x deterministic",
             "multihop tiebreak Gaussian conductance sigma=0.3 separation amplification >1000x"),
    ("HOT",  "CFMOTO 300SR ECU ignition advance fuel map AFR flash 292cc",
             "ECU flash 300SR 292cc ignition timing fuel map air-fuel-ratio advance degrees"),
    ("HOT",  "loci_symprev_streams SymPrevRouter rank_candidates slp_score",
             "SymPrevRouter rank_candidates returns slp_score per candidate loci_symprev_streams"),

    # ── MILD — adjacent concept, different specificity ────────────────────────
    ("MILD", "CFMOTO 300SR coolant temperature OBD",
             "liquid cooling system radiator thermostat automotive temperature sensor"),
    ("MILD", "Arc B70 GPU VRAM inference",
             "discrete graphics card dedicated video memory deep learning inference latency"),
    ("MILD", "LOCI prompt caching savings ledger",
             "LLM API cost reduction cache prefix reuse token pricing optimization"),
    ("MILD", "Michelin Road 6 tyre lean angle limit",
             "sport touring tyre cornering clearance geometry chassis ground clearance"),
    ("MILD", "nomic-embed-text cosine similarity retrieval",
             "dense vector retrieval approximate nearest neighbour FAISS index embedding"),
    ("MILD", "SLP heat routing MILD tier Haiku",
             "model selection routing low-cost inference small model budget API tier"),
    ("MILD", "ELM327 BLE OBD vehicle diagnostics",
             "vehicle on-board diagnostics CAN bus data logger scan tool protocol"),
    ("MILD", "Arc memory heat score cross-session",
             "contextual memory retrieval recency weighting session-aware knowledge graph"),
    ("MILD", "CFMOTO 300SR lean angle footpeg scrape",
             "motorcycle geometry rake trail suspension cornering clearance peg grind"),
    ("MILD", "ECU fuel map injection tuning",
             "engine management system power output performance chip tuning remap"),

    # ── COLD — entirely unrelated domain ─────────────────────────────────────
    ("COLD", "CFMOTO 300SR ignition timing advance degrees",
             "quarterly earnings per share dividend yield stock portfolio rebalancing"),
    ("COLD", "Arc B70 VRAM HOT-tier heat admission",
             "sourdough bread hydration ratio baking temperature Maillard crust"),
    ("COLD", "theta_hot routing threshold LOCI",
             "Atlantic hurricane season forecast sea surface temperature wind shear"),
    ("COLD", "nomic-embed MRL matryoshka 768-dim",
             "violin concerto key signature baroque counterpoint fugue subject"),
    ("COLD", "ISymPrev narrative episodic stream weight",
             "tax bracket marginal rate capital gains deduction itemised filing"),
    ("COLD", "LOCI 4-breakpoint prompt cache TTL",
             "competitive swimming stroke butterfly dolphin kick turn technique"),
    ("COLD", "2-hop Gaussian tiebreaker amplification separation",
             "French pastry cream puff choux dough piping bag buttercream"),
    ("COLD", "Brembo Corsa Corta braking distance Road 6 grip",
             "satellite orbital mechanics perigee apogee Hohmann transfer delta-v"),
    ("COLD", "ELM327 BLE OBD-II diagnostic scan CFMOTO",
             "Renaissance oil painting sfumato chiaroscuro perspective vanishing point"),
    ("COLD", "loci_symprev_streams 5-stream SymPrev scoring",
             "contract law consideration offer acceptance estoppel tort remedy"),
]

# ── Arc heat simulation distributions per band ───────────────────────────────
# HOT  items: high retrieval frequency + recent → arc_heat 0.55–0.90
# MILD items: occasional retrieval             → arc_heat 0.25–0.55
# COLD items: rarely seen by Arc               → arc_heat 0.02–0.20
# cross_session_count matches retrieval frequency expectations per band

ARC_HEAT_RANGE = {
    "HOT":  (0.55, 0.90),
    "MILD": (0.25, 0.55),
    "COLD": (0.02, 0.20),
}
CROSS_SESSION_RANGE = {
    "HOT":  (2, 8),
    "MILD": (1, 3),
    "COLD": (0, 1),
}


# ── Embedding helpers ─────────────────────────────────────────────────────────

def _embed(text: str, prefix: str, endpoint: str) -> list[float]:
    """Call nomic-embed with the stream task prefix, return raw 768-d vector."""
    payload = json.dumps({
        "model": NOMIC_MODEL,
        "input": f"{prefix}: {text}",
    }).encode()
    req = urllib.request.Request(
        f"{endpoint}/v1/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        d = json.loads(resp.read())
    return d["data"][0]["embedding"]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


# ── Score a single pair ───────────────────────────────────────────────────────

@dataclass
class PairResult:
    label:    str
    query:    str
    entity:   str
    stream_scores:   dict[str, float] = field(default_factory=dict)
    weighted_cosine: float = 0.0
    slp_score:       float = 0.0   # heat=0.0, cross_session=0 (cosine-only baseline)
    # Filled by apply_heat_simulation():
    sim_arc_heat:        float = 0.0
    sim_cross_session:   int   = 0
    slp_full:            float = 0.0  # SLP with simulated Arc heat
    latency_ms: float = 0.0

    def tier(self, theta_hot: float = 0.700, theta_mild: float = 0.350,
             use_full: bool = False) -> str:
        h = self.slp_full if use_full else self.slp_score
        if h >= theta_hot:  return "HOT"
        if h >= theta_mild: return "MILD"
        return "COLD"

    def apply_heat_simulation(self, seed_offset: int = 0) -> None:
        """Draw arc_heat and cross_session from band-appropriate distributions."""
        rng = random.Random(hash(self.query) + seed_offset)
        lo, hi = ARC_HEAT_RANGE[self.label]
        self.sim_arc_heat = rng.uniform(lo, hi)
        c_lo, c_hi = CROSS_SESSION_RANGE[self.label]
        self.sim_cross_session = rng.randint(c_lo, c_hi)
        h = max(ALPHA_HOPE, min(1.0, self.sim_arc_heat))
        wcos = self.weighted_cosine
        self.slp_full = (
            SLP_COSINE_W    * wcos +
            SLP_HEAT_W      * h +
            SLP_CROSS_W     * math.log1p(max(0, self.sim_cross_session))
        )


def score_pair(label: str, query: str, entity: str,
               endpoint: str) -> PairResult:
    t0 = time.monotonic()
    stream_cos: dict[str, float] = {}
    for stream, prefix in STREAM_PREFIX.items():
        dim = STREAM_DIM[stream]
        q_emb = _embed(query,  prefix, endpoint)[:dim]
        e_emb = _embed(entity, prefix, endpoint)[:dim]
        stream_cos[stream] = _cosine(q_emb, e_emb)

    wcos = sum(STREAM_WEIGHTS[s] * v for s, v in stream_cos.items())
    # SLP with heat=0 and cross_session_count=0 → pure cosine contribution
    slp  = SLP_COSINE_W * wcos + SLP_HEAT_W * ALPHA_HOPE + SLP_CROSS_W * math.log1p(0)
    ms   = (time.monotonic() - t0) * 1000

    return PairResult(
        label=label, query=query, entity=entity,
        stream_scores=stream_cos,
        weighted_cosine=wcos,
        slp_score=slp,
        latency_ms=ms,
    )


# ── Threshold sweep ───────────────────────────────────────────────────────────

def threshold_sweep(results: list[PairResult],
                    thresholds: list[float],
                    use_full: bool = False) -> list[dict]:
    rows = []
    for theta in thresholds:
        tp = fp = fn = tn = 0
        for r in results:
            score = r.slp_full if use_full else r.slp_score
            predicted_hot = score >= theta
            actual_hot    = r.label == "HOT"
            if predicted_hot and actual_hot:  tp += 1
            elif predicted_hot:               fp += 1
            elif actual_hot:                  fn += 1
            else:                             tn += 1
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        rows.append(dict(theta=theta, tp=tp, fp=fp, fn=fn, tn=tn,
                         precision=prec, recall=recall, f1=f1))
    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

def _print_sweep(sweep: list[dict], candidates: list[float],
                 label: str, results: list[PairResult], use_full: bool) -> None:
    best_f1 = max(s["f1"] for s in sweep)
    print(f"\n  ── {label} ──")
    print(f"  {'θ':>6s}  {'TP':>3s}  {'FP':>3s}  {'FN':>3s}  {'TN':>3s}  "
          f"{'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}")
    print(f"  {'─'*6}  {'─'*3}  {'─'*3}  {'─'*3}  {'─'*3}  {'─'*6}  {'─'*6}  {'─'*6}")
    for s in sweep:
        markers = []
        if s["f1"] == best_f1:         markers.append("best F1")
        if abs(s["theta"] - 0.700)  < 0.001: markers.append("0.700")
        if abs(s["theta"] - 0.7250) < 0.001: markers.append("0.7250")
        tag = ("  ◄ " + ", ".join(markers)) if markers else ""
        print(f"  {s['theta']:6.3f}  {s['tp']:3d}  {s['fp']:3d}  {s['fn']:3d}  {s['tn']:3d}  "
              f" {s['precision']:5.3f}   {s['recall']:5.3f}   {s['f1']:5.3f}{tag}")
    best_theta = max(sweep, key=lambda x: x["f1"])["theta"]
    s700  = next((s for s in sweep if abs(s["theta"] - 0.700)  < 0.001), None)
    s7250 = next((s for s in sweep if abs(s["theta"] - 0.7250) < 0.001), None)
    return best_theta, best_f1, s700, s7250


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate θ_HOT with real Nomic embeddings")
    ap.add_argument("--endpoint", default="http://localhost:8000",
                    help="OpenAI-compatible embeddings API (default: http://localhost:8000)")
    ap.add_argument("--json", action="store_true", help="Output JSON instead of table")
    ap.add_argument("--pairs", type=int, default=0,
                    help="Limit to N pairs per band (0 = all 10)")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for heat simulation (default: 42)")
    args = ap.parse_args()

    pairs = TEST_PAIRS
    if args.pairs:
        hot  = [p for p in pairs if p[0] == "HOT"][:args.pairs]
        mild = [p for p in pairs if p[0] == "MILD"][:args.pairs]
        cold = [p for p in pairs if p[0] == "COLD"][:args.pairs]
        pairs = hot + mild + cold

    print(f"Scoring {len(pairs)} pairs via {args.endpoint} …", file=sys.stderr)

    results: list[PairResult] = []
    for i, (label, query, entity) in enumerate(pairs, 1):
        print(f"  [{i:2d}/{len(pairs)}] {label:4s} {query[:55]!r}", file=sys.stderr,
              end="", flush=True)
        try:
            r = score_pair(label, query, entity, args.endpoint)
            r.apply_heat_simulation(seed_offset=args.seed)
            results.append(r)
            print(f"  wcos={r.weighted_cosine:.4f}  slp_cos={r.slp_score:.4f}  "
                  f"arc_h={r.sim_arc_heat:.3f}  slp_full={r.slp_full:.4f}  "
                  f"{r.latency_ms:.0f}ms", file=sys.stderr)
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)

    if not results:
        print("No results — is the Nomic endpoint reachable?", file=sys.stderr)
        sys.exit(1)

    # Sweep includes both 0.700 and 0.7250 as explicit candidates
    thresholds = sorted(set(
        [round(0.55 + i * 0.025, 3) for i in range(13)] +  # 0.550 … 0.850
        [0.700, 0.7250]
    ))

    sweep_cos  = threshold_sweep(results, thresholds, use_full=False)
    sweep_full = threshold_sweep(results, thresholds, use_full=True)

    if args.json:
        output = {
            "methodology": {
                "cosine_only": "SLP with heat=0, cross_session=0 — cosine floor",
                "full_slp": "SLP with simulated Arc heat + cross_session per band",
            },
            "pairs": [
                {
                    "label": r.label,
                    "query": r.query[:60],
                    "weighted_cosine": round(r.weighted_cosine, 4),
                    "slp_cosine_only": round(r.slp_score, 4),
                    "sim_arc_heat": round(r.sim_arc_heat, 4),
                    "sim_cross_session": r.sim_cross_session,
                    "slp_full": round(r.slp_full, 4),
                    "stream_scores": {k: round(v, 4) for k, v in r.stream_scores.items()},
                }
                for r in results
            ],
            "cosine_only_sweep": sweep_cos,
            "full_slp_sweep": sweep_full,
            "best_theta_cosine": max(sweep_cos, key=lambda x: x["f1"])["theta"],
            "best_theta_full":   max(sweep_full, key=lambda x: x["f1"])["theta"],
        }
        print(json.dumps(output, indent=2))
        return

    # ── text report ───────────────────────────────────────────────────────────
    print("\n╔══ θ_HOT Empirical Validation v2 — local nomic-embed-text ═══════════════════╗")
    print(f"  endpoint : {args.endpoint}  model : {NOMIC_MODEL}  seed : {args.seed}")
    print(f"  pairs    : {len(results)}  "
          f"({sum(r.label=='HOT' for r in results)} HOT / "
          f"{sum(r.label=='MILD' for r in results)} MILD / "
          f"{sum(r.label=='COLD' for r in results)} COLD)")
    print()
    print("  Per-pair scores (sorted by full SLP)")
    print(f"  {'LBL':4s}  {'wcos':>6s}  {'slp_cos':>7s}  {'arc_h':>5s}  "
          f"{'n_xs':>4s}  {'slp_full':>8s}  {'@0.700':>6s}  {'@0.725':>6s}")
    print(f"  {'─'*4}  {'─'*6}  {'─'*7}  {'─'*5}  {'─'*4}  {'─'*8}  {'─'*6}  {'─'*6}")
    for r in sorted(results, key=lambda x: x.slp_full, reverse=True):
        p700  = r.tier(0.700,  use_full=True)
        p7250 = r.tier(0.7250, use_full=True)
        ok700  = "✓" if p700  == r.label else "✗"
        ok7250 = "✓" if p7250 == r.label else "✗"
        print(f"  {r.label:4s}  {r.weighted_cosine:6.4f}  {r.slp_score:7.4f}  "
              f"{r.sim_arc_heat:5.3f}  {r.sim_cross_session:4d}  {r.slp_full:8.4f}  "
              f"{p700:4s}{ok700}  {p7250:4s}{ok7250}")

    # Band stats for full SLP
    print()
    print(f"  {'BAND':4s}  {'mean full SLP':>13s}  {'min':>6s}  {'max':>6s}  "
          f"{'mean arc_heat':>13s}")
    print(f"  {'─'*4}  {'─'*13}  {'─'*6}  {'─'*6}  {'─'*13}")
    for band in ("HOT", "MILD", "COLD"):
        br = [r for r in results if r.label == band]
        if br:
            slps = [r.slp_full for r in br]
            heats = [r.sim_arc_heat for r in br]
            print(f"  {band:4s}  {sum(slps)/len(slps):13.4f}  "
                  f"{min(slps):6.4f}  {max(slps):6.4f}  "
                  f"{sum(heats)/len(heats):13.4f}")

    # Threshold sweeps — both passes
    best_cos,  bf1_cos,  s700_cos,  s7250_cos  = _print_sweep(
        sweep_cos,  thresholds, "Pass 1 — cosine-only SLP (heat=0, n=0)", results, False)
    best_full, bf1_full, s700_full, s7250_full = _print_sweep(
        sweep_full, thresholds, "Pass 2 — full SLP (cosine + Arc heat sim + cross-session)", results, True)

    # Verdict
    print()
    print("  ── Verdict ─────────────────────────────────────────────────────────────────")
    for theta, f1_v, label_v in [
        (0.700,  s700_full["f1"]  if s700_full  else None, "0.700  (current)"),
        (0.7250, s7250_full["f1"] if s7250_full else None, "0.7250 (prior analytical)"),
        (best_full, bf1_full,                              f"{best_full} (best F1 on full SLP)"),
    ]:
        if f1_v is not None:
            mark = " ◄ RECOMMENDED" if abs(theta - best_full) < 0.001 else ""
            print(f"  θ = {theta:<6.4f}  full-SLP F1 = {f1_v:.3f}{mark}")
    print()
    if abs(best_full - 0.700) < 0.001:
        print("  ✓ θ_HOT = 0.700 confirmed optimal on full SLP.")
    elif abs(best_full - 0.7250) < 0.001:
        print("  ✓ θ_HOT = 0.7250 confirmed optimal on full SLP — revert warranted.")
    else:
        print(f"  ⚠  Neither 0.700 nor 0.7250 is optimal — best full-SLP θ = {best_full}.")
        if s700_full and s7250_full:
            winner = "0.700" if s700_full["f1"] >= s7250_full["f1"] else "0.7250"
            print(f"     Between the two, {winner} performs better on this corpus.")
    print("╚═══════════════════════════════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    main()
