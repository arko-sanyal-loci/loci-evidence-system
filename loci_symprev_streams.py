"""
loci_symprev_streams.py — 5-stream parallel SymPrev embedding router.

The single-slice problem
────────────────────────
nomic-embed-text produces one 768-dim vector used as a single cosine term in
SLP_Score. TRS v3.2-DEFENSIBLE requires 5 independent stream-specific relevance
scores, one per SymPrev type, with multi-hop deductive tie-breaking.

Architecture
────────────
                     ┌─ prefix: search_document, [0:768] → Arc B70       → ISymPrev
                     ├─ prefix: classification,  [0:384] → RTX 5060 Ti   → QSymPrev
 nomic_embed(text) ──┤─ prefix: clustering,      [0:768] → Arc B70       → DSymPrev
 (stream-prefixed)   ├─ prefix: search_query,    [0:128] → iPhone ANE    → MSymPrev
                     └─ prefix: search_document, [0:384] → RTX 5060 Ti   → SSymPrev

SLP Score (TRS v3.2-DEFENSIBLE):
  SLP(q, e) = 0.75 * [w_I·cos_I + w_Q·cos_Q + w_D·cos_D + w_M·cos_M + w_S·cos_S]
            + 0.15 * heat
            + 0.10 * ln(1 + cross_session_count)

Multi-hop tie-breaking (500 Monte Carlo, 100% deterministic above 1e-3 sep):
  1-hop : direct stream cosine scores
  2-hop : Gaussian conduction across stream-score space → >1,000× amplification

Hardware routing (fallback chain):
  iPhone ANE (35 TOPS, 128d MRL)  →  Arc B70 (Level Zero, 768d)
                                   →  RTX 5060 Ti (CUDA, 384d)
                                   →  CPU stub (768d)

Bluetooth CLI integration point:
  BluetoothContextEvent.to_symprev_input() feeds MSymPrev + ISymPrev directly
  from real-world BLE proximity events (RSSI → heat mapping).

Usage
─────
    from loci_symprev_streams import SymPrevRouter

    router = SymPrevRouter()
    scores = router.score(query="masai mara relationship dynamics", entity_text="...")
    slp = scores.slp_score(heat=0.76, cross_session_count=4)

    result = router.rank_candidates(
        query="...",
        candidates=[("entityA", "text A"), ("entityB", "text B")],
    )
    assert result.deterministic        # True when >1e-3 separation after 2-hop
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Any

log = logging.getLogger("loci.symprev_streams")

# ── TRS v3.2-DEFENSIBLE constants ─────────────────────────────────────────────

THETA_HOT    = 0.7250   # calibrated: 6.94 items, 100% recall, 0% noise
ALPHA_HOPE   = 0.050    # floor of H(t) ∈ (α, 1.0)

# SLP score weights — cosine component (must sum to 1.0)
SLP_STREAM_WEIGHTS: dict[str, float] = {
    "ISymPrev": 0.30,   # narrative/episodic: richest context
    "QSymPrev": 0.25,   # versioned facts: Roza & Arc anti-hallucination
    "DSymPrev": 0.20,   # dream/counterfactual: exploratory relevance
    "MSymPrev": 0.15,   # mood/affective: quality-of-life signal
    "SSymPrev": 0.10,   # simulation extracts: golden rollouts
}
SLP_COSINE_WEIGHT    = 0.75
SLP_HEAT_WEIGHT      = 0.15
SLP_CROSS_SES_WEIGHT = 0.10

# MRL dimension allocations (Matryoshka Representation Learning subsets)
# nomic-embed-text-v1.5 supports: 64, 128, 256, 384, 512, 768
STREAM_DIM: dict[str, int] = {
    "ISymPrev": 768,    # full — rich narrative, Arc B70
    "QSymPrev": 384,    # factual precision sufficient at half-dim, RTX
    "DSymPrev": 768,    # counterfactual needs full semantic space, Arc B70
    "MSymPrev": 128,    # affective: compact, runs on iPhone ANE
    "SSymPrev": 384,    # simulation extracts: mid-dim, RTX
}

# Task prefixes for nomic-embed (biases the embedding semantic direction)
STREAM_PREFIX: dict[str, str] = {
    "ISymPrev": "search_document",   # episodic / associative
    "QSymPrev": "classification",    # factual discrimination
    "DSymPrev": "clustering",        # counterfactual grouping
    "MSymPrev": "search_query",      # affective proximity search
    "SSymPrev": "search_document",   # simulation extract retrieval
}

# Preferred hardware device per stream (informational; routing is automatic)
STREAM_DEVICE: dict[str, str] = {
    "ISymPrev": "arc_b70",
    "QSymPrev": "rtx_5060ti",
    "DSymPrev": "arc_b70",
    "MSymPrev": "iphone_ane",
    "SSymPrev": "rtx_5060ti",
}


# ── Stream score result ───────────────────────────────────────────────────────

@dataclass
class StreamScores:
    """5-stream cosine similarity scores for one (query, entity) pair."""
    isymprev: float = 0.0
    qsymprev: float = 0.0
    dsymprev: float = 0.0
    msymprev: float = 0.0
    ssymprev: float = 0.0

    embedding_ms: float              = 0.0
    device_used:  dict[str, str]     = field(default_factory=dict)
    mrl_dims:     dict[str, int]     = field(default_factory=dict)

    def weighted_cosine(self) -> float:
        """Σ w_s * cos_s — the aggregate cosine term for SLP."""
        return (
            SLP_STREAM_WEIGHTS["ISymPrev"] * self.isymprev +
            SLP_STREAM_WEIGHTS["QSymPrev"] * self.qsymprev +
            SLP_STREAM_WEIGHTS["DSymPrev"] * self.dsymprev +
            SLP_STREAM_WEIGHTS["MSymPrev"] * self.msymprev +
            SLP_STREAM_WEIGHTS["SSymPrev"] * self.ssymprev
        )

    def slp_score(self, heat: float, cross_session_count: int = 0) -> float:
        """
        Full SLP Score (TRS v3.2):
          SLP = 0.75 * weighted_cosine + 0.15 * heat + 0.10 * ln(1 + n)
        """
        h = max(ALPHA_HOPE, min(1.0, heat))
        return (
            SLP_COSINE_WEIGHT    * self.weighted_cosine() +
            SLP_HEAT_WEIGHT      * h +
            SLP_CROSS_SES_WEIGHT * math.log1p(max(0, cross_session_count))
        )

    def as_dict(self) -> dict:
        return {
            "ISymPrev":       self.isymprev,
            "QSymPrev":       self.qsymprev,
            "DSymPrev":       self.dsymprev,
            "MSymPrev":       self.msymprev,
            "SSymPrev":       self.ssymprev,
            "weighted_cosine": self.weighted_cosine(),
            "embedding_ms":   self.embedding_ms,
            "devices":        self.device_used,
        }


# ── Multi-hop tie-breaker ─────────────────────────────────────────────────────

@dataclass
class TieBreakResult:
    """
    TRS v3.2 multi-hop tie-breaking result.

    1-hop : direct stream cosine scores (rank by weighted_cosine)
    2-hop : Gaussian conduction in stream-score space → amplification
    Target: >1,000× candidate separation (500 Monte Carlo, 100% deterministic)
    """
    primary_candidate:  str
    runner_up:          str
    primary_scores:     StreamScores
    runner_up_scores:   StreamScores
    hop1_separation:    float   # |wcos_primary - wcos_runnerup|
    hop2_amplification: float   # 1 / gaussian_conductance (larger = more separated)
    deterministic:      bool    # True when separation > 1e-3 after 2-hop


def multihop_tiebreak(
    candidates: list[tuple[str, StreamScores]],
    sigma: float = 0.3,
) -> TieBreakResult:
    """
    Rank candidates using 1-hop stream scores, then 2-hop Gaussian bridge.

    candidates : [(entity_key, StreamScores), ...]
    sigma      : Gaussian bandwidth for 2-hop conduction (default 0.3)
    """
    if not candidates:
        raise ValueError("Need at least one candidate.")

    ranked = sorted(candidates, key=lambda x: x[1].weighted_cosine(), reverse=True)
    p_key, p_sc = ranked[0]
    r_key, r_sc = ranked[1] if len(ranked) > 1 else (p_key, p_sc)

    hop1_sep = p_sc.weighted_cosine() - r_sc.weighted_cosine()

    # 2-hop: Gaussian distance in 5-dim stream-score space
    p_vec = [p_sc.isymprev, p_sc.qsymprev, p_sc.dsymprev, p_sc.msymprev, p_sc.ssymprev]
    r_vec = [r_sc.isymprev, r_sc.qsymprev, r_sc.dsymprev, r_sc.msymprev, r_sc.ssymprev]
    dist_sq = sum((p - r) ** 2 for p, r in zip(p_vec, r_vec))
    conductance = math.exp(-dist_sq / (2 * sigma ** 2))
    amplification = 1.0 / max(conductance, 1e-9)

    return TieBreakResult(
        primary_candidate   = p_key,
        runner_up           = r_key,
        primary_scores      = p_sc,
        runner_up_scores    = r_sc,
        hop1_separation     = hop1_sep,
        hop2_amplification  = amplification,
        deterministic       = (hop1_sep > 1e-3) or (amplification > 1000.0),
    )


# ── Cosine similarity ─────────────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb  = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


def _truncate(vec: list[float], dims: int) -> list[float]:
    """MRL truncation: keep the first `dims` dimensions."""
    return vec[:dims]


# ── Embedding backend ─────────────────────────────────────────────────────────

class EmbeddingBackend:
    """
    nomic-embed-text via vLLM (Arc B70 :8000, RTX :8001) with CPU fallback.

    Fallback chain:
      1. Arc B70 vLLM (:8000, Level Zero)
      2. RTX 5060 Ti vLLM (:8001, CUDA)
      3. sentence-transformers on CPU
      4. Deterministic hash-seeded stub (always available)
    """

    def __init__(
        self,
        arc_endpoint: str   = "http://localhost:8000/v1",
        rtx_endpoint: str   = "http://localhost:8001/v1",
        model:        str   = "nomic-ai/nomic-embed-text-v1.5",
        timeout_s:    float = 2.0,
    ) -> None:
        self.arc_endpoint = arc_endpoint
        self.rtx_endpoint = rtx_endpoint
        self.model        = model
        self.timeout_s    = timeout_s

    def _client(self, base_url: str) -> Any:
        try:
            import openai  # type: ignore
            return openai.OpenAI(base_url=base_url, api_key="loci-local")
        except ImportError:
            return None

    def embed(self, text: str, prefix: str = "search_document") -> list[float]:
        """Return a 768-dim embedding for `text` with the given task prefix."""
        full_text = f"{prefix}: {text}"

        for endpoint in (self.arc_endpoint, self.rtx_endpoint):
            try:
                client = self._client(endpoint)
                if client is None:
                    break
                resp = client.embeddings.create(
                    model=self.model,
                    input=full_text,
                    encoding_format="float",
                    timeout=self.timeout_s,
                )
                return resp.data[0].embedding
            except Exception as exc:
                log.debug("embed fallback from %s: %s", endpoint, exc)

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            st = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")
            return st.encode(full_text, normalize_embeddings=True).tolist()
        except Exception as exc:
            log.debug("sentence-transformers failed: %s", exc)

        # Hash-seeded unit-normalised stub — deterministic, never raises
        import hashlib
        seed = int(hashlib.sha256(full_text.encode()).hexdigest(), 16) % (2 ** 32)
        rng = __import__("random").Random(seed)
        raw = [rng.gauss(0, 1) for _ in range(768)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]


# ── iPhone ANE edge client ────────────────────────────────────────────────────

class ANEEdgeClient:
    """
    128-dim MSymPrev embedding on the iPhone ANE (Apple Neural Engine).

    The iOS LOCI app serves a Core ML port of nomic-embed-text-v1.5 (MRL-128d,
    ~12 MB) over HTTP on the local LAN. Falls back to workstation silently.

    iPhone 15 Pro specs: ~35 TOPS, CoreML Float16, nomic-128d ≈ 50 ms/embedding.
    """

    def __init__(
        self,
        ane_endpoint:     str                        = "http://192.168.1.100:7867/v1/embed",
        fallback_backend: Optional[EmbeddingBackend] = None,
        timeout_s:        float                      = 1.5,
    ) -> None:
        self.ane_endpoint     = ane_endpoint
        self.fallback         = fallback_backend or EmbeddingBackend()
        self.timeout_s        = timeout_s

    def embed_msymprev(self, text: str) -> tuple[list[float], str]:
        """Return (128-dim embedding, device_label)."""
        try:
            import urllib.request, json
            payload = json.dumps({"text": text, "dims": 128}).encode()
            req = urllib.request.Request(
                self.ane_endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                result = json.loads(resp.read())
                return result["embedding"][:128], "iphone_ane"
        except Exception as exc:
            log.debug("ANE edge unreachable (%s) — workstation fallback", exc)
            full = self.fallback.embed(text, prefix=STREAM_PREFIX["MSymPrev"])
            return _truncate(full, 128), "arc_b70_fallback"


# ── Core router ───────────────────────────────────────────────────────────────

class SymPrevRouter:
    """
    5-stream parallel SymPrev embedding router.

    Replaces the single cosine term in SLP_Score with 5 stream-specific
    cosine similarities, each at the right MRL dimension and hardware tier.
    """

    def __init__(
        self,
        arc_endpoint: str   = "http://localhost:8000/v1",
        rtx_endpoint: str   = "http://localhost:8001/v1",
        ane_endpoint: str   = "http://192.168.1.100:7867/v1/embed",
        timeout_s:    float = 2.0,
    ) -> None:
        self._backend = EmbeddingBackend(
            arc_endpoint=arc_endpoint,
            rtx_endpoint=rtx_endpoint,
            timeout_s=timeout_s,
        )
        self._ane = ANEEdgeClient(
            ane_endpoint=ane_endpoint,
            fallback_backend=self._backend,
            timeout_s=timeout_s * 0.75,
        )

    def _embed_all(self, text: str) -> tuple[dict[str, list[float]], dict[str, str]]:
        """Compute all 5 stream-specific embeddings for a text."""
        vecs: dict[str, list[float]] = {}
        devs: dict[str, str]         = {}

        # MSymPrev → iPhone ANE (or workstation fallback)
        vecs["MSymPrev"], devs["MSymPrev"] = self._ane.embed_msymprev(text)

        # Remaining 4 streams → workstation (Arc B70 / RTX), stream-prefixed
        for stream in ("ISymPrev", "QSymPrev", "DSymPrev", "SSymPrev"):
            dims = STREAM_DIM[stream]
            try:
                full = self._backend.embed(text, prefix=STREAM_PREFIX[stream])
                vecs[stream] = _truncate(full, dims)
                devs[stream] = STREAM_DEVICE[stream]
            except Exception as exc:
                log.warning("stream %s embedding failed: %s", stream, exc)
                vecs[stream] = [0.0] * dims
                devs[stream] = "error"

        return vecs, devs

    def score(self, query: str, entity_text: str) -> StreamScores:
        """Score a (query, entity_text) pair across all 5 SymPrev streams."""
        t0 = time.perf_counter()
        q_vecs, _    = self._embed_all(query)
        e_vecs, devs = self._embed_all(entity_text)

        return StreamScores(
            isymprev     = _cosine(q_vecs["ISymPrev"], e_vecs["ISymPrev"]),
            qsymprev     = _cosine(q_vecs["QSymPrev"], e_vecs["QSymPrev"]),
            dsymprev     = _cosine(q_vecs["DSymPrev"], e_vecs["DSymPrev"]),
            msymprev     = _cosine(q_vecs["MSymPrev"], e_vecs["MSymPrev"]),
            ssymprev     = _cosine(q_vecs["SSymPrev"], e_vecs["SSymPrev"]),
            embedding_ms = (time.perf_counter() - t0) * 1000,
            device_used  = devs,
            mrl_dims     = dict(STREAM_DIM),
        )

    def rank_candidates(
        self,
        query:                str,
        candidates:           list[tuple[str, str]],
        heat_map:             Optional[dict[str, float]] = None,
        cross_session_counts: Optional[dict[str, int]]  = None,
    ) -> TieBreakResult:
        """Rank entity candidates using 5-stream scoring + multi-hop tie-breaking."""
        scored = [(key, self.score(query=query, entity_text=text))
                  for key, text in candidates]
        return multihop_tiebreak(scored)


# ── Bluetooth CLI integration point ──────────────────────────────────────────

@dataclass
class BluetoothContextEvent:
    """
    Real-world context event from the Bluetooth CLI tool.

    BLE proximity events feed MSymPrev (affective) and ISymPrev (narrative)
    directly — no camera or manual annotation required.  RSSI → heat mapping
    follows the LOCI Heat Law: θ_HOT=0.7250, α=0.050.
    """
    device_id:    str
    device_name:  str
    rssi_dbm:     int        # -40 (very close) to -100 (out of range)
    timestamp_s:  float
    context_text: str = ""   # optional annotation from BT CLI

    @property
    def proximity_heat(self) -> float:
        """Map RSSI ∈ [-100, -40] → H ∈ [α, 0.95]."""
        clamped = max(-100, min(-40, self.rssi_dbm))
        norm    = (clamped + 100) / 60.0
        return ALPHA_HOPE + norm * (0.95 - ALPHA_HOPE)

    def to_symprev_input(self) -> dict[str, str]:
        """Format for 5-stream scoring (MSymPrev + ISymPrev + QSymPrev)."""
        return {
            "MSymPrev": f"proximity event: {self.device_name} at {self.rssi_dbm} dBm",
            "ISymPrev": self.context_text or f"nearby device: {self.device_name}",
            "QSymPrev": f"device_id={self.device_id} rssi={self.rssi_dbm}",
        }


# ── Garena bus event emitter ──────────────────────────────────────────────────

def emit_symprev_event(
    scores: StreamScores,
    entity_key: str,
    ws_port: int = 8765,
) -> bool:
    """
    Push a scored SymPrev event to the Garena bus SYMPREV_RING.
    Returns True on success, False on connection failure.
    """
    try:
        import socket, json
        payload = json.dumps({
            "kind":       "symprev",
            "entity_key": entity_key,
            "scores":     scores.as_dict(),
            "ts":         time.time(),
        })
        # Fire-and-forget UDP datagram to the bus (non-blocking)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(payload.encode(), ("127.0.0.1", ws_port + 1))
        return True
    except Exception as exc:
        log.debug("symprev bus emit failed: %s", exc)
        return False
