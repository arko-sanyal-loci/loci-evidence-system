"""
arc_token_offset.py — Fault-tolerant Arc GPU token offset service

The Arc B70 (Y=1) always provides local inference compute to offset API token
costs.  Rules:
  1. Never crash on a bad computation graph — catch everything, return fallback.
  2. If Arc VRAM is unavailable, fall back to RTX 5060 Ti (Y=0), then CPU.
  3. Expose a FastAPI status endpoint at :9099 so the IDE panel stays green.
  4. Log each offset event for the token economy ledger.

Run:  python arc_token_offset.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    import uvicorn
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

LOG_DIR = Path(os.getenv("ARC_LOG_DIR", os.path.expanduser("~/.local/state/arc-token-offset")))
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [token-offset] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "offset.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

LANE_ARC = 1   # Arc B70  32 GB
LANE_RTX = 0   # RTX 5060 Ti 16 GB


# ── Hardware query (never raises) ─────────────────────────────────────────────

def _query_free_mb(device: int) -> Optional[int]:
    """Query free VRAM; returns None on any failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={device}",
             "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=3,
        ).decode().strip()
        return int(out)
    except Exception:
        pass
    try:
        # Intel Arc via xpu-smi or sysfs
        sysfs = Path(f"/sys/class/drm/card{device}/device/mem_info_vram_used")
        if sysfs.exists():
            used_b = int(sysfs.read_text().strip())
            total_sysfs = Path(f"/sys/class/drm/card{device}/device/mem_info_vram_total")
            total_b = int(total_sysfs.read_text().strip()) if total_sysfs.exists() else 32 * 1024**3
            return int((total_b - used_b) / 1024**2)
    except Exception:
        pass
    return None


@dataclass
class DeviceStatus:
    device: int
    name: str
    free_mb: Optional[int]
    available: bool
    queried_at: float = field(default_factory=time.time)


def probe_devices() -> list[DeviceStatus]:
    devices = [
        DeviceStatus(device=LANE_ARC, name="Arc B70",      free_mb=_query_free_mb(LANE_ARC), available=False),
        DeviceStatus(device=LANE_RTX, name="RTX 5060 Ti",  free_mb=_query_free_mb(LANE_RTX), available=False),
    ]
    for d in devices:
        d.available = d.free_mb is not None and d.free_mb > 512
    return devices


# ── Offset event ──────────────────────────────────────────────────────────────

@dataclass
class OffsetEvent:
    request_id:   str
    prompt_tokens: int
    device_used:  str
    latency_ms:   float
    fallback:     bool
    error:        Optional[str] = None
    ts:           float = field(default_factory=time.time)

    def to_ledger_line(self) -> str:
        return json.dumps({
            "ts": self.ts,
            "req": self.request_id,
            "tokens": self.prompt_tokens,
            "device": self.device_used,
            "latency_ms": self.latency_ms,
            "fallback": self.fallback,
            "error": self.error,
        })


_ledger_path = LOG_DIR / "ledger.jsonl"
_events: list[OffsetEvent] = []


def _log_event(evt: OffsetEvent) -> None:
    _events.append(evt)
    with _ledger_path.open("a") as f:
        f.write(evt.to_ledger_line() + "\n")


# ── Fault-tolerant inference dispatch ─────────────────────────────────────────

async def offset_inference(
    request_id: str,
    prompt_tokens: int,
    payload: Any,
    arc_endpoint: str = "http://localhost:7860",
) -> dict[str, Any]:
    """
    Attempt inference on Arc B70.  If the graph is invalid or Arc is busy,
    fall back to RTX endpoint, then to a stub result — never raises.
    """
    t0 = time.monotonic()
    devices = probe_devices()
    arc_ok = any(d.device == LANE_ARC and d.available for d in devices)
    rtx_ok = any(d.device == LANE_RTX and d.available for d in devices)

    # ── Try Arc first ──────────────────────────────────────────────────────────
    if arc_ok:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=3.0)) as client:
                r = await client.post(f"{arc_endpoint}/api/offset", json={
                    "request_id": request_id,
                    "tokens": prompt_tokens,
                    "payload": payload,
                })
                r.raise_for_status()
                data = r.json()
                latency = (time.monotonic() - t0) * 1000
                evt = OffsetEvent(request_id, prompt_tokens, "arc-b70", latency, fallback=False)
                _log_event(evt)
                log.info("offset arc-b70  req=%s  tokens=%d  %.0f ms", request_id, prompt_tokens, latency)
                return data
        except Exception as exc:
            log.warning("arc offset failed (%s) — trying RTX fallback", exc)

    # ── Try RTX fallback ───────────────────────────────────────────────────────
    if rtx_ok:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=3.0)) as client:
                r = await client.post(f"{arc_endpoint}/api/offset_rtx", json={
                    "request_id": request_id,
                    "tokens": prompt_tokens,
                    "payload": payload,
                })
                r.raise_for_status()
                data = r.json()
                latency = (time.monotonic() - t0) * 1000
                evt = OffsetEvent(request_id, prompt_tokens, "rtx-5060ti", latency, fallback=True)
                _log_event(evt)
                log.info("offset rtx-5060ti  req=%s  tokens=%d  %.0f ms", request_id, prompt_tokens, latency)
                return data
        except Exception as exc:
            log.warning("rtx offset failed (%s) — using stub", exc)

    # ── Stub result — service never returns an error ───────────────────────────
    latency = (time.monotonic() - t0) * 1000
    stub_error = "both devices unavailable"
    evt = OffsetEvent(request_id, prompt_tokens, "stub-cpu", latency, fallback=True, error=stub_error)
    _log_event(evt)
    log.warning("offset stub  req=%s  tokens=%d  %.0f ms", request_id, prompt_tokens, latency)
    return {
        "request_id": request_id,
        "device": "stub",
        "result": None,
        "fallback": True,
        "error": stub_error,
    }


# ── FastAPI service ───────────────────────────────────────────────────────────

if _HAS_FASTAPI:
    app = FastAPI(title="Arc Token Offset", version="1.0")

    @app.get("/status")
    async def status() -> JSONResponse:
        devices = probe_devices()
        total_offset = sum(e.prompt_tokens for e in _events if not e.fallback)
        return JSONResponse({
            "service": "arc-token-offset",
            "devices": [
                {"name": d.name, "device": d.device,
                 "free_mb": d.free_mb, "available": d.available}
                for d in devices
            ],
            "total_tokens_offset": total_offset,
            "event_count": len(_events),
            "ledger": str(_ledger_path),
        })

    @app.post("/offset")
    async def offset(body: dict) -> JSONResponse:
        result = await offset_inference(
            request_id=body.get("request_id", "unknown"),
            prompt_tokens=body.get("tokens", 0),
            payload=body.get("payload"),
        )
        return JSONResponse(result)

    def serve(port: int = 9099) -> None:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

else:
    def serve(port: int = 9099) -> None:
        log.warning("fastapi/uvicorn not installed — status endpoint disabled; offset logic still runs")


if __name__ == "__main__":
    serve()
