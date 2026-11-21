#!/usr/bin/env python3
"""Send minimized, sequenced Arc workstation readiness to the private VPS chat."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen
from datetime import datetime, timezone


STATE_DIR = Path.home() / ".local/state/arc-telemetry"
SEQUENCE_FILE = STATE_DIR / "sequence"
STATUS_URL = os.environ.get(
    "ARC_VPS_TELEMETRY_URL", "https://billo.tech/arko/chat/api/telemetry"
)
TOKEN = os.environ.get("ARC_VPS_TELEMETRY_TOKEN", "")


def get_json(url: str, timeout: float = 4.0) -> dict | None:
    try:
        with urlopen(url, timeout=timeout) as response:
            value = json.load(response)
            return value if isinstance(value, dict) else None
    except Exception:
        return None


def next_sequence() -> int:
    STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        previous = int(SEQUENCE_FILE.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        previous = 0
    current = previous + 1
    temp = SEQUENCE_FILE.with_suffix(".tmp")
    temp.write_text(str(current), encoding="ascii")
    os.chmod(temp, 0o600)
    os.replace(temp, SEQUENCE_FILE)
    return current


def main() -> int:
    if not TOKEN:
        raise SystemExit("ARC_VPS_TELEMETRY_TOKEN is not configured")
    arc = get_json("http://127.0.0.1:7860/api/status")
    if arc is None:
        # Don't report a guessed failure as an authenticated state packet.
        raise SystemExit("Arc status endpoint is unavailable")
    media = get_json("http://127.0.0.1:11500/api/status") or {}
    devices = media.get("stream", {}) if isinstance(media.get("stream"), dict) else {}
    cameras = devices.get("camera_nodes", [])
    microphones = devices.get("audio_inputs", [])
    lanes = arc.get("inference_lanes", {})
    payload = {
        "agent_id": "workstation",
        "sequence": next_sequence(),
        "sent_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arc_api": True,
        "heartbeat_tick": (arc.get("heartbeat") or {}).get("tick_id"),
        "inference_lanes": {
            name: {"ready": bool((lanes.get(name) or {}).get("ready")),
                   "models": list((lanes.get(name) or {}).get("models", []))[:12]}
            for name in ("rtx", "arc")
        },
        "devices": {"camera_count": len(cameras), "microphone_count": len(microphones)},
    }
    body = json.dumps(payload, separators=(",", ":")).encode()
    request = Request(STATUS_URL, data=body,
                      headers={"Content-Type": "application/json",
                               "Authorization": f"Bearer {TOKEN}"}, method="POST")
    try:
        with urlopen(request, timeout=10) as response:
            result = json.load(response)
    except Exception as exc:
        raise SystemExit(f"VPS telemetry delivery failed ({type(exc).__name__})") from None
    print(json.dumps({"status": result.get("status"), "sequence": payload["sequence"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
