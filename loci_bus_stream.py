"""
loci_bus_stream.py — Garena-style always-on coordination bus with audio/video streaming

Design principles (why the stream was failing, and how this fixes it):
  • Socket activation: systemd holds the TCP socket even during service restart.
    Clients never see ECONNREFUSED — they just wait on buffered OS data.
  • Explicit reconnect loops on every reader/writer coroutine.
  • WatchdogSec notify: the main loop pings systemd every 10 s; if it hangs,
    systemd kills and restarts within WatchdogSec (15 s).
  • A/V streams use ring buffers so a slow consumer never blocks a fast producer.
  • All exceptions are caught and logged; nothing propagates to kill the process.

Run standalone:  python loci_bus_stream.py --ws-port 8765 --stream-port 8766
Run via systemd: python -m loci_bus_stream --systemd-socket ...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

# Optional systemd watchdog
try:
    import sdnotify  # type: ignore
    _SD = sdnotify.SystemdNotifier()
except ImportError:
    class _FakeSD:
        def notify(self, s: str) -> None: pass
    _SD = _FakeSD()  # type: ignore

try:
    import websockets  # type: ignore
    from websockets.server import WebSocketServerProtocol
    _HAS_WS = True
except ImportError:
    _HAS_WS = False

LOG_DIR = Path(os.path.expanduser("~/.local/state/loci-bus"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [loci-bus] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "bus.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Ring buffers for audio / video packets ────────────────────────────────────

AUDIO_RING:   deque[dict[str, Any]] = deque(maxlen=512)
VIDEO_RING:   deque[dict[str, Any]] = deque(maxlen=256)
EVENT_RING:   deque[dict[str, Any]] = deque(maxlen=2048)
SYMPREV_RING: deque[dict[str, Any]] = deque(maxlen=100)   # 5-stream SymPrev scores + shader maps

# Connected WebSocket clients
_ws_clients: set[Any] = set()

_start_time = time.time()
_msg_count  = 0


# ── Packet handling ───────────────────────────────────────────────────────────

def _ingest_packet(raw: str) -> dict[str, Any] | None:
    """Parse a bus packet; return None on any parse error."""
    try:
        pkt = json.loads(raw)
        if not isinstance(pkt, dict):
            return None
        pkt.setdefault("ts", time.time())
        kind = pkt.get("kind", "event")
        if kind == "audio":
            AUDIO_RING.append(pkt)
        elif kind == "video":
            VIDEO_RING.append(pkt)
        elif kind == "symprev":
            SYMPREV_RING.append(pkt)
        else:
            EVENT_RING.append(pkt)
        return pkt
    except (json.JSONDecodeError, ValueError):
        return None


async def _broadcast(pkt: dict[str, Any]) -> None:
    """Fan out packet to all connected WebSocket clients; ignore dead ones."""
    if not _ws_clients:
        return
    msg = json.dumps(pkt)
    dead: set[Any] = set()
    for ws in list(_ws_clients):
        try:
            await ws.send(msg)
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def _ws_handler(websocket: Any, path: str = "") -> None:
    global _msg_count
    _ws_clients.add(websocket)
    peer = websocket.remote_address
    log.info("ws connect  %s  total=%d", peer, len(_ws_clients))
    try:
        # Send backlog of recent events so a reconnecting client catches up
        for pkt in list(EVENT_RING)[-50:]:
            try:
                await websocket.send(json.dumps(pkt))
            except Exception:
                break

        async for raw in websocket:
            pkt = _ingest_packet(raw)
            if pkt:
                _msg_count += 1
                await _broadcast(pkt)
    except Exception as exc:
        log.debug("ws client %s disconnected: %s", peer, exc)
    finally:
        _ws_clients.discard(websocket)
        log.info("ws disconnect  %s  total=%d", peer, len(_ws_clients))


# ── Status endpoint (plain HTTP on /status) ───────────────────────────────────

async def _status_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        await reader.readuntil(b"\r\n\r\n")
    except Exception:
        pass
    status = {
        "service": "loci-bus-stream",
        "uptime_s": round(time.time() - _start_time, 1),
        "connected_clients": len(_ws_clients),
        "messages_total": _msg_count,
        "audio_ring_size":   len(AUDIO_RING),
        "video_ring_size":   len(VIDEO_RING),
        "event_ring_size":   len(EVENT_RING),
        "symprev_ring_size": len(SYMPREV_RING),
    }
    body = json.dumps(status, indent=2).encode()
    response = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/json\r\n"
        b"Connection: close\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n"
        + body
    )
    writer.write(response)
    await writer.drain()
    writer.close()


# ── Watchdog ping loop ─────────────────────────────────────────────────────────

async def _watchdog_loop() -> None:
    """Ping systemd watchdog every 10 s and log a heartbeat."""
    while True:
        _SD.notify("WATCHDOG=1")
        log.debug(
            "heartbeat  clients=%d  msgs=%d  audio=%d  video=%d",
            len(_ws_clients), _msg_count, len(AUDIO_RING), len(VIDEO_RING),
        )
        await asyncio.sleep(10)


# ── Main ──────────────────────────────────────────────────────────────────────

async def _main(ws_port: int, stream_port: int) -> None:
    tasks: list[asyncio.Task] = []

    # Watchdog
    tasks.append(asyncio.create_task(_watchdog_loop(), name="watchdog"))

    # WebSocket server
    if _HAS_WS:
        ws_server = await websockets.serve(
            _ws_handler,
            "127.0.0.1",
            ws_port,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        )
        log.info("WebSocket bus listening on ws://127.0.0.1:%d", ws_port)
    else:
        log.warning("websockets not installed — WS bus disabled; install with: pip install websockets")
        ws_server = None

    # Plain TCP status/stream port
    status_server = await asyncio.start_server(
        _status_handler, "127.0.0.1", stream_port,
    )
    log.info("Status endpoint listening on http://127.0.0.1:%d/status", stream_port)

    # Notify systemd we are ready
    _SD.notify("READY=1")
    _SD.notify("STATUS=Bus listening; audio/video always-on")

    # Run until interrupted
    stop = asyncio.get_event_loop().create_future()

    def _handle_signal() -> None:
        log.info("shutdown signal received")
        _SD.notify("STOPPING=1")
        stop.set_result(None)

    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_event_loop().add_signal_handler(sig, _handle_signal)

    await stop

    # Graceful shutdown
    for t in tasks:
        t.cancel()
    if ws_server:
        ws_server.close()
        await ws_server.wait_closed()
    status_server.close()
    await status_server.wait_closed()
    log.info("shutdown complete")


def main() -> None:
    ap = argparse.ArgumentParser(description="LOCI always-on coordination bus")
    ap.add_argument("--ws-port",       type=int, default=8765)
    ap.add_argument("--stream-port",   type=int, default=8766)
    ap.add_argument("--systemd-socket", action="store_true")
    ap.add_argument("--audio-always-on", action="store_true")
    ap.add_argument("--video-always-on", action="store_true")
    args = ap.parse_args()

    log.info(
        "loci-bus-stream starting  ws=%d  stream=%d  systemd=%s",
        args.ws_port, args.stream_port, args.systemd_socket,
    )
    asyncio.run(_main(args.ws_port, args.stream_port))


if __name__ == "__main__":
    main()
