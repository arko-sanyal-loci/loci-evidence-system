"""
loci_cfmoto_bt_cli.py — CFMOTO 300SR BLE/OBD-II diagnostic CLI + LOCI SymPrev bridge

Architecture
────────────
  CFMOTOBLEScanner   — async BLE scan for ELM327 adapters
  ELM327BLEClient    — async BLE transport (bleak); sends AT commands, reads responses
  OBDParser          — decodes ELM327 hex frames into typed Python values
  CFMOTOTelemetryStream — continuous poll loop; emits BluetoothContextEvent objects
  SymPrevBridge      — feeds events into the LOCI SymPrev pipeline via SYMPREV_RING

CLI commands
────────────
  scan               — list nearby BLE devices (filter by ELM327 name patterns)
  connect <addr>     — connect + init ELM327 + enter interactive REPL
  stream [--addr <addr>] [--interval 1.0] — continuous telemetry print + SymPrev feed
  dtc    [--addr <addr>] — read and decode stored fault codes (Mode 03)
  clear-dtc [--addr <addr>] — clear stored DTCs (Mode 04)

BLE UUIDs
─────────
  ELM327 BLE adapters (Vgate iCar Pro, OBDLink CX, etc.) typically expose:
    FFF0  service
    FFF1  write characteristic  (commands → adapter)
    FFF2  notify characteristic (responses ← adapter)

  Some newer adapters use the Nordic UART Service (NUS):
    6e400001-b5a3-f393-e0a9-e50e24dcca9e  service
    6e400002-b5a3-f393-e0a9-e50e24dcca9e  TX (write)
    6e400003-b5a3-f393-e0a9-e50e24dcca9e  RX (notify)

  The client tries FFF0/FFF1/FFF2 first, then NUS; set OBD_BLE_SERVICE_UUID /
  OBD_BLE_WRITE_UUID / OBD_BLE_NOTIFY_UUID env vars to override.

OBD-II PIDs polled
──────────────────
  0105  Coolant temperature      °C
  010C  Engine RPM               rpm
  010D  Vehicle speed            km/h
  010F  Intake air temperature   °C
  0111  Throttle position        %
  0104  Calculated engine load   %
  010E  Timing advance           °BTDC
  0142  Control module voltage   V
  Mode 03  Request stored DTCs

LOCI SymPrev integration
────────────────────────
  BluetoothContextEvent (stub in loci_symprev_streams.py) is populated with live
  telemetry and fed into SYMPREV_RING on the coordination bus:
    proximity_heat  ← RSSI [-100, -40] → heat [α, 0.95]
    to_symprev_input() → {"MSymPrev": ..., "ISymPrev": ..., "QSymPrev": ...}

Dependencies
────────────
  pip install bleak

Usage
─────
  python loci_cfmoto_bt_cli.py scan
  python loci_cfmoto_bt_cli.py connect AA:BB:CC:DD:EE:FF
  python loci_cfmoto_bt_cli.py stream --addr AA:BB:CC:DD:EE:FF --interval 2.0
  python loci_cfmoto_bt_cli.py dtc --addr AA:BB:CC:DD:EE:FF
  python loci_cfmoto_bt_cli.py clear-dtc --addr AA:BB:CC:DD:EE:FF
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

log = logging.getLogger("loci.cfmoto")

# ── BLE UUID configuration ────────────────────────────────────────────────────

_SVC_FFF0   = "0000fff0-0000-1000-8000-00805f9b34fb"
_CHR_FFF1   = "0000fff1-0000-1000-8000-00805f9b34fb"   # write (commands)
_CHR_FFF2   = "0000fff2-0000-1000-8000-00805f9b34fb"   # notify (responses)

_SVC_NUS    = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"   # Nordic UART Service
_CHR_NUS_TX = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"   # write
_CHR_NUS_RX = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"   # notify

# Allow override via env
BLE_SERVICE_UUID = os.getenv("OBD_BLE_SERVICE_UUID", "")
BLE_WRITE_UUID   = os.getenv("OBD_BLE_WRITE_UUID",   "")
BLE_NOTIFY_UUID  = os.getenv("OBD_BLE_NOTIFY_UUID",  "")

# Common ELM327 BLE adapter name fragments (case-insensitive)
_ELM327_NAME_HINTS = [
    "obd", "elm", "vgate", "obdlink", "veepeak", "konnwei",
    "icar", "bluedriver", "carista", "odb", "scan",
]

# ── ELM327 initialization sequence ───────────────────────────────────────────

_AT_INIT_SEQUENCE = [
    b"ATZ\r",    # reset all
    b"ATE0\r",   # echo off
    b"ATL0\r",   # linefeeds off
    b"ATS0\r",   # spaces off
    b"ATH0\r",   # headers off
    b"ATSP0\r",  # auto-detect protocol
    b"ATAT1\r",  # adaptive timing on
    b"ATST64\r", # set timeout 100ms (0x64)
]

# ── OBD-II PID definitions ────────────────────────────────────────────────────

@dataclass
class PIDDef:
    pid:    str    # e.g. "010C"
    name:   str
    unit:   str
    decode: Callable[[list[int]], float]


def _d_rpm(b: list[int]) -> float:
    return ((b[0] * 256) + b[1]) / 4.0


def _d_speed(b: list[int]) -> float:
    return float(b[0])


def _d_temp(b: list[int]) -> float:
    return float(b[0]) - 40.0


def _d_pct_a(b: list[int]) -> float:
    return b[0] * 100.0 / 255.0


def _d_throttle(b: list[int]) -> float:
    return b[0] * 100.0 / 255.0


def _d_timing(b: list[int]) -> float:
    return (b[0] - 128) / 2.0


def _d_voltage(b: list[int]) -> float:
    return ((b[0] * 256) + b[1]) / 1000.0


PIDS: list[PIDDef] = [
    PIDDef("010C", "RPM",             "rpm",  _d_rpm),
    PIDDef("010D", "Speed",           "km/h", _d_speed),
    PIDDef("0105", "Coolant",         "°C",   _d_temp),
    PIDDef("010F", "Intake Air",      "°C",   _d_temp),
    PIDDef("0111", "Throttle",        "%",    _d_throttle),
    PIDDef("0104", "Engine Load",     "%",    _d_pct_a),
    PIDDef("010E", "Timing Advance",  "°",    _d_timing),
    PIDDef("0142", "Ctrl Module V",   "V",    _d_voltage),
]

_PID_MAP: dict[str, PIDDef] = {p.pid: p for p in PIDS}

# ── DTC helpers ───────────────────────────────────────────────────────────────

_DTC_PREFIX = {0: "P", 1: "C", 2: "B", 3: "U"}

def _decode_dtc_pair(high: int, low: int) -> str:
    system = (high & 0xC0) >> 6
    digit1 = (high & 0x30) >> 4
    digits = f"{digit1:1X}{high & 0x0F:1X}{low >> 4:1X}{low & 0x0F:1X}"
    return f"{_DTC_PREFIX[system]}{digits}"


def _dtcs_from_response(raw: str) -> list[str]:
    """Parse Mode 03 ELM327 response into DTC string list."""
    tokens = raw.strip().replace("\r", " ").replace("\n", " ").split()
    tokens = [t for t in tokens if t not in ("43", ">", "NODATA", "OK")]
    dtcs: list[str] = []
    for i in range(0, len(tokens) - 1, 2):
        try:
            high = int(tokens[i],   16)
            low  = int(tokens[i+1], 16)
        except ValueError:
            continue
        if high == 0 and low == 0:
            continue
        dtcs.append(_decode_dtc_pair(high, low))
    return dtcs


# ── OBDParser ─────────────────────────────────────────────────────────────────

class OBDParser:
    """
    Parses raw ELM327 response strings into typed Python values.
    Returns None for NO DATA / error responses.
    """

    @staticmethod
    def parse_pid(pid: str, raw: str) -> Optional[float]:
        """Parse a standard Mode 01 PID response."""
        raw = raw.strip().upper().replace("\r", "").replace("\n", "").replace(" ", "")
        if any(err in raw for err in ("NODATA", "ERROR", "UNABLE", "STOPPED", "BUS")):
            return None
        pid_def = _PID_MAP.get(pid)
        if pid_def is None:
            return None
        # ELM327 response: "41 0C 1A F8" (without spaces after ATS0, ATH0)
        # Strip leading "41XX" response bytes
        try:
            # remove mode byte (41) and pid byte
            mode_pid_len = 4   # "41" + 2 hex pid chars
            data_hex = raw[mode_pid_len:] if raw.startswith("41") else raw
            data_bytes = [int(data_hex[i:i+2], 16) for i in range(0, len(data_hex), 2)]
            return pid_def.decode(data_bytes)
        except (ValueError, IndexError):
            return None

    @staticmethod
    def parse_dtcs(raw: str) -> list[str]:
        return _dtcs_from_response(raw)


# ── Telemetry snapshot ────────────────────────────────────────────────────────

@dataclass
class CFMOTOSnapshot:
    ts:            float = field(default_factory=time.time)
    rpm:           Optional[float] = None
    speed_kmh:     Optional[float] = None
    coolant_c:     Optional[float] = None
    intake_air_c:  Optional[float] = None
    throttle_pct:  Optional[float] = None
    engine_load:   Optional[float] = None
    timing_deg:    Optional[float] = None
    voltage_v:     Optional[float] = None
    rssi:          Optional[int]   = None
    addr:          str             = ""

    def pretty(self) -> str:
        def fmt(v: Optional[float], unit: str, fmt: str = ".1f") -> str:
            return f"{v:{fmt}} {unit}" if v is not None else "—"
        lines = [
            f"  RPM          {fmt(self.rpm, 'rpm', '.0f')}",
            f"  Speed        {fmt(self.speed_kmh, 'km/h', '.0f')}",
            f"  Coolant      {fmt(self.coolant_c, '°C')}",
            f"  Intake Air   {fmt(self.intake_air_c, '°C')}",
            f"  Throttle     {fmt(self.throttle_pct, '%')}",
            f"  Engine Load  {fmt(self.engine_load, '%')}",
            f"  Timing       {fmt(self.timing_deg, '°')}",
            f"  Voltage      {fmt(self.voltage_v, 'V')}",
        ]
        if self.rssi is not None:
            lines.append(f"  RSSI         {self.rssi} dBm")
        return "\n".join(lines)


# ── ELM327BLEClient ───────────────────────────────────────────────────────────

class ELM327BLEClient:
    """
    Async BLE client for ELM327 OBD-II adapters.

    Sends CR-terminated commands; collects notify callbacks until '>' prompt.
    """

    def __init__(self, address: str, timeout: float = 10.0) -> None:
        self._address   = address
        self._timeout   = timeout
        self._client    = None
        self._write_chr: Optional[str] = None
        self._buf        = bytearray()
        self._response_q: asyncio.Queue = asyncio.Queue()

    # ── connection ────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        try:
            from bleak import BleakClient  # type: ignore
        except ImportError:
            _die("bleak is not installed. Run: pip install bleak")

        self._client = BleakClient(self._address, timeout=self._timeout)
        await self._client.connect()

        notify_uuid, write_uuid = await self._discover_uuids()
        self._write_chr = write_uuid

        await self._client.start_notify(notify_uuid, self._on_notify)
        log.info("Connected to %s  write=%s  notify=%s", self._address, write_uuid, notify_uuid)

    async def disconnect(self) -> None:
        if self._client and self._client.is_connected:
            await self._client.disconnect()

    async def _discover_uuids(self) -> tuple[str, str]:
        """
        Try FFF0/FFF1/FFF2 first; fall back to Nordic UART Service.
        Env vars OBD_BLE_* override.
        """
        if BLE_SERVICE_UUID and BLE_WRITE_UUID and BLE_NOTIFY_UUID:
            return BLE_NOTIFY_UUID, BLE_WRITE_UUID

        svcs = {str(s.uuid).lower() for s in self._client.services}

        if _SVC_FFF0 in svcs:
            return _CHR_FFF2, _CHR_FFF1

        if _SVC_NUS in svcs:
            return _CHR_NUS_RX, _CHR_NUS_TX

        # Last resort: iterate characteristics and guess by properties
        for svc in self._client.services:
            chars = list(svc.characteristics)
            writes  = [c for c in chars if "write" in c.properties]
            notifs  = [c for c in chars if "notify" in c.properties]
            if writes and notifs:
                log.warning("Guessing UUIDs: write=%s notify=%s", writes[0].uuid, notifs[0].uuid)
                return str(notifs[0].uuid), str(writes[0].uuid)

        raise RuntimeError(
            f"No compatible OBD-II BLE service found on {self._address}.\n"
            "Set OBD_BLE_SERVICE_UUID / OBD_BLE_WRITE_UUID / OBD_BLE_NOTIFY_UUID if needed."
        )

    # ── notify callback ───────────────────────────────────────────────────────

    def _on_notify(self, _handle: int, data: bytearray) -> None:
        self._buf.extend(data)
        if b">" in self._buf:
            resp = self._buf.decode("ascii", errors="replace")
            self._buf.clear()
            asyncio.get_event_loop().call_soon_threadsafe(
                self._response_q.put_nowait, resp
            )

    # ── command interface ─────────────────────────────────────────────────────

    async def send(self, cmd: bytes, timeout: float = 5.0) -> str:
        """Send a command and wait for the ELM327 '>' prompt."""
        if not self._client or not self._client.is_connected:
            raise RuntimeError("BLE client not connected")

        # Clear any stale data
        while not self._response_q.empty():
            self._response_q.get_nowait()
        self._buf.clear()

        await self._client.write_gatt_char(self._write_chr, cmd, response=False)
        try:
            return await asyncio.wait_for(self._response_q.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return ""

    async def send_str(self, cmd: str, timeout: float = 5.0) -> str:
        return await self.send((cmd.upper().strip() + "\r").encode(), timeout=timeout)

    # ── ELM327 initialization ─────────────────────────────────────────────────

    async def init_elm327(self) -> str:
        """Send the AT initialization sequence; return the last response."""
        resp = ""
        for cmd in _AT_INIT_SEQUENCE:
            resp = await self.send(cmd, timeout=8.0)
            await asyncio.sleep(0.1)
        return resp

    # ── OBD-II query ─────────────────────────────────────────────────────────

    async def query_pid(self, pid: str) -> Optional[float]:
        """Query a single OBD-II PID and return the decoded float value."""
        raw = await self.send_str(pid, timeout=4.0)
        return OBDParser.parse_pid(pid, raw)

    async def request_dtcs(self) -> list[str]:
        """Mode 03: read stored DTCs."""
        raw = await self.send_str("03", timeout=6.0)
        return OBDParser.parse_dtcs(raw)

    async def clear_dtcs(self) -> str:
        """Mode 04: clear stored DTCs."""
        raw = await self.send_str("04", timeout=6.0)
        return raw.strip()

    # ── RSSI ─────────────────────────────────────────────────────────────────

    async def rssi(self) -> Optional[int]:
        try:
            return await self._client.get_rssi()
        except Exception:
            return None


# ── CFMOTOTelemetryStream ─────────────────────────────────────────────────────

class CFMOTOTelemetryStream:
    """
    Continuous OBD-II polling loop.

    Emits CFMOTOSnapshot objects to registered callbacks.
    Also constructs BluetoothContextEvent objects and feeds them to SymPrevBridge.
    """

    def __init__(
        self,
        client: ELM327BLEClient,
        interval: float = 1.0,
        symprev_bridge: Optional["SymPrevBridge"] = None,
    ) -> None:
        self._client   = client
        self._interval = interval
        self._bridge   = symprev_bridge
        self._running  = False
        self._callbacks: list[Callable[[CFMOTOSnapshot], None]] = []

    def add_callback(self, cb: Callable[[CFMOTOSnapshot], None]) -> None:
        self._callbacks.append(cb)

    async def _poll_once(self) -> CFMOTOSnapshot:
        snap = CFMOTOSnapshot(addr=self._client._address)
        snap.ts = time.time()

        # Poll all PIDs sequentially (ELM327 BLE is half-duplex)
        pid_to_attr = {
            "010C": "rpm",
            "010D": "speed_kmh",
            "0105": "coolant_c",
            "010F": "intake_air_c",
            "0111": "throttle_pct",
            "0104": "engine_load",
            "010E": "timing_deg",
            "0142": "voltage_v",
        }
        for pid, attr in pid_to_attr.items():
            val = await self._client.query_pid(pid)
            setattr(snap, attr, val)

        snap.rssi = await self._client.rssi()
        return snap

    async def run(self) -> None:
        self._running = True
        while self._running:
            t0 = time.time()
            try:
                snap = await self._poll_once()
            except Exception as exc:
                log.warning("Poll error: %s", exc)
                await asyncio.sleep(self._interval)
                continue

            for cb in self._callbacks:
                try:
                    cb(snap)
                except Exception as exc:
                    log.warning("Callback error: %s", exc)

            if self._bridge is not None:
                try:
                    await self._bridge.push(snap)
                except Exception as exc:
                    log.warning("SymPrev bridge error: %s", exc)

            elapsed = time.time() - t0
            await asyncio.sleep(max(0.0, self._interval - elapsed))

    def stop(self) -> None:
        self._running = False


# ── SymPrevBridge ─────────────────────────────────────────────────────────────

class SymPrevBridge:
    """
    Feeds CFMOTOSnapshot into the LOCI SymPrev pipeline via BluetoothContextEvent.

    Integrates with loci_symprev_streams.BluetoothContextEvent (the stub already
    written there). Falls back gracefully if loci_symprev_streams is unavailable.
    """

    def __init__(self) -> None:
        self._ring = None
        self._BluetoothContextEvent = None
        self._load_loci()

    def _load_loci(self) -> None:
        try:
            import importlib
            mod = importlib.import_module("loci_symprev_streams")
            self._BluetoothContextEvent = getattr(mod, "BluetoothContextEvent", None)
            # Try to get the shared SYMPREV_RING
            try:
                bus_mod = importlib.import_module("loci_bus_stream")
                self._ring = getattr(bus_mod, "SYMPREV_RING", None)
            except ImportError:
                pass
            log.info("SymPrev bridge loaded (BluetoothContextEvent=%s)", self._BluetoothContextEvent)
        except ImportError:
            log.info("loci_symprev_streams not in path — SymPrev bridge disabled")

    async def push(self, snap: CFMOTOSnapshot) -> None:
        if self._BluetoothContextEvent is None:
            return

        # Build a fake RSSI-based BT event
        rssi = snap.rssi if snap.rssi is not None else -70
        event = self._BluetoothContextEvent(
            rssi         = rssi,
            device_name  = "CFMOTO-300SR-OBD",
            device_addr  = snap.addr,
            payload      = {
                "rpm":          snap.rpm,
                "speed_kmh":    snap.speed_kmh,
                "coolant_c":    snap.coolant_c,
                "intake_air_c": snap.intake_air_c,
                "throttle_pct": snap.throttle_pct,
                "engine_load":  snap.engine_load,
                "voltage_v":    snap.voltage_v,
            },
        )

        symprev_input = event.to_symprev_input()
        symprev_input["_source"] = "cfmoto_obd"
        symprev_input["_ts"]     = snap.ts

        if self._ring is not None:
            try:
                self._ring.append(symprev_input)
            except Exception as exc:
                log.debug("SYMPREV_RING append: %s", exc)
        else:
            log.debug("SymPrev payload (no ring): %s", symprev_input)


# ── CFMOTOBLEScanner ─────────────────────────────────────────────────────────

class CFMOTOBLEScanner:
    """
    Async BLE scan that highlights likely ELM327 / OBD-II adapters.
    """

    @staticmethod
    async def scan(duration: float = 10.0) -> list[dict]:
        try:
            from bleak import BleakScanner  # type: ignore
        except ImportError:
            _die("bleak is not installed. Run: pip install bleak")

        print(f"Scanning for {duration:.0f} s …")
        devices = await BleakScanner.discover(timeout=duration, return_adv=True)

        results = []
        for addr, (device, adv) in devices.items():
            name  = device.name or ""
            rssi  = adv.rssi
            is_elm = any(h in name.lower() for h in _ELM327_NAME_HINTS)
            results.append({
                "addr":   addr,
                "name":   name or "(unnamed)",
                "rssi":   rssi,
                "is_elm": is_elm,
            })

        # Sort: ELM candidates first, then by RSSI
        results.sort(key=lambda d: (not d["is_elm"], -d["rssi"]))
        return results

    @staticmethod
    def print_results(devices: list[dict]) -> None:
        if not devices:
            print("No BLE devices found.")
            return
        print(f"\n{'Address':<20}  {'RSSI':>5}  {'Name'}")
        print("─" * 60)
        for d in devices:
            tag  = " ← OBD?" if d["is_elm"] else ""
            name = d["name"]
            print(f"{d['addr']:<20}  {d['rssi']:>5} dBm  {name}{tag}")
        print()


# ── CLI helpers ───────────────────────────────────────────────────────────────

def _die(msg: str) -> None:
    print(f"[error] {msg}", file=sys.stderr)
    sys.exit(1)


async def _connect_and_init(addr: str, timeout: float = 15.0) -> ELM327BLEClient:
    client = ELM327BLEClient(addr, timeout=timeout)
    print(f"Connecting to {addr} …", end="", flush=True)
    await client.connect()
    print(" connected.")
    print("Initialising ELM327 …", end="", flush=True)
    resp = await client.init_elm327()
    if resp.strip():
        print(f" {resp.strip()[:40]}")
    else:
        print(" done.")
    return client


# ── CLI commands ──────────────────────────────────────────────────────────────

async def cmd_scan(args: argparse.Namespace) -> None:
    devices = await CFMOTOBLEScanner.scan(duration=args.duration)
    CFMOTOBLEScanner.print_results(devices)
    if devices:
        candidates = [d for d in devices if d["is_elm"]]
        if candidates:
            print(f"Suggested address: {candidates[0]['addr']}")
            print(f"  python {sys.argv[0]} connect {candidates[0]['addr']}")


async def cmd_connect(args: argparse.Namespace) -> None:
    """Interactive REPL mode — type OBD-II commands directly."""
    client = await _connect_and_init(args.addr)
    print("\nInteractive OBD-II REPL (Ctrl-C or 'quit' to exit)")
    print("Examples: 010C  010D  03  04  ATRV")
    print()
    try:
        loop = asyncio.get_event_loop()
        while True:
            try:
                cmd = await loop.run_in_executor(None, input, "obd> ")
            except EOFError:
                break
            cmd = cmd.strip()
            if not cmd:
                continue
            if cmd.lower() in ("quit", "exit", "q"):
                break
            resp = await client.send_str(cmd)
            print(resp.strip() or "(no response)")
    except KeyboardInterrupt:
        pass
    finally:
        await client.disconnect()
        print("\nDisconnected.")


def _print_snap(snap: CFMOTOSnapshot) -> None:
    ts = time.strftime("%H:%M:%S", time.localtime(snap.ts))
    print(f"\n[{ts}]  addr={snap.addr}")
    print(snap.pretty())


async def cmd_stream(args: argparse.Namespace) -> None:
    addr = args.addr
    if not addr:
        devices = await CFMOTOBLEScanner.scan(duration=8.0)
        candidates = [d for d in devices if d["is_elm"]]
        if not candidates:
            _die("No ELM327 adapter found. Run 'scan' or pass --addr.")
        addr = candidates[0]["addr"]
        print(f"Auto-selected: {addr}")

    client = await _connect_and_init(addr)
    bridge = SymPrevBridge()
    stream = CFMOTOTelemetryStream(client, interval=args.interval, symprev_bridge=bridge)
    stream.add_callback(_print_snap)

    print(f"\nStreaming telemetry every {args.interval}s  (Ctrl-C to stop)\n")
    try:
        await stream.run()
    except KeyboardInterrupt:
        stream.stop()
        print("\nStopped.")
    finally:
        await client.disconnect()


async def cmd_dtc(args: argparse.Namespace) -> None:
    addr = args.addr
    if not addr:
        devices = await CFMOTOBLEScanner.scan(duration=8.0)
        candidates = [d for d in devices if d["is_elm"]]
        if not candidates:
            _die("No ELM327 adapter found. Run 'scan' or pass --addr.")
        addr = candidates[0]["addr"]

    client = await _connect_and_init(addr)
    try:
        print("Requesting stored DTCs (Mode 03) …")
        dtcs = await client.request_dtcs()
        if dtcs:
            print(f"\n  {len(dtcs)} DTC(s) found:")
            for code in dtcs:
                print(f"    {code}")
        else:
            print("  No DTCs stored.")
    finally:
        await client.disconnect()


async def cmd_clear_dtc(args: argparse.Namespace) -> None:
    addr = args.addr
    if not addr:
        devices = await CFMOTOBLEScanner.scan(duration=8.0)
        candidates = [d for d in devices if d["is_elm"]]
        if not candidates:
            _die("No ELM327 adapter found. Run 'scan' or pass --addr.")
        addr = candidates[0]["addr"]

    client = await _connect_and_init(addr)
    try:
        resp = await client.clear_dtcs()
        print(f"Clear DTCs response: {resp or 'OK'}")
    finally:
        await client.disconnect()


# ── Argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="loci_cfmoto_bt_cli",
        description="CFMOTO 300SR BLE/OBD-II diagnostic CLI + LOCI SymPrev bridge",
    )
    ap.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging"
    )
    sub = ap.add_subparsers(dest="cmd", metavar="COMMAND")
    sub.required = True

    # scan
    p_scan = sub.add_parser("scan", help="Scan for nearby BLE/ELM327 adapters")
    p_scan.add_argument(
        "--duration", type=float, default=10.0,
        help="Scan duration in seconds (default: 10)"
    )

    # connect
    p_conn = sub.add_parser("connect", help="Connect to ELM327 adapter and open interactive REPL")
    p_conn.add_argument("addr", help="BLE MAC address (e.g. AA:BB:CC:DD:EE:FF)")

    # stream
    p_stream = sub.add_parser("stream", help="Stream live telemetry to console + SymPrev ring")
    p_stream.add_argument("--addr", default="", help="BLE address (auto-detect if omitted)")
    p_stream.add_argument(
        "--interval", type=float, default=1.0,
        help="Poll interval in seconds (default: 1.0)"
    )

    # dtc
    p_dtc = sub.add_parser("dtc", help="Read stored fault codes (Mode 03)")
    p_dtc.add_argument("--addr", default="", help="BLE address (auto-detect if omitted)")

    # clear-dtc
    p_clr = sub.add_parser("clear-dtc", help="Clear stored fault codes (Mode 04)")
    p_clr.add_argument("--addr", default="", help="BLE address (auto-detect if omitted)")

    return ap


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = _build_parser()
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s  %(name)s  %(message)s",
    )

    _cmd_map = {
        "scan":      cmd_scan,
        "connect":   cmd_connect,
        "stream":    cmd_stream,
        "dtc":       cmd_dtc,
        "clear-dtc": cmd_clear_dtc,
    }

    handler = _cmd_map.get(args.cmd)
    if handler is None:
        ap.print_help()
        sys.exit(1)

    try:
        asyncio.run(handler(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except RuntimeError as exc:
        _die(str(exc))


if __name__ == "__main__":
    main()
