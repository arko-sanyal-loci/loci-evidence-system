#!/usr/bin/env python3
"""Record a non-secret Arc/LOCI readiness manifest after workstation boot."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


HOME = Path.home()
WORKSTATION = HOME / "src/loci-workstation"
STATE_DIR = HOME / ".local/state/loci-session"
STATE_FILE = STATE_DIR / "arc-memory-boot.json"
MANIFEST = WORKSTATION / "ARC-MEMORY-BOOT.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(path: Path) -> dict[str, str | bool]:
    try:
        branch = subprocess.run(
            ["git", "-C", str(path), "branch", "--show-current"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        head = subprocess.run(
            ["git", "-C", str(path), "log", "-1", "--format=%H"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        commit_at = subprocess.run(
            ["git", "-C", str(path), "log", "-1", "--format=%cI"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip() != ""
        return {"branch": branch, "head": head, "commit_at": commit_at, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"branch": "", "head": "", "commit_at": "", "dirty": True}


def service_state(name: str) -> str:
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", name],
            check=False, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def main() -> int:
    captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    required = {
        "arc_runtime": HOME / "src/arko-sanyal-loci/arc-canonical",
        "loci_engine": HOME / "src/arko-sanyal-loci/loci-memory-engine",
        "private_memory_repo": HOME / "src/arko-sanyal-loci/arc-private-memory",
        "session_state": STATE_DIR,
    }
    if not MANIFEST.is_file():
        raise SystemExit(f"boot manifest not found: {MANIFEST}")
    checks = {name: path.is_dir() for name, path in required.items()}
    if not all(checks.values()):
        missing = ", ".join(name for name, present in checks.items() if not present)
        raise SystemExit(f"required boot paths are missing: {missing}")
    STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    repositories = {
        "arc": HOME / "src/arko-sanyal-loci/arc-canonical",
        "loci_memory_engine": HOME / "src/arko-sanyal-loci/loci-memory-engine",
        "loci_workstation": WORKSTATION,
        "arc_private_memory": HOME / "src/arko-sanyal-loci/arc-private-memory",
    }
    session_state_names = {"context-journal.jsonl", "clock-calendar-context.json", "calendar-next.json"}
    session_files = sorted(
        (
            path for path in STATE_DIR.iterdir()
            if path.is_file()
            and path != STATE_FILE
            and (
                path.name in session_state_names
                or path.name.startswith("context-v") and path.suffix == ".json"
                or path.name.startswith("packet-v") and path.suffix == ".json"
            )
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    state = {
        "schema": "arc.memory_boot.v1",
        "booted_at": captured_at,
        "captured_at": captured_at,
        "manifest_sha256": sha256(MANIFEST),
        "checks": checks,
        "repositories": {name: git_state(path) for name, path in repositories.items()},
        "services": {
            name: service_state(name)
            for name in (
                "arc-interface.service",
                "arc-memory-sync.timer",
                "arc-telemetry-push.timer",
                "arc-memory-boot.service",
            )
        },
        "latest_session_state": (
            {
                "path": str(session_files[0]),
                "modified_at": datetime.fromtimestamp(
                    session_files[0].stat().st_mtime, timezone.utc
                ).isoformat(timespec="seconds"),
            }
            if session_files else None
        ),
        "orientation": {
            "current_direction": "Arc runtime with LOCI/Glassbox memory and isolated encrypted continuity backups",
            "next_objectives": [
                "Validate timestamped boot state and encrypted memory sync after power-up",
                "Verify VPS telemetry acceptance and the /arch interface",
                "Continue provenance-labeled CARD/TRS hybrid recall integration",
            ],
        },
        "time_policy": "All boot and memory handoffs use UTC ISO-8601 timestamps.",
        "encrypted_memory_sync_configured": (
            (HOME / ".config/loci/arc-memory-recipient").is_file()
            and (HOME / ".config/loci/workstation-backup.env").is_file()
        ),
        "secrets_read": False,
        "plaintext_memory_loaded": False,
    }
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, STATE_FILE)
    print(json.dumps({"status": "ready", "state_file": str(STATE_FILE),
                      "manifest_sha256": state["manifest_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
