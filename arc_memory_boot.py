#!/usr/bin/env python3
"""Record a non-secret Arc/LOCI readiness manifest after workstation boot."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


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


def main() -> int:
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
    state = {
        "schema": "arc.memory_boot.v1",
        "booted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest_sha256": sha256(MANIFEST),
        "checks": checks,
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
