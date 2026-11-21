# LOCI Workstation Projects

This public source repository collects workstation tools and project guides that do not
belong to the individual Arc, LOCI Memory Engine, or Garena repositories. It is
maintained by Arko Sanyal. It contains no camera or microphone recordings,
calendar exports, operating-system event logs, credentials, or model weights.

## Start here

- [Workspace guide](WORKSPACE-GUIDE.md): project map and day-to-day commands.
- [Install guide](docs/INSTALL.md): dependencies and local setup.
- [System architecture](docs/ARCHITECTURE.md): local data flow and trust boundaries.
- [Credits](CREDITS.md): authorship boundaries and session-specific contribution record.
- [Evidence journal guide](docs/EVIDENCE-JOURNAL.md): private handling of existing files.
- `evidence_journal.py`: hash-chained journal and metadata-only Arc index exporter.
- `evidence_worker.py`: persistent, consent-bound intake from `Downloads/evidence`.
- `memory_store_sync.py`: age-encrypted, versioned Arc memory snapshots for the
  separate private `arc-private-memory` repository.

The evidence journal is outside Git in `~/.local/share/loci-evidence/` with
owner-only permissions. A five-minute persistent user timer ingests only stable
files the user puts in `~/Downloads/evidence`, encrypts a snapshot for the VPS,
and sends metadata-only references to Arc at `127.0.0.1:7860`. It never
activates a camera or microphone. If the store reaches 500 MiB, it stages a
proxy-only review package and notifies the user; an rclone Drive destination
must be configured before background Drive upload can occur.

Arc runs as a separate loopback-only persistent service on port 7860. Arc's
SQLite memory database and local context journal are copied into a private
GitHub repository only as age-encrypted, content-addressed snapshots. The
decryption identity stays on the Arc workstation. The public source repository
contains code and documentation only, never memory records or original media.

## Quick start

```bash
uv sync --locked
uv run python evidence_journal.py init
uv run python evidence_journal.py verify
```

See [INSTALL.md](docs/INSTALL.md) before enabling background sync.

## Safety and attribution

This repository does not capture media. Every capture must begin through a
visible user action in the browser. Existing third-party code retains its own
license, copyright notices, and history. See [CREDITS.md](CREDITS.md).
