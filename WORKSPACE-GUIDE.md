# Workstation User Guide

## Workspace layout

- `~/Desktop/Home` opens the workstation home directory.
- `~/Desktop/Git Repos/` contains shallow clones of the nine repositories in
  the `arko-sanyal-loci` GitHub organization. The branch and issue snapshots
  are in that folder's `README.md` and `OPEN-ISSUES.md`.
- `~/src/arko-sanyal-loci/` contains the active Arc, LOCI, vision/hearing, and
  Mr Calendar working trees. A few older copies remain under `~/Documents/`;
  they are not automatically synchronized.
- `~/loci_*.sh`, `~/loci_calendar_bridge.py`, and `~/loci-web/` are workstation
  orchestration and local site files.

## Git workflow

Inspect work before publishing:

```bash
git -C ~/src/arko-sanyal-loci/arc-canonical status --short --branch
git -C ~/src/arko-sanyal-loci/loci-memory-engine status --short --branch
```

Each Git Repos clone has remote branch tips under `origin/*`. Start a branch
from a remote branch with:

```bash
git switch -c my-work --track origin/main
```

Use the repository's own `AUTHORS.md`, `CONTRIBUTING.md`, and license as the
source of attribution and contribution rules. Do not commit private activity
logs, credentials, media, runtime databases, or model caches.

## Local A/V monitor

The opt-in workstation capture page is at <http://127.0.0.1:11500/capture>.
Press **Start local A/V** and grant browser permission to select a camera and
microphone. It shows a local audio spectrum, measured signal, Garena gradient
regions, XY motion, local Whisper text, and local Qwen scene descriptions.
Press **Stop** to release the tracks. Nothing is captured until Start is
pressed. Use the explicit device selector; the USB audio endpoint appears as
`GK51` on this machine. The current monitor does not synchronize devices on
other computers, Echo/HomePod, or cameras not exposed to this workstation.

## Evidence journal

See [the evidence journal guide](docs/EVIDENCE-JOURNAL.md). Put selected
existing files in `~/Downloads/Evidence-Intake`; the persistent watcher waits until
each file is stable, then writes originals to
`~/.local/share/loci-evidence/objects/`, makes a video-only H.264 proxy using
NVENC where available, and uploads age-encrypted snapshots to the VPS. Arc
receives only hashes, labels, byte counts and exact source-frame references
over a bearer-authenticated loopback endpoint.

## Persistent services

The user session has linger enabled, so the services continue when the desktop
session closes:

- `arc-monitor.service`: local HUD and opt-in A/V monitor on port 11500.
- `arc-interface.service`: Arc web/API UI at <http://127.0.0.1:7860/>.
- `loci-evidence-watch.timer`: checks the evidence inbox every five minutes.
- `arc-memory-sync.timer`: versions an encrypted Arc memory snapshot every
  fifteen minutes in the private `arc-private-memory` repository.

Check status and logs with:

```bash
systemctl --user --no-pager status arc-monitor.service arc-interface.service
systemctl --user --no-pager status loci-evidence-watch.timer arc-memory-sync.timer
journalctl --user -u loci-evidence-watch.service -u arc-memory-sync.service -n 50 --no-pager
```

The watcher currently has no evidence clips to ingest. Havit/GK51 was detected,
but its input was silent in earlier direct checks, so it is not a verified
recording source. No capture runs automatically.
