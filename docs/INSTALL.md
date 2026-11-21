# Install and run

This project is a local Linux workstation toolset. The command-line journal can
be used by itself; the systemd units add periodic intake and synchronization.

## Dependencies

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- `age`, `ssh`, and `scp` for encrypted VPS sync
- `rclone` for optional Google Drive uploads
- systemd user services for persistent automation
- NVIDIA driver with NVENC for GPU H.264 proxy encoding; libx264 is the CPU
  fallback

Clone and lock the Python environment:

```bash
git clone https://github.com/arko-sanyal-loci/loci-evidence-system.git
cd loci-evidence-system
uv sync --locked
```

Initialize and validate a private journal:

```bash
uv run python evidence_journal.py init
uv run python evidence_journal.py verify
```

`evidence_journal.py add FILE --source LABEL` preserves an existing file and
checks its SHA-256. `proxy SHA256 --start SECONDS --end SECONDS` creates a
video-only, compact review proxy and a source-PTS frame map. The manual CLI
does not access a camera or microphone.

## Optional persistent watcher

The watcher processes only media extensions under `~/Downloads/evidence`.
It waits for each file to remain unchanged for five minutes, then creates a
private local copy. To enable automatic sync, first configure a VPS SSH key,
age recipient and Arc bearer token in owner-only files under
`~/.config/loci/`. The actual workstation units and configuration are
documented in [WORKSPACE-GUIDE.md](../WORKSPACE-GUIDE.md). Never put keys or
tokens in this repository.

Google Drive upload requires a configured rclone remote and a private
destination variable in `~/.config/loci/evidence-worker.env`. For example,
create a file-scoped remote and point it at the authorized archive folder:

```bash
rclone config create locievidence drive scope drive.file root_folder_id=FOLDER_ID
printf '%s\n' 'LOCI_EVIDENCE_RCLONE_DEST=locievidence:' \
  >> ~/.config/loci/evidence-worker.env
chmod 600 ~/.config/loci/evidence-worker.env
```

Complete Google's sign-in prompt for rclone. Keep `rclone.conf`, OAuth tokens,
folder IDs and the environment file out of Git. Without a usable remote and
destination, the worker stages a proxy-only review package and sends a desktop
notification when the evidence store reaches 500 MiB.

The provided systemd units are Linux-specific. On another machine, run the
manual commands first; adapt the service and OS-specific key paths before
enabling background sync.
