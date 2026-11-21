# Private Evidence Journal

This tool is for preserving existing files supplied by the user. It does not
turn on a microphone or camera, collect operating-system activity, infer a
speaker's identity, or change media bytes.

## Current state

The journal at `~/.local/share/loci-evidence/journal.jsonl` is owner-only and
hash-chain verified. No user-supplied audio or video clips have been added. An
administrative non-media inbox note was recorded by an earlier watcher version
and then excluded with an append-only correction; it does not appear in Arc's
reference index. The Havit device is detected as GK51, but earlier direct
captures contained zero-valued PCM, so it is not a verified audio source.

## Add and verify an existing original

```bash
python3 ~/src/loci-workstation/evidence_journal.py init
python3 ~/src/loci-workstation/evidence_journal.py add /path/to/original.mp4 \
  --source 'connected camera file'
python3 ~/src/loci-workstation/evidence_journal.py verify
python3 ~/src/loci-workstation/evidence_journal.py arc-index
```

`add` copies exact bytes to the private object store, checks the copy against
SHA-256, and appends a timestamped record to the hash chain. It stores the
original filename and filesystem modification timestamp privately. Re-adding
the same hash is idempotent. Do not rename or edit the source before intake.
The journal time means when the file was added to this workstation; it does not
validate when a camera recorded it.

## Compact review clips and exact frames

After intake, create a small video-only proxy for the time range you want Arc to
review. Start and end offsets are seconds from the clip's first decoded video
frame:

```bash
~/src/arko-sanyal-loci/mr-calendar/.venv/bin/python \
  ~/src/loci-workstation/evidence_journal.py proxy <original-sha256> \
  --start 12.5 --end 25 --fps 8 --max-width 640
```

The H.264 proxy samples no more than 8 frames per second and scales to at most
640 pixels wide by default. Its `.frames.json` sidecar maps every proxy frame to
the exact source PTS, source time base, and source-relative timestamp. Both
derived files are hashed and linked to the original in the journal and Arc
index. Proxy audio is intentionally omitted; the original, including its audio,
remains unchanged and is the only source of truth. The index omits the original
filename and local path.

## Automatic intake and Arc references

The persistent `loci-evidence-watch.timer` checks `~/Downloads/Evidence-Intake` every
five minutes and waits for an unchanged file for five minutes before intake.
It accepts regular files only and does not follow symlinks. Video clips are
split into review-proxy intervals of at most ten minutes. The worker encrypts a
snapshot with age, sends it to the VPS, and verifies the remote SHA-256 before
posting references to Arc. The VPS destination is root-owned and mode 700.

Arc's `/api/evidence/reference` route accepts loopback requests only and
requires a bearer token held in an owner-only config file. The Arc index omits
the original filename, local path, notes and original bytes. It stores event
ID, timestamp, neutral source label, size, hash, VPS object hash, proxy paths,
proxy hashes and exact frame-map references.

The 500 MiB total threshold stages a ZIP containing the metadata index,
video-only proxies and frame maps, then sends a desktop notification. Automatic
Drive upload occurs only when a Google Drive `rclone` remote and destination
are configured. This workstation now has a file-scoped `locievidence:` remote
bound to the private archive folder. A synthetic upload check confirmed write
access; no media or personal evidence was uploaded by that check. On systems
without a configured remote, the review ZIP stays in the owner-only local
outbox.

The encrypted VPS copy is the source of truth; Arc is a searchable reference,
never a replacement for the original. The app keeps its API bound to loopback.

## Limits

A hash chain can reveal later changes; it cannot prove who recorded a clip,
whether its clock was correct, or establish legal admissibility. Keep the
original device and any original storage where possible. Do not use AI output
as evidence. Seek qualified local support for evidence-handling or legal
questions.
