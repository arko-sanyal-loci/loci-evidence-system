# System architecture

## Evidence path

```text
User places an existing media file
       │
       ▼
Downloads/evidence ── stable-file timer ──► private SHA-256 object store
                                              │
                         ┌────────────────────┴──────────────────┐
                         ▼                                       ▼
              video-only proxy + PTS map              age-encrypted VPS bundle
                         │                                       │
                         └──── metadata-only reference ─────────┘
                                              │
                                              ▼
                                 Arc loopback SQLite index
```

The worker does not open a device or capture a stream. It only considers
regular files with known media extensions that the user has placed in the
inbox. Stability is checked across timer runs before intake.

The original file is copied byte-for-byte and hashed. Video proxies use H.264
NVENC if available and fall back to libx264. Proxy audio is omitted. Each
proxy frame maps to source PTS, source time base and source-relative time.
Arc receives hashes, byte size, a neutral label, and proxy/frame-map metadata;
it never receives the original media bytes or original filename.

## Memory path

The Arc SQLite database and the local context journal are copied as a
consistent, content-addressed snapshot. `age` encrypts the archive before it
enters the private `arko-sanyal-loci/arc-private-memory` repository. Arc's
workstation-only age identity decrypts it. A repository-scoped GitHub deploy
key lets the workstation update that repository; no public source repository
receives snapshots or keys.

## Trust boundaries

- Local state, object store, tokens and age identity are owner-only files.
- Arc and the evidence monitor bind to loopback, not the LAN.
- The Arc reference endpoint requires both loopback and a bearer token.
- VPS snapshots are encrypted before transfer; the remote directory is
  root-owned and mode 700.
- The Drive review ZIP contains only proxies, frame maps and the Arc index.
- Drive automation requires an rclone remote. Without one, the ZIP stays in a
  private local outbox and the desktop notification points to the missing
  configuration.

Hash chains and SHA-256 provide tamper evidence for this journal. They do not
prove who recorded media, its original time, or its legal admissibility.
