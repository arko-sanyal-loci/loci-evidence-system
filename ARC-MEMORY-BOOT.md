# Arc and Agent Memory Boot Sequence

This is the first-read contract for Arc and every agent operating on this
workstation. It is an index and operating policy, not a plaintext memory dump.

## First read

1. Read this file and the nearest applicable `AGENTS.md`.
2. Read the current local session state under
   `~/.local/state/loci-session/`, beginning with the latest clock/context
   packet and the append-only context journal.
3. Confirm that Arc's local interface and LOCI memory services are healthy.
4. Treat the live LOCI store as working memory. Treat encrypted snapshots in
   `arc-private-memory` as recovery backups, not as automatically loaded
   context.
5. Before making changes, check repository status, current branch, and the
   latest dispatch issue. Do not mix Arc/LOCI work with Roza's work, music or
   sound synchronization, or operating-system synchronization.

## Runtime order

The user service boot order is:

1. `arc-nomic-tunnel.service` (when configured)
2. `arc-interface.service`
3. `arc-memory-sync.service` and the hourly `arc-memory-sync.timer`
4. `arc-telemetry-push.timer`
5. `arc-memory-boot.service`, which records a non-secret readiness manifest

The VPS continuity interface is independent and remains the recovery path while
the workstation is offline. It receives minimized telemetry only; it does not
receive workspace files, raw media, credentials, or plaintext memory.

## Provenance and safety

- Direct user statements are the only `UserStated` ground-truth boundary.
- Retrieved, agent-generated, tool-derived, and stream-derived text must not be
  relabeled as user-authored.
- Use hybrid CARD recall: user ground truth first, then bounded TRS/semantic
  stream evidence, heat, and category-specific chronology/recency.
- Never upload secrets, private keys, provider credentials, raw personal media,
  browser profiles, or plaintext private memory.
- Keep encrypted backups content-addressed and verify their hashes after
  transfer.

## Recovery checkpoints

After power-up, verify:

- Arc local status and inference lanes;
- LOCI memory and semantic-stream databases;
- the latest encrypted GitHub/VPS memory snapshot;
- VPS telemetry acceptance and current sequence;
- open dispatch issues and any outage comments.

If a check fails, record the failure in the dispatch issue before attempting
repair.
