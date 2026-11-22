#!/usr/bin/env python3
"""Version encrypted Arc-memory snapshots in a dedicated private Git repo."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import os
from pathlib import Path
import shutil
import shlex
import sqlite3
import subprocess
import tarfile
import tempfile
from pathlib import PurePosixPath


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_sqlite(source: Path, target: Path) -> None:
    if not source.is_file():
        raise RuntimeError(f"memory database not found: {source}")
    source_db = sqlite3.connect(f"file:{source}?mode=ro", uri=True, timeout=20)
    target_db = sqlite3.connect(target)
    try:
        source_db.backup(target_db, pages=256, sleep=0.05)
    finally:
        target_db.close()
        source_db.close()
    os.chmod(target, 0o600)


def add_to_tar(tar: tarfile.TarFile, source: Path, name: str) -> None:
    data = source.read_bytes()
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o600
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    import io
    tar.addfile(info, io.BytesIO(data))


def prune_snapshots(directory: Path, retention: int) -> int:
    if retention < 1:
        raise ValueError("retention must be at least one snapshot")
    snapshots = sorted(
        directory.glob("*.tar.gz.age"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    removed = snapshots[retention:]
    for path in removed:
        path.unlink()
    return len(removed)


def run(argv: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(argv, cwd=cwd, env=env, check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip().replace("\n", " ")
        raise RuntimeError(f"command failed ({exc.returncode}): {detail[:220]}") from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.home() / "src/arko-sanyal-loci/arc-private-memory")
    parser.add_argument("--db", type=Path, default=Path.home() / "src/arko-sanyal-loci/arc-canonical/data/arc.db")
    parser.add_argument("--streams-db", type=Path,
                        default=Path.home() / "src/arko-sanyal-loci/arc-canonical/data/semantic_streams.sqlite3")
    parser.add_argument("--context-journal", type=Path,
                        default=Path.home() / ".local/state/loci-session/context-journal.jsonl")
    parser.add_argument("--recipient-file", type=Path,
                        default=Path.home() / ".config/loci/arc-memory-recipient")
    parser.add_argument("--ssh-key", type=Path, default=Path.home() / ".ssh/arc-memory-deploy")
    parser.add_argument("--vps-ssh-key", type=Path,
                        default=Path.home() / ".ssh/codex_billo_backup_20261120")
    parser.add_argument("--known-hosts", type=Path, default=Path.home() / ".ssh/known_hosts")
    parser.add_argument("--vps-host", default=os.environ.get("LOCI_BACKUP_SSH_HOST", ""))
    parser.add_argument("--vps-dir", default=os.environ.get("LOCI_MEMORY_VPS_DIR", ""))
    parser.add_argument("--retention", type=int, default=500,
                        help="maximum encrypted snapshots retained in each backup store")
    parser.add_argument("--min-free-kib", type=int, default=2 * 1024 * 1024)
    args = parser.parse_args()
    if args.retention < 1:
        raise SystemExit("retention must be at least one snapshot")
    repo = args.repo.expanduser().resolve()
    if not (repo / ".git").is_dir():
        raise SystemExit("private memory repository is not initialized")
    if args.min_free_kib < 0 or shutil.disk_usage(repo).free // 1024 < args.min_free_kib:
        raise SystemExit("local free space is below the configured memory snapshot reserve")
    if args.vps_host:
        if not args.vps_ssh_key.expanduser().is_file() or not args.known_hosts.expanduser().is_file():
            raise SystemExit("VPS memory backup SSH key or known_hosts file is missing")
        vps_base = [
            "ssh", "-F", "/dev/null", "-i", str(args.vps_ssh_key.expanduser()),
            "-o", "IdentitiesOnly=yes", "-o", f"UserKnownHostsFile={args.known_hosts.expanduser()}",
            "-o", "StrictHostKeyChecking=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        ]
        backup_root = os.environ.get("LOCI_BACKUP_REMOTE_DIR", "/root/arko-workstation-backups")
        free_output = subprocess.run(
            [*vps_base, args.vps_host,
             f"df -Pk {shlex.quote(backup_root)} | tail -1 | awk '{{print $4}}'"],
            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()
        try:
            remote_free_kib = int(free_output)
        except ValueError:
            raise SystemExit("could not read available VPS storage") from None
        if remote_free_kib < args.min_free_kib:
            raise SystemExit("VPS free space is below the configured memory snapshot reserve")
    recipient = args.recipient_file.read_text(encoding="ascii").strip()
    if not recipient.startswith("age1"):
        raise SystemExit("invalid Arc memory encryption recipient")
    snapshots = repo / "snapshots"
    snapshots.mkdir(mode=0o700, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="arc-memory-snapshot-") as directory:
        temp = Path(directory)
        db_copy = temp / "arc-memory.sqlite3"
        snapshot_sqlite(args.db.expanduser(), db_copy)
        sources = [(db_copy, "arc-memory.sqlite3")]
        streams_source = args.streams_db.expanduser()
        if streams_source.is_file():
            streams_copy = temp / "semantic-streams.sqlite3"
            snapshot_sqlite(streams_source, streams_copy)
            sources.append((streams_copy, "semantic-streams.sqlite3"))
        if args.context_journal.expanduser().is_file():
            journal = args.context_journal.expanduser()
            before = journal.stat()
            journal_copy = temp / "context-journal.jsonl"
            shutil.copyfile(journal, journal_copy)
            after = journal.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise SystemExit("context journal changed while snapshot was being read")
            os.chmod(journal_copy, 0o600)
            sources.append((journal_copy, "context-journal.jsonl"))
        archive_path = temp / "arc-memory.tar.gz"
        with archive_path.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                    for source, name in sources:
                        add_to_tar(tar, source, name)
        os.chmod(archive_path, 0o600)
        digest = digest_file(archive_path)
        encrypted = snapshots / f"{digest}.tar.gz.age"
        if not encrypted.exists():
            subprocess.run(["age", "-r", recipient, "-o", str(encrypted), str(archive_path)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            os.chmod(encrypted, 0o600)
    git_env = os.environ.copy()
    git_env["GIT_SSH_COMMAND"] = (
        f"ssh -F /dev/null -i {args.ssh_key.expanduser()} -o IdentitiesOnly=yes "
        "-o BatchMode=yes -o StrictHostKeyChecking=yes"
    )
    run(["git", "add", "snapshots"], cwd=repo, env=git_env)
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo,
                            env=git_env, check=False)
    if staged.returncode == 1:
        run(["git", "-c", "user.name=Arc Memory Sync", "-c",
             "user.email=arc-memory@localhost", "commit", "-m",
             f"Add encrypted memory snapshot {digest[:12]}"], cwd=repo, env=git_env)
        run(["git", "push", "origin", "HEAD:main"], cwd=repo, env=git_env)
        action = "pushed"
    elif staged.returncode == 0:
        action = "unchanged"
    else:
        raise SystemExit("could not inspect staged memory snapshot")
    vps_state = "not_configured"
    if args.vps_host:
        backup_root = os.environ.get("LOCI_BACKUP_REMOTE_DIR", "/root/arko-workstation-backups")
        remote_dir = args.vps_dir or os.environ.get(
            "LOCI_MEMORY_VPS_DIR", str(PurePosixPath(backup_root) / "arc-memory")
        )
        remote_root = PurePosixPath(remote_dir)
        if not remote_root.is_absolute() or ".." in remote_root.parts:
            raise SystemExit("unsafe VPS memory snapshot directory")
        remote_path = str(remote_root / encrypted.name)
        remote_part = remote_path + ".part"
        ssh_base = [
            "ssh", "-F", "/dev/null", "-i", str(args.vps_ssh_key.expanduser()),
            "-o", "IdentitiesOnly=yes", "-o", f"UserKnownHostsFile={args.known_hosts.expanduser()}",
            "-o", "StrictHostKeyChecking=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        ]
        quote = shlex.quote
        run([*ssh_base, args.vps_host, f"install -d -m 700 {quote(remote_dir)}"], cwd=repo)
        check = subprocess.run(
            [*ssh_base, args.vps_host,
             f"if test -f {quote(remote_path)}; then sha256sum {quote(remote_path)}; fi"],
            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()
        cipher_hash = digest_file(encrypted)
        if check:
            if check.split()[0] != cipher_hash:
                raise SystemExit("VPS memory snapshot exists but failed content hash verification")
            vps_state = "already_present"
        else:
            run([
                "scp", "-F", "/dev/null", "-i", str(args.vps_ssh_key.expanduser()),
                "-o", "IdentitiesOnly=yes", "-o", f"UserKnownHostsFile={args.known_hosts.expanduser()}",
                "-o", "StrictHostKeyChecking=yes", "-o", "BatchMode=yes",
                str(encrypted), f"{args.vps_host}:{remote_part}",
            ], cwd=repo)
            run([*ssh_base, args.vps_host,
                 f"set -eu; chmod 600 {quote(remote_part)}; "
                 f"printf '%s  %s\\n' {quote(cipher_hash)} {quote(remote_part)} | sha256sum --check --status; "
                 f"mv -T {quote(remote_part)} {quote(remote_path)}; "
                 f"printf '%s  %s\\n' {quote(cipher_hash)} {quote(remote_path)} | sha256sum --check --status"], cwd=repo)
            vps_state = "uploaded"
        remote_prune = (
                 f"find {shlex.quote(remote_dir)} -maxdepth 1 -type f "
                 f"-name '*.tar.gz.age' -printf '%T@ %p\\n' | sort -nr | "
                 f"tail -n +{args.retention + 1} | cut -d' ' -f2- | "
                 "while IFS= read -r path; do test -z \"$path\" || rm -f -- \"$path\"; done"
        )
        run([*ssh_base, args.vps_host, remote_prune], cwd=repo)
        vps_state = f"{vps_state};retained<={args.retention}"
    removed = prune_snapshots(snapshots, args.retention)
    if removed:
        git_env = os.environ.copy()
        git_env["GIT_SSH_COMMAND"] = (
                 f"ssh -F /dev/null -i {args.ssh_key.expanduser()} -o IdentitiesOnly=yes "
                 "-o BatchMode=yes -o StrictHostKeyChecking=yes"
        )
        run(["git", "add", "snapshots"], cwd=repo, env=git_env)
        run(["git", "-c", "user.name=Arc Memory Sync", "-c",
                  "user.email=arc-memory@localhost", "commit", "-m",
                  f"Prune encrypted memory snapshots beyond {args.retention}"], cwd=repo, env=git_env)
        run(["git", "push", "origin", "HEAD:main"], cwd=repo, env=git_env)
        action = f"{action};pruned={removed}"
    print(f"memory snapshot {action}; vps={vps_state}; retained<={args.retention}; "
          f"sha256={digest[:16]}; files={len(sources)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
