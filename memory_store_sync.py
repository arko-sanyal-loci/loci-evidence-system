#!/usr/bin/env python3
"""Version encrypted Arc-memory snapshots in a dedicated private Git repo."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tarfile
import tempfile


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
    parser.add_argument("--context-journal", type=Path,
                        default=Path.home() / ".local/state/loci-session/context-journal.jsonl")
    parser.add_argument("--recipient-file", type=Path,
                        default=Path.home() / ".config/loci/arc-memory-recipient")
    parser.add_argument("--ssh-key", type=Path, default=Path.home() / ".ssh/arc-memory-deploy")
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    if not (repo / ".git").is_dir():
        raise SystemExit("private memory repository is not initialized")
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
    print(f"memory snapshot {action}; sha256={digest[:16]}; files={len(sources)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
