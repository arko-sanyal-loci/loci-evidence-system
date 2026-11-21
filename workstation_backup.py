#!/usr/bin/env python3
"""Create a weekly encrypted, content-addressed backup of workstation source."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import tempfile

HOME = Path.home()
BACKUP_ROOT = Path(os.environ.get(
    "LOCI_BACKUP_ROOT", str(HOME / ".local/state/loci-workstation-backup")
)).expanduser()
IDENTITY = Path(os.environ.get(
    "LOCI_BACKUP_AGE_IDENTITY", str(HOME / ".config/loci/backup-key.txt")
)).expanduser()
SSH_KEY = Path(os.environ.get(
    "LOCI_BACKUP_SSH_KEY", str(HOME / ".ssh/codex_billo_backup_20261120")
)).expanduser()
SSH_HOST = os.environ.get("LOCI_BACKUP_SSH_HOST", "")
REMOTE_DIR = os.environ.get("LOCI_BACKUP_REMOTE_DIR", "")
KNOWN_HOSTS = Path(os.environ.get("LOCI_BACKUP_KNOWN_HOSTS", str(HOME / ".ssh/known_hosts")))
MIN_REMOTE_FREE_KIB = int(os.environ.get("LOCI_BACKUP_MIN_REMOTE_FREE_KIB", "2097152"))

EXCLUDES = (
    ".runtime", ".venv", "venv", "node_modules", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache", "cache", "caches",
    "firefox-beta", "voices", ".env", ".env.*", "*.env", "*.agekey", "*.pem", "*.p12", "*.pfx",
    "credentials.json", "*credentials*.json", "client_secret.json", "token.json",
    "oauth*.json", "secrets.json", ".git-credentials", ".netrc", ".npmrc", ".pypirc",
    "*.key", "*.key.*", "id_rsa", "id_ed25519", "*.wav", "*.mp3", "*.m4a", "*.aac", "*.flac",
    "*.mp4", "*.mov", "*.mkv", "*.webm", "*.avi", "*.sqlite", "*.sqlite3", "*.db",
)


def run(argv: list[str], *, capture: bool = False, timeout: int = 180) -> str:
    result = subprocess.run(
        argv,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return result.stdout.strip() if capture else ""


def ssh_base() -> list[str]:
    if not SSH_KEY.is_file() or not KNOWN_HOSTS.is_file():
        raise RuntimeError("backup SSH key or known_hosts file is missing")
    return [
        "ssh", "-F", "/dev/null", "-i", str(SSH_KEY), "-o", "IdentitiesOnly=yes",
        "-o", f"UserKnownHostsFile={KNOWN_HOSTS}", "-o", "StrictHostKeyChecking=yes",
        "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
    ]


def include_paths() -> list[str]:
    selected = [
        "src", "Desktop/Git Repos", "loci-web", "loci-web-deploy",
        "loci-web-deploy (2)", "AGENTS.md", "Downloads/evidence/README.txt",
    ]
    for pattern in ("loci_*.sh", "loci_*.py"):
        selected.extend(path.name for path in HOME.glob(pattern) if path.is_file() and not path.is_symlink())
    downloads = HOME / "Downloads"
    for suffix in (".py", ".sh", ".html", ".md"):
        selected.extend(
            path.relative_to(HOME).as_posix()
            for path in downloads.glob(f"*{suffix}")
            if path.is_file() and not path.is_symlink()
        )
    result = []
    for relative in sorted(set(selected)):
        path = HOME / relative
        if path.is_symlink():
            continue
        if path.exists():
            result.append(relative)
    if not result:
        raise RuntimeError("no workstation source paths were found")
    return result


def create_plain_archive(output: Path, paths: list[str]) -> None:
    tar_cmd = [
        "tar", "--sort=name", "--format=gnu", "--mtime=@0", "--owner=0", "--group=0",
        "--numeric-owner", "--wildcards-match-slash",
    ]
    for pattern in EXCLUDES:
        tar_cmd.extend(("--exclude", pattern))
    tar_cmd.extend(("-C", str(HOME), "-cf", "-", *paths))

    with tempfile.TemporaryFile() as tar_errors, tempfile.TemporaryFile() as gzip_errors:
        tar = subprocess.Popen(tar_cmd, stdout=subprocess.PIPE, stderr=tar_errors)
        assert tar.stdout is not None
        with output.open("wb") as compressed:
            gzip = subprocess.Popen(["gzip", "-n", "-6"], stdin=tar.stdout,
                                    stdout=compressed, stderr=gzip_errors)
            tar.stdout.close()
            gzip_code = gzip.wait()
            tar_code = tar.wait()
        if tar_code or gzip_code:
            tar_errors.seek(0)
            gzip_errors.seek(0)
            detail = (tar_errors.read() + gzip_errors.read()).decode("utf-8", "replace")[:300]
            raise RuntimeError(f"archive creation failed (tar={tar_code}, gzip={gzip_code}): {detail}")
    os.chmod(output, 0o600)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_cmd(command: str, *, capture: bool = False) -> str:
    return run([*ssh_base(), SSH_HOST, command], capture=capture, timeout=120)


def remote_free_kib() -> int:
    quoted = shlex.quote(REMOTE_DIR)
    output = remote_cmd(f"df -Pk {quoted} | tail -1 | awk '{{print $4}}'", capture=True)
    try:
        return int(output)
    except ValueError as exc:
        raise RuntimeError("could not read available VPS space") from exc


def publish(local_archive: Path, archive_name: str, cipher_hash: str) -> str:
    remote_root = PurePosixPath(REMOTE_DIR)
    if remote_root.is_absolute() is False or ".." in remote_root.parts:
        raise RuntimeError("unsafe VPS backup directory")
    remote_final = str(remote_root / archive_name)
    remote_part = f"{remote_final}.part"
    quote = shlex.quote
    remote_cmd(f"install -d -m 700 {quote(REMOTE_DIR)}")

    remote_existing = remote_cmd(
        f"if test -f {quote(remote_final)}; then sha256sum {quote(remote_final)}; fi",
        capture=True,
    )
    if remote_existing:
        existing_hash = remote_existing.split()[0]
        if existing_hash == cipher_hash:
            return "already_present"
        raise RuntimeError("existing content-addressed VPS archive failed its hash check")

    if remote_free_kib() < MIN_REMOTE_FREE_KIB:
        raise RuntimeError("VPS backup directory has less than the configured free-space reserve")
    run([
        "scp", "-F", "/dev/null", "-i", str(SSH_KEY), "-o", "IdentitiesOnly=yes",
        "-o", f"UserKnownHostsFile={KNOWN_HOSTS}", "-o", "StrictHostKeyChecking=yes",
        "-o", "BatchMode=yes", str(local_archive), f"{SSH_HOST}:{remote_part}",
    ], timeout=3600)
    remote_cmd(
        f"set -eu; chmod 600 {quote(remote_part)}; "
        f"printf '%s  %s\\n' {quote(cipher_hash)} {quote(remote_part)} | sha256sum --check --status; "
        f"mv -T {quote(remote_part)} {quote(remote_final)}; "
        f"printf '%s  %s\\n' {quote(cipher_hash)} {quote(remote_final)} | sha256sum --check --status"
    )
    return "uploaded"


def main() -> int:
    if not IDENTITY.is_file():
        raise RuntimeError("backup age identity is missing")
    if not SSH_HOST or not REMOTE_DIR:
        raise RuntimeError("LOCI_BACKUP_SSH_HOST and LOCI_BACKUP_REMOTE_DIR must be configured privately")
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(BACKUP_ROOT, 0o700)
    paths = include_paths()
    local_space = shutil.disk_usage(BACKUP_ROOT).free
    if local_space < 2 * 1024 * 1024 * 1024:
        raise RuntimeError("less than 2 GiB is free in the local backup directory")

    recipient = run(["age-keygen", "-y", str(IDENTITY)], capture=True, timeout=15)
    if not recipient.startswith("age1"):
        raise RuntimeError("could not derive the backup age recipient")

    with tempfile.TemporaryDirectory(prefix="workstation-backup-", dir=BACKUP_ROOT) as temp_name:
        temp_dir = Path(temp_name)
        os.chmod(temp_dir, 0o700)
        plain = temp_dir / "source.tar.gz"
        encrypted = temp_dir / "source.tar.gz.age"
        create_plain_archive(plain, paths)
        source_hash = sha256(plain)
        archive_name = f"workstation-code-{source_hash}.tar.gz.age"
        local_archive = BACKUP_ROOT / archive_name
        if not local_archive.exists():
            run(["age", "-r", recipient, "-o", str(encrypted), str(plain)], timeout=3600)
            os.chmod(encrypted, 0o600)
            os.replace(encrypted, local_archive)
        cipher_hash = sha256(local_archive)
        state = publish(local_archive, archive_name, cipher_hash)
        metadata = {
            "schema": "loci.workstation.source_backup.v1",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "archive": archive_name,
            "archive_sha256": cipher_hash,
            "source_archive_sha256": source_hash,
            "archive_bytes": local_archive.stat().st_size,
            "remote_state": state,
            "included_roots": paths,
            "exclusions": list(EXCLUDES),
            "encrypted_with": "age; decryption identity remains on the workstation",
        }
        pointer = BACKUP_ROOT / "WORKSTATION-CODE-LATEST.json"
        temporary_pointer = BACKUP_ROOT / ".WORKSTATION-CODE-LATEST.json.part"
        temporary_pointer.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.chmod(temporary_pointer, 0o600)
        os.replace(temporary_pointer, pointer)

        remote_pointer = str(PurePosixPath(REMOTE_DIR) / pointer.name)
        remote_part = remote_pointer + ".part"
        run([
            "scp", "-F", "/dev/null", "-i", str(SSH_KEY), "-o", "IdentitiesOnly=yes",
            "-o", f"UserKnownHostsFile={KNOWN_HOSTS}", "-o", "StrictHostKeyChecking=yes",
            "-o", "BatchMode=yes", str(pointer), f"{SSH_HOST}:{remote_part}",
        ], timeout=120)
        quote = shlex.quote
        pointer_hash = sha256(pointer)
        remote_cmd(
            f"set -eu; chmod 600 {quote(remote_part)}; "
            f"printf '%s  %s\\n' {quote(pointer_hash)} {quote(remote_part)} | sha256sum --check --status; "
            f"mv -T {quote(remote_part)} {quote(remote_pointer)}; "
            f"printf '%s  %s\\n' {quote(pointer_hash)} {quote(remote_pointer)} | sha256sum --check --status"
        )
        print(json.dumps({"status": state, "archive": archive_name,
                          "sha256": cipher_hash, "bytes": local_archive.stat().st_size}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"workstation backup: {str(exc)[:300]}", file=__import__("sys").stderr)
        raise SystemExit(1)
