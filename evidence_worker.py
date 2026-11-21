#!/usr/bin/env python3
"""Consent-bound worker for files the user places in Downloads/evidence.

No capture, transcription, voice identification, or profile inference occurs.
It waits until files are stable, records them, creates video-only review
proxies, encrypts the evidence bundle, syncs it to the VPS, and posts
metadata-only pointers to the loopback Arc API.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.request import Request, urlopen

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
ACCEPTED_EXTENSIONS = VIDEO_EXTENSIONS | {".wav", ".mp3", ".m4a", ".aac", ".flac",
                                          ".jpg", ".jpeg", ".png"}
DEFAULT_WATCH = Path.home() / "Downloads/evidence"
DEFAULT_ROOT = Path.home() / ".local/share/loci-evidence"
DEFAULT_STATE = Path.home() / ".local/state/loci-evidence"


def load_journal_module():
    path = Path(__file__).with_name("evidence_journal.py")
    spec = importlib.util.spec_from_file_location("loci_evidence_journal", path)
    if not spec or not spec.loader:
        raise RuntimeError("evidence journal module unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def digest_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_new_stable_files(watch: Path, state_path: Path, stable_seconds: int) -> list[Path]:
    try:
        old = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        old = {}
    now = time.time()
    next_state: dict[str, dict] = {}
    ready: list[Path] = []
    if watch.is_dir() and not watch.is_symlink():
        for path in sorted(watch.iterdir()):
            if path.name.startswith(".") or path.is_symlink() or not path.is_file():
                continue
            if path.suffix.lower() not in ACCEPTED_EXTENSIONS:
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            key = hashlib.sha256(os.fsencode(path.name)).hexdigest()
            signature = [st.st_size, st.st_mtime_ns]
            previous = old.get(key, {})
            if previous.get("signature") == signature:
                stable_since = float(previous.get("stable_since", now))
            else:
                stable_since = now
            next_state[key] = {"signature": signature, "stable_since": stable_since}
            if now - stable_since >= stable_seconds:
                ready.append(path)
    atomic_json(state_path, next_state)
    return ready


def video_duration(path: Path) -> float | None:
    try:
        import av
        with av.open(str(path)) as container:
            if not container.streams.video:
                return None
            if container.duration:
                return max(0.0, float(container.duration / av.time_base))
            stream = container.streams.video[0]
            if stream.duration is not None and stream.time_base is not None:
                return max(0.0, float(stream.duration * stream.time_base))
    except Exception:
        return None
    return None


def run_checked(argv: list[str], *, capture: bool = False):
    return subprocess.run(argv, check=True,
                          stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                          stderr=subprocess.PIPE, text=capture)


def encrypt_vps_snapshot(root: Path, state_dir: Path) -> str:
    host = os.environ.get("LOCI_EVIDENCE_VPS_HOST", "")
    remote_dir = os.environ.get("LOCI_EVIDENCE_VPS_DIR", "")
    ssh_key = Path(os.environ.get("LOCI_EVIDENCE_SSH_KEY", ""))
    recipient = os.environ.get("LOCI_EVIDENCE_AGE_RECIPIENT", "")
    if not (host and remote_dir and ssh_key.is_file() and recipient.startswith("age1")):
        raise RuntimeError("VPS sync is not configured")
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(state_dir, 0o700)
    with tempfile.TemporaryDirectory(prefix="loci-evidence-", dir=state_dir) as temp:
        archive = Path(temp) / "evidence.tar.gz"
        encrypted = Path(temp) / "evidence.tar.gz.age"
        with tarfile.open(archive, "w:gz") as tar:
            for source in (root / "journal.jsonl", root / "objects", root / "proxies", root / "arc-index"):
                if source.exists():
                    tar.add(source, arcname=source.name, recursive=True)
        os.chmod(archive, 0o600)
        run_checked(["age", "-r", recipient, "-o", str(encrypted), str(archive)])
        digest = hashlib.sha256(encrypted.read_bytes()).hexdigest()
        remote_name = f"evidence-{digest}.tar.gz.age"
        run_checked(["ssh", "-F", "/dev/null", "-i", str(ssh_key), "-o", "BatchMode=yes",
                     "-o", "StrictHostKeyChecking=yes", host,
                     "install", "-d", "-m", "700", remote_dir])
        run_checked(["scp", "-F", "/dev/null", "-i", str(ssh_key), "-o", "BatchMode=yes",
                     "-o", "StrictHostKeyChecking=yes", str(encrypted),
                     f"{host}:{remote_dir}/{remote_name}"])
        remote_check = run_checked(["ssh", "-F", "/dev/null", "-i", str(ssh_key), "-o", "BatchMode=yes",
                                    "-o", "StrictHostKeyChecking=yes", host,
                                    "sha256sum", f"{remote_dir}/{remote_name}"], capture=True)
        if not remote_check.stdout.startswith(digest):
            raise RuntimeError("VPS snapshot checksum did not match")
        return digest


def post_arc_references(root: Path) -> int:
    token = os.environ.get("ARC_EVIDENCE_TOKEN", "")
    if not token:
        raise RuntimeError("Arc evidence token is unavailable")
    count = 0
    with (root / "arc-index/evidence-index.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            request = Request(
                "http://127.0.0.1:7860/api/evidence/reference",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
                method="POST",
            )
            with urlopen(request, timeout=10) as response:
                if response.status not in (200, 201):
                    raise RuntimeError("Arc reference request failed")
            count += 1
    return count


def stage_drive_review(root: Path, state_dir: Path, journal, threshold_bytes: int,
                       allow_upload: bool = True) -> tuple[int, bool]:
    records = journal._records()
    rejected = {record["sha256"] for record in records if record.get("kind") == "evidence.file_rejected"}
    active_sources = {record["sha256"] for record in records
                      if record.get("kind") == "evidence.file_added" and record["sha256"] not in rejected}
    evidence_bytes = sum(int(record.get("size_bytes", 0)) for record in records
                         if record.get("kind") == "evidence.file_added" and record["sha256"] not in rejected)
    evidence_bytes += sum((root / record["proxy"]).stat().st_size for record in records
                          if record.get("kind") == "evidence.proxy_created"
                          and record.get("source_sha256") in active_sources
                          and (root / record["proxy"]).is_file())
    if evidence_bytes < threshold_bytes:
        return evidence_bytes, False
    outbox = state_dir / "drive-outbox"
    outbox.mkdir(parents=True, exist_ok=True, mode=0o700)
    index_path = root / "arc-index/evidence-index.jsonl"
    bundle = outbox / "evidence-review.zip"
    temporary = outbox / ".evidence-review.zip.part"
    import zipfile
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(index_path, "evidence-index.jsonl")
        for record in records:
            if (record.get("kind") == "evidence.proxy_created"
                    and record.get("source_sha256") in active_sources):
                for field in ("proxy", "frame_map"):
                    path = root / record[field]
                    archive.write(path, record[field])
    os.chmod(temporary, 0o600)
    os.replace(temporary, bundle)
    bundle_digest = digest_path(bundle)
    marker = state_dir / "last-drive-review-hash"
    try:
        if marker.read_text(encoding="ascii").strip() == bundle_digest:
            return evidence_bytes, False
    except OSError:
        pass
    destination = os.environ.get("LOCI_EVIDENCE_RCLONE_DEST", "").strip()
    uploaded = False
    if destination and allow_upload:
        run_checked(["rclone", "copyto", str(bundle), f"{destination.rstrip('/')}/{bundle.name}"])
        uploaded = True
    if uploaded:
        marker.write_text(bundle_digest + "\n", encoding="ascii")
        os.chmod(marker, 0o600)
    notify = shutil.which("notify-send")
    if notify:
        message = "Google Drive review package uploaded." if uploaded else \
                  "Review package staged locally; configure an rclone Google Drive remote to upload."
        subprocess.run([notify, "LOCI evidence archive threshold reached", message],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return evidence_bytes, uploaded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", type=Path, default=DEFAULT_WATCH)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--stable-seconds", type=int, default=300)
    parser.add_argument("--threshold-mib", type=int, default=500)
    parser.add_argument("--test-mode", action="store_true", help="disable VPS, Arc, and Drive network actions")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.state.mkdir(parents=True, exist_ok=True, mode=0o700)
    journal_module = load_journal_module()
    journal = journal_module.Journal(args.root)
    journal.init()
    candidates = list_new_stable_files(args.watch, args.state / "worker-state.json", args.stable_seconds)
    records_added = 0
    try:
        for path in candidates:
            result = journal.add(path, "user-supplied evidence file")
            if result.get("already_recorded"):
                continue
            records_added += 1
            duration = video_duration(args.root / f"objects/{result['sha256']}.original")
            if duration and duration > 0:
                start = 0.0
                while start < duration:
                    end = min(start + 600.0, duration)
                    if end - start < 0.1:
                        break
                    journal.create_proxy(result["sha256"], start, end)
                    start = end
        index_path = journal.export_arc_index()
        valid, message = journal.verify()
        if not valid:
            raise RuntimeError(message)
        journal_digest = digest_path(journal.journal)
        excluded_hashes = {record["sha256"] for record in journal._records()
                           if record.get("kind") == "evidence.file_rejected"}
        evidence_exists = any(record.get("kind") == "evidence.file_added"
                              and record.get("sha256") not in excluded_hashes
                              for record in journal._records())
        evidence_file_count = sum(
            record.get("kind") == "evidence.file_added"
            and record.get("sha256") not in excluded_hashes
            for record in journal._records()
        )
        vps_marker = args.state / "last-vps-journal-hash"
        if evidence_exists and not args.test_mode:
            try:
                last_vps_digest = vps_marker.read_text(encoding="ascii").strip()
            except OSError:
                last_vps_digest = ""
            if last_vps_digest != journal_digest:
                encrypt_vps_snapshot(args.root, args.state)
                vps_marker.write_text(journal_digest + "\n", encoding="ascii")
                os.chmod(vps_marker, 0o600)
            post_arc_references(args.root)
        usage, uploaded = stage_drive_review(journal.root, args.state,
                                              journal, args.threshold_mib * 1024 * 1024,
                                              allow_upload=not args.test_mode)
        print(json.dumps({"new_files": records_added, "journal_records": len(journal._records()),
                          "eligible_evidence_files": evidence_file_count,
                          "vps_synced": bool(evidence_exists and not args.test_mode),
                          "arc_references": bool(evidence_exists and not args.test_mode),
                          "sync_state": ("no_eligible_evidence" if not evidence_exists else
                                         "test_mode" if args.test_mode else "synced"),
                          "drive_threshold_bytes": usage,
                          "drive_uploaded": uploaded,
                          "journal_verified": True}, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"new_files": records_added, "error": str(exc)[:180]}, sort_keys=True), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
