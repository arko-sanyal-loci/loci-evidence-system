#!/usr/bin/env python3
"""Private, hash-chained index for user-supplied evidence files.

This tool never captures audio or video. It copies exact files into a private
local object store and records hashes plus acquisition metadata. It does not
establish legal authenticity or replace a professional chain-of-custody process.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import uuid

SCHEMA = "loci.evidence.v1"
GENESIS = "0" * 64
DEFAULT_ROOT = Path.home() / ".local/share/loci-evidence"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Journal:
    def __init__(self, root: Path = DEFAULT_ROOT):
        self.root = root.expanduser()
        self.objects = self.root / "objects"
        self.proxies = self.root / "proxies"
        self.journal = self.root / "journal.jsonl"
        self.arc_dir = self.root / "arc-index"

    def init(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.is_symlink() or self.root.stat().st_uid != os.geteuid():
            raise RuntimeError("Evidence root must be owned by the current user and cannot be a symlink")
        os.chmod(self.root, 0o700)
        self.objects.mkdir(mode=0o700, exist_ok=True)
        self.proxies.mkdir(mode=0o700, exist_ok=True)
        self.arc_dir.mkdir(mode=0o700, exist_ok=True)
        for directory in (self.objects, self.proxies, self.arc_dir):
            if directory.is_symlink() or directory.stat().st_uid != os.geteuid():
                raise RuntimeError(f"Unsafe evidence directory: {directory}")
            os.chmod(directory, 0o700)
        if not self.journal.exists():
            fd = os.open(self.journal, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(fd)
            self._append({"kind": "journal.created", "schema": SCHEMA, "created_at": utc_now()})
        elif self.journal.is_symlink() or self.journal.stat().st_uid != os.geteuid():
            raise RuntimeError("Journal must be owned by the current user and cannot be a symlink")
        os.chmod(self.journal, 0o600)

    def _records(self) -> list[dict]:
        if not self.journal.exists():
            return []
        result = []
        with self.journal.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid journal JSON at line {line_no}") from exc
                if not isinstance(item, dict):
                    raise RuntimeError(f"Invalid journal record at line {line_no}")
                result.append(item)
        return result

    def _append(self, payload: dict) -> dict:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.journal.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            records = self._records()
            previous = records[-1]["record_hash"] if records else GENESIS
            record = {**payload, "event_id": str(uuid.uuid4()), "previous_hash": previous}
            record["record_hash"] = hashlib.sha256(canonical(record)).hexdigest()
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return record

    def add(self, source: Path, source_label: str) -> dict:
        self.init()
        source = source.expanduser()
        if not source_label.strip() or len(source_label) > 120:
            raise RuntimeError("Provide a short source label, such as 'connected camera clip'")
        source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        source_handle = os.fdopen(source_fd, "rb")
        before = os.fstat(source_handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            source_handle.close()
            raise RuntimeError("Source must be a regular file, not a symlink or directory")
        digest_state = hashlib.sha256()
        for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
            digest_state.update(chunk)
        digest = digest_state.hexdigest()
        source_handle.seek(0)
        for existing in self._records():
            if existing.get("kind") == "evidence.file_added" and existing.get("sha256") == digest:
                source_handle.close()
                return {"event_id": existing["event_id"], "sha256": digest,
                        "size_bytes": before.st_size, "already_recorded": True}
        target = self.objects / f"{digest}.original"
        created_target = False
        if target.exists():
            if target.is_symlink() or sha256_file(target) != digest:
                source_handle.close()
                raise RuntimeError("Existing evidence object failed its hash check")
        else:
            temporary = self.objects / f".{digest}.{uuid.uuid4().hex}.part"
            try:
                with temporary.open("xb") as output_file:
                    os.chmod(temporary, 0o600)
                    shutil.copyfileobj(source_handle, output_file, 1024 * 1024)
                    output_file.flush()
                    os.fsync(output_file.fileno())
                if sha256_file(temporary) != digest:
                    raise RuntimeError("Copied object hash mismatch; evidence was not recorded")
                os.replace(temporary, target)
                os.chmod(target, 0o600)
                created_target = True
            finally:
                temporary.unlink(missing_ok=True)
        after = os.fstat(source_handle.fileno())
        source_handle.close()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            if created_target:
                target.unlink(missing_ok=True)
            raise RuntimeError("Source changed while being read; nothing was journaled")
        entry = {
            "kind": "evidence.file_added", "schema": SCHEMA,
            "recorded_at": utc_now(), "source_label": source_label.strip(),
            "original_name": source.name, "size_bytes": before.st_size,
            "source_mtime_ns": before.st_mtime_ns, "sha256": digest,
            "object": f"objects/{digest}.original",
            "statement": "Original bytes copied without media conversion or analysis.",
        }
        record = self._append(entry)
        return {"event_id": record["event_id"], "sha256": digest, "size_bytes": before.st_size}

    def reject(self, source_sha256: str, reason: str) -> dict:
        """Append an exclusion without rewriting or deleting an earlier record."""
        if len(source_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_sha256):
            raise RuntimeError("Source SHA-256 is invalid")
        if not reason.strip() or len(reason) > 240:
            raise RuntimeError("Provide a short correction reason")
        records = self._records()
        source = next((r for r in records if r.get("kind") == "evidence.file_added"
                       and r.get("sha256") == source_sha256), None)
        if source is None:
            raise RuntimeError("No recorded source matches that SHA-256")
        if any(r.get("kind") == "evidence.file_rejected" and r.get("sha256") == source_sha256
               for r in records):
            return {"sha256": source_sha256, "already_rejected": True}
        record = self._append({"kind": "evidence.file_rejected", "schema": SCHEMA,
                               "recorded_at": utc_now(), "source_event_id": source["event_id"],
                               "sha256": source_sha256, "reason": reason.strip()})
        return {"event_id": record["event_id"], "sha256": source_sha256, "rejected": True}

    def verify(self, verify_objects: bool = True) -> tuple[bool, str]:
        records = self._records()
        previous = GENESIS
        for index, record in enumerate(records, 1):
            digest = record.get("record_hash")
            payload = {key: value for key, value in record.items() if key != "record_hash"}
            if record.get("previous_hash") != previous or hashlib.sha256(canonical(payload)).hexdigest() != digest:
                return False, f"Hash chain failed at record {index}"
            previous = digest
            if verify_objects and record.get("kind") == "evidence.file_added":
                obj = self.root / record["object"]
                if obj.is_symlink() or not obj.is_file() or sha256_file(obj) != record.get("sha256"):
                    return False, f"Evidence object failed verification at record {index}"
            if verify_objects and record.get("kind") == "evidence.proxy_created":
                for field, digest_field in (("proxy", "proxy_sha256"), ("frame_map", "frame_map_sha256")):
                    obj = self.root / record[field]
                    if obj.is_symlink() or not obj.is_file() or sha256_file(obj) != record.get(digest_field):
                        return False, f"Derived proxy failed verification at record {index}"
        return True, f"{len(records)} records verified; original objects match"

    def create_proxy(self, source_sha256: str, start_seconds: float,
                     end_seconds: float, fps: int = 8, max_width: int = 640) -> dict:
        self.init()
        if len(source_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_sha256):
            raise RuntimeError("Source SHA-256 is invalid")
        if not (math.isfinite(start_seconds) and math.isfinite(end_seconds)
                and 0 <= start_seconds < end_seconds and end_seconds - start_seconds <= 600):
            raise RuntimeError("Choose a clip interval from 0 to 600 seconds")
        if not (1 <= fps <= 15 and 160 <= max_width <= 1280):
            raise RuntimeError("Proxy limits: 1–15 fps and 160–1280 pixels wide")
        source = self.objects / f"{source_sha256}.original"
        if source.is_symlink() or not source.is_file() or sha256_file(source) != source_sha256:
            raise RuntimeError("Verified original for that SHA-256 was not found")
        start_ms, end_ms = round(start_seconds * 1000), round(end_seconds * 1000)
        stem = f"{source_sha256}_{start_ms}-{end_ms}"
        proxy = self.proxies / f"{stem}.mp4"
        frame_map_path = self.proxies / f"{stem}.frames.json"
        if proxy.exists() or frame_map_path.exists():
            raise RuntimeError("That proxy interval already exists; verify it instead of overwriting")
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("PyAV is required; run this command with the Mr Calendar virtualenv") from exc
        temp_video = self.proxies / f".{stem}.{uuid.uuid4().hex}.part.mp4"
        temp_map = self.proxies / f".{stem}.{uuid.uuid4().hex}.part.json"
        frame_map = []
        try:
            with av.open(str(source)) as probe:
                if not probe.streams.video:
                    raise RuntimeError("Source file has no video stream")
                width, height = probe.streams.video[0].width, probe.streams.video[0].height
            scale = min(1.0, max_width / width)
            out_width = max(2, int(width * scale) // 2 * 2)
            out_height = max(2, int(height * scale) // 2 * 2)
            time_base = Fraction(1, fps)

            def encode_proxy(codec: str, options: dict) -> None:
                frame_map.clear()
                with av.open(str(source)) as container:
                    input_stream = container.streams.video[0]
                    next_sample = start_seconds
                    with av.open(str(temp_video), mode="w", format="mp4") as output:
                        output_stream = output.add_stream(codec, rate=fps)
                        output_stream.width = out_width
                        output_stream.height = out_height
                        output_stream.pix_fmt = "yuv420p"
                        output_stream.options = options
                        output_stream.time_base = time_base
                        origin = None
                        for frame in container.decode(input_stream):
                            if frame.pts is None:
                                continue
                            absolute_time = float(frame.pts * frame.time_base)
                            if origin is None:
                                origin = absolute_time
                            source_time = absolute_time - origin
                            if source_time < start_seconds:
                                continue
                            if source_time > end_seconds:
                                break
                            if source_time + 1e-7 < next_sample:
                                continue
                            proxy_index = len(frame_map)
                            converted = frame.reformat(width=out_width, height=out_height, format="yuv420p")
                            converted.pts = proxy_index
                            converted.time_base = time_base
                            for packet in output_stream.encode(converted):
                                output.mux(packet)
                            frame_map.append({"proxy_frame": proxy_index,
                                              "proxy_seconds": round(proxy_index / fps, 6),
                                              "source_pts": int(frame.pts),
                                              "source_time_base": str(frame.time_base),
                                              "source_seconds": round(source_time, 6)})
                            next_sample += 1 / fps
                        for packet in output_stream.encode(None):
                            output.mux(packet)

            used_codec = "h264_nvenc"
            try:
                encode_proxy(used_codec, {"preset": "p4", "cq": "35", "tune": "hq"})
            except Exception:
                temp_video.unlink(missing_ok=True)
                frame_map.clear()
                used_codec = "libx264"
                encode_proxy(used_codec, {"crf": "32", "preset": "ultrafast"})
            if not frame_map:
                raise RuntimeError("The selected interval contains no decodable video frames")
            map_record = {"schema": "loci.evidence.frame-map.v1", "source_sha256": source_sha256,
                          "start_seconds": start_seconds, "end_seconds": end_seconds,
                          "fps_cap": fps, "proxy_width": out_width, "proxy_height": out_height,
                          "audio_included": False, "frames": frame_map}
            temp_map.write_text(json.dumps(map_record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            os.chmod(temp_video, 0o600)
            os.chmod(temp_map, 0o600)
            proxy_hash, map_hash = sha256_file(temp_video), sha256_file(temp_map)
            os.replace(temp_video, proxy)
            os.replace(temp_map, frame_map_path)
            record = self._append({"kind": "evidence.proxy_created", "schema": SCHEMA,
                                   "recorded_at": utc_now(), "source_sha256": source_sha256,
                                   "start_seconds": start_seconds, "end_seconds": end_seconds,
                                   "proxy": f"proxies/{proxy.name}", "proxy_sha256": proxy_hash,
                                   "frame_map": f"proxies/{frame_map_path.name}",
                                   "frame_map_sha256": map_hash, "frame_count": len(frame_map),
                                   "video_codec": "H.264/NVENC" if used_codec == "h264_nvenc" else "H.264/libx264",
                                   "audio_included": False,
                                   "statement": "Review-only derivative; original remains the evidence source."})
            return {"event_id": record["event_id"], "proxy": str(proxy), "frame_map": str(frame_map_path),
                    "frame_count": len(frame_map), "proxy_sha256": proxy_hash}
        finally:
            temp_video.unlink(missing_ok=True)
            temp_map.unlink(missing_ok=True)

    def export_arc_index(self) -> Path:
        self.init()
        records = self._records()
        valid, message = self.verify()
        if not valid:
            raise RuntimeError(message)
        rejected = {record["sha256"] for record in records
                    if record.get("kind") == "evidence.file_rejected"}
        proxies_by_source: dict[str, list[dict]] = {}
        for record in records:
            if record.get("kind") == "evidence.proxy_created":
                proxies_by_source.setdefault(record["source_sha256"], []).append({
                    "event_id": record["event_id"], "start_seconds": record["start_seconds"],
                    "end_seconds": record["end_seconds"], "proxy": record["proxy"],
                    "proxy_sha256": record["proxy_sha256"], "frame_map": record["frame_map"],
                    "frame_map_sha256": record["frame_map_sha256"],
                    "frame_count": record["frame_count"], "audio_included": False,
                })
        output = self.arc_dir / "evidence-index.jsonl"
        temporary = output.with_suffix(".jsonl.part")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                if record.get("kind") != "evidence.file_added":
                    continue
                if record["sha256"] in rejected:
                    continue
                ref = {"schema": "loci.arc.evidence_reference.v1",
                       "event_id": record["event_id"], "recorded_at": record["recorded_at"],
                       "source_label": record["source_label"], "size_bytes": record["size_bytes"],
                       "sha256": record["sha256"], "vps_object": record["sha256"],
                       "frame_references": proxies_by_source.get(record["sha256"], [])}
                handle.write(json.dumps(ref, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, output)
        return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="private journal directory")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="create an empty private journal")
    add = commands.add_parser("add", help="preserve and hash one existing file")
    add.add_argument("file", type=Path)
    add.add_argument("--source", required=True, help="neutral source label")
    proxy = commands.add_parser("proxy", help="make a compact, video-only review clip with exact source PTS mapping")
    proxy.add_argument("source_sha256")
    proxy.add_argument("--start", type=float, required=True, help="seconds from the clip's first video frame")
    proxy.add_argument("--end", type=float, required=True)
    proxy.add_argument("--fps", type=int, default=8)
    proxy.add_argument("--max-width", type=int, default=640)
    commands.add_parser("verify", help="verify the hash chain and object hashes")
    commands.add_parser("arc-index", help="write a metadata-only Arc reference index")
    reject = commands.add_parser("reject", help="append a correction excluding a recorded source")
    reject.add_argument("sha256")
    reject.add_argument("--reason", required=True)
    args = parser.parse_args()
    journal = Journal(args.root)
    try:
        if args.command == "init":
            journal.init()
            print(f"Empty evidence journal ready: {journal.root}")
        elif args.command == "add":
            print(json.dumps(journal.add(args.file, args.source), sort_keys=True))
        elif args.command == "proxy":
            print(json.dumps(journal.create_proxy(args.source_sha256, args.start, args.end,
                                                  args.fps, args.max_width), sort_keys=True))
        elif args.command == "verify":
            valid, message = journal.verify()
            print(message)
            return 0 if valid else 1
        elif args.command == "arc-index":
            print(journal.export_arc_index())
        elif args.command == "reject":
            print(json.dumps(journal.reject(args.sha256, args.reason), sort_keys=True))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"evidence journal: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
