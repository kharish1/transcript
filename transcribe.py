"""
Transcribe videos from S3 or public URLs using AWS Transcribe.
SRT files are always saved locally. No output S3 bucket required.

Modes:
  --bucket   Scan an S3 bucket/prefix and transcribe all videos found.
  --urls     Transcribe one or more public HTTPS video URLs directly.
  Both modes can be combined in a single run.

Usage:
    pip install boto3 tqdm

    # Public URLs only — no S3 bucket needed for output
    python transcribe_s3_videos.py \
        --urls https://example.com/video1.mp4 https://example.com/video2.mp4

    # S3 bucket scan
    python transcribe_s3_videos.py \
        --bucket my-bucket --input-prefix videos/

    # Mix both, custom output dir
    python transcribe_s3_videos.py \
        --bucket my-bucket \
        --urls https://example.com/extra.mp4 \
        --download-dir ./my-srts
"""

import argparse
import hashlib
import json
import logging
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import boto3
from tqdm import tqdm

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".m4v", ".webm"}
SUPPORTED_FORMATS = {"mp4", "mov", "avi", "mkv", "flv", "wmv", "m4v", "webm", "wav", "mp3", "ogg", "amr"}
DEFAULT_FORMAT = "mp4"

POLL_INTERVAL_SECONDS = 15
WORDS_PER_SRT_BLOCK = 10

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def safe_job_name(label: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]", "-", label)
    return name[:200]


def stem_from_url(url: str) -> str:
    path = urllib.parse.urlparse(url).path
    name = path.rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0] if "." in name else name
    stem = re.sub(r"[^a-zA-Z0-9_-]", "_", stem) or "video"
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"{stem}_{url_hash}"


def format_from_url(url: str) -> str:
    path = urllib.parse.urlparse(url).path
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return ext if ext in SUPPORTED_FORMATS else DEFAULT_FORMAT


def seconds_to_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def build_srt(transcript_json: dict) -> str:
    items = [
        item for item in transcript_json["results"]["items"]
        if item["type"] == "pronunciation"
    ]
    if not items:
        return ""

    blocks = []
    for i in range(0, len(items), WORDS_PER_SRT_BLOCK):
        chunk = items[i : i + WORDS_PER_SRT_BLOCK]
        start = float(chunk[0]["start_time"])
        end = float(chunk[-1]["end_time"])
        text = " ".join(w["alternatives"][0]["content"] for w in chunk)
        blocks.append((start, end, text))

    lines = []
    for idx, (start, end, text) in enumerate(blocks, start=1):
        lines.append(str(idx))
        lines.append(f"{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


# ── Source collection ──────────────────────────────────────────────────────────

def collect_s3_sources(s3, bucket: str, prefix: str) -> list[dict]:
    log.info("Scanning s3://%s/%s for video files ...", bucket, prefix)
    sources = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            ext = Path(key).suffix.lower()
            if ext in VIDEO_EXTENSIONS:
                sources.append({
                    "uri": f"s3://{bucket}/{key}",
                    "stem": Path(key).stem,
                    "format": ext.lstrip("."),
                })
                log.debug("  Found: %s", key)
    log.info("Found %d video(s) in S3.", len(sources))
    return sources


def collect_url_sources(urls: list[str]) -> list[dict]:
    sources = []
    for url in urls:
        src = {"uri": url, "stem": stem_from_url(url), "format": format_from_url(url)}
        sources.append(src)
        log.info("URL source: %s  (stem=%s, format=%s)", url, src["stem"], src["format"])
    return sources


# ── Transcribe jobs ────────────────────────────────────────────────────────────

def start_jobs(transcribe, sources: list[dict]) -> dict[str, dict]:
    """
    Start one Transcribe job per source without specifying an output bucket.
    Transcribe stores results in a service-managed bucket and returns a
    pre-signed URL when the job completes — no S3 write permissions needed.
    Returns {job_name: source_dict}.
    """
    jobs = {}
    with tqdm(sources, desc="Starting jobs", unit="job", colour="cyan") as bar:
        for src in bar:
            job_name = safe_job_name(src["stem"])
            bar.set_postfix_str(job_name[:40])
            try:
                transcribe.start_transcription_job(
                    TranscriptionJobName=job_name,
                    Media={"MediaFileUri": src["uri"]},
                    MediaFormat=src["format"],
                    LanguageCode="en-US",
                    # No OutputBucketName → service-managed bucket, pre-signed URL on completion
                )
                log.info("Job started: %s", job_name)
            except transcribe.exceptions.ConflictException:
                log.warning("Job already exists, will track it: %s", job_name)
            jobs[job_name] = src

    return jobs


def wait_for_jobs(transcribe, jobs: dict[str, dict]) -> dict[str, str]:
    """Poll until all jobs finish. Returns {job_name: status_string}."""
    total = len(jobs)
    results: dict[str, str] = {}
    pending = set(jobs.keys())

    log.info("Polling %d job(s) every %ds ...", total, POLL_INTERVAL_SECONDS)

    with tqdm(total=total, desc="Transcribing", unit="job", colour="green") as bar:
        while pending:
            still_pending = set()
            for job_name in list(pending):
                resp = transcribe.get_transcription_job(TranscriptionJobName=job_name)
                job_info = resp["TranscriptionJob"]
                status = job_info["TranscriptionJobStatus"]

                if status == "COMPLETED":
                    results[job_name] = "COMPLETED"
                    bar.update(1)
                    log.info("✓ Completed: %s", job_name)

                elif status == "FAILED":
                    reason = job_info.get("FailureReason", "unknown")
                    results[job_name] = f"FAILED: {reason}"
                    bar.update(1)
                    log.error("✗ Failed: %s — %s", job_name, reason)

                else:
                    still_pending.add(job_name)

            pending = still_pending
            if pending:
                bar.set_postfix_str(f"{len(pending)} pending, next check in {POLL_INTERVAL_SECONDS}s")
                log.debug("Still pending: %s", ", ".join(sorted(pending)))
                time.sleep(POLL_INTERVAL_SECONDS)

    return results


# ── Download transcript & save SRT ────────────────────────────────────────────

def fetch_transcript_json(transcribe, job_name: str) -> dict:
    resp = transcribe.get_transcription_job(TranscriptionJobName=job_name)
    presigned_url = resp["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
    log.debug("Fetching transcript from pre-signed URL for: %s", job_name)
    with urllib.request.urlopen(presigned_url) as response:
        return json.loads(response.read())


def save_srts(transcribe, jobs: dict[str, dict], download_dir: Path):
    download_dir.mkdir(parents=True, exist_ok=True)
    with tqdm(jobs.items(), desc="Saving SRTs", unit="file", colour="yellow") as bar:
        for job_name, src in bar:
            local_srt = download_dir / (src["stem"] + ".srt")
            bar.set_postfix_str(local_srt.name[:40])
            try:
                transcript_json = fetch_transcript_json(transcribe, job_name)
                srt_content = build_srt(transcript_json)
                local_srt.write_text(srt_content, encoding="utf-8")
                log.info("Saved: %s", local_srt)
            except Exception as e:
                log.error("Failed to save SRT for %s: %s", job_name, e)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe videos from S3 or public URLs → local SRT files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bucket", help="S3 bucket to scan for videos")
    parser.add_argument("--input-prefix", default="", help="S3 key prefix filter (used with --bucket)")
    parser.add_argument("--urls", nargs="+", metavar="URL",
                        help="One or more public HTTPS video URLs to transcribe")
    parser.add_argument("--download-dir", default="./transcripts",
                        help="Local directory to save .srt files (default: ./transcripts)")
    parser.add_argument("--region", default=None, help="AWS region override")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG-level logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.bucket and not args.urls:
        parser.error("Provide at least one of --bucket or --urls.")

    log.info("=== Video Transcription Started ===")
    start_time = time.monotonic()

    session = boto3.Session(region_name=args.region)
    s3 = session.client("s3")
    transcribe = session.client("transcribe")

    sources: list[dict] = []

    if args.bucket:
        sources.extend(collect_s3_sources(s3, args.bucket, args.input_prefix))

    if args.urls:
        sources.extend(collect_url_sources(args.urls))

    if not sources:
        log.warning("No video sources found. Nothing to do.")
        return

    log.info("Total sources to transcribe: %d", len(sources))

    jobs = start_jobs(transcribe, sources)
    results = wait_for_jobs(transcribe, jobs)

    completed = {j: src for j, src in jobs.items() if results.get(j) == "COMPLETED"}
    failed = {j: results[j] for j in jobs if results.get(j, "").startswith("FAILED")}

    log.info("Results — completed: %d, failed: %d", len(completed), len(failed))

    if completed:
        save_srts(transcribe, completed, Path(args.download_dir))

    if failed:
        log.error("Failed jobs:")
        for j, reason in failed.items():
            log.error("  %s: %s", j, reason)

    elapsed = time.monotonic() - start_time
    log.info("=== Done in %.1fs ===", elapsed)


if __name__ == "__main__":
    main()
