#!/usr/bin/env python3
"""
Generic movie dataset preprocessor for DiffSynth-Studio style training data.

Creates short high-quality clips and an empty metadata.csv from a feature film
or similar long-form video source.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TARGET_MIN = 20
DEFAULT_TARGET_MAX = 35
DEFAULT_CLIP_MIN = 3.0
DEFAULT_CLIP_MAX = 5.0
DEFAULT_CLIP_TARGET = 4.0
DEFAULT_OUTPUT_W = 1920
DEFAULT_OUTPUT_H = 1080
DEFAULT_SCENE_THRESH = 0.2
DEFAULT_SUB_BUFFER = 0.8
DEFAULT_MIN_GAP = 150
DEFAULT_SKIP_START = 4 * 60
DEFAULT_SKIP_END = 5 * 60
DEFAULT_CRF = 17
BLACK_BAR_THRESHOLD = 18.0
MIN_CROP_MARGIN = 4


@dataclass
class Interval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlaps(self, other: "Interval", buf: float = 0.0) -> bool:
        return self.start - buf < other.end and self.end + buf > other.start


@dataclass
class Clip:
    start: float
    end: float
    brightness: float = 0.0
    detail: float = 0.0
    motion: float = 0.0
    skin_ratio: float = 0.0
    is_bw: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2

    @property
    def has_face(self) -> bool:
        return self.skin_ratio > 0.04

    @property
    def is_empty_shot(self) -> bool:
        return not self.has_face

    @property
    def score(self) -> float:
        base = self.detail * 0.5 + self.brightness * 0.3 + self.motion * 0.2
        return base * (1.0 if self.has_face else 2.5)


@dataclass
class CropRegion:
    width: int
    height: int
    x: int = 0
    y: int = 0

    def ffmpeg_expr(self) -> str:
        return f"{self.width}:{self.height}:{self.x}:{self.y}"


@dataclass
class VideoInfo:
    path: Path
    duration: float
    width: int
    height: int
    subtitle_streams: list[dict[str, Any]]


@dataclass
class Settings:
    input_video: Path
    output_dir: Path
    cache_dir: Path
    target_min: int
    target_max: int
    clip_min: float
    clip_max: float
    clip_target: float
    output_w: int
    output_h: int
    scene_thresh: float
    subtitle_index: int | None
    sub_buffer: float
    min_gap: int
    skip_start: float
    skip_end: float
    crf: int
    crop_mode: str
    crop_region: CropRegion
    cache_prefix: str
    total_dur: float


def banner(msg: str) -> None:
    print(f"\n{'─' * 60}\n  {msg}\n{'─' * 60}", flush=True)


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, **kwargs)


def srt_to_sec(value: str) -> float:
    value = value.replace(",", ".")
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def fmt_time(sec: float) -> str:
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    seconds = sec % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def slugify_stem(path: Path) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", path.stem).strip("_").lower()
    return slug or "movie"


def cache_name(prefix: str, kind: str) -> str:
    return f"{prefix}_{kind}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate training clips from a movie file."
    )
    parser.add_argument("--input-video", type=Path, required=True, help="Input movie file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for exported clips and metadata.csv. Defaults to dataset_<movie> beside the input file.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Directory for reusable preprocessing caches. Defaults to <input-dir>/.cache.",
    )
    parser.add_argument("--target-min", type=int, default=DEFAULT_TARGET_MIN)
    parser.add_argument("--target-max", type=int, default=DEFAULT_TARGET_MAX)
    parser.add_argument("--clip-min", type=float, default=DEFAULT_CLIP_MIN)
    parser.add_argument("--clip-max", type=float, default=DEFAULT_CLIP_MAX)
    parser.add_argument("--clip-target", type=float, default=DEFAULT_CLIP_TARGET)
    parser.add_argument("--output-width", type=int, default=DEFAULT_OUTPUT_W)
    parser.add_argument("--output-height", type=int, default=DEFAULT_OUTPUT_H)
    parser.add_argument("--scene-thresh", type=float, default=DEFAULT_SCENE_THRESH)
    parser.add_argument(
        "--subtitle-stream",
        default="auto",
        help="Subtitle stream index among subtitle streams, 'auto', or 'none'.",
    )
    parser.add_argument("--sub-buffer", type=float, default=DEFAULT_SUB_BUFFER)
    parser.add_argument("--min-gap", type=int, default=DEFAULT_MIN_GAP)
    parser.add_argument("--skip-start", type=float, default=DEFAULT_SKIP_START)
    parser.add_argument("--skip-end", type=float, default=DEFAULT_SKIP_END)
    parser.add_argument("--crf", type=int, default=DEFAULT_CRF)
    parser.add_argument(
        "--crop-mode",
        choices=("auto", "manual", "none"),
        default="auto",
        help="Auto-detect black bars, use --crop manually, or disable cropping.",
    )
    parser.add_argument(
        "--crop",
        help="Manual crop as width:height:x:y. Only used with --crop-mode manual.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.target_min <= 0 or args.target_max <= 0:
        raise SystemExit("target clip counts must be positive")
    if args.target_min > args.target_max:
        raise SystemExit("--target-min must be <= --target-max")
    if not (0 < args.clip_min <= args.clip_max):
        raise SystemExit("clip duration bounds are invalid")
    if not (args.clip_min <= args.clip_target <= args.clip_max):
        raise SystemExit("--clip-target must be within [clip-min, clip-max]")
    if args.output_width <= 0 or args.output_height <= 0:
        raise SystemExit("output dimensions must be positive")
    if args.crop_mode == "manual" and not args.crop:
        raise SystemExit("--crop is required when --crop-mode=manual")
    if args.crop_mode != "manual" and args.crop:
        raise SystemExit("--crop can only be used when --crop-mode=manual")


def parse_crop(value: str) -> CropRegion:
    match = re.fullmatch(r"(\d+):(\d+):(\d+):(\d+)", value.strip())
    if not match:
        raise SystemExit("--crop must be width:height:x:y")
    width, height, x, y = (int(group) for group in match.groups())
    if width <= 0 or height <= 0:
        raise SystemExit("manual crop width and height must be positive")
    return CropRegion(width=width, height=height, x=x, y=y)


def run_checked(command: list[str], *, text: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(command, capture_output=True, text=text)
    if result.returncode != 0:
        stderr = result.stderr if text else result.stderr.decode("utf-8", "ignore")
        raise SystemExit(stderr.strip() or f"Command failed: {' '.join(command)}")
    return result


def probe_video(path: Path) -> VideoInfo:
    result = run_checked(
        [
            "ffprobe",
            "-v",
            "error",
            "-probesize",
            "50000000",
            "-analyzeduration",
            "100000000",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ],
        text=True,
    )
    payload = json.loads(result.stdout)
    video_stream = next(
        (stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise SystemExit(f"No video stream found in {path}")

    subtitle_streams = [
        stream
        for stream in payload.get("streams", [])
        if stream.get("codec_type") == "subtitle"
    ]
    duration = float(payload.get("format", {}).get("duration") or video_stream.get("duration") or 0.0)
    if duration <= 0:
        raise SystemExit(f"Could not determine duration for {path}")

    return VideoInfo(
        path=path,
        duration=duration,
        width=int(video_stream.get("width") or 0),
        height=int(video_stream.get("height") or 0),
        subtitle_streams=subtitle_streams,
    )


def choose_subtitle_stream(raw: str, info: VideoInfo) -> int | None:
    value = raw.strip().lower()
    if value == "none":
        return None
    if value == "auto":
        if not info.subtitle_streams:
            return None
        for index, stream in enumerate(info.subtitle_streams):
            codec_name = str(stream.get("codec_name") or "").lower()
            if codec_name in {"subrip", "ass", "ssa", "mov_text", "webvtt"}:
                return index
        return 0
    try:
        selected = int(raw)
    except ValueError as exc:
        raise SystemExit("--subtitle-stream must be auto, none, or an integer") from exc
    if selected < 0 or selected >= len(info.subtitle_streams):
        raise SystemExit(
            f"Subtitle stream index {selected} is out of range; found {len(info.subtitle_streams)} subtitle streams."
        )
    return selected


def build_cache_prefix(info: VideoInfo, args: argparse.Namespace, subtitle_index: int | None) -> str:
    identity = {
        "file": str(info.path.resolve()),
        "size": info.path.stat().st_size,
        "mtime_ns": info.path.stat().st_mtime_ns,
        "scene_thresh": args.scene_thresh,
        "crop_mode": args.crop_mode,
        "crop": args.crop or "",
        "subtitle_index": subtitle_index,
        "sub_buffer": args.sub_buffer,
    }
    digest = hashlib.sha1(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return f"{slugify_stem(info.path)}_{digest}"


def default_output_dir(input_video: Path) -> Path:
    return input_video.parent / f"dataset_{slugify_stem(input_video)}"


def detect_black_bars(info: VideoInfo, output_w: int, output_h: int) -> CropRegion:
    banner("Detecting crop region")
    sample_points = [info.duration * frac for frac in (0.1, 0.25, 0.5, 0.75, 0.9)]
    cropdetect_values: list[tuple[int, int, int, int]] = []

    for timestamp in sample_points:
        result = run(
            [
                "ffmpeg",
                "-hide_banner",
                "-probesize",
                "50000000",
                "-analyzeduration",
                "100000000",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(info.path),
                "-frames:v",
                "12",
                "-vf",
                "cropdetect=limit=24:round=2:skip=2,metadata=mode=print",
                "-an",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
        )
        for line in result.stderr.splitlines():
            match = re.search(r"crop=(\d+):(\d+):(\d+):(\d+)", line)
            if match:
                cropdetect_values.append(tuple(int(group) for group in match.groups()))

    if not cropdetect_values:
        log("Crop detection unavailable, using full frame")
        return CropRegion(width=info.width, height=info.height, x=0, y=0)

    widths = sorted(value[0] for value in cropdetect_values)
    heights = sorted(value[1] for value in cropdetect_values)
    xs = sorted(value[2] for value in cropdetect_values)
    ys = sorted(value[3] for value in cropdetect_values)
    candidate = CropRegion(
        width=widths[len(widths) // 2],
        height=heights[len(heights) // 2],
        x=xs[len(xs) // 2],
        y=ys[len(ys) // 2],
    )

    if candidate.width <= 0 or candidate.height <= 0:
        return CropRegion(width=info.width, height=info.height, x=0, y=0)

    x_margin = max(candidate.x, info.width - (candidate.x + candidate.width))
    y_margin = max(candidate.y, info.height - (candidate.y + candidate.height))
    if x_margin < MIN_CROP_MARGIN and y_margin < MIN_CROP_MARGIN:
        log("No meaningful black bars detected, using full frame")
        return CropRegion(width=info.width, height=info.height, x=0, y=0)

    # Reject aggressive crops that distort the frame more than target framing would need.
    target_aspect = output_w / output_h
    candidate_aspect = candidate.width / candidate.height
    full_aspect = info.width / info.height
    if abs(candidate_aspect - full_aspect) < 0.02 and x_margin < MIN_CROP_MARGIN * 2 and y_margin < MIN_CROP_MARGIN * 2:
        return CropRegion(width=info.width, height=info.height, x=0, y=0)
    if candidate.width < info.width * 0.7 or candidate.height < info.height * 0.7:
        log("Auto crop looked too aggressive, using full frame")
        return CropRegion(width=info.width, height=info.height, x=0, y=0)

    # Keep full frame if bars are minor and target center crop already covers it.
    if abs(candidate_aspect - target_aspect) < 0.02 and x_margin < MIN_CROP_MARGIN * 2 and y_margin < MIN_CROP_MARGIN * 2:
        return CropRegion(width=info.width, height=info.height, x=0, y=0)

    log(f"Detected crop: {candidate.ffmpeg_expr()}")
    return candidate


def load_dialogue(video: str, cache: Path, subtitle_index: int | None, sub_buffer: float) -> list[Interval]:
    if subtitle_index is None:
        banner("Subtitle timeline")
        log("No subtitle stream selected, dialogue filtering disabled")
        return []
    if cache.exists():
        banner("Subtitle timeline  [cached]")
        with cache.open() as handle:
            return [Interval(item["s"], item["e"]) for item in json.load(handle)]

    banner("Phase 1: Extracting subtitle dialogue timeline")
    result = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-probesize",
            "50000000",
            "-analyzeduration",
            "100000000",
            "-i",
            video,
            "-map",
            f"0:s:{subtitle_index}",
            "-f",
            "srt",
            "pipe:1",
        ],
        capture_output=True,
        text=True,
    )
    srt = result.stdout
    if not srt.strip():
        log("WARNING: Selected subtitle stream returned no SRT text; dialogue filtering disabled")
        return []

    pattern = r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})"
    intervals: list[Interval] = []
    for match in re.finditer(pattern, srt):
        start = max(0.0, srt_to_sec(match.group(1)) - sub_buffer)
        end = srt_to_sec(match.group(2)) + sub_buffer
        intervals.append(Interval(start, end))

    intervals.sort(key=lambda interval: interval.start)
    merged: list[Interval] = []
    for interval in intervals:
        if merged and interval.start <= merged[-1].end:
            merged[-1].end = max(merged[-1].end, interval.end)
        else:
            merged.append(interval)

    total_seconds = sum(interval.duration for interval in merged)
    log(f"Dialogue segments : {len(merged)}")
    log(f"Total dialogue    : {total_seconds / 60:.1f} min")

    with cache.open("w") as handle:
        json.dump([{"s": item.start, "e": item.end} for item in merged], handle)

    return merged


def has_dialogue(start: float, end: float, dialogue: list[Interval]) -> bool:
    clip = Interval(start, end)
    return any(clip.overlaps(interval) for interval in dialogue)


def detect_scenes(video: str, total_dur: float, cache: Path, scene_thresh: float) -> list[float]:
    if cache.exists():
        banner("Scene detection  [cached]")
        with cache.open() as handle:
            return json.load(handle)

    banner(f"Phase 2: Scene detection  ({total_dur / 60:.0f} min video)")
    result = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-probesize",
            "50000000",
            "-analyzeduration",
            "100000000",
            "-i",
            video,
            "-vf",
            f"scale=160:90,select='gt(scene,{scene_thresh})',showinfo",
            "-vsync",
            "0",
            "-an",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )

    timestamps: list[float] = [0.0]
    for line in result.stderr.splitlines():
        if "Parsed_showinfo" in line and "pts_time" in line:
            match = re.search(r"pts_time:(\d+\.?\d*)", line)
            if match:
                timestamps.append(float(match.group(1)))
    timestamps.append(total_dur)
    timestamps = sorted(set(timestamps))

    log(f"Scene cuts found  : {len(timestamps) - 2}")
    with cache.open("w") as handle:
        json.dump(timestamps, handle)
    return timestamps


def make_candidates(cuts: list[float], settings: Settings) -> list[Clip]:
    skip_end = settings.total_dur - settings.skip_end
    candidates: list[Clip] = []

    for index in range(len(cuts) - 1):
        seg_s = cuts[index]
        seg_e = cuts[index + 1]
        seg_d = seg_e - seg_s

        if seg_s < settings.skip_start or seg_e > skip_end:
            continue
        if seg_d < settings.clip_min:
            continue

        if seg_d <= settings.clip_max:
            clip_start, clip_end = seg_s, seg_e
        else:
            midpoint = (seg_s + seg_e) / 2
            clip_start = max(seg_s, midpoint - settings.clip_target / 2)
            clip_end = min(seg_e, clip_start + settings.clip_target)
            if clip_end - clip_start < settings.clip_min:
                continue

        candidates.append(Clip(clip_start, clip_end))

    return candidates


def compute_clip_signature(clip: Clip, video: str, crop_region: CropRegion) -> tuple[float, bool]:
    width, height = 160, 74
    pixel_count = width * height
    max_skin = 0.0
    any_bw = False

    for frac in (0.2, 0.5, 0.8):
        timestamp = clip.start + clip.duration * frac
        result = run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "quiet",
                "-ss",
                str(timestamp),
                "-i",
                video,
                "-vframes",
                "1",
                "-vf",
                f"crop={crop_region.ffmpeg_expr()},scale={width}:{height}",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            capture_output=True,
        )
        raw = result.stdout
        if len(raw) < pixel_count * 3:
            continue

        skin_count = 0
        total_sat = 0.0
        for index in range(0, pixel_count * 3 - 2, 3):
            red, green, blue = raw[index], raw[index + 1], raw[index + 2]
            max_channel = max(red, green, blue)
            min_channel = min(red, green, blue)
            total_sat += (max_channel - min_channel) / max(max_channel, 1)
            cr = int(0.5000 * red - 0.4187 * green - 0.0813 * blue + 128)
            cb = int(-0.1687 * red - 0.3313 * green + 0.5000 * blue + 128)
            if 133 <= cr <= 173 and 77 <= cb <= 127:
                skin_count += 1

        max_skin = max(max_skin, skin_count / pixel_count)
        if (total_sat / pixel_count) < 0.12:
            any_bw = True

    return max_skin, any_bw


def has_black_frames(clip: Clip, video: str, crop_region: CropRegion, max_black_s: float = 0.3) -> bool:
    result = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-probesize",
            "50000000",
            "-analyzeduration",
            "100000000",
            "-ss",
            str(clip.start),
            "-i",
            video,
            "-t",
            str(clip.duration),
            "-vf",
            f"crop={crop_region.ffmpeg_expr()},scale=160:74,blackdetect=d={max_black_s}:pix_th=0.1",
            "-an",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    return "black_start" in result.stderr


def score_clip(clip: Clip, settings: Settings, dialogue: list[Interval]) -> Clip:
    if has_black_frames(clip, str(settings.input_video), settings.crop_region):
        return clip

    clip.skin_ratio, clip.is_bw = compute_clip_signature(
        clip,
        str(settings.input_video),
        settings.crop_region,
    )
    in_dialogue = has_dialogue(clip.start, clip.end, dialogue)
    if (clip.has_face or clip.is_bw) and in_dialogue:
        return clip

    width, height = 160, 74
    result = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "quiet",
            "-ss",
            str(clip.start),
            "-i",
            str(settings.input_video),
            "-t",
            str(clip.duration),
            "-vf",
            f"crop={settings.crop_region.ffmpeg_expr()},scale={width}:{height},fps=2.0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ],
        capture_output=True,
    )

    raw = result.stdout
    frame_size = width * height
    if len(raw) < frame_size:
        return clip

    frames = [raw[index : index + frame_size] for index in range(0, len(raw) - frame_size + 1, frame_size)]
    bright_list: list[float] = []
    detail_list: list[float] = []
    for frame in frames:
        pixels = list(frame)
        count = len(pixels)
        average = sum(pixels) / count
        variance = sum((pixel - average) ** 2 for pixel in pixels) / count
        bright_list.append(average)
        detail_list.append(math.sqrt(variance))

    avg_brightness = sum(bright_list) / len(bright_list)
    avg_detail = sum(detail_list) / len(detail_list)

    if avg_brightness < 20:
        brightness_score = 0.05
    elif avg_brightness < 40:
        brightness_score = 0.4
    elif avg_brightness > 230:
        brightness_score = 0.2
    else:
        brightness_score = 1.0 - abs(avg_brightness - 120) / 120

    detail_score = min(1.0, avg_detail / 55.0)
    motion_score = 0.0
    if len(frames) >= 2:
        diffs = []
        for left, right in zip(frames, frames[1:]):
            diffs.append(sum(abs(a - b) for a, b in zip(left, right)) / frame_size)
        avg_diff = sum(diffs) / len(diffs)
        max_diff = max(diffs)
        if max_diff > 45:
            return clip
        if avg_diff < 1:
            motion_score = 0.15
        elif avg_diff > 35:
            motion_score = 0.25
        else:
            motion_score = min(1.0, avg_diff / 20.0)

    clip.brightness = brightness_score
    clip.detail = detail_score
    clip.motion = motion_score
    return clip


def score_all(clips: list[Clip], settings: Settings, dialogue: list[Interval]) -> list[Clip]:
    banner(f"Phase 4: Scoring {len(clips)} candidates")
    scored: list[Clip] = []
    for index, clip in enumerate(clips):
        if index % 50 == 0:
            log(f"{index}/{len(clips)} ...")
        scored.append(score_clip(clip, settings, dialogue))

    passed = [clip for clip in scored if clip.score > 0.05]
    empty = sum(1 for clip in passed if clip.is_empty_shot and not clip.is_bw)
    bw_person = sum(1 for clip in passed if clip.is_bw)
    color_person = sum(1 for clip in passed if clip.has_face and not clip.is_bw)
    log(f"After scoring -> environment: {empty} | B&W person: {bw_person} | color silent person: {color_person}")
    return scored


def select(clips: list[Clip], settings: Settings) -> list[Clip]:
    banner("Phase 5: Selecting best clips")
    ranked = sorted((clip for clip in clips if clip.score > 0.05), key=lambda clip: clip.score, reverse=True)
    log(f"Candidates after quality filter: {len(ranked)}")

    selected: list[Clip] = []
    gap = settings.min_gap
    while True:
        selected = []
        for clip in ranked:
            if len(selected) >= settings.target_max:
                break
            if all(abs(clip.midpoint - other.midpoint) >= gap for other in selected):
                selected.append(clip)
        if len(selected) >= settings.target_min:
            break
        if gap < 30:
            log("WARNING: Cannot reach target clip count even with tight spacing")
            break
        gap = max(30, int(gap * 0.65))
        log(f"Only {len(selected)} clips -> relaxing gap to {gap}s")

    selected.sort(key=lambda clip: clip.start)
    third = settings.total_dur / 3
    thirds = [
        sum(1 for clip in selected if clip.midpoint < third),
        sum(1 for clip in selected if third <= clip.midpoint < 2 * third),
        sum(1 for clip in selected if clip.midpoint >= 2 * third),
    ]
    empty = sum(1 for clip in selected if clip.is_empty_shot)
    people = len(selected) - empty
    log(f"Selected {len(selected)} clips -> environment: {empty} | silent person: {people}")
    log(f"Distribution (thirds): {thirds[0]} / {thirds[1]} / {thirds[2]}")
    return selected


def export_clip(clip: Clip, settings: Settings, out_path: Path) -> bool:
    filters: list[str] = []
    if settings.crop_region.width != settings.output_w or settings.crop_region.height != settings.output_h or settings.crop_region.x != 0 or settings.crop_region.y != 0:
        filters.append(f"crop={settings.crop_region.ffmpeg_expr()}")
    filters.append(f"scale=-2:{settings.output_h}")
    filters.append(f"crop={settings.output_w}:{settings.output_h}")
    vf = ",".join(filters)

    result = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-probesize",
            "50000000",
            "-analyzeduration",
            "100000000",
            "-ss",
            str(clip.start),
            "-i",
            str(settings.input_video),
            "-t",
            str(clip.duration),
            "-avoid_negative_ts",
            "make_zero",
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-crf",
            str(settings.crf),
            "-preset",
            "slow",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-y",
            str(out_path),
        ]
    )
    return result.returncode == 0


def export_all(clips: list[Clip], settings: Settings) -> list[tuple[str, Clip]]:
    banner(f"Phase 6: Exporting {len(clips)} clips")
    results: list[tuple[str, Clip]] = []
    for index, clip in enumerate(clips, start=1):
        name = f"clip_{index:03d}.mp4"
        path = settings.output_dir / name
        if clip.is_empty_shot:
            tag = "environment"
        elif clip.is_bw:
            tag = f"b&w-person(skin={clip.skin_ratio:.2f})"
        else:
            tag = f"person(skin={clip.skin_ratio:.2f})"
        log(
            f"[{index:2d}/{len(clips)}] {name}  "
            f"{fmt_time(clip.start)} -> {fmt_time(clip.end)}  "
            f"({clip.duration:.1f}s)  score={clip.score:.3f}  {tag}"
        )
        if export_clip(clip, settings, path):
            results.append((name, clip))
        else:
            log(f"!! FAILED: {name}")
    return results


def write_metadata(exported: list[tuple[str, Clip]], out_dir: Path) -> None:
    banner("Phase 7: Writing metadata.csv")
    path = out_dir / "metadata.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video", "prompt"])
        for name, _clip in exported:
            writer.writerow([name, ""])
    log(f"Saved: {path}")
    log("Prompts are blank. Fill metadata.csv before training.")


def build_settings(args: argparse.Namespace) -> Settings:
    validate_args(args)
    input_video = args.input_video.expanduser().resolve()
    if not input_video.exists():
        raise SystemExit(f"Input not found: {input_video}")

    info = probe_video(input_video)
    subtitle_index = choose_subtitle_stream(args.subtitle_stream, info)
    output_dir = (args.output_dir.expanduser() if args.output_dir else default_output_dir(input_video)).resolve()
    cache_dir = (args.cache_dir.expanduser() if args.cache_dir else input_video.parent / ".cache").resolve()
    cache_prefix = build_cache_prefix(info, args, subtitle_index)

    if args.crop_mode == "manual":
        crop_region = parse_crop(args.crop)
        if crop_region.x + crop_region.width > info.width or crop_region.y + crop_region.height > info.height:
            raise SystemExit("manual crop exceeds source frame bounds")
    elif args.crop_mode == "none":
        crop_region = CropRegion(width=info.width, height=info.height, x=0, y=0)
    else:
        crop_region = detect_black_bars(info, args.output_width, args.output_height)

    return Settings(
        input_video=input_video,
        output_dir=output_dir,
        cache_dir=cache_dir,
        target_min=args.target_min,
        target_max=args.target_max,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        clip_target=args.clip_target,
        output_w=args.output_width,
        output_h=args.output_height,
        scene_thresh=args.scene_thresh,
        subtitle_index=subtitle_index,
        sub_buffer=args.sub_buffer,
        min_gap=args.min_gap,
        skip_start=args.skip_start,
        skip_end=args.skip_end,
        crf=args.crf,
        crop_mode=args.crop_mode,
        crop_region=crop_region,
        cache_prefix=cache_prefix,
        total_dur=info.duration,
    )


def main() -> int:
    settings = build_settings(parse_args())
    print("=" * 60)
    print(" Generic Movie -> Dataset")
    print("=" * 60)

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)

    log(f"Input video      : {settings.input_video}")
    log(f"Duration         : {settings.total_dur / 60:.1f} min")
    log(f"Output dir       : {settings.output_dir}")
    log(f"Cache dir        : {settings.cache_dir}")
    log(f"Subtitle stream  : {settings.subtitle_index if settings.subtitle_index is not None else 'disabled'}")
    log(f"Crop region      : {settings.crop_region.ffmpeg_expr()}")

    dialogue = load_dialogue(
        str(settings.input_video),
        settings.cache_dir / cache_name(settings.cache_prefix, "subtitles"),
        settings.subtitle_index,
        settings.sub_buffer,
    )
    cuts = detect_scenes(
        str(settings.input_video),
        settings.total_dur,
        settings.cache_dir / cache_name(settings.cache_prefix, "scenes"),
        settings.scene_thresh,
    )

    banner("Phase 3: Generating candidates from stable segments")
    candidates = make_candidates(cuts, settings)
    log(f"Stable segment candidates: {len(candidates)}")
    if len(candidates) < settings.target_min:
        log("WARNING: few candidates; consider lowering scene threshold or subtitle buffer")

    scored = score_all(candidates, settings, dialogue)
    selected = select(scored, settings)
    exported = export_all(selected, settings)
    write_metadata(exported, settings.output_dir)

    print("\n" + "=" * 60)
    print(f"  DONE -> {len(exported)} clips exported to:")
    print(f"  {settings.output_dir}")
    print("=" * 60)
    print("\nDataset structure:")
    print("  dataset/")
    print("  ├── metadata.csv")
    for name, _clip in exported[:6]:
        print(f"  ├── {name}")
    if len(exported) > 6:
        print(f"  └── ... ({len(exported) - 6} more)")
    print("\nNext steps:")
    print(f"  1. Review clips in {settings.output_dir}")
    print("  2. Run caption_dataset.py to generate prompts")
    print(f"  3. Point DiffSynth-Studio at {settings.output_dir / 'metadata.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
