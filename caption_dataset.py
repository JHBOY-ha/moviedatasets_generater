#!/usr/bin/env python3
"""
Batch caption short video clips for generic movie LoRA training datasets.

The script reads metadata.csv, samples frames from each listed video, sends
those frames to an OpenAI-compatible vision endpoint, validates the returned
caption, and writes the accepted result back to the prompt column.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib import error as urlerror
from urllib import request as urlrequest


DEFAULT_METADATA = Path("dataset/metadata.csv")
DEFAULT_REPORT_NAME = "caption_report.csv"
FRAME_COUNT = 5
FRAME_HEIGHT = 480
ANALYSIS_WIDTH = 160
ANALYSIS_HEIGHT = 90
LOW_CONFIDENCE_THRESHOLD = 0.75
RETRYABLE_HTTP_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

CAPTION_API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CAPTION_API_KEY = ""
CAPTION_MODEL = "qwen3-vl-plus"
CAPTION_TIMEOUT = 60.0

BANNED_STYLE_TERMS = (
    "cinematic",
    "masterpiece",
    "epic",
    "film still",
    "highly detailed",
    "dramatic lighting",
    "stunning",
    "gorgeous",
    "beautiful",
    "photorealistic",
    "ultra detailed",
    "4k",
    "8k",
    "award-winning",
    "best quality",
    "aesthetic",
    "moody color grading",
)

UNCERTAIN_TERMS = (
    "maybe",
    "possibly",
    "probably",
    "unclear",
    "perhaps",
    "appears to be",
    "seems to be",
    "likely",
    "suggests that",
)

RELATIONSHIP_TERMS = (
    "father",
    "mother",
    "husband",
    "wife",
    "son",
    "daughter",
    "boss",
    "leader",
    "commander",
    "scientist",
    "politician",
    "judge",
    "witness",
    "suspect",
    "hero",
    "villain",
)

STORY_INFERENCE_TERMS = (
    "argues",
    "confronts",
    "confesses",
    "plans",
    "realizes",
    "discovers",
    "decides",
    "investigates",
    "celebrates",
    "mourns",
    "threatens",
    "warns",
)

TITLE_CUES = (
    "from the film",
    "from the movie",
    "in the film",
    "in the movie",
    "scene from",
    "movie scene",
    "film scene",
)

ENVIRONMENT_WORDS = {
    "office",
    "room",
    "street",
    "road",
    "city",
    "valley",
    "mountain",
    "mountains",
    "sky",
    "clouds",
    "forest",
    "trees",
    "building",
    "buildings",
    "interior",
    "exterior",
    "indoors",
    "outdoors",
    "laboratory",
    "hallway",
    "desert",
    "field",
    "house",
    "town",
    "camp",
    "night",
    "daylight",
    "yard",
    "courtroom",
    "apartment",
}

ACTION_WORDS = {
    "standing",
    "sitting",
    "walking",
    "moving",
    "driving",
    "looking",
    "watching",
    "leaning",
    "turning",
    "drifting",
    "rising",
    "falling",
    "crossing",
    "running",
    "waiting",
    "listening",
    "holding",
    "talking",
    "working",
    "glancing",
    "gesturing",
}

OBJECT_WORDS = {
    "car",
    "cars",
    "vehicle",
    "vehicles",
    "lights",
    "light",
    "desk",
    "window",
    "hat",
    "glasses",
    "suit",
    "uniform",
    "trees",
    "peaks",
    "clouds",
    "road",
    "street",
    "building",
    "buildings",
    "chair",
    "door",
    "smoke",
    "machine",
    "tower",
    "papers",
    "table",
    "microphone",
}

SUBJECT_WORDS = {
    "man",
    "men",
    "woman",
    "women",
    "person",
    "people",
    "figure",
    "figures",
    "group",
    "crowd",
    "soldier",
    "soldiers",
    "worker",
    "workers",
    "elderly",
    "young",
    "child",
    "audience",
}


SYSTEM_PROMPT = """
You write factual training captions for short movie clips.

Your job is to describe visible facts in English with enough detail for model
training while keeping style language minimal.

Rules:
- Focus on subject, setting, action, visible objects, and spatial relationships.
- Use generic identities only. Never use character names, actor names, real-person names, movie titles, or franchise names.
- Do not infer plot, relationships, professions, or motivations unless they are directly visible.
- Do not quote dialogue, subtitles, or text on screen.
- Prefer one detailed sentence. Two sentences maximum.
- Keep the caption between 22 and 55 English words.
- Use concrete visual details, not aesthetic praise.
- Return JSON only with keys:
  caption, confidence, entities, scene_type, style_terms_used, reason

Example environment:
{
  "caption": "Low clouds drift over a broad valley, with dark tree lines in the foreground and layered ridges fading into a pale overcast sky.",
  "confidence": 0.91,
  "entities": ["clouds", "valley", "tree line", "ridges"],
  "scene_type": "environment",
  "style_terms_used": [],
  "reason": "Describes only visible landscape details."
}

Example person-focused:
{
  "caption": "A person wearing a dark coat and brimmed hat stands near a window in a modest interior, turning slightly while soft light falls across nearby furniture and pale walls.",
  "confidence": 0.9,
  "entities": ["person", "dark coat", "brimmed hat", "window", "interior"],
  "scene_type": "person-focused",
  "style_terms_used": [],
  "reason": "Uses generic identity and visible clothing, pose, and setting."
}

Example black-and-white:
{
  "caption": "A black-and-white indoor scene shows several people seated at long tables with papers and microphones, while one figure gestures forward under even overhead lighting.",
  "confidence": 0.88,
  "entities": ["people", "tables", "papers", "microphones"],
  "scene_type": "black-and-white",
  "style_terms_used": ["black-and-white"],
  "reason": "Covers layout, objects, and action without any proper names or plot claims."
}
""".strip()


class CaptionError(RuntimeError):
    """Recoverable caption-generation failure for one row."""


class FatalAPIError(RuntimeError):
    """Configuration or authentication failure that should abort the run."""


@dataclass
class APIConfig:
    base_url: str
    api_key: str
    model: str
    timeout: float


@dataclass
class VideoInfo:
    duration: float
    width: int
    height: int


@dataclass
class FrameAnalysis:
    timestamp: float
    image_url: str
    brightness: float
    saturation: float
    skin_ratio: float


@dataclass
class VideoFeatures:
    duration: float
    width: int
    height: int
    timestamps: list[float]
    frames: list[FrameAnalysis]
    brightness_min: float
    brightness_max: float
    is_bw: bool
    person_present: bool
    group_present: bool

    @property
    def scene_hint(self) -> str:
        if self.is_bw:
            return "black-and-white"
        if self.group_present:
            return "group-interior"
        if self.person_present:
            return "person-focused"
        return "environment"


@dataclass
class CaptionCandidate:
    caption: str
    confidence: float
    entities: list[str]
    scene_type: str
    style_terms_used: list[str]
    reason: str
    repaired: bool = False
    fallback_used: bool = False


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate factual captions for movie clip datasets."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="Path to metadata.csv (default: dataset/metadata.csv)",
    )
    parser.add_argument(
        "--overwrite",
        choices=("blank", "all"),
        default="blank",
        help="Overwrite only blank prompts or rewrite all prompts.",
    )
    parser.add_argument(
        "--only",
        help="Comma-separated list of video filenames to process.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate captions and report output without updating metadata.csv.",
    )
    parser.add_argument("--api-base-url", help="Override API base URL.")
    parser.add_argument("--api-key", help="Override API key.")
    parser.add_argument("--model", help="Override model name.")
    parser.add_argument("--timeout", type=float, help="Override API timeout in seconds.")
    return parser.parse_args()


def load_api_config(args: argparse.Namespace) -> APIConfig:
    base_url = (args.api_base_url or os.environ.get("CAPTION_API_BASE_URL", CAPTION_API_BASE_URL)).strip()
    api_key = (args.api_key or os.environ.get("CAPTION_API_KEY", CAPTION_API_KEY)).strip()
    model = (args.model or os.environ.get("CAPTION_MODEL", CAPTION_MODEL)).strip()

    missing = []
    if not base_url:
        missing.append("CAPTION_API_BASE_URL")
    if not api_key:
        missing.append("CAPTION_API_KEY")
    if not model:
        missing.append("CAPTION_MODEL")
    if missing:
        raise SystemExit(
            "Missing API config. Set values via CLI flags, env vars, or the defaults at the top of caption_dataset.py: "
            + ", ".join(missing)
        )

    timeout_value = args.timeout if args.timeout is not None else os.environ.get("CAPTION_TIMEOUT", str(CAPTION_TIMEOUT))
    try:
        timeout = float(timeout_value)
    except ValueError as exc:
        raise SystemExit("CAPTION_TIMEOUT/--timeout must be a number") from exc

    return APIConfig(base_url=base_url, api_key=api_key, model=model, timeout=timeout)


def run_command(command: list[str], *, text: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(command, capture_output=True, text=text)
    if result.returncode != 0:
        stderr = result.stderr.strip() if text else result.stderr.decode("utf-8", "ignore").strip()
        raise CaptionError(stderr or f"Command failed: {' '.join(command)}")
    return result


def read_metadata(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise SystemExit(f"metadata file not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"metadata file is empty: {path}")
        fieldnames = list(reader.fieldnames)
        for column in ("video", "prompt"):
            if column not in fieldnames:
                raise SystemExit(f"metadata.csv must contain '{column}' column")
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def resolve_video_path(metadata_path: Path, video_value: str) -> Path:
    candidate = Path(video_value)
    if candidate.is_absolute():
        return candidate
    return metadata_path.parent / candidate


def parse_only_filter(raw_value: str | None) -> set[str] | None:
    if not raw_value:
        return None
    values = {item.strip() for item in raw_value.split(",") if item.strip()}
    return values or None


def should_process_row(
    row: dict[str, str],
    *,
    overwrite: str,
    only_filter: set[str] | None,
) -> bool:
    video_name = row.get("video", "").strip()
    if only_filter is not None and video_name not in only_filter:
        return False
    if overwrite == "blank":
        return not row.get("prompt", "").strip()
    return True


def probe_video(video_path: Path) -> VideoInfo:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]
    result = run_command(command, text=True)
    payload = json.loads(result.stdout)
    video_stream = next(
        (stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise CaptionError(f"No video stream found in {video_path}")

    duration = float(payload.get("format", {}).get("duration") or video_stream.get("duration") or 0.0)
    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    if duration <= 0:
        raise CaptionError(f"Could not determine duration for {video_path}")
    return VideoInfo(duration=duration, width=width, height=height)


def build_sample_timestamps(duration: float, count: int = FRAME_COUNT) -> list[float]:
    if count <= 1:
        return [max(0.0, min(duration - 0.05, duration / 2))]
    start = 0.12
    end = 0.88
    timestamps: list[float] = []
    for index in range(count):
        frac = start + (end - start) * (index / (count - 1))
        moment = max(0.0, min(duration - 0.05, duration * frac))
        timestamps.append(round(moment, 3))
    deduped = sorted(set(timestamps))
    return deduped or [round(max(0.0, duration / 2), 3)]


def extract_jpeg_frame(video_path: Path, timestamp: float) -> bytes:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        f"scale=-2:{FRAME_HEIGHT}",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]
    result = run_command(command, text=False)
    if not result.stdout:
        raise CaptionError(f"Could not extract JPEG frame from {video_path} at {timestamp:.3f}s")
    return result.stdout


def extract_raw_frame(video_path: Path, timestamp: float) -> bytes:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        f"scale={ANALYSIS_WIDTH}:{ANALYSIS_HEIGHT}:flags=lanczos",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    result = run_command(command, text=False)
    expected_size = ANALYSIS_WIDTH * ANALYSIS_HEIGHT * 3
    if len(result.stdout) < expected_size:
        raise CaptionError(f"Could not extract analysis frame from {video_path} at {timestamp:.3f}s")
    return result.stdout[:expected_size]


def rgb_to_data_url(frame_bytes: bytes) -> str:
    encoded = base64.b64encode(frame_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def analyze_raw_frame(raw: bytes) -> tuple[float, float, float]:
    pixel_count = ANALYSIS_WIDTH * ANALYSIS_HEIGHT
    total_brightness = 0.0
    total_saturation = 0.0
    skin_pixels = 0

    for index in range(0, pixel_count * 3, 3):
        red = raw[index]
        green = raw[index + 1]
        blue = raw[index + 2]
        brightness = 0.299 * red + 0.587 * green + 0.114 * blue
        total_brightness += brightness

        max_channel = max(red, green, blue)
        min_channel = min(red, green, blue)
        if max_channel:
            total_saturation += (max_channel - min_channel) / max_channel

        cr = int(0.5000 * red - 0.4187 * green - 0.0813 * blue + 128)
        cb = int(-0.1687 * red - 0.3313 * green + 0.5000 * blue + 128)
        if 133 <= cr <= 173 and 77 <= cb <= 127:
            skin_pixels += 1

    avg_brightness = total_brightness / pixel_count
    avg_saturation = total_saturation / pixel_count
    skin_ratio = skin_pixels / pixel_count
    return avg_brightness, avg_saturation, skin_ratio


def gather_video_features(video_path: Path) -> VideoFeatures:
    info = probe_video(video_path)
    timestamps = build_sample_timestamps(info.duration, FRAME_COUNT)
    frames: list[FrameAnalysis] = []

    for timestamp in timestamps:
        jpeg_bytes = extract_jpeg_frame(video_path, timestamp)
        raw_bytes = extract_raw_frame(video_path, timestamp)
        brightness, saturation, skin_ratio = analyze_raw_frame(raw_bytes)
        frames.append(
            FrameAnalysis(
                timestamp=timestamp,
                image_url=rgb_to_data_url(jpeg_bytes),
                brightness=brightness,
                saturation=saturation,
                skin_ratio=skin_ratio,
            )
        )

    brightness_values = [frame.brightness for frame in frames]
    saturation_values = [frame.saturation for frame in frames]
    skin_ratios = [frame.skin_ratio for frame in frames]
    max_skin_ratio = max(skin_ratios)

    is_bw = (sum(saturation_values) / len(saturation_values)) < 0.12 or sum(
        1 for value in saturation_values if value < 0.12
    ) >= max(2, len(saturation_values) // 2)
    person_frames = sum(1 for ratio in skin_ratios if ratio > 0.04)
    strong_person_frames = sum(1 for ratio in skin_ratios if ratio > 0.08)
    person_present = max_skin_ratio > 0.04
    group_present = strong_person_frames >= 2 and person_frames >= 3

    return VideoFeatures(
        duration=info.duration,
        width=info.width,
        height=info.height,
        timestamps=timestamps,
        frames=frames,
        brightness_min=min(brightness_values),
        brightness_max=max(brightness_values),
        is_bw=is_bw,
        person_present=person_present,
        group_present=group_present,
    )


def build_generation_messages(video_name: str, features: VideoFeatures) -> list[dict[str, Any]]:
    feature_summary = {
        "video": video_name,
        "duration_seconds": round(features.duration, 3),
        "resolution": f"{features.width}x{features.height}",
        "sample_timestamps_seconds": features.timestamps,
        "likely_black_and_white": features.is_bw,
        "likely_person_visible": features.person_present,
        "likely_group_visible": features.group_present,
        "brightness_range": [round(features.brightness_min, 1), round(features.brightness_max, 1)],
        "scene_hint": features.scene_hint,
    }

    prompt = (
        "Generate one detailed English caption for this short training clip.\n"
        "Requirements:\n"
        "- 22 to 55 words\n"
        "- One sentence preferred, two sentences maximum\n"
        "- Describe visible facts only\n"
        "- Use generic identities only\n"
        "- Do not mention character names, actor names, real-person names, titles, or franchises\n"
        "- Do not infer profession, relationship, plot, or motivation unless directly visible\n"
        "- Prioritize subject, environment, action, objects, and layout\n"
        "- Keep style wording minimal and neutral\n"
        "- Do not mention subtitles, dialogue, or text on screen\n"
        "- Return JSON only\n\n"
        f"Clip metadata:\n{json.dumps(feature_summary, ensure_ascii=True)}"
    )

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for frame in features.frames:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": frame.image_url, "detail": "low"},
            }
        )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def build_repair_messages(candidate: CaptionCandidate, errors: list[str]) -> list[dict[str, str]]:
    repair_prompt = {
        "caption": candidate.caption,
        "confidence": candidate.confidence,
        "entities": candidate.entities,
        "scene_type": candidate.scene_type,
        "style_terms_used": candidate.style_terms_used,
        "reason": candidate.reason,
        "validation_errors": errors,
    }
    user_prompt = (
        "Rewrite only the caption so it satisfies the rules.\n"
        "Keep the same visible content.\n"
        "Do not add names, titles, dialogue, plot, relationship guesses, or aesthetic filler.\n"
        "Keep the JSON schema unchanged.\n"
        "Return JSON only.\n\n"
        f"{json.dumps(repair_prompt, ensure_ascii=True)}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def endpoint_for_base_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/chat/completions"):
        return trimmed
    return f"{trimmed}/chat/completions"


def parse_chat_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not choices:
        raise CaptionError("API response contained no choices")
    message = choices[0].get("message", {})
    content = message.get("content")

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        joined = "\n".join(part.strip() for part in parts if part.strip())
        if joined:
            return joined
    raise CaptionError("API response did not contain text content")


def post_chat_completion(
    config: APIConfig,
    messages: list[dict[str, Any]],
    *,
    max_retries: int = 2,
) -> str:
    payload = json.dumps(
        {
            "model": config.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 350,
        }
    ).encode("utf-8")

    endpoint = endpoint_for_base_url(config.base_url)
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        req = urlrequest.Request(endpoint, data=payload, headers=headers, method="POST")
        try:
            with urlrequest.urlopen(req, timeout=config.timeout) as response:
                body = response.read().decode("utf-8")
            parsed = json.loads(body)
            return parse_chat_content(parsed)
        except urlerror.HTTPError as exc:
            body = exc.read().decode("utf-8", "ignore")
            if exc.code in (401, 403):
                raise FatalAPIError(f"Authentication failed ({exc.code}): {body or exc.reason}") from exc
            if exc.code in RETRYABLE_HTTP_CODES and attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                last_error = exc
                continue
            raise CaptionError(f"HTTP {exc.code}: {body or exc.reason}") from exc
        except urlerror.URLError as exc:
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                last_error = exc
                continue
            raise CaptionError(f"Network error: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise CaptionError(f"API returned invalid JSON payload: {exc}") from exc
    raise CaptionError(f"Request failed after retries: {last_error}")


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    for candidate in (stripped, _slice_json_candidate(stripped)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise CaptionError("Model output was not a valid JSON object")


def _slice_json_candidate(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start : end + 1]


def normalize_list(value: Any) -> list[str]:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def clean_caption_text(text: str) -> str:
    cleaned = text.strip().strip("`").strip()
    cleaned = cleaned.replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip("\"' ")
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def build_candidate_from_payload(payload: dict[str, Any]) -> CaptionCandidate:
    caption = clean_caption_text(str(payload.get("caption", "")))
    confidence_raw = payload.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(confidence_raw)))
    except (TypeError, ValueError):
        confidence = 0.0

    return CaptionCandidate(
        caption=caption,
        confidence=confidence,
        entities=normalize_list(payload.get("entities")),
        scene_type=str(payload.get("scene_type", "")).strip(),
        style_terms_used=normalize_list(payload.get("style_terms_used")),
        reason=str(payload.get("reason", "")).strip(),
    )


def caption_word_count(caption: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", caption))


def sentence_count(caption: str) -> int:
    parts = re.split(r"[.!?]+", caption)
    return sum(1 for item in parts if item.strip())


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text))


def ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(1 for char in text if 32 <= ord(char) <= 126)
    return printable / len(text)


def has_dialogue_markers(text: str) -> bool:
    return bool(
        re.search(
            r"['\"“”‘’]|"
            r"\b(says|said|speaks|speaking|subtitle|subtitles|captioned|text on screen|dialogue)\b",
            text,
            re.IGNORECASE,
        )
    )


def match_any(text: str, terms: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def infer_categories(caption: str, entities: list[str]) -> set[str]:
    lowered = caption.lower()
    categories: set[str] = set()

    if match_any(lowered, SUBJECT_WORDS) or any(match_any(entity.lower(), SUBJECT_WORDS) for entity in entities):
        categories.add("subject")
    if match_any(lowered, ENVIRONMENT_WORDS) or any(
        match_any(entity.lower(), ENVIRONMENT_WORDS) for entity in entities
    ):
        categories.add("environment")
    if match_any(lowered, ACTION_WORDS):
        categories.add("action")
    if match_any(lowered, OBJECT_WORDS) or any(match_any(entity.lower(), OBJECT_WORDS) for entity in entities):
        categories.add("object")
    return categories


def contains_proper_name(text: str) -> bool:
    sanitized = re.sub(r"\b(?:A|An|The|Black-and-white)\b", "", text)
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", sanitized):
        return True
    if re.search(r"\b(?:Mr|Mrs|Ms|Dr|Professor|President|General)\.?\s+[A-Z][a-z]+\b", sanitized):
        return True
    return False


def validate_candidate(candidate: CaptionCandidate) -> list[str]:
    errors: list[str] = []
    caption = candidate.caption

    if not caption:
        errors.append("caption is empty")
        return errors

    words = caption_word_count(caption)
    if words < 22 or words > 55:
        errors.append("caption must contain 22 to 55 English words")
    if sentence_count(caption) > 2:
        errors.append("caption must contain one or two sentences")
    if contains_cjk(caption):
        errors.append("caption must be English only")
    if ascii_ratio(caption) < 0.95:
        errors.append("caption contains too many non-ASCII characters")
    if contains_proper_name(caption) or any(contains_proper_name(entity) for entity in candidate.entities):
        errors.append("caption contains a probable proper name or title")
    if match_any(caption, TITLE_CUES):
        errors.append("caption references the film or movie explicitly")
    if match_any(caption, UNCERTAIN_TERMS):
        errors.append("caption contains uncertain wording")
    if match_any(caption, BANNED_STYLE_TERMS):
        errors.append("caption contains banned style filler")
    if has_dialogue_markers(caption):
        errors.append("caption contains dialogue or quote markers")
    if match_any(caption, RELATIONSHIP_TERMS):
        errors.append("caption contains relationship or role inference")
    if match_any(caption, STORY_INFERENCE_TERMS):
        errors.append("caption contains plot or motivation inference")
    categories = infer_categories(caption, candidate.entities)
    if len(categories) < 2:
        errors.append("caption must include at least two information categories")
    if not candidate.scene_type:
        errors.append("scene_type is missing")
    if not candidate.reason:
        errors.append("reason is missing")
    return errors


def fallback_caption(features: VideoFeatures) -> tuple[str, str]:
    if features.is_bw:
        return (
            "A black-and-white scene shows people and furnishings arranged in an interior space, with strong tonal contrast, steady framing, and visible tables, walls, or other set details shaping the composition.",
            "black-and-white",
        )
    if features.group_present:
        return (
            "Several people occupy an indoor setting, with seated or standing figures arranged around furniture and objects while the camera holds on their positions, gestures, and the surrounding room layout.",
            "group-interior",
        )
    if features.person_present:
        return (
            "A person remains the main focus within a built environment, with visible clothing, posture, and nearby objects defining the shot as the figure shifts slightly within the frame.",
            "person-focused",
        )
    if features.brightness_max - features.brightness_min > 45:
        return (
            "An exterior view shows open space, layered background elements, and changing light across the frame, with landscape or street details providing depth and gentle motion.",
            "street-exterior",
        )
    return (
        "An environment-focused shot shows open space, structural elements, and background depth, with lighting, surfaces, and distant forms providing most of the visible detail rather than a single dominant person.",
        "environment",
    )


def generate_candidate(
    config: APIConfig,
    video_name: str,
    features: VideoFeatures,
) -> CaptionCandidate:
    response_text = post_chat_completion(config, build_generation_messages(video_name, features))
    payload = extract_json_object(response_text)
    candidate = build_candidate_from_payload(payload)
    errors = validate_candidate(candidate)
    if not errors:
        return candidate

    repair_text = post_chat_completion(config, build_repair_messages(candidate, errors))
    repaired_payload = extract_json_object(repair_text)
    repaired = build_candidate_from_payload(repaired_payload)
    repaired.repaired = True
    repair_errors = validate_candidate(repaired)
    if not repair_errors:
        return repaired

    fallback_text, fallback_scene_type = fallback_caption(features)
    return CaptionCandidate(
        caption=fallback_text,
        confidence=min(candidate.confidence, repaired.confidence, 0.5),
        entities=repaired.entities or candidate.entities,
        scene_type=repaired.scene_type or candidate.scene_type or fallback_scene_type,
        style_terms_used=[],
        reason="Fallback used after validation failure: " + "; ".join(repair_errors),
        repaired=True,
        fallback_used=True,
    )


def write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with tempfile.NamedTemporaryFile(
        "w",
        newline="",
        encoding="utf-8",
        delete=False,
        dir=path.parent,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def write_report(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["video", "status", "confidence", "low_confidence", "caption", "reason"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_rows(
    rows: list[dict[str, str]],
    *,
    metadata_path: Path,
    fieldnames: list[str],
    report_path: Path,
    overwrite: str,
    only_filter: set[str] | None,
    dry_run: bool,
    config: APIConfig,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    report_rows: list[dict[str, str]] = []
    updated_rows = [dict(row) for row in rows]

    for index, row in enumerate(updated_rows, start=1):
        video_name = row.get("video", "").strip()
        if not video_name:
            raise SystemExit("Every metadata row must include a video value")

        if not should_process_row(row, overwrite=overwrite, only_filter=only_filter):
            report_rows.append(
                {
                    "video": video_name,
                    "status": "skipped",
                    "confidence": "",
                    "low_confidence": "",
                    "caption": row.get("prompt", ""),
                    "reason": "Skipped by filters or overwrite mode",
                }
            )
            continue

        log(f"[{index}/{len(updated_rows)}] {video_name}")
        video_path = resolve_video_path(metadata_path, video_name)
        if not video_path.exists():
            report_rows.append(
                {
                    "video": video_name,
                    "status": "error",
                    "confidence": "",
                    "low_confidence": "",
                    "caption": row.get("prompt", ""),
                    "reason": f"Video file not found: {video_path}",
                }
            )
            continue

        try:
            features = gather_video_features(video_path)
            candidate = generate_candidate(config, video_name, features)
        except FatalAPIError:
            raise
        except Exception as exc:
            features = None
            fallback_text, fallback_scene = fallback_caption(
                VideoFeatures(
                    duration=0.0,
                    width=0,
                    height=0,
                    timestamps=[],
                    frames=[],
                    brightness_min=0.0,
                    brightness_max=0.0,
                    is_bw=False,
                    person_present=False,
                    group_present=False,
                )
            )
            candidate = CaptionCandidate(
                caption=fallback_text,
                confidence=0.0,
                entities=[],
                scene_type=fallback_scene,
                style_terms_used=[],
                reason=f"Fallback used after processing error: {exc}",
                fallback_used=True,
            )

        row["prompt"] = candidate.caption
        low_confidence = candidate.fallback_used or candidate.repaired or candidate.confidence < LOW_CONFIDENCE_THRESHOLD
        status = "fallback" if candidate.fallback_used else "generated"
        if candidate.repaired and not candidate.fallback_used:
            status = "repaired"

        report_rows.append(
            {
                "video": video_name,
                "status": status,
                "confidence": f"{candidate.confidence:.3f}",
                "low_confidence": "yes" if low_confidence else "no",
                "caption": candidate.caption,
                "reason": candidate.reason,
            }
        )

    if not dry_run:
        write_csv_atomic(metadata_path, fieldnames, updated_rows)
    write_report(report_path, report_rows)
    return updated_rows, report_rows


def main() -> int:
    args = parse_args()
    metadata_path = args.metadata.expanduser().resolve()
    report_path = metadata_path.parent / DEFAULT_REPORT_NAME
    fieldnames, rows = read_metadata(metadata_path)
    config = load_api_config(args)
    only_filter = parse_only_filter(args.only)

    try:
        _updated_rows, report_rows = process_rows(
            rows,
            metadata_path=metadata_path,
            fieldnames=fieldnames,
            report_path=report_path,
            overwrite=args.overwrite,
            only_filter=only_filter,
            dry_run=args.dry_run,
            config=config,
        )
    except FatalAPIError as exc:
        raise SystemExit(str(exc)) from exc

    generated = sum(1 for row in report_rows if row["status"] == "generated")
    repaired = sum(1 for row in report_rows if row["status"] == "repaired")
    fallback = sum(1 for row in report_rows if row["status"] == "fallback")
    low_confidence = sum(1 for row in report_rows if row["low_confidence"] == "yes")

    log("")
    log(f"Report written to {report_path}")
    log(
        "Summary: "
        f"generated={generated}, repaired={repaired}, fallback={fallback}, "
        f"low_confidence={low_confidence}, total={len(report_rows)}"
    )
    if args.dry_run:
        log("Dry run: metadata.csv was not modified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
