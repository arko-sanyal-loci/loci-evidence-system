"""
loci_shader_map.py — Multi-X-axis person imagery → 5-channel SymPrev shader map.

What it does
────────────
Takes N images of a person at different horizontal rotation angles
(X-axis sweep: 0°, 45°, 90°, 135°, 180°) and produces a per-stream
semantic shader map: a compact texture encoding which SymPrev streams
each spatial region of the person activates.

Shader map channels (5):
  0: ISymPrev — narrative identity features (face, posture, expression cues)
  1: QSymPrev — factual/versioned features (build, height, distinguishing marks)
  2: DSymPrev — counterfactual uncertainty (occluded regions, missing angles)
  3: MSymPrev — affective mood features (micro-expressions, proxemics)
  4: SSymPrev — simulation/motion anchors (gait, gesture, motion blur)

Spatial layout (H × W):
  Top 1/3     — face/head  → ISymPrev + MSymPrev
  Middle 1/3  — torso      → ISymPrev + QSymPrev
  Bottom 1/3  — lower body → SSymPrev
  Everywhere  — DSymPrev edges (occluded boundary Gaussian)

Integration with ChatGPT X-axis preset engine
──────────────────────────────────────────────
The ChatGPT custom GPT extracts structured PersonFeatures from multi-angle
imagery. This module consumes that JSON output via from_chatgpt_preset().
Raw image paths are also supported via from_images() (requires CLIP/ViT).

Garena bus integration
──────────────────────
ShaderMaps are serialised as loci.shader_map.v1 JSON and published to the
SYMPREV_RING on the Garena coordination bus (ws://127.0.0.1:8765).

Usage
─────
    from loci_shader_map import ShaderMapGenerator

    gen = ShaderMapGenerator()

    # From ChatGPT preset output (no vision model required)
    shader = gen.from_chatgpt_preset(preset_json_dict)
    shader.dominant_stream_at(x=0.5, y=0.1)   # → "MSymPrev" (face region)

    # From raw image files (requires torch + transformers + PIL)
    shader = gen.from_images(["0deg.jpg", "45deg.jpg", "90deg.jpg"])

    # Publish to Garena bus
    shader.publish_to_bus()
"""

from __future__ import annotations

import base64
import json
import math
import struct
import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("loci.shader_map")

DEFAULT_WIDTH  = 64
DEFAULT_HEIGHT = 36
N_STREAMS      = 5
STREAM_NAMES   = ["ISymPrev", "QSymPrev", "DSymPrev", "MSymPrev", "SSymPrev"]


# ── ChatGPT X-axis preset output schema ──────────────────────────────────────

@dataclass
class PersonFeatures:
    """
    Structured person features extracted by the ChatGPT X-axis preset engine.

    JSON keys mirror what the ChatGPT custom GPT emits after processing
    multi-angle (X-axis rotation) person imagery.  Each field maps to one
    or more SymPrev stream inputs for the shader map computation.
    """
    angles_captured:       list[float]        = field(default_factory=list)

    # ISymPrev inputs — narrative & identity
    narrative_description: str                = ""
    identity_tags:         list[str]          = field(default_factory=list)

    # QSymPrev inputs — versioned facts
    height_estimate_m:     Optional[float]    = None
    build:                 str                = ""
    distinguishing:        list[str]          = field(default_factory=list)

    # DSymPrev inputs — counterfactual / occlusion
    angle_coverage_pct:    float              = 0.0   # 0–1
    occluded_regions:      list[str]          = field(default_factory=list)

    # MSymPrev inputs — affective
    expression:            str                = ""
    posture_energy:        float              = 0.5   # 0=closed → 1=open
    gaze_direction:        str                = ""

    # SSymPrev inputs — simulation / motion
    detected_gesture:      str                = ""
    motion_blur_score:     float              = 0.0   # 0=still → 1=high motion

    @classmethod
    def from_chatgpt_json(cls, data: dict) -> "PersonFeatures":
        f = cls()
        f.angles_captured         = data.get("angles_captured", [])
        f.narrative_description   = data.get("narrative_description", "")
        f.identity_tags           = data.get("identity_tags", [])
        f.height_estimate_m       = data.get("height_estimate_m")
        f.build                   = data.get("build", "")
        f.distinguishing          = data.get("distinguishing", [])
        f.angle_coverage_pct      = float(data.get("angle_coverage_pct", 0))
        f.occluded_regions        = data.get("occluded_regions", [])
        f.expression              = data.get("expression", "")
        f.posture_energy          = float(data.get("posture_energy", 0.5))
        f.gaze_direction          = data.get("gaze_direction", "")
        f.detected_gesture        = data.get("detected_gesture", "")
        f.motion_blur_score       = float(data.get("motion_blur_score", 0.0))
        return f


# ── Shader map ────────────────────────────────────────────────────────────────

class ShaderMap:
    """
    5-channel SymPrev semantic shader map for a person.

    Internal: flat list of float32, shape (H, W, 5), row-major, channel-last.
    Channel order: ISymPrev=0, QSymPrev=1, DSymPrev=2, MSymPrev=3, SSymPrev=4.
    All values ∈ [0, 1].
    """

    def __init__(self, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT) -> None:
        self.width  = width
        self.height = height
        self._data  = [0.0] * (width * height * N_STREAMS)

    def _idx(self, x: int, y: int, ch: int) -> int:
        return (y * self.width + x) * N_STREAMS + ch

    def set_pixel(self, x: int, y: int, channel: int, value: float) -> None:
        self._data[self._idx(x, y, channel)] = max(0.0, min(1.0, value))

    def get_pixel(self, x: int, y: int, channel: int) -> float:
        return self._data[self._idx(x, y, channel)]

    def stream_activation(self, stream: str) -> list[list[float]]:
        """Return (H×W) 2D grid of activation values for one named stream."""
        ch = STREAM_NAMES.index(stream)
        return [[self.get_pixel(col, row, ch) for col in range(self.width)]
                for row in range(self.height)]

    def dominant_stream_at(self, x_norm: float, y_norm: float) -> str:
        """Named stream with highest activation at normalised coordinates."""
        x    = int(x_norm * (self.width  - 1))
        y    = int(y_norm * (self.height - 1))
        vals = [self.get_pixel(x, y, ch) for ch in range(N_STREAMS)]
        return STREAM_NAMES[vals.index(max(vals))]

    def to_dict(self) -> dict:
        buf = struct.pack(f"{len(self._data)}f", *self._data)
        return {
            "schema":   "loci.shader_map.v1",
            "width":    self.width,
            "height":   self.height,
            "channels": STREAM_NAMES,
            "data_b64": base64.b64encode(buf).decode(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ShaderMap":
        sm  = cls(d["width"], d["height"])
        buf = base64.b64decode(d["data_b64"])
        sm._data = list(struct.unpack(f"{len(sm._data)}f", buf))
        return sm

    def blend(self, other: "ShaderMap", alpha: float = 0.5) -> "ShaderMap":
        """Linear blend: α·self + (1−α)·other."""
        out = ShaderMap(self.width, self.height)
        for i, (a, b) in enumerate(zip(self._data, other._data)):
            out._data[i] = alpha * a + (1 - alpha) * b
        return out

    def publish_to_bus(self, entity_key: str = "person", ws_port: int = 8765) -> bool:
        """Publish this shader map to the Garena bus SYMPREV_RING."""
        try:
            import socket
            payload = json.dumps({
                "kind":       "symprev",
                "sub_kind":   "shader_map",
                "entity_key": entity_key,
                "shader":     self.to_dict(),
            }).encode()
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.sendto(payload, ("127.0.0.1", ws_port + 1))
            return True
        except Exception as exc:
            log.debug("shader_map bus publish failed: %s", exc)
            return False


# ── Helper: edge Gaussian ─────────────────────────────────────────────────────

def _edge_gaussian(x: int, y: int, w: int, h: int, sigma: float = 0.25) -> float:
    """
    Gaussian that peaks at the boundary and falls off toward centre.
    Used for DSymPrev (counterfactual uncertainty) at occlusion edges.
    """
    cx = w / 2.0
    cy = h / 2.0
    dx = abs(x - cx) / cx if cx > 0 else 0.0
    dy = abs(y - cy) / cy if cy > 0 else 0.0
    d  = max(dx, dy)
    return math.exp(-(1.0 - d) ** 2 / (2 * sigma ** 2))


# ── Generator ─────────────────────────────────────────────────────────────────

class ShaderMapGenerator:
    """
    Builds a 5-channel SymPrev shader map from multi-angle person imagery.

    Two modes:
      from_chatgpt_preset(json) — structured ChatGPT X-axis extraction output;
                                   no vision model, works fully offline.
      from_images(paths)        — raw image files; uses local CLIP ViT-B/32
                                   (requires torch + transformers + PIL).
    """

    def __init__(
        self,
        width:  int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
    ) -> None:
        self.width  = width
        self.height = height

    def from_chatgpt_preset(self, preset_data: "dict | str") -> ShaderMap:
        """
        Build shader map from the ChatGPT X-axis preset output.

        preset_data: dict or JSON string from the ChatGPT custom GPT that
                     processed multiple X-axis (yaw-rotation) person images.
        """
        if isinstance(preset_data, str):
            preset_data = json.loads(preset_data)
        features = PersonFeatures.from_chatgpt_json(preset_data)
        return self._features_to_shader(features)

    def _features_to_shader(self, f: PersonFeatures) -> ShaderMap:
        sm = ShaderMap(self.width, self.height)

        face_top  = 0
        face_bot  = self.height // 3
        torso_top = self.height // 3
        torso_bot = 2 * self.height // 3
        lower_top = 2 * self.height // 3
        lower_bot = self.height

        # ── ISymPrev: narrative identity — face + torso ───────────────────────
        isym = min(1.0, len(f.identity_tags) * 0.15 + 0.3
                   + (0.2 if f.narrative_description else 0.0))
        self._fill(sm, 0, face_top,  self.width, face_bot,  0, isym)
        self._fill(sm, 0, torso_top, self.width, torso_bot, 0, isym * 0.6)

        # ── QSymPrev: versioned facts — torso (build/height) ─────────────────
        qsym = 0.4 + (0.2 if f.height_estimate_m else 0.0) + (0.2 if f.build else 0.0)
        self._fill(sm, 0, torso_top, self.width, torso_bot, 1, min(1.0, qsym))

        # ── DSymPrev: counterfactual — edges + occluded regions ───────────────
        base_uncertainty = 1.0 - f.angle_coverage_pct
        for x in range(self.width):
            for y in range(self.height):
                edge = _edge_gaussian(x, y, self.width, self.height, sigma=0.25)
                sm.set_pixel(x, y, 2, min(1.0, base_uncertainty * 0.5 + edge * 0.5))

        # ── MSymPrev: affective — face region (expression + posture) ─────────
        EXPRESSION_HEAT = {
            "smiling": 0.9, "laughing": 0.95, "neutral": 0.5,
            "tense":   0.7, "sad":      0.65, "surprised": 0.8, "angry": 0.75,
        }
        msym_face = EXPRESSION_HEAT.get(f.expression.lower(), 0.5)
        msym      = (msym_face + f.posture_energy) / 2.0
        self._fill(sm, 0, face_top,  self.width, face_bot,  3, msym)
        self._fill(sm, 0, torso_top, self.width, torso_bot, 3, f.posture_energy * 0.4)

        # ── SSymPrev: simulation/motion — lower body ──────────────────────────
        ssym = min(1.0, 0.3 + f.motion_blur_score * 0.4
                   + (0.3 if f.detected_gesture else 0.0))
        self._fill(sm, 0, lower_top, self.width, lower_bot, 4, ssym)

        return sm

    def _fill(
        self,
        sm: ShaderMap,
        x0: int, y0: int, x1: int, y1: int,
        channel: int,
        value: float,
    ) -> None:
        for y in range(y0, y1):
            for x in range(x0, x1):
                sm.set_pixel(x, y, channel, value)

    def from_images(self, image_paths: list[str]) -> ShaderMap:
        """
        Build shader map from raw image files (multi-X-axis angles).
        Requires torch + transformers + PIL; falls back to neutral map.
        """
        try:
            return self._from_images_clip(image_paths)
        except Exception as exc:
            log.warning("CLIP encoding failed (%s) — returning neutral shader map", exc)
            sm = ShaderMap(self.width, self.height)
            for y in range(self.height):
                for x in range(self.width):
                    for ch in range(N_STREAMS):
                        sm.set_pixel(x, y, ch, 0.5)
            return sm

    def _from_images_clip(self, image_paths: list[str]) -> ShaderMap:
        import torch
        from PIL import Image
        from transformers import CLIPProcessor, CLIPModel  # type: ignore

        model_name = "openai/clip-vit-base-patch32"
        model      = CLIPModel.from_pretrained(model_name)
        processor  = CLIPProcessor.from_pretrained(model_name)
        model.eval()

        # One descriptive prompt per SymPrev stream
        stream_prompts = {
            "ISymPrev": "a person with distinct identity features",
            "QSymPrev": "factual physical description of a person",
            "DSymPrev": "partially visible or occluded person",
            "MSymPrev": "facial expression and emotional mood of a person",
            "SSymPrev": "person in motion, gesturing or walking",
        }

        # Average image features across all captured angles
        angle_features = []
        for path in image_paths:
            img    = Image.open(path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                feats = model.get_image_features(**inputs)
                angle_features.append(feats / feats.norm(dim=-1, keepdim=True))

        avg_img = torch.stack(angle_features).mean(dim=0)

        text_inputs = processor(
            text=list(stream_prompts.values()),
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            text_feats = model.get_text_features(**text_inputs)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        sims = (avg_img @ text_feats.T).squeeze(0).tolist()   # 5 cosines ∈ [-1, 1]
        activations = {s: (c + 1) / 2.0 for s, c in zip(stream_prompts, sims)}
        log.info("CLIP stream activations from %d images: %s", len(image_paths), activations)

        sm = ShaderMap(self.width, self.height)
        for ch, stream in enumerate(STREAM_NAMES):
            val = activations.get(stream, 0.5)
            for y in range(self.height):
                for x in range(self.width):
                    sm.set_pixel(x, y, ch, val)
        return sm
