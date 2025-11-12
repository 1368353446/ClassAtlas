from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import fitz
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

import logging

logger = logging.getLogger(__name__)

def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class _ImageEmbedder:
    def __init__(self):
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = torch.nn.Sequential(*list(backbone.children())[:-1]).to(_device()).eval()
        self.transform = transforms.Compose(  # type: ignore[attr-defined]
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def embed(self, image: Image.Image) -> torch.Tensor:
        tensor = self.transform(image).unsqueeze(0).to(_device())
        feat = self.encoder(tensor).flatten(1)
        return feat / (feat.norm(dim=1, keepdim=True) + 1e-9)


def render_pdf_pages(pdf_path: Path, slides_dir: Path) -> List[Path]:
    if not pdf_path or not pdf_path.exists():
        return []
    doc = fitz.open(pdf_path)
    slides_dir.mkdir(parents=True, exist_ok=True)
    page_paths: List[Path] = []
    for idx, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=150)
        image_path = slides_dir / f"slide_{idx:03d}.png"
        pix.save(str(image_path))
        page_paths.append(image_path)
    return page_paths


def sample_video_frames(
    video_path: Path, embedder: _ImageEmbedder, interval_seconds: float = 2.0
) -> Tuple[List[dict], float]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps if total_frames > 0 else 0.0
    frames: List[dict] = []
    if total_duration == 0.0:
        cap.release()
        return frames, total_duration
    next_capture_time = 0.0
    while next_capture_time <= total_duration + 0.5:
        cap.set(cv2.CAP_PROP_POS_MSEC, next_capture_time * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
        feature = embedder.embed(pil_img)
        frames.append({"timestamp": float(next_capture_time), "feature": feature})
        next_capture_time += interval_seconds
    cap.release()
    if not frames and total_duration > 0:
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        if ret:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
            feature = embedder.embed(pil_img)
            frames.append({"timestamp": 0.0, "feature": feature})
        cap.release()
    return frames, total_duration


def assign_pages_to_frames(page_features: List[torch.Tensor], frame_features: List[dict]) -> List[float]:
    if not page_features or not frame_features:
        return [0.0 for _ in page_features]
    assignments: List[float] = []
    frame_idx = 0
    for feat in page_features:
        candidates = frame_features[frame_idx:]
        if not candidates:
            assignments.append(frame_features[-1]["timestamp"])
            continue
        sims = [F.cosine_similarity(feat, cand["feature"]).item() for cand in candidates]
        best_rel_idx = int(max(range(len(sims)), key=lambda i: sims[i]))
        best_idx = frame_idx + best_rel_idx
        best_time = frame_features[best_idx]["timestamp"]
        assignments.append(best_time)
        frame_idx = best_idx
    return assignments


def build_slide_manifest(
    pdf_path: Path | None,
    video_path: Path,
    slides_dir: Path,
    *,
    interval_seconds: float = 2.0,
) -> List[dict]:
    if pdf_path is None or not pdf_path.exists():
        return []
    embedder = _ImageEmbedder()
    page_paths = render_pdf_pages(pdf_path, slides_dir)
    if not page_paths:
        return []
    page_features: List[torch.Tensor] = []
    for path in page_paths:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            page_features.append(embedder.embed(rgb))
    frame_features, total_duration = sample_video_frames(video_path, embedder, interval_seconds=interval_seconds)
    if not frame_features:
        frame_features = [{"timestamp": 0.0, "feature": page_features[0]}]
        total_duration = max(total_duration, interval_seconds * len(page_paths))
    assignments = assign_pages_to_frames(page_features, frame_features)
    manifest: List[dict] = []
    for page_index, assigned_time in enumerate(assignments, start=1):
        if page_index < len(assignments):
            next_time = assignments[page_index]
        else:
            next_time = total_duration if total_duration > 0 else assigned_time + interval_seconds
        end_time = max(next_time, assigned_time + 0.5)
        manifest.append(
            {
                "slide_index": page_index,
                "start_time": float(assigned_time),
                "end_time": float(end_time),
                "image_path": str(page_paths[page_index - 1]),
            }
        )
    logger.info(
        "Slide manifest built | pages=%d | video=%.1fs | interval=%.1fs",
        len(manifest),
        total_duration,
        interval_seconds,
    )
    return manifest
