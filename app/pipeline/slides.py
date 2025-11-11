from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def extract_slide_keyframes(
    video_path: Path,
    *,
    interval_seconds: float = 2.0,
    similarity_threshold: float = 0.95,
) -> List[dict]:
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1]).to(_device()).eval()
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps if total_frames > 0 else 0.0
    step = max(1, int(fps * interval_seconds))

    slides: List[dict] = []
    current_feature = None
    current_image = None
    current_start = 0.0
    slide_index = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            timestamp = frame_idx / fps
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
            frame_tensor = preprocess(pil_img).unsqueeze(0).to(_device())
            with torch.no_grad():
                feat = resnet(frame_tensor).flatten(1)
                feat = feat / feat.norm(dim=1, keepdim=True)
            if current_feature is None:
                slide_index = 1
                current_feature = feat
                current_image = pil_img
                current_start = timestamp
            else:
                similarity = F.cosine_similarity(feat, current_feature).item()
                if similarity < similarity_threshold:
                    slides.append(
                        {
                            "slide_index": slide_index,
                            "start_time": float(current_start),
                            "end_time": float(timestamp),
                            "image": current_image,
                        }
                    )
                    slide_index += 1
                    current_feature = feat
                    current_image = pil_img
                    current_start = timestamp
        frame_idx += 1

    cap.release()

    if current_feature is not None:
        end_time = total_duration if total_duration > current_start else current_start
        slides.append(
            {
                "slide_index": slide_index,
                "start_time": float(current_start),
                "end_time": float(end_time),
                "image": current_image,
            }
        )
    return slides


def persist_slides_and_build_pdf(
    slides: List[dict],
    slides_dir: Path,
    pdf_path: Path,
) -> Tuple[List[dict], str | None]:
    slides_dir.mkdir(parents=True, exist_ok=True)
    pdf_images: List[Image.Image] = []
    manifest: List[dict] = []
    for slide in slides:
        image: Image.Image = slide["image"]
        slide_path = slides_dir / f"slide_{slide['slide_index']:02d}.png"
        image.save(slide_path)
        manifest.append(
            {
                "slide_index": slide["slide_index"],
                "start_time": slide["start_time"],
                "end_time": slide["end_time"],
                "image_path": str(slide_path),
            }
        )
        pdf_images.append(image)

    pdf_file = None
    if pdf_images:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_images[0].save(pdf_path, save_all=True, append_images=pdf_images[1:])
        pdf_file = str(pdf_path)
    return manifest, pdf_file
