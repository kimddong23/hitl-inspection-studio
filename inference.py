"""모델 로드 + 추론 모듈 — classifier (YOLO-cls) + segmenter (YOLO-seg)."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_yolo_model(weights_path: str | Path):
    """ultralytics YOLO 로드. classifier/segmenter 모두 동일 인터페이스."""
    from ultralytics import YOLO
    return YOLO(str(weights_path))


def predict_classifier(model, img: Image.Image, device: str = "auto") -> dict:
    """Classifier 추론. {top_class, top_conf, all_probs:[(name,prob),...]}"""
    if device == "auto":
        device = detect_device()
    res = model.predict(img, verbose=False, device=device)[0]
    names = res.names
    probs = res.probs
    if probs is None:
        return {"top_class": None, "top_conf": 0.0, "all_probs": []}
    top_idx = int(probs.top1)
    top_conf = float(probs.top1conf)
    all_p = [(names[i], float(probs.data[i].cpu())) for i in range(len(names))]
    all_p.sort(key=lambda x: -x[1])
    return {"top_class": names[top_idx], "top_conf": top_conf, "all_probs": all_p}


def predict_segmenter(model, img: Image.Image, device: str = "auto",
                      conf: float = 0.25, imgsz: int = 640) -> dict:
    """Segmenter 추론. {polygons, class_ids, class_names, confs, boxes}.

    polygons: list of [(x,y), ...] in image pixel coordinates
    class_ids: list of int
    boxes: list of [x1,y1,x2,y2]
    """
    if device == "auto":
        device = detect_device()
    res = model.predict(img, verbose=False, device=device, conf=conf, imgsz=imgsz)[0]
    names = res.names

    polygons = []
    class_ids = []
    class_names = []
    confs = []
    boxes = []
    if res.masks is not None and res.boxes is not None:
        for poly, box, cls_id, c in zip(res.masks.xy, res.boxes.xyxy.cpu().numpy(),
                                          res.boxes.cls.cpu().numpy(),
                                          res.boxes.conf.cpu().numpy()):
            pts = [(float(x), float(y)) for x, y in poly.tolist()]
            polygons.append(pts)
            class_ids.append(int(cls_id))
            class_names.append(names[int(cls_id)])
            confs.append(float(c))
            boxes.append([float(x) for x in box.tolist()])
    return {
        "polygons": polygons,
        "class_ids": class_ids,
        "class_names": class_names,
        "confs": confs,
        "boxes": boxes,
        "names_dict": dict(names),
    }


def draw_segmenter_result(img: Image.Image, seg_result: dict,
                          alpha: int = 80, line_width: int = 3) -> Image.Image:
    """Segmenter 결과 polygon + bbox 오버레이."""
    out = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    colors = [
        (255, 64, 64),   # red
        (64, 200, 64),   # green
        (64, 128, 255),  # blue
        (255, 160, 64),  # orange
        (180, 64, 255),  # purple
        (64, 200, 200),  # cyan
        (255, 200, 64),  # yellow
    ]

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for i, (poly, cid, cname, conf, box) in enumerate(zip(
        seg_result["polygons"], seg_result["class_ids"],
        seg_result["class_names"], seg_result["confs"], seg_result["boxes"]
    )):
        color = colors[cid % len(colors)]
        # polygon fill (반투명) + 외곽선
        if len(poly) >= 3:
            draw.polygon(poly, fill=(*color, alpha), outline=(*color, 255))
        # bbox
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=(*color, 255), width=line_width)
        # label
        text = f"{cname} {conf:.2f}"
        bbox = draw.textbbox((x1 + 3, y1 + 3), text, font=font)
        draw.rectangle(bbox, fill=(*color, 220))
        draw.text((x1 + 3, y1 + 3), text, fill="white", font=font)

    return Image.alpha_composite(out, overlay).convert("RGB")


def image_from_uploaded(uploaded_file) -> Image.Image:
    """Streamlit file_uploader 객체에서 PIL Image."""
    data = uploaded_file.read()
    return Image.open(BytesIO(data)).convert("RGB")


def image_from_path(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")
