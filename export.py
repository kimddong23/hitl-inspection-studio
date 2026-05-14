"""라벨 export — YOLO + COCO format. zip stream 으로 다운로드.

옵션:
- Format: YOLO / COCO / both
- 범위: 정정된 case 만 / 전체 (inspected_at not NULL)
- 모델: classifier / segmenter / both
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

from PIL import Image

from db import get_labels_by_session


# ============================================================
# YOLO Segmentation export
# ============================================================
def to_yolo_segment_lines(human_polygons: list, image_w: int, image_h: int) -> list[str]:
    """human_polygons 리스트 → YOLO segment .txt 라인 리스트.

    human_polygons = [{"class_id": int, "polygon": [[x,y], ...] (pixel 좌표)}, ...]
    """
    lines = []
    if not human_polygons:
        return lines
    for p in human_polygons:
        cid = int(p.get("class_id", 0))
        poly = p.get("polygon", [])
        if len(poly) < 3:
            continue
        coords = []
        for x, y in poly:
            nx = max(0.0, min(1.0, x / image_w))
            ny = max(0.0, min(1.0, y / image_h))
            coords.append(f"{nx:.6f}")
            coords.append(f"{ny:.6f}")
        lines.append(f"{cid} " + " ".join(coords))
    return lines


def export_yolo_segment(labels: list[dict], images_by_filename: dict[str, Image.Image],
                        sample_weight_threshold: float = 0.0) -> dict[str, bytes]:
    """YOLO segment dataset zip 파일들.

    returns: dict {path_in_zip: bytes}
    - images/<filename>.jpg
    - labels/<filename>.txt
    - data.yaml
    - weights.csv (sample weight per image, for weighted training)
    """
    files: dict[str, bytes] = {}
    class_id_to_name: dict[int, str] = {}
    weight_csv = ["filename,sample_weight"]

    for L in labels:
        fn = L["image_filename"]
        img = images_by_filename.get(fn)
        if img is None:
            continue
        # human polygons 우선, 없으면 model polygons
        polys = L.get("seg_human_polygons") or L.get("seg_model_polygons") or []
        names_dict = L.get("seg_names_dict") or {}

        # human 라벨이 None 이고 model 결과만 있으면 그대로 export (model 정답 가정)
        # human 정정한 case 만 원하면 seg_verdict 체크
        for p in polys:
            cid = int(p.get("class_id", 0))
            if cid not in class_id_to_name:
                name = p.get("class_name") or names_dict.get(str(cid)) or names_dict.get(cid) or f"class_{cid}"
                class_id_to_name[cid] = name

        # 이미지 저장
        stem = Path(fn).stem
        ext = Path(fn).suffix.lower() or ".jpg"
        if ext not in (".jpg", ".jpeg", ".png"):
            ext = ".jpg"
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG" if ext in (".jpg", ".jpeg") else "PNG")
        files[f"images/{stem}{ext}"] = img_bytes.getvalue()

        # 라벨 저장
        lines = to_yolo_segment_lines(polys, L["image_w"], L["image_h"])
        files[f"labels/{stem}.txt"] = ("\n".join(lines)).encode("utf-8")

        # sample weight
        weight = float(L.get("seg_correction_weight", 1.0) or 1.0)
        weight_csv.append(f"{stem}{ext},{weight}")

    # data.yaml
    sorted_classes = sorted(class_id_to_name.items())
    names_list = [n for _, n in sorted_classes]
    yaml_content = (
        f"# YOLO segmentation export (HITL prototype) — {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"path: .\n"
        f"train: images\n"
        f"val: images\n"
        f"nc: {len(names_list)}\n"
        f"names: {names_list}\n"
    )
    files["data.yaml"] = yaml_content.encode("utf-8")
    files["weights.csv"] = ("\n".join(weight_csv)).encode("utf-8")
    return files


# ============================================================
# YOLO Classification export (folder per class)
# ============================================================
def export_yolo_classify(labels: list[dict], images_by_filename: dict[str, Image.Image]) -> dict[str, bytes]:
    """YOLO classification dataset (folder structure).

    train/<class_name>/<filename>.jpg
    """
    files: dict[str, bytes] = {}
    weight_csv = ["filename,class,sample_weight"]
    classes_seen = set()
    for L in labels:
        fn = L["image_filename"]
        img = images_by_filename.get(fn)
        if img is None:
            continue
        # human 라벨 우선
        label = L.get("cls_human_label") or L.get("cls_model_top")
        if not label:
            continue
        classes_seen.add(label)
        stem = Path(fn).stem
        ext = Path(fn).suffix.lower() or ".jpg"
        if ext not in (".jpg", ".jpeg", ".png"):
            ext = ".jpg"
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG" if ext in (".jpg", ".jpeg") else "PNG")
        files[f"train/{label}/{stem}{ext}"] = img_bytes.getvalue()
        weight = float(L.get("cls_correction_weight", 1.0) or 1.0)
        weight_csv.append(f"{stem}{ext},{label},{weight}")

    # placeholder val/ — 사용자가 split 결정
    for cls in classes_seen:
        files[f"val/{cls}/.keep"] = b""

    readme = (
        f"# YOLO Classification export (HITL prototype) — {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"# Structure:\n#   train/<class>/<image>\n#   val/<class>/<image>  (사용자가 split)\n"
        f"# Classes: {sorted(classes_seen)}\n"
    )
    files["README.md"] = readme.encode("utf-8")
    files["weights.csv"] = ("\n".join(weight_csv)).encode("utf-8")
    return files


# ============================================================
# COCO export
# ============================================================
def export_coco(labels: list[dict], images_by_filename: dict[str, Image.Image]) -> dict[str, bytes]:
    """COCO format single JSON (+ images)."""
    files: dict[str, bytes] = {}
    coco = {
        "info": {
            "description": "HITL prototype export",
            "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }
    cat_map: dict[int, str] = {}
    ann_id = 1

    for img_id, L in enumerate(labels, 1):
        fn = L["image_filename"]
        img = images_by_filename.get(fn)
        if img is None:
            continue
        stem = Path(fn).stem
        ext = Path(fn).suffix.lower() or ".jpg"
        if ext not in (".jpg", ".jpeg", ".png"):
            ext = ".jpg"

        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG" if ext in (".jpg", ".jpeg") else "PNG")
        files[f"images/{stem}{ext}"] = img_bytes.getvalue()

        coco["images"].append({
            "id": img_id,
            "file_name": f"{stem}{ext}",
            "width": L["image_w"],
            "height": L["image_h"],
            "human_label": L.get("cls_human_label"),
            "model_label": L.get("cls_model_top"),
            "cls_correction_weight": float(L.get("cls_correction_weight") or 1.0),
            "seg_correction_weight": float(L.get("seg_correction_weight") or 1.0),
        })

        polys = L.get("seg_human_polygons") or L.get("seg_model_polygons") or []
        names_dict = L.get("seg_names_dict") or {}
        for p in polys:
            cid = int(p.get("class_id", 0))
            if cid not in cat_map:
                name = p.get("class_name") or names_dict.get(str(cid)) or names_dict.get(cid) or f"class_{cid}"
                cat_map[cid] = name
            poly_pts = p.get("polygon", [])
            if len(poly_pts) < 3:
                continue
            seg_flat = [coord for pt in poly_pts for coord in pt]
            xs = [pt[0] for pt in poly_pts]
            ys = [pt[1] for pt in poly_pts]
            x_min, y_min = min(xs), min(ys)
            x_max, y_max = max(xs), max(ys)
            w, h = x_max - x_min, y_max - y_min
            # shoelace area
            n = len(poly_pts)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += poly_pts[i][0] * poly_pts[j][1] - poly_pts[j][0] * poly_pts[i][1]
            area = abs(area) / 2

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cid,
                "segmentation": [seg_flat],
                "bbox": [x_min, y_min, w, h],
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1

    for cid, name in sorted(cat_map.items()):
        coco["categories"].append({"id": cid, "name": name, "supercategory": "defect"})

    files["annotations.json"] = json.dumps(coco, ensure_ascii=False, indent=2).encode("utf-8")
    return files


# ============================================================
# Zip 빌더
# ============================================================
def build_export_zip(
    session_id: int,
    images_by_filename: dict[str, Image.Image],
    formats: list[str],          # ["yolo_seg", "yolo_cls", "coco"]
    only_inspected: bool = False,
    only_wrong: bool = False,
) -> bytes:
    """선택된 format 으로 zip 만들어 bytes 반환."""
    all_labels = get_labels_by_session(session_id)
    labels = []
    for L in all_labels:
        if only_inspected and not L.get("inspected_at"):
            continue
        if only_wrong and L.get("cls_verdict") != "wrong" and L.get("seg_verdict") != "wrong":
            continue
        labels.append(L)

    out_buf = io.BytesIO()
    with zipfile.ZipFile(out_buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        if "yolo_seg" in formats:
            files = export_yolo_segment(labels, images_by_filename)
            for path, data in files.items():
                zf.writestr(f"yolo_segment/{path}", data)
        if "yolo_cls" in formats:
            files = export_yolo_classify(labels, images_by_filename)
            for path, data in files.items():
                zf.writestr(f"yolo_classify/{path}", data)
        if "coco" in formats:
            files = export_coco(labels, images_by_filename)
            for path, data in files.items():
                zf.writestr(f"coco/{path}", data)
        # 메타 파일
        meta = {
            "session_id": session_id,
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_labels": len(labels),
            "formats": formats,
            "only_inspected": only_inspected,
            "only_wrong": only_wrong,
        }
        zf.writestr("export_meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
    return out_buf.getvalue()
