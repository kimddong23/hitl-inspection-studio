"""Label Studio 통합 모듈.

기능:
1. NAS 이미지를 로컬 카피 (Label Studio docker 가 접근 가능한 폴더)
2. 모델 폴리곤을 Label Studio pre-annotation JSON 으로 변환
3. Label Studio API 로 project 생성 + task import
4. Label Studio export JSON → 우리 DB 라벨로 변환

전부 read-only NAS 약속 유지 — NAS 에서 읽기만, 쓰기는 로컬에만.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Iterable

import requests


# Label Studio (pip install) 의 LOCAL_FILES_DOCUMENT_ROOT 하위 'images/' 폴더
# 즉 LS 실행 시 LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=~/Documents 설정 → 'images/...' 가 url 의 d= 값
LOCAL_IMAGES_DIR = Path.home() / "Documents" / "images"
LS_INTERNAL_IMAGES_PREFIX = "/data/local-files/?d=images"

LABEL_CONFIG_TEMPLATE = """
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <PolygonLabels name="seg" toName="image" strokeWidth="3" pointSize="medium" opacity="0.4">
    {labels_xml}
  </PolygonLabels>
  <Choices name="cls_verdict" toName="image" showInLine="true">
    <Choice value="correct"/>
    <Choice value="wrong"/>
    <Choice value="uncertain"/>
  </Choices>
  <Choices name="cls_human_label" toName="image">
    {cls_choices_xml}
  </Choices>
</View>
"""


def make_label_config(seg_class_names: Iterable[str], cls_class_names: Iterable[str]) -> str:
    """Label Studio project 의 label config XML 생성."""
    labels_xml = "\n    ".join(
        f'<Label value="{n}" background="{_color_for(i)}"/>'
        for i, n in enumerate(seg_class_names)
    )
    cls_choices_xml = "\n    ".join(f'<Choice value="{n}"/>' for n in cls_class_names)
    return LABEL_CONFIG_TEMPLATE.format(labels_xml=labels_xml, cls_choices_xml=cls_choices_xml)


def _color_for(i: int) -> str:
    palette = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6", "#1abc9c", "#f1c40f"]
    return palette[i % len(palette)]


def copy_images_to_local(images_by_filename: dict, session_id: int) -> Path:
    """NAS 이미지를 로컬 폴더로 카피 (Label Studio docker 접근용).
    NAS 원본은 절대 안 건드림 — 메모리 PIL.Image 를 새 파일로 저장.
    """
    sub_dir = LOCAL_IMAGES_DIR / f"session_{session_id}"
    sub_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for fn, img in images_by_filename.items():
        out = sub_dir / fn
        if not out.exists():
            img.save(out, "JPEG", quality=90)
        saved.append(out)
    return sub_dir


def build_predictions_for_image(seg_polygons: list[dict], image_w: int, image_h: int,
                                  seg_class_names: list[str]) -> list[dict]:
    """모델 폴리곤 → Label Studio pre-annotation prediction.
    좌표는 % (0~100).
    """
    if not seg_polygons:
        return []
    results = []
    for i, p in enumerate(seg_polygons):
        poly_pts = p.get("polygon", [])
        if len(poly_pts) < 3:
            continue
        cname = p.get("class_name", "defect")
        if cname not in seg_class_names:
            # config 에 없는 클래스는 첫 번째로 fallback
            cname = seg_class_names[0] if seg_class_names else "defect"
        # Label Studio 폴리곤 points: [[x_pct, y_pct], ...] (0~100)
        pts_pct = [[(float(x) / image_w) * 100.0, (float(y) / image_h) * 100.0]
                   for x, y in poly_pts]
        results.append({
            "id": f"poly_{i}",
            "from_name": "seg",
            "to_name": "image",
            "type": "polygonlabels",
            "value": {
                "points": pts_pct,
                "polygonlabels": [cname],
            },
            "score": float(p.get("conf", 0.0)),
            "origin": "manual",
        })
    return results


def build_task_for_image(image_local_path: str, image_filename: str,
                          predictions: list[dict],
                          cls_top: str | None = None,
                          cls_conf: float | None = None) -> dict:
    """Label Studio 단일 task 생성. local-files URL 사용."""
    rel = Path(image_local_path).relative_to(LOCAL_IMAGES_DIR)
    ls_url = f"/data/local-files/?d=images/{rel.as_posix()}"
    return {
        "data": {
            "image": ls_url,
            "filename": image_filename,
            "cls_model_top": cls_top or "",
            "cls_model_conf": float(cls_conf) if cls_conf is not None else 0.0,
        },
        "predictions": [{"result": predictions, "score": 1.0, "model_version": "v1"}] if predictions else [],
    }


# ──────────────────────────────────────────────────────────────
# Label Studio HTTP API
# ──────────────────────────────────────────────────────────────
class LSClient:
    def __init__(self, base_url: str, token: str):
        self.base = base_url.rstrip("/")
        self.token = token
        # 토큰 형식 자동 감지: JWT (eyJ...) vs Legacy
        if token.startswith("eyJ") and token.count(".") == 2:
            # JWT — refresh 시도해서 access token 발급
            try:
                r = requests.post(f"{self.base}/api/token/refresh/",
                                   json={"refresh": token}, timeout=10)
                if r.status_code == 200:
                    access = r.json().get("access", token)
                    self.headers = {"Authorization": f"Bearer {access}"}
                else:
                    # 이미 access token일 수도
                    self.headers = {"Authorization": f"Bearer {token}"}
            except Exception:
                self.headers = {"Authorization": f"Bearer {token}"}
        else:
            # Legacy personal access token
            self.headers = {"Authorization": f"Token {token}"}

    def ping(self) -> bool:
        # Label Studio 2.x — /api/projects 호출로 인증 확인
        try:
            r = requests.get(f"{self.base}/api/projects", headers=self.headers, timeout=5,
                              params={"page_size": 1})
            return r.status_code == 200
        except Exception:
            return False

    def create_project(self, title: str, label_config: str, description: str = "") -> dict:
        r = requests.post(
            f"{self.base}/api/projects",
            json={"title": title, "description": description, "label_config": label_config},
            headers=self.headers, timeout=15,
        )
        r.raise_for_status()
        return r.json()

    def import_tasks(self, project_id: int, tasks: list[dict]) -> dict:
        r = requests.post(
            f"{self.base}/api/projects/{project_id}/import",
            json=tasks,
            headers={**self.headers, "Content-Type": "application/json"},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()

    def export_annotations(self, project_id: int, fmt: str = "JSON") -> list[dict]:
        r = requests.get(
            f"{self.base}/api/projects/{project_id}/export?exportType={fmt}",
            headers=self.headers, timeout=60,
        )
        r.raise_for_status()
        return r.json()

    def list_projects(self) -> list[dict]:
        r = requests.get(f"{self.base}/api/projects", headers=self.headers, timeout=10)
        r.raise_for_status()
        return r.json().get("results", []) if isinstance(r.json(), dict) else r.json()


def annotation_to_label_polygons(annotation_result: list[dict],
                                  image_w: int, image_h: int) -> list[dict]:
    """Label Studio annotation result → 우리 polygon dict 포맷."""
    polys = []
    for r in annotation_result:
        if r.get("type") != "polygonlabels":
            continue
        val = r.get("value", {})
        pts_pct = val.get("points", [])
        labels = val.get("polygonlabels", [])
        cname = labels[0] if labels else "defect"
        pts_abs = [[x_pct / 100.0 * image_w, y_pct / 100.0 * image_h]
                   for x_pct, y_pct in pts_pct]
        polys.append({
            "class_id": 0,  # 호출자가 클래스 매핑 후 채움
            "class_name": cname,
            "polygon": pts_abs,
            "conf": 1.0,
            "source": "labelstudio_human",
        })
    return polys


def parse_export(export_json: list[dict],
                  seg_class_id_map: dict[str, int]) -> dict[str, dict]:
    """Label Studio export JSON → filename → {cls_verdict, cls_human_label, seg_polys} dict."""
    out = {}
    for item in export_json:
        data = item.get("data", {})
        fn = data.get("filename") or Path(data.get("image", "")).name
        image_w = item.get("image_width") or 0
        image_h = item.get("image_height") or 0
        # 안전: 첫 annotation 만 사용 (multi-annotator 미사용 PoC)
        anns = item.get("annotations") or []
        if not anns:
            continue
        ann = anns[0]
        results = ann.get("result", [])
        cls_verdict = None
        cls_human_label = None
        seg_polys = []
        for r in results:
            ftype = r.get("type")
            from_name = r.get("from_name")
            val = r.get("value", {})
            if ftype == "choices" and from_name == "cls_verdict":
                cs = val.get("choices") or []
                if cs:
                    cls_verdict = cs[0]
            elif ftype == "choices" and from_name == "cls_human_label":
                cs = val.get("choices") or []
                if cs:
                    cls_human_label = cs[0]
            elif ftype == "polygonlabels":
                pts_pct = val.get("points") or []
                labels = val.get("polygonlabels") or []
                cname = labels[0] if labels else "defect"
                if image_w and image_h and len(pts_pct) >= 3:
                    pts_abs = [[x / 100.0 * image_w, y / 100.0 * image_h]
                               for x, y in pts_pct]
                    seg_polys.append({
                        "class_id": int(seg_class_id_map.get(cname, 0)),
                        "class_name": cname,
                        "polygon": pts_abs,
                        "conf": 1.0,
                        "source": "labelstudio_human",
                    })
        out[fn] = {
            "cls_verdict": cls_verdict,
            "cls_human_label": cls_human_label,
            "seg_polygons": seg_polys,
        }
    return out
