"""Streamlit Custom Component — Polygon Editor (Konva.js).

전체 polygon vertex 편집을 지원하는 streamlit component.
- vertex drag / add / delete
- 새 polygon / rect 그리기
- polygon 이동, 선택, 삭제
- 클래스 변경
- zoom / pan / undo / redo
"""

from __future__ import annotations

import os
from typing import Any

import streamlit.components.v1 as components

# 개발 중에는 False, frontend/dist 빌드 완료 후 True 로 전환
_RELEASE = True

if not _RELEASE:
    _component_func = components.declare_component(
        "polygon_editor",
        url="http://localhost:3001",  # Vite dev server
    )
else:
    _parent_dir = os.path.dirname(os.path.abspath(__file__))
    _build_dir = os.path.join(_parent_dir, "frontend", "dist")
    _component_func = components.declare_component(
        "polygon_editor",
        path=_build_dir,
    )


def polygon_editor(
    image_b64: str,
    polygons: list[dict[str, Any]],
    class_names: list[str],
    image_w: int,
    image_h: int,
    max_width: int = 1100,
    key: str | None = None,
    theme: str = "dark",
) -> dict[str, Any]:
    """Konva-based polygon editor.

    Args:
        image_b64: data URI (e.g. "data:image/jpeg;base64,...")
        polygons: list of {class_id, class_name, polygon: [[x,y],...], conf?, source?}
        class_names: list of available class names (인덱스 = class_id)
        image_w, image_h: 원본 이미지 해상도
        max_width: 컴포넌트가 가져갈 최대 폭 (이미지 좌표 그대로 반환)
        key: streamlit component key
        theme: "dark" | "light" — frontend root data-theme 속성으로 전달

    Returns:
        {"polygons": [{class_id, class_name, polygon, source, conf?}, ...]}
        값은 원본 이미지 좌표 기준.
    """
    default = {"polygons": polygons or []}
    return _component_func(
        image_b64=image_b64,
        polygons=polygons or [],
        class_names=class_names or [],
        image_w=int(image_w),
        image_h=int(image_h),
        max_width=int(max_width),
        theme=str(theme or "dark").lower(),
        key=key,
        default=default,
    )
