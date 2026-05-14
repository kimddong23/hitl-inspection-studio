"""검수 UI 컴포넌트 (v2) — 모드별 분리 + 고품질 UX."""
from __future__ import annotations

import base64
import io
import json
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from polygon_editor import polygon_editor


COLORS_RGB = [
    (231, 76, 60),    # red
    (46, 204, 113),   # green
    (52, 152, 219),   # blue
    (243, 156, 18),   # orange
    (155, 89, 182),   # purple
    (26, 188, 156),   # teal
    (241, 196, 15),   # yellow
]


def _bi_icon(name: str, color: str = "808080", size: int = 16) -> str:
    """Bootstrap Icon (Iconify CDN) — Streamlit markdown 안에서 인라인 사용."""
    return (
        f'<img src="https://api.iconify.design/bi/{name}.svg?color=%23{color}" '
        f'width="{size}" height="{size}" style="vertical-align:-3px;" alt="{name}"/>'
    )


_SVG_CHECK = ('<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" viewBox="0 0 16 16">'
              '<path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0zm-3.97-3.03a.75.75 0 0 0-1.08.022L7.477 9.417'
              ' 5.384 7.323a.75.75 0 0 0-1.06 1.06L6.97 11.03a.75.75 0 0 0 1.079-.02l3.992-4.99a.75.75'
              ' 0 0 0-.01-1.05z"/></svg>')
_SVG_X = ('<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" viewBox="0 0 16 16">'
          '<path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0zM5.354 4.646a.5.5 0 1 0-.708.708L7.293 8'
          ' 4.646 10.646a.5.5 0 0 0 .708.708L8 8.707l2.646 2.647a.5.5 0 0 0 .708-.708L8.707 8l2.647'
          '-2.646a.5.5 0 0 0-.708-.708L8 7.293 5.354 4.646z"/></svg>')
_SVG_DASH = ('<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" viewBox="0 0 16 16">'
             '<path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0zM4 7.5a.5.5 0 0 0 0 1h8a.5.5 0 0 0 0-1H4z"/></svg>')
_SVG_CIRCLE = ('<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" viewBox="0 0 16 16">'
               '<path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z"/></svg>')


def status_badge(label: dict | None) -> str:
    """검수 상태 배지 — Bootstrap Icons SVG inline (CDN 의존 X, 안정)."""
    if not label or not label.get("inspected_at"):
        svg, text, color, bg = _SVG_CIRCLE, "미검수", "#9aa4b2", "rgba(128,128,128,0.15)"
    else:
        cv = label.get("cls_verdict")
        sv = label.get("seg_verdict")
        if cv == "wrong" or sv == "wrong":
            svg, text, color, bg = _SVG_X, "정정 필요", "#d1242f", "rgba(209,36,47,0.15)"
        elif cv == "uncertain" or sv == "uncertain":
            svg, text, color, bg = _SVG_DASH, "애매", "#d68800", "rgba(214,136,0,0.15)"
        else:
            svg, text, color, bg = _SVG_CHECK, "정상", "#2da44e", "rgba(45,164,78,0.15)"
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;'
        f'background:{bg};color:{color};font-weight:600;padding:4px 12px;'
        f'border-radius:14px;font-size:0.9em;">'
        f'{svg}{text}</span>'
    )


def status_text(label: dict | None) -> str:
    if not label or not label.get("inspected_at"):
        return "○ 미검수"
    cv = label.get("cls_verdict")
    sv = label.get("seg_verdict")
    if cv == "wrong" or sv == "wrong":
        return "✗ 정정 필요"
    if cv == "uncertain" or sv == "uncertain":
        return "△ 애매"
    return "✓ 정상"


# ============================================================
# 큰 verdict 버튼 (컴팩트 라디오 대체)
# ============================================================
def verdict_buttons(key_prefix: str, saved_verdict: str | None,
                    auto_default: str = "correct") -> str:
    """3개 컬러 버튼으로 verdict 빠른 선택. 큰 버튼 + 색상 강조."""
    current = saved_verdict or auto_default
    if f"{key_prefix}_verdict_state" not in st.session_state:
        st.session_state[f"{key_prefix}_verdict_state"] = current

    selected = st.session_state[f"{key_prefix}_verdict_state"]
    cols = st.columns(3)
    # 의미별 표기 — 정확 O · 오류 X · 애매 △
    # 사용자 요청 — 명시 O / X / △ (정확한 unicode 모양)
    button_data = [
        ("정확", "correct", "#2da44e", "check-circle-fill", "○"),  # U+25CB white circle
        ("오류", "wrong",   "#d1242f", "x-circle-fill",     "✕"),  # U+2715 multiplication X
        ("애매", "uncertain","#d68800", "dash-circle-fill",  "△"),  # U+25B3 white triangle
    ]
    for col, (label, key, color, icon, mark) in zip(cols, button_data):
        with col:
            is_selected = selected == key
            btn_label = f"{mark}  {label}"
            btn_type = "primary" if is_selected else "secondary"
            if st.button(
                btn_label,
                key=f"{key_prefix}_verdict_btn_{key}",
                use_container_width=True,
                type=btn_type,
            ):
                st.session_state[f"{key_prefix}_verdict_state"] = key
                st.rerun()
    return st.session_state[f"{key_prefix}_verdict_state"]


# ============================================================
# Sample weight slider + preset
# ============================================================
def sample_weight_input(key_prefix: str, verdict: str,
                         saved_weight: float = 1.0) -> float:
    """가중치 입력 — preset 버튼 + 슬라이더."""
    weight_default = float(saved_weight or 1.0)
    if verdict == "wrong" and weight_default == 1.0:
        weight_default = 5.0

    # preset 버튼
    presets = [("×1", 1.0), ("×3", 3.0), ("×5", 5.0), ("×10", 10.0)]
    preset_cols = st.columns(len(presets))
    weight_key = f"{key_prefix}_weight_value"
    if weight_key not in st.session_state:
        st.session_state[weight_key] = weight_default
    for col, (label, val) in zip(preset_cols, presets):
        with col:
            if st.button(label, key=f"{key_prefix}_preset_{val}",
                          use_container_width=True,
                          type="primary" if abs(st.session_state[weight_key] - val) < 0.01 else "secondary"):
                st.session_state[weight_key] = val
                st.rerun()

    weight = st.slider(
        "정정 가중치 (재학습 시 강조 배율)",
        min_value=1.0, max_value=10.0,
        value=st.session_state[weight_key], step=0.5,
        help="오류 case는 ×5 권장 — 재학습 시 5배 자주 등장",
        key=f"{key_prefix}_weight_slider",
        label_visibility="visible",
    )
    if weight != st.session_state[weight_key]:
        st.session_state[weight_key] = weight
    return weight


# ============================================================
# Classifier 검수 패널 (컴팩트)
# ============================================================
def classifier_review_panel(
    cls_result: dict,
    saved_label: dict | None,
    available_classes: list[str],
    panel_key: str = "cls",
) -> dict:
    if cls_result is None or "error" in (cls_result or {}):
        st.warning("Classifier 결과 없음")
        return {"human_label": None, "verdict": None, "weight": 1.0, "changed": False}

    top = cls_result.get("top_class")
    conf = cls_result.get("top_conf", 0.0)

    # 모델 예측 — 큰 카드
    conf_color = "#2da44e" if conf >= 0.8 else ("#d68800" if conf >= 0.5 else "#d1242f")
    st.markdown(
        f"""
        <div style="background:rgba(128,128,128,0.08);padding:14px 16px;border-radius:10px;
                    border-left:4px solid {conf_color};">
            <div style="color:var(--text-color);opacity:0.65;font-size:0.85em;margin-bottom:4px;">모델 예측</div>
            <div style="font-size:1.4em;font-weight:600;color:var(--text-color);">{top}</div>
            <div style="margin-top:6px;">
                <span style="color:{conf_color};font-weight:600;">{conf:.1%}</span>
                <span style="color:var(--text-color);opacity:0.55;">신뢰도</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top-5 표
    with st.expander("Top-5 확률 보기"):
        for name, p in cls_result.get("all_probs", [])[:5]:
            bar_w = int(p * 100)
            highlight = "background:rgba(31,111,235,0.15);" if name == top else ""
            st.markdown(
                f'<div style="display:flex;align-items:center;padding:4px;{highlight}border-radius:4px;">'
                f'<div style="width:140px;color:var(--text-color);">{name}</div>'
                f'<div style="flex:1;background:rgba(128,128,128,0.18);height:14px;border-radius:7px;overflow:hidden;">'
                f'  <div style="width:{bar_w}%;background:{conf_color};height:100%;"></div>'
                f'</div>'
                f'<div style="width:60px;text-align:right;color:var(--text-color);opacity:0.75;">{p:.3f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # 정답 클래스 select
    saved_verdict = (saved_label or {}).get("cls_verdict")
    saved_human = (saved_label or {}).get("cls_human_label")
    initial_label = saved_human or top
    classes_options = list(dict.fromkeys(available_classes + [top]))
    if initial_label not in classes_options:
        classes_options.insert(0, initial_label)
    classes_options_with_other = classes_options + ["__OTHER__"]
    idx_default = (
        classes_options_with_other.index(initial_label)
        if initial_label in classes_options_with_other else 0
    )
    selected = st.selectbox(
        "정답 클래스 (운영자 판단)",
        classes_options_with_other,
        index=idx_default,
        format_func=lambda x: "(직접 입력)" if x == "__OTHER__" else x,
        key=f"{panel_key}_label",
    )
    human_label = selected
    if selected == "__OTHER__":
        human_label = st.text_input("새 클래스 이름",
                                     value=saved_human or "",
                                     key=f"{panel_key}_other")

    # Auto verdict
    if human_label == top:
        auto_verdict = "correct"
    else:
        auto_verdict = "wrong"

    st.markdown("**판정** (1 정확 · 2 오류 · 3 애매)")
    verdict = verdict_buttons(panel_key, saved_verdict, auto_default=auto_verdict)

    # Weight (오류일 때만)
    weight = 1.0
    if verdict == "wrong":
        st.markdown("**정정 가중치**")
        weight = sample_weight_input(
            panel_key, verdict,
            saved_weight=float((saved_label or {}).get("cls_correction_weight") or 1.0),
        )

    return {
        "human_label": human_label,
        "verdict": verdict,
        "weight": weight,
        "changed": True,
    }


# ============================================================
# ROI 크롭 (작은 폴리곤 확대해서 보기)
# ============================================================
def _roi_crop(img: Image.Image, polygon: list[tuple[float, float]],
              padding_ratio: float = 0.4, max_side: int = 360) -> Image.Image | None:
    """폴리곤 bbox 주위를 crop. padding 추가로 주변 컨텍스트 포함."""
    if not polygon or len(polygon) < 2:
        return None
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    px = w * padding_ratio
    py = h * padding_ratio
    cx0 = max(0, int(x0 - px))
    cy0 = max(0, int(y0 - py))
    cx1 = min(img.size[0], int(x1 + px))
    cy1 = min(img.size[1], int(y1 + py))
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    crop = img.crop((cx0, cy0, cx1, cy1))
    # overlay polygon outline
    from PIL import ImageDraw
    crop_rgba = crop.convert("RGBA")
    overlay = Image.new("RGBA", crop_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    shifted = [(x - cx0, y - cy0) for x, y in polygon]
    if len(shifted) >= 3:
        draw.polygon(shifted, fill=(231, 76, 60, 60), outline=(231, 76, 60, 255))
    out = Image.alpha_composite(crop_rgba, overlay).convert("RGB")
    if max(out.size) > max_side:
        ratio = max_side / max(out.size)
        new_w = int(out.size[0] * ratio)
        new_h = int(out.size[1] * ratio)
        out = out.resize((new_w, new_h), Image.LANCZOS)
    elif max(out.size) < 120:
        ratio = 120 / max(out.size)
        new_w = int(out.size[0] * ratio)
        new_h = int(out.size[1] * ratio)
        out = out.resize((new_w, new_h), Image.LANCZOS)
    return out


# ============================================================
# Segmenter 폴리곤 카드 (모델 결과별)
# ============================================================
def _polygon_card(
    idx: int,
    polygon_info: dict,
    panel_key: str,
    classes_options: list[str],
    available_class_names: dict[int, str],
    img: Image.Image | None = None,
    show_roi: bool = True,
) -> tuple[str, dict | None]:
    """폴리곤 카드 — ROI 크롭 + 정보 + 조치(유지/삭제/애매) + 클래스 정정.

    좌표 편집은 캔버스에서 (드래그). 여기는 의사결정만.
    Returns (decision, kept_dict|None).
    """
    cid = polygon_info["class_id"]
    r, g, b = COLORS_RGB[cid % len(COLORS_RGB)]
    cname = polygon_info["class_name"]
    conf = polygon_info["conf"]
    poly = polygon_info["polygon"]
    n_pts = len(poly)

    if poly:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        bw = max(xs) - min(xs)
        bh = max(ys) - min(ys)
    else:
        bw = bh = 0

    # 컬러 헤더 카드
    st.markdown(
        f'<div style="background:rgba({r},{g},{b},0.08);padding:10px 14px;border-radius:8px;'
        f'border-left:4px solid rgb({r},{g},{b});margin-bottom:6px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'  <span style="color:rgb({r},{g},{b});font-weight:600;font-size:1.05em;">#{idx+1} {cname}</span>'
        f'  <span style="color:var(--text-color);opacity:0.7;font-size:0.85em;">신뢰도 {conf:.3f} · {n_pts}점 · {int(bw)}×{int(bh)}px</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 본문: ROI + 컨트롤
    if show_roi and img is not None and poly:
        col_roi, col_ctrl = st.columns([1, 3])
        with col_roi:
            crop = _roi_crop(img, poly)
            if crop:
                st.image(crop, use_column_width=True)
    else:
        col_ctrl = st.container()

    with col_ctrl:
        cc1, cc2 = st.columns([3, 2])
        with cc1:
            decision = st.radio(
                "조치",
                ["keep", "delete", "uncertain"],
                horizontal=True,
                format_func=lambda x: {"keep": "유지", "delete": "삭제", "uncertain": "애매"}[x],
                key=f"{panel_key}_decision_{idx}",
                label_visibility="collapsed",
            )
        with cc2:
            new_cname = st.selectbox(
                "클래스",
                classes_options or [cname],
                index=(classes_options.index(cname) if cname in classes_options else 0),
                key=f"{panel_key}_class_{idx}",
                label_visibility="collapsed",
            )

    if decision == "keep":
        new_cid = next((cid_ for cid_, n in available_class_names.items() if n == new_cname), cid)
        return decision, {
            "class_id": int(new_cid),
            "class_name": new_cname,
            "polygon": poly,
            "conf": polygon_info["conf"],
            "source": "model_kept",
        }
    return decision, None


# ============================================================
# Segmenter 검수 패널 (캔버스 + 카드)
# ============================================================
def segmenter_review_panel(
    img: Image.Image,
    seg_result: dict,
    saved_label: dict | None,
    available_class_names: dict[int, str],
    max_canvas_width: int = 1100,
    panel_key: str = "seg",
    show_roi: bool = True,
) -> dict:
    if seg_result is None or "error" in (seg_result or {}):
        st.warning("Segmenter 결과 없음")
        return {"human_polygons": None, "verdict": None, "weight": 1.0, "changed": False}

    model_polys = []
    for poly, cid, cname, conf in zip(
        seg_result.get("polygons", []),
        seg_result.get("class_ids", []),
        seg_result.get("class_names", []),
        seg_result.get("confs", []),
    ):
        model_polys.append({
            "class_id": int(cid),
            "class_name": cname,
            "polygon": [(float(x), float(y)) for x, y in poly],
            "conf": float(conf),
        })

    classes_options = sorted(set(available_class_names.values()) | {p["class_name"] for p in model_polys})
    classes_options = [c for c in classes_options if c]

    # ──────────────────────────────────────────────────────
    # Header — 모델 폴리곤 요약 + 클래스별 범례
    # ──────────────────────────────────────────────────────
    legend_html_parts = []
    class_counts: dict[int, int] = {}
    for p in model_polys:
        class_counts[p["class_id"]] = class_counts.get(p["class_id"], 0) + 1
    for cid_, n in class_counts.items():
        r, g, b = COLORS_RGB[cid_ % len(COLORS_RGB)]
        cname = available_class_names.get(cid_, f"class_{cid_}")
        legend_html_parts.append(
            f'<span style="display:inline-block;padding:2px 10px;margin:2px;'
            f'background:rgba({r},{g},{b},0.2);border:1px solid rgb({r},{g},{b});'
            f'border-radius:12px;color:rgb({r},{g},{b});">'
            f'{cname} × {n}</span>'
        )
    st.markdown(
        f'<div style="margin-bottom:8px;">'
        f'<b>모델 폴리곤 {len(model_polys)}개</b> {"".join(legend_html_parts)}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ──────────────────────────────────────────────────────
    # 1. 모델 폴리곤별 조치 (카드) — 의사결정 (keep/delete/uncertain) + 클래스 정정
    # ──────────────────────────────────────────────────────
    kept_model_polys: list[dict] = []
    decisions: list[str] = []
    if model_polys:
        st.markdown("##### 모델 폴리곤 의사결정")
        for i, p in enumerate(model_polys):
            decision, kept = _polygon_card(
                i, p, panel_key, classes_options, available_class_names,
                img=img, show_roi=show_roi,
            )
            decisions.append(decision)
            if kept:
                kept_model_polys.append(kept)

    # ──────────────────────────────────────────────────────
    # 2. Polygon Editor (Konva) — 정점 편집 + 새 영역 추가
    #    카드에서 keep된 모델 폴리곤 + 사용자가 직접 추가/편집한 폴리곤
    #    좌표는 IMAGE 좌표 그대로 입출력 (component 내부에서 max_width로 표시 스케일)
    # ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f'<h5 style="margin-bottom:6px;">'
        f'{_bi_icon("vector-pen","808080",18)} &nbsp;폴리곤 편집</h5>',
        unsafe_allow_html=True,
    )

    with st.expander("사용법 (조작 방법 + 단축키)", expanded=False):
        st.markdown(
            f"""
<style>
.pe-help-grid {{ display:grid; grid-template-columns: 110px 1fr; gap:6px 14px; font-size:0.9em; }}
.pe-help-grid b {{ color:#1f6feb; }}
.pe-help-section {{ color:var(--text-color); opacity:0.65; font-weight:600; margin-top:8px; margin-bottom:4px; font-size:0.85em; }}
kbd {{ background:rgba(128,128,128,0.12); border:1px solid rgba(128,128,128,0.35); border-radius:4px; padding:1px 6px;
       font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:0.8em; color:var(--text-color); }}
</style>

<div class="pe-help-section">{_bi_icon("pencil-square","58a6ff",13)}&nbsp;편집 (선택 모드)</div>
<div class="pe-help-grid">
  <b>정점 이동</b><div>흰 원을 마우스로 드래그</div>
  <b>정점 추가</b><div>엣지(선분)에 마우스 hover → <span style="color:#2da44e;">+</span> 원 클릭</div>
  <b>정점 삭제</b><div>정점 <kbd>우클릭</kbd> 또는 정점 선택 후 <kbd>Delete</kbd> / <kbd>Backspace</kbd></div>
  <b>폴리곤 이동</b><div>폴리곤 내부 드래그</div>
  <b>폴리곤 삭제</b><div>폴리곤 클릭 선택 후 <kbd>Delete</kbd> 또는 툴바 <b>삭제</b> 버튼</div>
  <b>클래스 변경</b><div>폴리곤 선택 → 툴바의 <b>선택된 클래스</b> 드롭다운</div>
</div>

<div class="pe-help-section">{_bi_icon("plus-square","2da44e",13)}&nbsp;새 영역 추가</div>
<div class="pe-help-grid">
  <b>사각형</b><div>툴바 <b>+ 사각형</b> → 캔버스 클릭+드래그 한 번</div>
  <b>폴리곤</b><div>툴바 <b>+ 폴리곤</b> → 점 클릭 (3개 이상) → <b>첫 점 (초록 원)</b> 다시 클릭 또는 <kbd>Enter</kbd>로 닫기</div>
  <b>취소</b><div>그리던 중 <kbd>Esc</kbd></div>
</div>

<div class="pe-help-section">{_bi_icon("arrow-counterclockwise","d68800",13)}&nbsp;Undo / Redo / 보기</div>
<div class="pe-help-grid">
  <b>Undo</b><div><kbd>Ctrl</kbd>+<kbd>Z</kbd> (Mac: <kbd>⌘</kbd>+<kbd>Z</kbd>) · 툴바 <b>↶ Undo</b></div>
  <b>Redo</b><div><kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Z</kbd> (Mac: <kbd>⌘</kbd>+<kbd>Shift</kbd>+<kbd>Z</kbd>) · 툴바 <b>↷ Redo</b></div>
  <b>줌</b><div>마우스 휠</div>
  <b>줌 초기화</b><div>툴바 <b>↺ 보기</b></div>
</div>
            """,
            unsafe_allow_html=True,
        )

    W, H = img.size

    # class_names: 인덱스 = class_id 가 되도록 list 생성
    # 1순위: seg_result["names_dict"] (추론 시 모델 전체 names 저장) - source of truth
    # 2순위: available_class_names (st.session_state.segmenter_classes)
    # 3순위: classes_options (모델 결과 + 사용자 추가 클래스 합집합)
    names_source: dict[int, str] = {}
    nd_raw = (seg_result or {}).get("names_dict") or {}
    if nd_raw:
        names_source = {int(k): v for k, v in nd_raw.items() if v}
    if not names_source and available_class_names:
        names_source = {int(k): v for k, v in available_class_names.items() if v}

    if names_source:
        _max_cid = max(names_source.keys())
        class_names_list: list[str] = [
            names_source.get(cid, f"class_{cid}") for cid in range(_max_cid + 1)
        ]
    else:
        class_names_list = []
    # classes_options 에만 있는 이름도 추가 (fallback)
    for cname in classes_options:
        if cname and cname not in class_names_list:
            class_names_list.append(cname)
    if not class_names_list:
        class_names_list = ["defect"]

    # 이미지 → data URI
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    # 편집기에 넣을 폴리곤 리스트: 카드에서 keep된 것만 (delete/uncertain은 제외)
    editor_input_polys: list[dict] = []
    for kp in kept_model_polys:
        editor_input_polys.append({
            "class_id": int(kp["class_id"]),
            "class_name": kp["class_name"],
            "polygon": [[float(x), float(y)] for (x, y) in kp["polygon"]],
            "conf": float(kp.get("conf", 1.0)),
            "source": kp.get("source", "model_kept"),
        })

    editor_result = polygon_editor(
        image_b64=img_b64,
        polygons=editor_input_polys,
        class_names=class_names_list,
        image_w=W,
        image_h=H,
        max_width=max_canvas_width,
        key=f"{panel_key}_polyedit",
        theme=st.session_state.get("theme_choice", "dark"),
    )

    # editor_result.polygons = 편집기에서 최종 반환된 폴리곤 (편집된 모델 + 새로 그린 것)
    final_polys: list[dict] = []
    new_user_polys: list[dict] = []
    for ep in (editor_result or {}).get("polygons", []) or []:
        poly_pts = [(float(x), float(y)) for x, y in ep.get("polygon", [])]
        if len(poly_pts) < 3:
            continue
        cid = int(ep.get("class_id", 0))
        cname = ep.get("class_name") or available_class_names.get(cid, f"class_{cid}")
        src = ep.get("source") or "user_added"
        conf = ep.get("conf")
        if conf is None:
            conf = 1.0
        entry = {
            "class_id": cid,
            "class_name": cname,
            "polygon": poly_pts,
            "conf": float(conf),
            "source": src,
        }
        final_polys.append(entry)
        if src in ("user_added", "user_drawn"):
            new_user_polys.append(entry)

    # 통계 — 카드 grid
    n_deleted = decisions.count("delete")
    n_uncertain = decisions.count("uncertain")
    n_kept_final = len(final_polys) - len(new_user_polys)
    n_total_final = len(final_polys)

    stat_items = [
        ("유지/편집", n_kept_final, "#2da44e", "check-circle-fill"),
        ("삭제",      n_deleted,    "#d1242f", "x-circle-fill"),
        ("애매",      n_uncertain,  "#d68800", "dash-circle-fill"),
        ("사용자 추가", len(new_user_polys), "#3498db", "plus-square-fill"),
    ]

    cards_html = ""
    for label, value, color, icon in stat_items:
        cards_html += (
            f'<div style="background:rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.10);'
            f'border-left:3px solid {color};border-radius:6px;padding:8px 12px;">'
            f'<div style="color:var(--text-color, #6e7681);font-size:0.82em;display:flex;align-items:center;gap:4px;opacity:0.85;">'
            f'{_bi_icon(icon, color[1:], 12)}<span>{label}</span></div>'
            f'<div style="color:{color};font-size:1.6em;font-weight:700;line-height:1.2;margin-top:2px;">{value}</div>'
            f'</div>'
        )

    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(4, 1fr) 1.2fr;gap:8px;margin-top:10px;">'
        f'{cards_html}'
        f'<div style="background:rgba(128,128,128,0.08);border:1px solid rgba(128,128,128,0.25);border-radius:6px;padding:8px 12px;">'
        f'<div style="color:var(--text-color, #6e7681);font-size:0.82em;opacity:0.85;">총 폴리곤</div>'
        f'<div style="color:var(--text-color, #262730);font-size:1.6em;font-weight:700;line-height:1.2;margin-top:2px;">'
        f'{n_total_final}<span style="opacity:0.55;font-size:0.55em;font-weight:400;">&nbsp;개</span></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ──────────────────────────────────────────────────────
    # Verdict + weight
    # ──────────────────────────────────────────────────────
    st.markdown("---")
    if n_deleted == 0 and len(new_user_polys) == 0 and n_uncertain == 0:
        auto_verdict = "correct"
    elif n_uncertain > 0 and n_deleted == 0:
        auto_verdict = "uncertain"
    else:
        auto_verdict = "wrong"
    saved_verdict = (saved_label or {}).get("seg_verdict") or auto_verdict

    st.markdown("**판정** (1 정확 · 2 오류 · 3 애매)")
    verdict = verdict_buttons(panel_key, saved_verdict, auto_default=auto_verdict)

    weight = 1.0
    if verdict == "wrong":
        st.markdown("**정정 가중치**")
        weight = sample_weight_input(
            panel_key, verdict,
            saved_weight=float((saved_label or {}).get("seg_correction_weight") or 1.0),
        )

    return {
        "human_polygons": final_polys,
        "verdict": verdict,
        "weight": weight,
        "changed": True,
    }
