"""AI 추론 검수 스튜디오 — Streamlit 메인 앱.

운영자가 Classifier + Segmenter 두 종류의 YOLO 모델 추론 결과를
직관적으로 검수·정정하고, SQLite DB 에 라벨을 누적해 학습 데이터로
즉시 export 할 수 있는 Human-in-the-Loop 시스템.

구성:
- 사이드바: 세션 / 모델·이미지 업로드 / 추론 실행 / 보기 모드
- 메인: 검수 패널 / 통계 / 내보내기
- 자동 복구: 새로고침 시 마지막 작업 상태 복원
- 단축키: ← / → / Cmd+S / Cmd+Z

실행:
  streamlit run app.py --server.port=8501
"""
from __future__ import annotations

import os
import tempfile
import time
import zipfile
from io import BytesIO
from pathlib import Path

import streamlit as st
import streamlit_shortcuts as ssh
from PIL import Image

from inference import (
    detect_device,
    draw_segmenter_result,
    image_from_uploaded,
    load_yolo_model,
    predict_classifier,
    predict_segmenter,
)
from labeling import (
    _bi_icon,
    classifier_review_panel,
    segmenter_review_panel,
    status_badge,
    status_text,
)
from db import (
    DB_PATH,
    create_session,
    get_label,
    get_labels_by_session,
    get_recent_history,
    get_session,
    get_undo_count,
    init_db,
    list_sessions,
    save_label_with_history,
    session_stats,
    undo_last,
    upsert_inference_result,
    upsert_label,
    delete_session,
    delete_sessions,
)
from export import build_export_zip
from labelstudio_bridge import (
    LSClient,
    build_predictions_for_image,
    build_task_for_image,
    copy_images_to_local,
    make_label_config,
    parse_export,
)


# ============================================================
# 정렬 키 (Active Learning)
# ============================================================
SORT_OPTIONS = {
    "default": "기본 (입력순)",
    "low_conf": "신뢰도 낮은 것부터 (AL)",
    "uninspected": "검수 안 한 것부터",
    "wrong_first": "정정 필요한 것부터",
    "many_polys": "폴리곤 많은 것부터",
}


def compute_sort_order(results: list[dict], saved_labels_by_fn: dict, mode: str) -> list[int]:
    """원본 results 인덱스를 정렬 mode 에 따라 재배열한 list 반환."""
    n = len(results)
    if mode == "default" or n == 0:
        return list(range(n))

    def key_fn(i):
        r = results[i]
        fn = r["filename"]
        L = saved_labels_by_fn.get(fn) or {}
        cls_conf = (r.get("classifier") or {}).get("top_conf", 1.0) or 1.0
        seg_polys = (r.get("segmenter") or {}).get("polygons", []) or []
        seg_confs = (r.get("segmenter") or {}).get("confs", []) or []
        min_seg_conf = min(seg_confs) if seg_confs else 1.0
        n_polys = len(seg_polys)
        inspected = 1 if L.get("inspected_at") else 0
        is_wrong = 1 if (L.get("cls_verdict") == "wrong" or L.get("seg_verdict") == "wrong") else 0

        if mode == "low_conf":
            # 모델 최저 신뢰도 (cls + seg 중 작은 쪽) 오름차순. inspected = 뒤로.
            return (inspected, min(cls_conf, min_seg_conf))
        if mode == "uninspected":
            return (inspected, i)
        if mode == "wrong_first":
            return (-is_wrong, inspected, i)
        if mode == "many_polys":
            return (-n_polys, inspected, i)
        return i

    return sorted(range(n), key=key_fn)


# ============================================================
# 자동 저장 fingerprint
# ============================================================
def _autosave_fp(cls_review: dict, seg_review: dict) -> str:
    """변경 감지용 fingerprint. JSON serialize 가능한 안정적인 hash."""
    import hashlib
    payload = {
        "cls_label": cls_review.get("human_label"),
        "cls_verdict": cls_review.get("verdict"),
        "cls_weight": round(float(cls_review.get("weight") or 1.0), 3),
        "seg_verdict": seg_review.get("verdict"),
        "seg_weight": round(float(seg_review.get("weight") or 1.0), 3),
        "seg_polys": [
            {
                "cid": int(p.get("class_id", 0)),
                "n": len(p.get("polygon", [])),
                "pts": [(round(x, 1), round(y, 1)) for x, y in p.get("polygon", [])],
                "src": p.get("source", "model_kept"),
            }
            for p in (seg_review.get("human_polygons") or [])
        ],
    }
    return hashlib.md5(__import__("json").dumps(payload, sort_keys=True).encode()).hexdigest()


# ============================================================
# Streamlit 설정
# ============================================================
st.set_page_config(
    page_title="AI 추론 검수 스튜디오",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.metric-card { background:rgba(128,128,128,0.08); padding:8px 12px; border-radius:8px; }
.stRadio > label { margin-right: 0.5rem; }
/* number input 가운데 정렬 + step buttons: - 왼쪽 / 숫자 가운데 / + 오른쪽 (대칭) */
[data-testid="stNumberInputField"] { text-align: center !important; }
[data-testid="stNumberInputContainer"] {
    display: flex !important;
    align-items: stretch !important;
}
[data-testid="stNumberInputContainer"] > div[data-baseweb="input"] {
    flex: 1 1 auto !important;
}
[data-testid="stNumberInput"] [data-testid="stNumberInputStepDown"] { order: -1 !important; }
[data-testid="stNumberInput"] [data-testid="stNumberInputStepUp"] { order: 1 !important; }
/* vm_toggle (기능 설명 페이지 보기) 체크 시 말차색 */
.vm-toggle-matcha [role="checkbox"][aria-checked="true"],
.vm-toggle-matcha [data-baseweb="checkbox"] > div[data-checked="true"],
.vm-toggle-matcha label > div:first-child[aria-checked="true"] {
    background-color: #91a464 !important;
    border-color: #91a464 !important;
}
/* progress bar 텍스트 가운데 정렬 */
[data-testid="stProgress"] [data-testid="stMarkdownContainer"],
[data-testid="stProgress"] [data-testid="stMarkdownContainer"] *,
[data-testid="stProgress"] p { text-align: center !important; }
/* Streamlit 기본 사이드바 collapse/expand 버튼 숨김 — 자체 switch 로 교체 */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    display: none !important;
}

/* 자체 사이드바 토글 switch (uiverse Bodyhc 디자인 기반) */
#hitl-sb-switch-host {
    position: absolute !important;
    top: 18px !important;
    left: 18px !important;
    z-index: 9999999 !important;
    background: rgba(74, 79, 92, 0.92);
    padding: 8px 14px;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    backdrop-filter: blur(6px);
    display: flex;
    gap: 16px;
    align-items: center;
}
.checkbox-wrapper-35 .switch { display: none; }
.checkbox-wrapper-35 .switch + label {
    align-items: center; color: #ffffff; cursor: pointer;
    display: flex; font-family: -apple-system, Helvetica, Arial, sans-serif;
    font-size: 12px; line-height: 16px; position: relative;
    user-select: none; font-weight: 600;
}
.checkbox-wrapper-35 .switch + label::before,
.checkbox-wrapper-35 .switch + label::after { content: ''; display: block; }
.checkbox-wrapper-35 .switch + label::before {
    background-color: #1c1f29; border-radius: 500px; height: 16px;
    margin-right: 10px; transition: background-color 0.15s ease-out; width: 30px;
    flex-shrink: 0;
}
.checkbox-wrapper-35 .switch + label::after {
    background-color: #fff; border-radius: 13px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    height: 13px; left: 1.5px; position: absolute; top: 1.5px;
    transition: transform 0.18s ease-out; width: 13px;
}
.checkbox-wrapper-35 .switch:checked + label::before { background-color: #8b5cf6; }
.checkbox-wrapper-35 .switch:checked + label::after { transform: translate3d(14px, 0, 0); }
/* OFF/ON 라벨 부드러운 슬라이드 (uiverse 원본) */
.checkbox-wrapper-35 .switch-x-text { display: inline-block; margin-right: 4px; }
.checkbox-wrapper-35 .switch-x-toggletext {
    display: inline-block; font-weight: 700; height: 16px; line-height: 16px;
    overflow: hidden; position: relative; min-width: 28px;
}
.checkbox-wrapper-35 .switch-x-unchecked,
.checkbox-wrapper-35 .switch-x-checked {
    left: 0; position: absolute; top: 0;
    transition: transform 0.18s ease-out, opacity 0.18s ease-out;
}
.checkbox-wrapper-35 .switch-x-unchecked { opacity: 1; transform: none; }
.checkbox-wrapper-35 .switch-x-checked { opacity: 0; transform: translate3d(0, 100%, 0); }
.checkbox-wrapper-35 .switch:checked + label .switch-x-unchecked {
    opacity: 0; transform: translate3d(0, -100%, 0);
}
.checkbox-wrapper-35 .switch:checked + label .switch-x-checked { opacity: 1; transform: none; }
/* three-body 로딩 (Uiverse.io by dovatgabriel) — 멀티 컬러 (회색·파랑·보라) */
.three-body { --uib-size:55px; --uib-speed:0.8s; position:relative; display:inline-block; height:var(--uib-size); width:var(--uib-size); animation:spin78236 calc(var(--uib-speed)*2.5) infinite linear; }
.three-body__dot { position:absolute; height:100%; width:30%; }
.three-body__dot:after { content:''; position:absolute; height:0%; width:100%; padding-bottom:100%; border-radius:50%; }
.three-body__dot:nth-child(1)::after { background-color:#6366f1; }
.three-body__dot:nth-child(2)::after { background-color:#8b5cf6; }
.three-body__dot:nth-child(3)::after { background-color:#a78bfa; }
.three-body__dot:nth-child(1) { bottom:5%; left:0; transform:rotate(60deg); transform-origin:50% 85%; }
.three-body__dot:nth-child(1)::after { bottom:0; left:0; animation:wobble1 var(--uib-speed) infinite ease-in-out; animation-delay:calc(var(--uib-speed)*-0.3); }
.three-body__dot:nth-child(2) { bottom:5%; right:0; transform:rotate(-60deg); transform-origin:50% 85%; }
.three-body__dot:nth-child(2)::after { bottom:0; left:0; animation:wobble1 var(--uib-speed) infinite calc(var(--uib-speed)*-0.15) ease-in-out; }
.three-body__dot:nth-child(3) { bottom:-5%; left:0; transform:translateX(116.666%); }
.three-body__dot:nth-child(3)::after { top:0; left:0; animation:wobble2 var(--uib-speed) infinite ease-in-out; }
@keyframes spin78236 { 0% { transform:rotate(0deg); } 100% { transform:rotate(360deg); } }
@keyframes wobble1 { 0%,100% { transform:translateY(0%) scale(1); opacity:1; } 50% { transform:translateY(-66%) scale(0.65); opacity:0.8; } }
@keyframes wobble2 { 0%,100% { transform:translateY(0%) scale(1); opacity:1; } 50% { transform:translateY(66%) scale(0.65); opacity:0.8; } }
.hitl-loader-box { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:14px; padding:48px 0 24px 0; }
.hitl-loader-title { font-size:1.15em; font-weight:600; color:var(--text-color); }
.hitl-loader-sub { font-size:0.92em; opacity:0.75; color:var(--text-color); text-align:center; max-width:540px; }
.hitl-loader-skip { font-size:0.82em; opacity:0.6; color:var(--text-color); margin-top:6px; }
.hitl-loader-skip code { background:rgba(128,128,128,0.18); padding:2px 6px; border-radius:4px; font-size:0.95em; }
.hitl-loader-cancel { display:inline-block; margin-top:8px; padding:8px 18px; background:rgba(128,128,128,0.15); color:var(--text-color) !important; text-decoration:none; border-radius:6px; font-size:0.9em; border:1px solid rgba(128,128,128,0.35); transition:all 0.15s; }
.hitl-loader-cancel:hover { background:rgba(214,136,0,0.18); border-color:#d68800; }
/* 사이드바: 펼친 상태에서만 폭 420 강제. 접힘 상태는 streamlit 기본 동작 */
section[data-testid="stSidebar"][aria-expanded="true"] {
    min-width: 420px !important;
    max-width: 420px !important;
    width: 420px !important;
}
section[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
    width: 420px !important;
}
section[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 0 !important;
    max-width: 0 !important;
    width: 0 !important;
    overflow: hidden !important;
}
/* resize handle 숨김 (드래그 비활성) */
section[data-testid="stSidebar"] > div[class*="resizer"],
section[data-testid="stSidebar"] [data-testid="stSidebarResizeHandle"],
section[data-testid="stSidebar"] [class*="ResizeHandle"] {
    display: none !important;
    pointer-events: none !important;
    cursor: default !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# 좌측 상단 visual switch — 사이드바 열기/닫기 (Page 토글은 native streamlit toggle 을 JS로 옆으로 이동)
st.markdown(
    """
<div id="hitl-sb-switch-host">
  <div class="checkbox-wrapper-35 sw-sidebar">
    <input value="sb" id="hitl-sb-sw" type="checkbox" class="switch" checked>
    <label for="hitl-sb-sw">
      <span class="switch-x-text">Sidebar </span>
      <span class="switch-x-toggletext">
        <span class="switch-x-unchecked">Off</span>
        <span class="switch-x-checked">On</span>
      </span>
    </label>
  </div>
</div>
    """,
    unsafe_allow_html=True,
)

# ── 키보드 단축키 + 사이드바 switch JS — st.stop() 위로 올려서 추론 전에도 동작
import streamlit.components.v1 as _stc_top
_stc_top.html(
    """
<script>
(function() {
  try {
    const doc = window.parent.document;
    const winp = window.parent;
    // === 사이드바 토글 switch ===
    function wireSwitch() {
      const host = doc.getElementById('hitl-sb-switch-host');
      // host 를 body 직접 자식으로 이동 — stHeader stacking context 회피
      if (host && host.parentElement !== doc.body) {
        doc.body.appendChild(host);
      }
      const sw = doc.getElementById('hitl-sb-sw');
      const sb = doc.querySelector('section[data-testid="stSidebar"]');
      if (!sw || !sb) return false;

      // 사이드바 안 "기능 설명 페이지" 토글에 custom class 추가 (말차색 styling — 현재 button 이지만 잔존 가능)
      for (const t of doc.querySelectorAll('[data-testid="stCheckbox"]')) {
        if ((t.innerText || '').includes('기능 설명 페이지')) {
          t.classList.add('vm-toggle-matcha');
          break;
        }
      }



      // ── 사이드바 toggle switch
      if (!sw.__wired) {
        sw.__wired = true;
        function syncSb() { sw.checked = sb.getAttribute('aria-expanded') === 'true'; }
        syncSb();
        sw.addEventListener('change', (e) => {
          const cb = doc.querySelector('[data-testid="stSidebarCollapseButton"] button');
          const eb = doc.querySelector('[data-testid="stSidebarCollapsedControl"] button');
          if (e.target.checked) { if (eb) eb.click(); }
          else { if (cb) cb.click(); }
        });
        new MutationObserver(syncSb).observe(sb, {attributes:true, attributeFilter:['aria-expanded']});
      }

      return true;
    }
    if (!wireSwitch()) {
      const obs = new MutationObserver(() => { if (wireSwitch()) obs.disconnect(); });
      obs.observe(doc.body, {childList:true, subtree:true});
    }

    // === number_input 의 stepDown(-) 을 input 좌측으로 이동 (영구 observer) ===
    if (!winp.__hitlNumInpFixed) {
      winp.__hitlNumInpFixed = true;
      function fixNumberInputs() {
        for (const c of doc.querySelectorAll('[data-testid="stNumberInputContainer"]')) {
          const stepDown = c.querySelector('[data-testid="stNumberInputStepDown"]');
          const stepUp = c.querySelector('[data-testid="stNumberInputStepUp"]');
          if (!stepDown || !stepUp) continue;
          const stepUpWrap = stepUp.parentElement;  // .st-emotion-cache-XXX (wrapper)
          // stepDown 이 wrapper 안에 있지 않으면 동일 className wrapper 생성 + 이동
          if (stepDown.parentElement === c) {
            const wrap = doc.createElement('div');
            wrap.className = stepUpWrap ? stepUpWrap.className : '';
            wrap.dataset.hitlStepDownWrap = '1';
            c.insertBefore(wrap, c.firstChild);
            wrap.appendChild(stepDown);
          }
        }
      }
      fixNumberInputs();
      new MutationObserver(fixNumberInputs).observe(doc.body, {childList:true, subtree:true});
    }

    // === 키보드 단축키 (한 번만) ===
    if (winp.__hitlShortcutsV3) return;
    winp.__hitlShortcutsV3 = true;
    function inEditable(e) {
      const t = e.target; if (!t) return false;
      const tag = (t.tagName||'').toUpperCase();
      return tag === 'INPUT' || tag === 'TEXTAREA' || t.isContentEditable;
    }
    function clickBtnText(txt) {
      for (const b of doc.querySelectorAll('button')) {
        if (b.disabled) continue;
        if ((b.innerText||'').trim().includes(txt)) { b.click(); return true; }
      }
      return false;
    }
    winp.addEventListener('keydown', function(e) {
      if (inEditable(e)) return;
      const k = e.key, cmd = e.metaKey || e.ctrlKey, shift = e.shiftKey;
      if (!cmd && !shift && k === 'ArrowLeft') { if (clickBtnText('이전')) e.preventDefault(); }
      else if (!cmd && !shift && k === 'ArrowRight') { if (clickBtnText('다음')) e.preventDefault(); }
      else if (cmd && !shift && k.toLowerCase() === 's') { e.preventDefault(); clickBtnText('저장+다음'); }
      else if (cmd && !shift && k.toLowerCase() === 'z') { e.preventDefault(); clickBtnText('Undo'); }
    }, { capture: true });
  } catch(err) {}
})();
</script>
    """,
    height=0,
)

# 테마 override — session_state.theme_choice 기반 (종합 CSS, in-page toggle)
# 새로고침 후에도 마지막 테마 선택 유지 — .last_config.json 에서 read
if "theme_choice" not in st.session_state:
    try:
        import json as _json_th
        _cfg_path_th = Path(__file__).parent / ".last_config.json"
        if _cfg_path_th.exists():
            _saved_th = _json_th.loads(_cfg_path_th.read_text()).get("theme")
            if _saved_th in ("light", "dark"):
                st.session_state.theme_choice = _saved_th
    except Exception:
        pass
_theme_pref = st.session_state.get("theme_choice", "dark")
# theme 분기 icon/link color 변수 — 회색 fixed 회수 일괄 교체용
# light 배경에서 너무 옅은 808080 회색은 가독성 떨어짐 → 더 진한 중성 회색 사용
st.session_state["_icon_default"] = "57606a" if _theme_pref == "light" else "808080"
st.session_state["_link_color"] = "#0969da" if _theme_pref == "light" else "#58a6ff"
if _theme_pref == "light":
    st.markdown(
        """
<style>
:root, html, body {
    --background-color: #ffffff !important;
    --secondary-background-color: #f0f2f6 !important;
    --text-color: #262730 !important;
    color-scheme: light !important;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #ffffff !important; color:#262730 !important;
}
[data-testid="stHeader"] { background: rgba(255,255,255,0.85) !important; }
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div { background-color: #f0f2f6 !important; }
section[data-testid="stSidebar"] *:not(svg):not(path):not(input):not(textarea) { color:#262730 !important; }
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *:not(code):not(svg):not(path) {
    color:#262730 !important;
}
.stTextInput input, .stNumberInput input, .stTextArea textarea,
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea,
.stSelectbox div[data-baseweb="select"], .stMultiSelect div[data-baseweb="select"],
.stSelectbox > div, .stMultiSelect > div {
    background:#ffffff !important; color:#262730 !important; border-color:#d0d7de !important;
}
/* selectbox 내부 텍스트 강제 (data-baseweb 안 잡힐 때 대비) */
.stSelectbox, .stSelectbox *, .stMultiSelect, .stMultiSelect * { color:#262730 !important; }
.stSelectbox div[value], .stMultiSelect div[value] {
    color:#262730 !important; background:transparent !important;
}
.stSelectbox input, .stMultiSelect input { color:#262730 !important; background:#ffffff !important; }
.stSelectbox svg, .stMultiSelect svg { fill:#262730 !important; color:#262730 !important; }
/* 빈 indicator/separator 영역도 흰색으로 (검정 fill 방지) */
.stSelectbox [class*="st-"][class*="st-"] { background-color:transparent !important; }
.stSelectbox > div > div { background:#ffffff !important; }
[data-baseweb="popover"], [data-baseweb="popover"] ul, [data-baseweb="popover"] li,
[data-baseweb="menu"], [data-baseweb="menu"] li {
    background:#ffffff !important; color:#262730 !important;
}
[data-baseweb="menu"] li:hover, [data-baseweb="popover"] li:hover { background:#f0f2f6 !important; }
[data-baseweb="tag"] { background:#eaeef2 !important; color:#262730 !important; }
[data-baseweb="tag"] * { color:#262730 !important; }
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="base-input"] {
    background:#ffffff !important; color:#262730 !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder,
[data-baseweb="input"] input::placeholder { color:#6e7681 !important; opacity:1 !important; }
.stButton button { background:#ffffff !important; color:#262730 !important; border-color:#d0d7de !important; }
[data-testid="stDownloadButton"] button, [data-testid="stDownloadButton"] button * {
    background:#ffffff !important; color:#262730 !important; border-color:#d0d7de !important;
}
[data-testid="stFileUploaderDropzone"] {
    background:#ffffff !important; border:1px dashed #d0d7de !important; color:#262730 !important;
}
[data-testid="stFileUploaderDropzone"] * { color:#262730 !important; }
[data-testid="stFileUploaderDropzone"] svg { fill:#262730 !important; }
[data-testid="stFileUploaderDropzone"] button { background:#f6f8fa !important; color:#262730 !important; border:1px solid #d0d7de !important; }
[data-testid="stFileUploaderDropzone"] small { color:#57606a !important; }
[data-testid="stFileUploader"] section { background:#ffffff !important; }
[data-testid="stFileUploaderFile"], [data-testid="stFileUploaderFileName"] { color:#262730 !important; }
.stButton button[kind="primary"],
.stButton button[kind="primary"] *,
.stButton button[kind="primary"] [data-testid="stMarkdownContainer"],
.stButton button[kind="primary"] [data-testid="stMarkdownContainer"] *,
.stButton button[kind="primary"] [data-testid="stMarkdownContainer"] p {
    color:#ffffff !important; border-color:#ff4b4b !important;
}
.stButton button[kind="primary"] { background:#ff4b4b !important; }
.stButton button[kind="primary"] [data-testid="stMarkdownContainer"] { background:transparent !important; }
[data-testid="stExpander"] details { background:#f7f8fa !important; border-color:#dde !important; }
[data-testid="stCheckbox"] label { color:#262730 !important; }
[data-testid="stRadio"] label { color:#262730 !important; }
[data-testid="stDataFrame"] { background:#fff !important; }
[data-testid="stMetric"] { color:#262730 !important; }
[data-testid="stTabs"] button { color:#262730 !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#ff4b4b !important; }
[data-testid="stProgress"] { background:transparent !important; }
[data-testid="stProgress"] > div { background:transparent !important; }
[data-testid="stProgress"] > div > div { background:#e6ebf1 !important; }
[data-testid="stProgress"] > div > div > div { background:linear-gradient(90deg,#ff4b4b 0%,#ff8a4b 100%) !important; }
/* slider/progress 류 모든 빈 검정 트랙 강제 */
.stApp div[role="progressbar"], .stApp div[role="progressbar"] > div { background:#e6ebf1 !important; }
/* slider thumb / track */
[data-testid="stSlider"] [role="slider"] { background:#ff4b4b !important; }
[data-testid="stSlider"] [data-baseweb="slider"] div[role="presentation"] { background:#d0d7de !important; }
[data-testid="stSlider"] label { color:#262730 !important; }
/* number input */
[data-testid="stNumberInput"] input { color:#262730 !important; background:#ffffff !important; }
[data-testid="stNumberInput"] button { background:#f6f8fa !important; color:#262730 !important; border-color:#d0d7de !important; }
[data-testid="stNumberInput"] button svg { fill:#262730 !important; }
/* code blocks */
[data-testid="stCodeBlock"], [data-testid="stCodeBlock"] pre, [data-testid="stCodeBlock"] code {
    background:#f6f8fa !important; color:#262730 !important;
}
/* tooltip — dark on light background for legibility */
[data-baseweb="tooltip"] { background:#262730 !important; color:#fafafa !important; }
/* spinner */
[data-testid="stSpinner"] { color:#262730 !important; }
[data-testid="stSpinner"] svg { stroke:#ff4b4b !important; }
/* image caption */
[data-testid="stImageCaption"], [data-testid="stImage"] figcaption { color:#262730 !important; opacity:0.7 !important; }
/* dataframe — canvas-glide 은 streamlit 기본 theme 따라가는 한계 있음 (CSS override 한계) */
[data-testid="stDataFrame"], [data-testid="stDataFrame"] [data-testid="data-grid-canvas"] {
    background:#ffffff !important;
}
/* toast — light gray for memory rule (no yellow, prefer gray tone) */
[data-testid="stToast"] {
    background:#f6f8fa !important; color:#262730 !important;
    border:1px solid #d0d7de !important;
}
/* st.dialog (portal로 렌더됨 — 별도 강제) */
[data-testid="stDialog"], [role="dialog"], [data-baseweb="modal"], [data-baseweb="dialog"] {
    background:#ffffff !important; color:#262730 !important;
}
[data-testid="stDialog"] *, [role="dialog"] *, [data-baseweb="modal"] *, [data-baseweb="dialog"] * {
    color:#262730 !important;
}
[data-testid="stDialog"] svg, [role="dialog"] svg { fill:#262730 !important; }
[data-testid="stDialog"] [data-testid="stMarkdownContainer"] *,
[role="dialog"] [data-testid="stMarkdownContainer"] * { color:#262730 !important; }
[data-testid="stDialog"] .stButton button[kind="primary"],
[data-testid="stDialog"] .stButton button[kind="primary"] *,
[role="dialog"] .stButton button[kind="primary"],
[role="dialog"] .stButton button[kind="primary"] * { color:#ffffff !important; }
[data-testid="stAlertContainer"] {
    background:#f6f8fa !important; border:1px solid #d0d7de !important;
    border-left:3px solid #6e7681 !important; border-radius:6px !important;
}
[data-testid="stAlertContainer"] *,
[data-testid="stAlertContainer"] p { color:#262730 !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) { border-left-color:#2da44e !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"])    { border-left-color:#1f6feb !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) { border-left-color:#d68800 !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"])   { border-left-color:#d1242f !important; }
hr { border-color: rgba(0,0,0,0.1) !important; }
code { background:#f0f2f6 !important; color:#262730 !important; }
kbd { background:#e1e4e8 !important; color:#262730 !important; border-color:#d0d7de !important; }
a, a:visited { color:#0969da !important; }
</style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
<style>
:root, html, body {
    --background-color: #0e1117 !important;
    --secondary-background-color: #262730 !important;
    --text-color: #fafafa !important;
    color-scheme: dark !important;
}
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #0e1117 !important; color:#fafafa !important;
}
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div { background-color: #262730 !important; }
/* alert override — gray base + type별 left-border (no yellow, gray-leaning per memory rule) */
[data-testid="stAlertContainer"] {
    background:#161b22 !important; border:1px solid #30363d !important;
    border-left:3px solid #6e7681 !important; border-radius:6px !important;
}
[data-testid="stAlertContainer"] *, [data-testid="stAlertContainer"] p { color:#e6edf3 !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) { border-left-color:#2da44e !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"])    { border-left-color:#1f6feb !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) { border-left-color:#d68800 !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"])   { border-left-color:#d1242f !important; }
/* dark mode 누락 selector 보강 */
code { background:#1c2128 !important; color:#e6edf3 !important; }
kbd { background:#21262d !important; color:#e6edf3 !important; border-color:#30363d !important; }
a, a:visited { color:#58a6ff !important; }
hr { border-color: rgba(255,255,255,0.1) !important; }
/* light CSS counterpart — light→dark 전환 시 잔존하는 light rule 대응 */
section[data-testid="stSidebar"] *:not(svg):not(path):not(input):not(textarea) { color:#fafafa !important; }
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *:not(code):not(svg):not(path) {
    color:#fafafa !important;
}
.stTextInput input, .stNumberInput input, .stTextArea textarea,
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea,
.stSelectbox div[data-baseweb="select"], .stMultiSelect div[data-baseweb="select"],
.stSelectbox > div, .stMultiSelect > div {
    background:#262730 !important; color:#fafafa !important; border-color:#30363d !important;
}
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="base-input"] {
    background:#262730 !important; color:#fafafa !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder,
[data-baseweb="input"] input::placeholder { color:#9aa4b2 !important; opacity:1 !important; }
.stButton button { background:#262730 !important; color:#fafafa !important; border-color:#30363d !important; }
[data-testid="stDownloadButton"] button, [data-testid="stDownloadButton"] button * {
    background:#262730 !important; color:#fafafa !important; border-color:#30363d !important;
}
.stButton button[kind="primary"],
.stButton button[kind="primary"] *,
.stButton button[kind="primary"] [data-testid="stMarkdownContainer"],
.stButton button[kind="primary"] [data-testid="stMarkdownContainer"] *,
.stButton button[kind="primary"] [data-testid="stMarkdownContainer"] p {
    color:#ffffff !important; border-color:#ff4b4b !important;
}
.stButton button[kind="primary"] { background:#ff4b4b !important; }
.stButton button[kind="primary"] [data-testid="stMarkdownContainer"] { background:transparent !important; }
[data-testid="stFileUploaderDropzone"] {
    background:#1c2128 !important; border:1px dashed #30363d !important; color:#e6edf3 !important;
}
[data-testid="stFileUploaderDropzone"] * { color:#e6edf3 !important; }
[data-testid="stFileUploaderDropzone"] svg { fill:#e6edf3 !important; }
[data-testid="stFileUploaderDropzone"] button { background:#262730 !important; color:#e6edf3 !important; border:1px solid #30363d !important; }
[data-testid="stFileUploaderDropzone"] small { color:#9aa4b2 !important; }
[data-testid="stFileUploader"] section { background:#1c2128 !important; }
[data-testid="stFileUploaderFile"], [data-testid="stFileUploaderFileName"] { color:#e6edf3 !important; }
.stSelectbox, .stSelectbox *, .stMultiSelect, .stMultiSelect * { color:#fafafa !important; }
.stSelectbox div[value], .stMultiSelect div[value] { color:#fafafa !important; background:transparent !important; }
.stSelectbox input, .stMultiSelect input { color:#fafafa !important; background:#262730 !important; }
.stSelectbox svg, .stMultiSelect svg { fill:#fafafa !important; color:#fafafa !important; }
.stSelectbox > div > div { background:#262730 !important; }
[data-baseweb="popover"], [data-baseweb="popover"] ul, [data-baseweb="popover"] li,
[data-baseweb="menu"], [data-baseweb="menu"] li {
    background:#262730 !important; color:#fafafa !important;
}
[data-baseweb="menu"] li:hover, [data-baseweb="popover"] li:hover { background:#30363d !important; }
[data-baseweb="tag"] { background:#30363d !important; color:#fafafa !important; }
[data-baseweb="tag"] * { color:#fafafa !important; }
[data-testid="stExpander"] details { background:#161b22 !important; border-color:#30363d !important; }
[data-testid="stExpander"] details summary, [data-testid="stExpander"] details summary * { color:#fafafa !important; }
[data-testid="stCheckbox"] label, [data-testid="stRadio"] label { color:#fafafa !important; }
[data-testid="stMetric"], [data-testid="stMetric"] * { color:#fafafa !important; }
[data-testid="stTabs"] button { color:#9aa4b2 !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#ff4b4b !important; }
[data-testid="stDataFrame"], [data-testid="stDataFrame"] [data-testid="data-grid-canvas"] {
    background:#0e1117 !important;
}
[data-testid="stToast"] { background:#262730 !important; color:#fff !important; border:1px solid #30363d !important; }
[data-testid="stDialog"], [role="dialog"], [data-baseweb="modal"], [data-baseweb="dialog"] {
    background:#161b22 !important; color:#fafafa !important;
}
[data-testid="stDialog"] *, [role="dialog"] *, [data-baseweb="modal"] *, [data-baseweb="dialog"] * { color:#fafafa !important; }
[data-testid="stDialog"] svg, [role="dialog"] svg { fill:#fafafa !important; }
[data-testid="stDialog"] [data-testid="stMarkdownContainer"] *,
[role="dialog"] [data-testid="stMarkdownContainer"] * { color:#fafafa !important; }
[data-testid="stDialog"] .stButton button[kind="primary"],
[data-testid="stDialog"] .stButton button[kind="primary"] *,
[role="dialog"] .stButton button[kind="primary"],
[role="dialog"] .stButton button[kind="primary"] * { color:#ffffff !important; }
[data-testid="stSlider"] [role="slider"] { background:#ff4b4b !important; }
[data-testid="stSlider"] label { color:#fafafa !important; }
[data-testid="stNumberInput"] input { color:#fafafa !important; background:#262730 !important; }
[data-testid="stNumberInput"] button { background:#262730 !important; color:#fafafa !important; border-color:#30363d !important; }
[data-testid="stNumberInput"] button svg { fill:#fafafa !important; }
[data-testid="stCodeBlock"], [data-testid="stCodeBlock"] pre, [data-testid="stCodeBlock"] code {
    background:#1c2128 !important; color:#e6edf3 !important;
}
[data-baseweb="tooltip"] { background:#f6f8fa !important; color:#262730 !important; }
[data-testid="stSpinner"] { color:#fafafa !important; }
[data-testid="stImageCaption"], [data-testid="stImage"] figcaption { color:#fafafa !important; opacity:0.7 !important; }
[data-testid="stProgress"] { background:transparent !important; }
[data-testid="stProgress"] > div { background:transparent !important; }
[data-testid="stProgress"] > div > div { background:#30363d !important; }
[data-testid="stProgress"] > div > div > div { background:linear-gradient(90deg,#ff4b4b 0%,#ff8a4b 100%) !important; }
</style>
        """,
        unsafe_allow_html=True,
    )


def _render_intro_cards() -> None:
    """기능·사용 안내 카드 — 세로 1열. 메인 페이지 진입 전 + 보기 모드 '안내' 양쪽에서 호출."""
    st.markdown(
        """
<style>
.intro-grid { display:grid; grid-template-columns:1fr; gap:12px; margin-top:18px; }
.intro-card { background:rgba(128,128,128,0.08); border:1px solid rgba(128,128,128,0.18);
              border-radius:10px; padding:16px 20px; color:var(--text-color); }
.intro-card h4 { margin:0 0 8px 0; font-size:1.05em; font-weight:600; }
.intro-card p, .intro-card li { font-size:0.9em; opacity:0.85; line-height:1.6; margin:4px 0; }
.intro-card ul { padding-left:20px; margin:6px 0; }
.intro-section-title { font-size:1.1em; font-weight:600; margin:28px 0 8px 0;
                        color:var(--text-color); opacity:0.92; }
.intro-kbd { background:rgba(128,128,128,0.2); border:1px solid rgba(128,128,128,0.4);
             border-radius:4px; padding:1px 8px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
             font-size:0.85em; color:var(--text-color); }
.intro-pill { display:inline-flex; align-items:center; gap:5px;
              padding:2px 10px; border-radius:12px; font-size:0.85em; margin-right:6px; }
</style>

<div class="intro-section-title">시작하기</div>
<div class="intro-grid">
  <div class="intro-card">
    <h4>1. 모델 업로드</h4>
    <p>사이드바에서 Classifier · Segmenter <code>.pt</code> 파일을 업로드하거나, 서버/NAS 경로를 직접 입력하세요.</p>
    <p>두 모델 모두 필요한 건 아닙니다 — 둘 중 하나만 있어도 검수 가능합니다.</p>
  </div>
  <div class="intro-card">
    <h4>2. 이미지 로드</h4>
    <p><b>폴더 경로 입력</b>을 권장합니다. NAS 수백 장도 즉시 로드 가능.</p>
    <p>작은 셋트는 zip 또는 다중 파일 업로드도 지원합니다.</p>
  </div>
  <div class="intro-card">
    <h4>3. 추론 시작</h4>
    <p>사이드바 빨강 <b>추론 시작</b> 버튼 클릭. 자동으로 GPU(MPS/CUDA) 또는 CPU 선택, 진행률 표시 후 검수 화면 진입.</p>
  </div>
</div>

<div class="intro-section-title">검수 작업</div>
<div class="intro-grid">
  <div class="intro-card">
    <h4>판정 (Verdict)</h4>
    <p>이미지마다 세 가지 판정을 줍니다:</p>
    <p><span class="intro-pill" style="background:rgba(45,164,78,0.18);color:#2da44e;"><b>○ 정확</b></span> 모델 결과 그대로 OK</p>
    <p><span class="intro-pill" style="background:rgba(209,36,47,0.18);color:#d1242f;"><b>✕ 오류</b></span> 정정 필요 — 클래스/폴리곤 직접 수정</p>
    <p><span class="intro-pill" style="background:rgba(214,136,0,0.18);color:#d68800;"><b>△ 애매</b></span> 판단 보류 — 다음 라운드에 다시 보기</p>
  </div>
  <div class="intro-card">
    <h4>폴리곤 편집</h4>
    <ul>
      <li><b>정점 이동</b> — 흰 원을 마우스로 드래그</li>
      <li><b>정점 추가</b> — 선분(엣지) hover 시 + 클릭</li>
      <li><b>정점 삭제</b> — 정점 우클릭 또는 정점 선택 후 Delete</li>
      <li><b>새 영역 추가</b> — 툴바 <b>[+ 폴리곤]</b> / <b>[+ 사각형]</b></li>
      <li><b>클래스 변경</b> — 폴리곤 선택 후 드롭다운에서 선택</li>
    </ul>
  </div>
  <div class="intro-card">
    <h4>이동 + 저장</h4>
    <p>판정 + (필요시) 정정 후 <b>저장+다음</b> 클릭. DB에 자동 누적되고 다음 이미지로 이동합니다.</p>
    <p>중간에 멈춰도 OK — 다음 진입 시 마지막 미검수 위치로 자동 점프합니다.</p>
  </div>
</div>

<div class="intro-section-title">자동 복구 + 학습 데이터 Export</div>
<div class="intro-grid">
  <div class="intro-card">
    <h4>새로고침해도 안전</h4>
    <p>브라우저를 닫거나 새로고침해도 마지막 작업 상태가 그대로 복원됩니다.</p>
    <p>모델 + 이미지 + 검수 결과 + 마지막 위치 모두 보존.</p>
  </div>
  <div class="intro-card">
    <h4>Export — 학습 데이터</h4>
    <p>사이드바 <b>보기 모드 → 내보내기</b> 선택 후 Zip 만들기.</p>
    <p>포맷: YOLO Segmentation · YOLO Classification · COCO (3종 동시 가능).</p>
  </div>
  <div class="intro-card">
    <h4>이어가기 / 새로 시작</h4>
    <p>이전 세션 selectbox 로 다른 작업도 이어갈 수 있습니다.</p>
    <p>완전히 새로 시작하려면 <b>새 세션 시작</b> 클릭.</p>
  </div>
</div>

<div class="intro-section-title">단축키 <span style="font-size:0.7em;color:#d68800;background:rgba(214,136,0,0.18);padding:2px 8px;border-radius:10px;margin-left:6px;vertical-align:middle;">검수 필요</span></div>
<div class="intro-grid">
  <div class="intro-card">
    <h4>이미지 이동</h4>
    <p>Windows · macOS 공통 — <span class="intro-kbd">←</span> 이전 &nbsp;·&nbsp; <span class="intro-kbd">→</span> 다음</p>
  </div>
  <div class="intro-card">
    <h4>저장 + 다음</h4>
    <p>Windows — <span class="intro-kbd">Ctrl</span> + <span class="intro-kbd">S</span></p>
    <p>macOS — <span class="intro-kbd">⌘ Cmd</span> + <span class="intro-kbd">S</span></p>
  </div>
  <div class="intro-card">
    <h4>되돌리기 (Undo)</h4>
    <p>Windows — <span class="intro-kbd">Ctrl</span> + <span class="intro-kbd">Z</span></p>
    <p>macOS — <span class="intro-kbd">⌘ Cmd</span> + <span class="intro-kbd">Z</span></p>
    <p style="opacity:0.6;font-size:0.82em;">현재 이미지 직전 상태로 복귀</p>
  </div>
</div>
<div style="margin-top:8px;padding:10px 14px;background:rgba(214,136,0,0.1);border-left:3px solid #d68800;border-radius:6px;font-size:0.85em;color:var(--text-color);opacity:0.85;">
  ⚠ 단축키 기능은 브라우저 환경에 따라 동작이 달라질 수 있어 운영자 검수가 필요합니다.
  input 필드에 포커스가 있을 때는 단축키가 텍스트 입력으로 작동하니, 빈 영역 클릭 후 사용해 주세요.
</div>
        """,
        unsafe_allow_html=True,
    )


def init_state():
    defaults = {
        "classifier_model": None,
        "segmenter_model": None,
        "classifier_name": None,
        "segmenter_name": None,
        "classifier_classes": [],   # 모델 names dict
        "segmenter_classes": {},    # cid → name
        "images": [],               # [(filename, PIL.Image), ...]
        "images_by_filename": {},   # filename → PIL.Image
        "results": [],              # [{filename, classifier, segmenter}, ...]
        "current_idx": 0,
        "device": detect_device(),
        "inference_done": False,
        "session_id": None,
        "session_meta": None,
        "view_mode": "inspect",     # inspect / export
        "sort_mode": "default",
        "autosave_enabled": False,  # default OFF — 매 행동마다 rerun 부담 회피
        "show_roi_panel": True,
        # 자동 복구용 — 마지막 사용 경로 저장
        "last_cls_path": "",
        "last_seg_path": "",
        "last_folder_path": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ★ 마지막 설정 디스크 저장 (streamlit 재시작 후 자동 복구)
_LAST_CFG_PATH = Path(__file__).parent / ".last_config.json"


def save_last_config():
    import json as _json
    cfg = {
        "cls_path": st.session_state.get("last_cls_path", ""),
        "seg_path": st.session_state.get("last_seg_path", ""),
        "folder_path": st.session_state.get("last_folder_path", ""),
        "session_id": st.session_state.get("session_id"),
        "theme": st.session_state.get("theme_choice", "dark"),
    }
    try:
        _LAST_CFG_PATH.write_text(_json.dumps(cfg, indent=2))
    except Exception:
        pass


def load_last_config() -> dict:
    import json as _json
    if not _LAST_CFG_PATH.exists():
        return {}
    try:
        return _json.loads(_LAST_CFG_PATH.read_text())
    except Exception:
        return {}


# 모델 캐시 — streamlit 재시작 후 cache 도 사라지지만 같은 세션 내에선 1회 로드
@st.cache_resource(show_spinner="모델 로드 중...")
def _cached_load_yolo(path: str):
    from inference import load_yolo_model as _load
    return _load(path)


init_state()
init_db()

# 첫 진입 시 last config 자동 복구
if "_last_cfg_loaded" not in st.session_state:
    _last = load_last_config()
    if _last.get("cls_path"):
        st.session_state.last_cls_path = _last["cls_path"]
    if _last.get("seg_path"):
        st.session_state.last_seg_path = _last["seg_path"]
    if _last.get("folder_path"):
        st.session_state.last_folder_path = _last["folder_path"]
    if _last.get("session_id"):
        st.session_state.session_id = _last["session_id"]
    st.session_state._last_cfg_loaded = True


def _run_auto_restore_logic() -> dict:
    """모델/이미지/DB 추론 결과 한 번에 복원. three-body 로더 + progress 위젯."""
    # 복구 진행 중에는 좌측 상단 switch 숨김 (다음 rerun 시 자동 해제)
    st.markdown(
        "<style>#hitl-sb-switch-host { display:none !important; }</style>",
        unsafe_allow_html=True,
    )
    summary = {"cls": None, "seg": None, "imgs": 0, "labels": 0, "errors": []}
    loader = st.empty()

    def _show(title: str, sub: str = ""):
        loader.markdown(
            '<div class="hitl-loader-box">'
            '<div class="three-body"><div class="three-body__dot"></div>'
            '<div class="three-body__dot"></div><div class="three-body__dot"></div></div>'
            f'<div class="hitl-loader-title">{title}</div>'
            + (f'<div class="hitl-loader-sub">{sub}</div>' if sub else '')
            + '<a href="?skip_auto=1" class="hitl-loader-cancel">복구 중단하고 초기 화면으로</a>'
            + '</div>',
            unsafe_allow_html=True,
        )

    _show(
        "인프라엑스 자비스가 복구 중",
        "<b>실수로 새로고침을 누르셨네요!</b><br><br>"
        "걱정 마세요.<br>"
        "마지막 작업을 그대로 이어갈 수 있도록<br>"
        "<b>모델 + 이미지 + 검수 결과</b>를 복원하는 중입니다.",
    )
    prog = st.progress(0.0, text="자동 복구 시작...")

    # 1) Classifier 로드 (0~10%)
    if st.session_state.last_cls_path:
        if os.path.exists(st.session_state.last_cls_path):
            prog.progress(0.02, text="Classifier 모델 로드 중...")
            try:
                m = _cached_load_yolo(st.session_state.last_cls_path)
                st.session_state.classifier_model = m
                name = os.path.basename(st.session_state.last_cls_path)
                st.session_state.classifier_name = name
                if hasattr(m, "names"):
                    st.session_state.classifier_classes = list(m.names.values()) if isinstance(m.names, dict) else list(m.names)
                summary["cls"] = name
            except Exception as e:
                summary["errors"].append(f"Classifier 로드 실패: {e}")
        else:
            summary["errors"].append(f"Classifier 경로 없음: {st.session_state.last_cls_path}")
    prog.progress(0.10, text="Classifier 로드 완료")

    # 2) Segmenter 로드 (10~20%)
    if st.session_state.last_seg_path:
        if os.path.exists(st.session_state.last_seg_path):
            prog.progress(0.12, text="Segmenter 모델 로드 중...")
            try:
                m = _cached_load_yolo(st.session_state.last_seg_path)
                st.session_state.segmenter_model = m
                name = os.path.basename(st.session_state.last_seg_path)
                st.session_state.segmenter_name = name
                if hasattr(m, "names"):
                    nd = m.names if isinstance(m.names, dict) else dict(enumerate(m.names))
                    st.session_state.segmenter_classes = {int(k): v for k, v in nd.items()}
                summary["seg"] = name
            except Exception as e:
                summary["errors"].append(f"Segmenter 로드 실패: {e}")
        else:
            summary["errors"].append(f"Segmenter 경로 없음: {st.session_state.last_seg_path}")
    prog.progress(0.20, text="Segmenter 로드 완료")

    # 3) 이미지 폴더 스캔 + 로드 (20~90%)
    fp = st.session_state.last_folder_path
    n_failed = 0
    if fp:
        folder = Path(fp).expanduser()
        if folder.exists() and folder.is_dir():
            prog.progress(0.22, text="폴더 스캔 중...")
            exts = (".jpg", ".jpeg", ".png")
            paths = [p for p in sorted(folder.rglob("*"))
                     if p.is_file() and p.suffix.lower() in exts]
            n_total = len(paths) or 1
            loaded = []
            for i, p in enumerate(paths):
                try:
                    loaded.append((p.name, Image.open(p).convert("RGB")))
                except Exception:
                    n_failed += 1
                if i % 5 == 0 or i == n_total - 1:
                    ratio = 0.25 + 0.65 * ((i + 1) / n_total)
                    prog.progress(ratio, text=f"이미지 로드 중 {i+1}/{n_total}")
            st.session_state.images = loaded
            st.session_state.images_by_filename = {fn: img for fn, img in loaded}
            summary["imgs"] = len(loaded)
            if n_failed:
                summary["errors"].append(f"이미지 {n_failed}장 로드 실패 (파일 손상 또는 권한)")
            if not loaded:
                summary["errors"].append(f"폴더에 이미지 0장 — 경로 또는 권한 문제: {fp}")
        else:
            summary["errors"].append(f"폴더 없음 또는 접근 불가: {fp}")
    prog.progress(0.92, text="이미지 로드 완료")

    # 4) DB 추론 결과 복원 (92~100%)
    if st.session_state.session_id and st.session_state.images_by_filename:
        prog.progress(0.95, text="DB 검수 결과 복원 중...")
        n = restore_results_from_db(st.session_state.session_id,
                                      st.session_state.images_by_filename)
        summary["labels"] = n
        if n == 0:
            summary["errors"].append(
                f"DB 추론 결과 0건 (세션 #{st.session_state.session_id} 가 삭제됐거나 라벨 없음) — 추론 시작 재실행 필요"
            )
    prog.progress(1.0, text="자동 복구 완료")
    loader.empty()
    return summary


def restore_results_from_db(session_id: int, images_by_filename: dict) -> int:
    """DB에 저장된 모델 추론 결과를 메모리 results 로 복원.

    ★ 폴더 로드된 전체 이미지를 그대로 보존.
       DB 라벨이 있는 이미지만 모델 결과를 채우고, 없는 이미지는 빈 result.
       (사용자가 추론을 다시 누르면 빈 result 도 채워짐)
    Returns: DB 복원 성공한 이미지 수 (모델 결과 있는 것)
    """
    if not session_id or not images_by_filename:
        return 0
    db_labels = get_labels_by_session(session_id)
    db_label_map = {L["image_filename"]: L for L in db_labels}

    # 폴더 로드 순서 그대로 — 전체 이미지 보존
    all_image_items = [(fn, images_by_filename[fn]) for fn in images_by_filename]

    results_all: list[dict] = []
    n_db_restored = 0
    for fn, img in all_image_items:
        L = db_label_map.get(fn)
        cls_r = None
        seg_r = None
        if L:
            if L.get("cls_model_top"):
                cls_r = {
                    "top_class": L["cls_model_top"],
                    "top_conf": L.get("cls_model_conf") or 0.0,
                    "all_probs": [],
                }
            seg_polys_db = L.get("seg_model_polygons") or []
            if seg_polys_db:
                polygons, class_ids, class_names, confs, boxes = [], [], [], [], []
                for p in seg_polys_db:
                    poly = p.get("polygon", [])
                    polygons.append([(float(x), float(y)) for x, y in poly])
                    class_ids.append(int(p.get("class_id", 0)))
                    class_names.append(p.get("class_name") or f"class_{p.get('class_id', 0)}")
                    confs.append(float(p.get("conf", 0.0)))
                    if len(poly) >= 3:
                        xs = [pt[0] for pt in poly]
                        ys = [pt[1] for pt in poly]
                        boxes.append([min(xs), min(ys), max(xs), max(ys)])
                    else:
                        boxes.append([0, 0, 0, 0])
                names_d = L.get("seg_names_dict") or {}
                seg_r = {
                    "polygons": polygons,
                    "class_ids": class_ids,
                    "class_names": class_names,
                    "confs": confs,
                    "boxes": boxes,
                    "names_dict": {int(k): v for k, v in names_d.items()} if names_d else {},
                }
            if cls_r or seg_r:
                n_db_restored += 1
        results_all.append({"filename": fn, "classifier": cls_r, "segmenter": seg_r})

    # 전체 이미지 + 전체 results 둘 다 보존
    st.session_state.images = all_image_items
    st.session_state.results = results_all
    st.session_state.inference_done = True

    # ★ 검수 안 끝난 첫 이미지로 자동 점프 (작업 이어가기)
    first_pending = None
    for idx, (fn, _img) in enumerate(all_image_items):
        L = db_label_map.get(fn)
        if not L or not L.get("inspected_at"):
            first_pending = idx
            break
    if first_pending is None:
        first_pending = max(0, len(all_image_items) - 1)
    st.session_state.current_idx = first_pending
    return n_db_restored


# ★ 사용자 명시 "마지막 설정 자동 로드" 클릭 — 메인 페이지 새로고침 화면으로 진입
if st.session_state.pop("_pending_manual_restore", False):
    _restore_summary = _run_auto_restore_logic()
    st.session_state._last_restore_summary = _restore_summary
    st.rerun()

# ★ 새로고침 후 자동 복구 — 한 번만 시도 (path 있고 모델 미로드 상태에서).
# URL 에 ?skip_auto=1 붙이면 자동 복구 건너뜀 (자동 복구가 멈췄을 때 탈출용)
_skip_auto_restore = st.query_params.get("skip_auto") == "1"
if "_auto_restore_attempted" not in st.session_state:
    st.session_state._auto_restore_attempted = True
    _has_paths = bool(
        st.session_state.get("last_cls_path") or
        st.session_state.get("last_seg_path") or
        st.session_state.get("last_folder_path")
    )
    if ((not _skip_auto_restore)
            and _has_paths
            and st.session_state.get("session_id") is not None
            and st.session_state.classifier_model is None
            and st.session_state.segmenter_model is None):
        _restore_summary = _run_auto_restore_logic()
        st.session_state._last_restore_summary = _restore_summary
        st.rerun()
    elif _skip_auto_restore:
        st.session_state._skip_auto_notice = True


# ============================================================
# Sidebar — 세션 + 업로드 + 실행
# ============================================================
_icon_default = st.session_state.get("_icon_default", "808080")
_link_color = st.session_state.get("_link_color", "#58a6ff")
_link_color_hex = _link_color.lstrip("#")  # _bi_icon expects hex without '#'

with st.sidebar:
    st.markdown(
        f'<h2 style="margin-bottom:2px;">{_bi_icon("patch-check-fill", "8b5cf6", 24)} AI 추론 검수 스튜디오</h2>'
        f'<div style="opacity:0.7;font-size:0.82em;margin-bottom:8px;">모델 결과 운영자 정정 → 학습 데이터 누적</div>',
        unsafe_allow_html=True,
    )
    _dev_label = {
        "cuda": "NVIDIA GPU (cuda)",
        "mps":  "Apple GPU (mps)",
        "cpu":  "CPU",
    }.get(st.session_state.device, st.session_state.device)
    st.caption(
        f"추론 디바이스: **{_dev_label}** (자동 선택) · "
        f"DB: `{DB_PATH.name}`"
    )

    # 테마 토글 — 클릭 1번으로 light/dark 전환 (CSS override 강제)
    if "theme_choice" not in st.session_state:
        st.session_state.theme_choice = "dark"
    _is_light = st.session_state.theme_choice == "light"
    if st.button(
        ("Dark 테마로 전환" if _is_light else "Light 테마로 전환"),
        use_container_width=True, key="theme_toggle_btn",
    ):
        st.session_state.theme_choice = "dark" if _is_light else "light"
        save_last_config()
        st.rerun()

    # ── 마지막 설정 자동 로드 (최상단 — 검수 시작 전 한 번에 복구)
    if (st.session_state.get("last_cls_path") or st.session_state.get("last_seg_path") or
            st.session_state.get("last_folder_path")):
        st.markdown(
            f'<div style="background:rgba(31,111,235,0.08);padding:6px 10px;border-radius:6px;'
            f'border-left:3px solid #1f6feb;color:var(--text-color);font-size:0.82em;margin:6px 0;">'
            f'{_bi_icon("arrow-clockwise",_link_color_hex,12)} &nbsp;마지막 작업 이어가기 — 모델 + 이미지 + DB 검수 결과까지 한 번에 복원 (새로고침 시 자동 시도됨)'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("마지막 설정 자동 로드", use_container_width=True, type="primary",
                      key="auto_restore_top_btn"):
            # flag 만 set, 다음 rerun 의 main 영역에서 로더 화면으로 실행
            st.session_state._pending_manual_restore = True
            st.rerun()

    # ── 작업 ↔ 안내 view-mode 토글 (button 으로 — st.button 은 streamlit 보장 동작)
    st.divider()
    _is_guide = st.session_state.view_mode == "guide"
    _btn_label = "← 작업 화면으로 돌아가기" if _is_guide else "기능 설명 페이지 보기"
    if st.button(_btn_label, use_container_width=True, key="vm_toggle_btn"):
        if _is_guide:
            st.session_state.view_mode = st.session_state.get("_prev_view_mode", "inspect")
        else:
            st.session_state._prev_view_mode = st.session_state.view_mode
            st.session_state.view_mode = "guide"
        st.rerun()

    # 세션 관리
    st.divider()
    st.subheader("세션")
    sessions = list_sessions()

    # 현재 활성 세션
    current_sid = st.session_state.session_id
    if current_sid:
        cur_info = next((s for s in sessions if s["id"] == current_sid), None)
        if cur_info:
            st.success(
                f"활성 세션 **#{current_sid}** · "
                f"{cur_info['n_inspected']}/{cur_info['n_images']} 검수"
            )
        else:
            st.warning(f"활성 세션 #{current_sid} (DB 정보 없음 — 추론 시 새로 생성)")
    else:
        st.info("활성 세션 없음 — 추론 시작 시 자동 생성")

    # 이전 세션 로드 (있을 때만)
    if sessions:
        session_options = ["(선택 안 함)"] + [
            f"#{s['id']} [{s['started_at'][:16]}] {s['n_inspected']}/{s['n_images']} 검수"
            for s in sessions
        ]
        # 기본 인덱스 = 현재 활성 세션
        default_idx = 0
        if current_sid:
            for i, s in enumerate(sessions):
                if s["id"] == current_sid:
                    default_idx = i + 1
                    break
        sel = st.selectbox(
            "이전 세션 이어서 진행",
            session_options,
            index=default_idx,
            key="session_select",
            label_visibility="collapsed",
        )

        # 다중 선택 삭제
        with st.expander("세션 삭제 (다중 선택 가능)", expanded=False):
            del_format = {
                s["id"]: f"#{s['id']} [{s['started_at'][:16]}] {s['n_inspected']}/{s['n_images']} 검수"
                for s in sessions
            }
            sel_to_delete = st.multiselect(
                "삭제할 세션",
                options=list(del_format.keys()),
                format_func=lambda sid: del_format.get(sid, str(sid)),
                key="sessions_to_delete",
                label_visibility="collapsed",
                placeholder="삭제할 세션 선택...",
            )
            if st.button(
                f"선택한 {len(sel_to_delete)}개 세션 삭제" if sel_to_delete else "삭제할 세션을 선택하세요",
                use_container_width=True,
                disabled=not sel_to_delete,
                type="primary" if sel_to_delete else "secondary",
                help="선택된 세션을 영구 삭제 (라벨 + history 포함)",
                key="sidebar_session_multidelete_btn",
            ):
                st.session_state["_pending_delete_sids"] = list(sel_to_delete)

        if sel != "(선택 안 함)":
            sid_loaded = int(sel.split()[0].lstrip("#"))
            if sid_loaded != current_sid:
                st.session_state.session_id = sid_loaded
                st.session_state.results = []
                st.session_state.current_idx = 0
                st.session_state.inference_done = False

                # ★ 세션 record에 경로 저장돼 있으면 완전 자동 복구
                with st.spinner(f"세션 #{sid_loaded} 이어가기 — 모델·이미지·추론 결과 복구 중..."):
                    sess_rec = get_session(sid_loaded) or {}
                    # 기존 세션 (구 스키마) 은 path NULL — last_config 의 값으로 fallback
                    cls_p = sess_rec.get("cls_path") or st.session_state.get("last_cls_path") or ""
                    seg_p = sess_rec.get("seg_path") or st.session_state.get("last_seg_path") or ""
                    fld_p = sess_rec.get("folder_path") or st.session_state.get("last_folder_path") or ""

                    # 1) 모델 로드 (없거나 경로 바뀌었으면)
                    if cls_p and os.path.exists(cls_p):
                        if st.session_state.get("last_cls_path") != cls_p or st.session_state.classifier_model is None:
                            try:
                                m = _cached_load_yolo(cls_p)
                                st.session_state.classifier_model = m
                                st.session_state.classifier_name = os.path.basename(cls_p)
                                if hasattr(m, "names"):
                                    st.session_state.classifier_classes = list(m.names.values()) if isinstance(m.names, dict) else list(m.names)
                                st.session_state.last_cls_path = cls_p
                            except Exception as e:
                                st.warning(f"Classifier 로드 실패: {e}")
                    if seg_p and os.path.exists(seg_p):
                        if st.session_state.get("last_seg_path") != seg_p or st.session_state.segmenter_model is None:
                            try:
                                m = _cached_load_yolo(seg_p)
                                st.session_state.segmenter_model = m
                                st.session_state.segmenter_name = os.path.basename(seg_p)
                                if hasattr(m, "names"):
                                    nd = m.names if isinstance(m.names, dict) else dict(enumerate(m.names))
                                    st.session_state.segmenter_classes = {int(k): v for k, v in nd.items()}
                                st.session_state.last_seg_path = seg_p
                            except Exception as e:
                                st.warning(f"Segmenter 로드 실패: {e}")

                    # 2) 이미지 로드
                    if fld_p:
                        folder = Path(fld_p).expanduser()
                        if folder.exists() and folder.is_dir():
                            exts = (".jpg", ".jpeg", ".png")
                            loaded = []
                            for p in sorted(folder.rglob("*")):
                                if p.is_file() and p.suffix.lower() in exts:
                                    try:
                                        loaded.append((p.name, Image.open(p).convert("RGB")))
                                    except Exception:
                                        pass
                            st.session_state.images = loaded
                            st.session_state.images_by_filename = {fn: img for fn, img in loaded}
                            st.session_state.last_folder_path = fld_p

                    # 3) 추론 결과 DB 복원
                    if st.session_state.images_by_filename:
                        n = restore_results_from_db(sid_loaded,
                                                      st.session_state.images_by_filename)
                        st.toast(f"세션 #{sid_loaded} 복구 완료: 모델 + {len(st.session_state.images)}장 이미지 + 추론 {n}장")
                    else:
                        st.toast(f"세션 #{sid_loaded} 활성. 이미지 경로 없거나 폴더 비어 — 수동 로드 필요")
                save_last_config()
                st.rerun()

    # 세션 삭제 확인 dialog (다중)
    _pending_sids = st.session_state.get("_pending_delete_sids") or []
    if _pending_sids:
        @st.dialog(f"세션 {len(_pending_sids)}개 삭제 확인")
        def _confirm_session_delete():
            sids = list(st.session_state.get("_pending_delete_sids") or [])
            if not sids:
                return
            rows = []
            total_labels = 0
            for sid_d in sids:
                sess_rec = get_session(sid_d) or {}
                n_labels = len(get_labels_by_session(sid_d))
                total_labels += n_labels
                rows.append(
                    f"- #{sid_d} ({sess_rec.get('started_at', '')[:16]}) · 라벨 {n_labels}건"
                )
            st.markdown(
                "다음 세션을 영구 삭제합니다:<br>"
                + "<br>".join(rows)
                + f"<br><br>총 라벨 <b>{total_labels}</b>건 + 검수 history 가 함께 삭제됩니다."
                + "<br><span style='color:#d1242f;font-weight:600;'>되돌릴 수 없습니다.</span>",
                unsafe_allow_html=True,
            )
            cd1, cd2 = st.columns(2)
            with cd1:
                if st.button("취소", use_container_width=True, key="confirm_del_cancel"):
                    st.session_state["_pending_delete_sids"] = None
                    st.rerun()
            with cd2:
                if st.button(
                    f"{len(sids)}개 삭제 진행",
                    use_container_width=True,
                    type="primary",
                    key="confirm_del_ok",
                ):
                    n_del = delete_sessions(sids)
                    if st.session_state.session_id in sids:
                        st.session_state.session_id = None
                        st.session_state.inference_done = False
                        st.session_state.results = []
                        st.session_state.current_idx = 0
                    st.session_state["_pending_delete_sids"] = None
                    st.session_state["sessions_to_delete"] = []
                    st.toast(f"세션 {n_del}개 삭제 완료")
                    st.rerun()
        _confirm_session_delete()

    # 명시적 새 세션 시작 버튼 — 완전 초기화
    if st.button("새 세션 시작", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.inference_done = False
        st.session_state.results = []
        st.session_state.current_idx = 0
        st.session_state.images = []
        st.session_state.images_by_filename = {}
        # 모델/경로 입력 필드도 비움
        st.session_state.last_cls_path = ""
        st.session_state.last_seg_path = ""
        st.session_state.last_folder_path = ""
        st.session_state.cls_path = ""
        st.session_state.seg_path = ""
        st.session_state.folder_path = ""
        st.session_state.classifier_model = None
        st.session_state.segmenter_model = None
        st.session_state.classifier_name = None
        st.session_state.segmenter_name = None
        # .last_config.json 도 모든 path + session_id 비움
        save_last_config()
        st.toast("새 세션 시작 — 모든 경로/모델/세션 초기화됨.")
        st.rerun()
        st.rerun()

    # 모델 업로드
    st.divider()
    st.subheader("1. 모델 업로드")

    cls_upload = st.file_uploader("Classifier (.pt)", type=["pt"], key="cls_up")
    cls_path_input = st.text_input("Classifier 경로", key="cls_path",
                                   value=st.session_state.get("last_cls_path", ""),
                                   placeholder="/path/to/classifier.pt",
                                   label_visibility="collapsed")

    seg_upload = st.file_uploader("Segmenter (.pt)", type=["pt"], key="seg_up")
    seg_path_input = st.text_input("Segmenter 경로", key="seg_path",
                                   value=st.session_state.get("last_seg_path", ""),
                                   placeholder="/path/to/segmenter.pt",
                                   label_visibility="collapsed")

    if st.button("모델 로드", use_container_width=True):
        with st.spinner("모델 로드 중..."):
            try:
                # Classifier
                if cls_upload is not None:
                    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
                    tmp.write(cls_upload.read()); tmp.close()
                    m = load_yolo_model(tmp.name)
                    st.session_state.classifier_model = m
                    st.session_state.classifier_name = cls_upload.name
                    if hasattr(m, "names"):
                        st.session_state.classifier_classes = list(m.names.values()) if isinstance(m.names, dict) else list(m.names)
                elif cls_path_input.strip() and os.path.exists(cls_path_input.strip()):
                    p = cls_path_input.strip()
                    m = load_yolo_model(p)
                    st.session_state.classifier_model = m
                    st.session_state.classifier_name = os.path.basename(p)
                    if hasattr(m, "names"):
                        st.session_state.classifier_classes = list(m.names.values()) if isinstance(m.names, dict) else list(m.names)

                # Segmenter
                if seg_upload is not None:
                    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
                    tmp.write(seg_upload.read()); tmp.close()
                    m = load_yolo_model(tmp.name)
                    st.session_state.segmenter_model = m
                    st.session_state.segmenter_name = seg_upload.name
                    if hasattr(m, "names"):
                        nd = m.names if isinstance(m.names, dict) else dict(enumerate(m.names))
                        st.session_state.segmenter_classes = {int(k): v for k, v in nd.items()}
                elif seg_path_input.strip() and os.path.exists(seg_path_input.strip()):
                    p = seg_path_input.strip()
                    m = load_yolo_model(p)
                    st.session_state.segmenter_model = m
                    st.session_state.segmenter_name = os.path.basename(p)
                    if hasattr(m, "names"):
                        nd = m.names if isinstance(m.names, dict) else dict(enumerate(m.names))
                        st.session_state.segmenter_classes = {int(k): v for k, v in nd.items()}

                if st.session_state.classifier_model or st.session_state.segmenter_model:
                    st.success("모델 로드 완료")
                    # 마지막 경로 저장
                    if cls_path_input.strip():
                        st.session_state.last_cls_path = cls_path_input.strip()
                    if seg_path_input.strip():
                        st.session_state.last_seg_path = seg_path_input.strip()
                    save_last_config()
            except Exception as e:
                st.error(f"모델 로드 실패: {e}")

    if st.session_state.classifier_name:
        st.caption(f"✓ classifier: `{st.session_state.classifier_name}`  ({len(st.session_state.classifier_classes)} 클래스)")
    if st.session_state.segmenter_name:
        st.caption(f"✓ segmenter: `{st.session_state.segmenter_name}`  ({len(st.session_state.segmenter_classes)} 클래스)")

    # 이미지 업로드
    st.divider()
    st.subheader("2. 이미지 업로드")

    st.markdown(
        f'<div style="background:rgba(128,128,128,0.08);padding:10px 14px;border-radius:8px;'
        f'border-left:4px solid #2da44e;margin-bottom:8px;">'
        f'<div style="color:#2da44e;font-weight:600;margin-bottom:6px;">'
        f'{_bi_icon("folder2-open", "2da44e", 16)} &nbsp;폴더 경로 권장</div>'
        f'<table style="width:100%;font-size:0.88em;color:var(--text-color);">'
        f'<tr style="opacity:0.65;"><td></td><td style="text-align:center;">파일 업로드</td>'
        f'<td style="text-align:center;color:#2da44e;opacity:1;">폴더 경로</td></tr>'
        f'<tr><td style="opacity:0.65;">속도</td><td style="text-align:center;">전송 대기</td>'
        f'<td style="text-align:center;color:#2da44e;">즉시</td></tr>'
        f'<tr><td style="opacity:0.65;">한도</td><td style="text-align:center;">~100MB 안정</td>'
        f'<td style="text-align:center;color:#2da44e;">무제한</td></tr>'
        f'<tr><td style="opacity:0.65;">메모리</td><td style="text-align:center;">2배 사용</td>'
        f'<td style="text-align:center;color:#2da44e;">읽기 전용</td></tr>'
        f'<tr><td style="opacity:0.65;">NAS 직접</td><td style="text-align:center;">불가</td>'
        f'<td style="text-align:center;color:#2da44e;">가능</td></tr>'
        f'</table></div>',
        unsafe_allow_html=True,
    )

    folder_path_input = st.text_input(
        "폴더 경로",
        key="folder_path",
        value=st.session_state.get("last_folder_path", ""),
        placeholder="/mnt/nas_data/images",
        help="하위 폴더까지 재귀로 jpg/jpeg/png 전부 로드. NAS 경로 직접 입력 가능.",
    )

    with st.expander(f"파일 업로드 (소규모, 100MB 이내)"):
        img_uploads = st.file_uploader(
            "파일/Zip 업로드",
            type=["jpg", "jpeg", "png", "zip"],
            accept_multiple_files=True,
            key="img_up",
            label_visibility="collapsed",
        )

    if st.button("이미지 로드", use_container_width=True):
        loaded = []
        load_errors = []
        # 1) 파일 업로드 처리
        for up in img_uploads or []:
            name = up.name.lower()
            if name.endswith(".zip"):
                try:
                    with zipfile.ZipFile(BytesIO(up.read())) as zf:
                        for zname in zf.namelist():
                            if zname.lower().endswith((".jpg", ".jpeg", ".png")):
                                try:
                                    img = Image.open(BytesIO(zf.read(zname))).convert("RGB")
                                    loaded.append((os.path.basename(zname), img))
                                except Exception as e:
                                    load_errors.append(f"zip {zname}: {e}")
                except Exception as e:
                    load_errors.append(f"zip 열기 실패: {e}")
            else:
                try:
                    img = image_from_uploaded(up)
                    loaded.append((up.name, img))
                except Exception as e:
                    load_errors.append(f"{up.name}: {e}")
        # 2) 폴더 경로 처리
        fp = folder_path_input.strip()
        if fp:
            folder = Path(fp).expanduser()
            if not folder.exists():
                st.error(f"폴더 없음: {folder}")
            elif not folder.is_dir():
                st.error(f"폴더가 아님: {folder}")
            else:
                exts = (".jpg", ".jpeg", ".png")
                # 1단계: 파일 목록 수집 (PermissionError 캐치 + 진행 표시)
                try:
                    with st.spinner(f"{folder.name} 폴더 스캔 중... (NAS는 시간 걸림)"):
                        all_files = []
                        for p in folder.rglob("*"):
                            if p.is_file() and p.suffix.lower() in exts:
                                all_files.append(p)
                        all_files.sort()
                    n_found = len(all_files)
                    if n_found == 0:
                        st.markdown(
                            f'<div style="background:rgba(214,136,0,0.1);padding:8px 12px;'
                            f'border-left:4px solid #d68800;border-radius:6px;color:#d68800;">'
                            f'{_bi_icon("exclamation-triangle-fill","d68800",16)} &nbsp;'
                            f'폴더는 있지만 jpg/jpeg/png 파일 없음: <code>{folder}</code></div>',
                            unsafe_allow_html=True,
                        )
                except PermissionError as e:
                    st.error(f"권한 차단: {e} — Python 권한 부여 + streamlit 재시작 필요")
                    all_files = []
                    n_found = 0
                except Exception as e:
                    st.error(f"폴더 스캔 실패: {type(e).__name__}: {e}")
                    all_files = []
                    n_found = 0

                # 2단계: 이미지 로드 (진행바)
                if all_files:
                    progress = st.progress(0, text=f"이미지 로드 중 0/{n_found}")
                    for i, p in enumerate(all_files):
                        try:
                            img = Image.open(p).convert("RGB")
                            loaded.append((p.name, img))
                        except Exception as e:
                            load_errors.append(f"{p.name}: {e}")
                        if (i + 1) % max(1, n_found // 50) == 0 or i == n_found - 1:
                            progress.progress((i + 1) / n_found,
                                              text=f"이미지 로드 중 {i+1}/{n_found}")
                    progress.empty()
        st.session_state.images = loaded
        st.session_state.images_by_filename = {fn: img for fn, img in loaded}
        st.session_state.results = []
        st.session_state.current_idx = 0
        st.session_state.inference_done = False
        if loaded:
            st.markdown(
                f'<div style="background:rgba(45,164,78,0.12);padding:8px 12px;'
                f'border-left:4px solid #2da44e;border-radius:6px;color:#2da44e;font-weight:600;">'
                f'{_bi_icon("check-circle-fill","2da44e",16)} &nbsp;'
                f'{len(loaded)}장 로드 완료</div>',
                unsafe_allow_html=True,
            )
            if fp:
                st.session_state.last_folder_path = fp
                save_last_config()
            if load_errors:
                with st.expander(f"일부 파일 실패 ({len(load_errors)}건)"):
                    for err in load_errors[:20]:
                        st.caption(err)
            if st.session_state.session_id:
                n = restore_results_from_db(
                    st.session_state.session_id,
                    st.session_state.images_by_filename,
                )
                if n > 0:
                    st.markdown(
                        f'<div style="background:rgba(31,111,235,0.12);padding:8px 12px;'
                        f'border-left:4px solid #1f6feb;border-radius:6px;color:{_link_color};margin-top:6px;">'
                        f'{_bi_icon("arrow-clockwise",_link_color_hex,16)} &nbsp;'
                        f'세션 #{st.session_state.session_id} 추론 결과 {n}장 자동 복원 — 즉시 검수 가능</div>',
                        unsafe_allow_html=True,
                    )
        else:
            if not fp and not img_uploads:
                msg = "폴더 경로도 안 적었고 파일도 업로드 안 함"
            elif fp:
                msg = f"로드된 이미지 0장 — 경로 확인: <code>{fp}</code>"
            else:
                msg = "로드된 이미지 없음"
            st.markdown(
                f'<div style="background:rgba(214,136,0,0.1);padding:8px 12px;'
                f'border-left:4px solid #d68800;border-radius:6px;color:#d68800;">'
                f'{_bi_icon("exclamation-triangle-fill","d68800",16)} &nbsp;{msg}</div>',
                unsafe_allow_html=True,
            )
            if load_errors:
                with st.expander(f"에러 {len(load_errors)}건"):
                    for err in load_errors[:20]:
                        st.caption(err)

    if st.session_state.images:
        st.caption(f"✓ 이미지 {len(st.session_state.images)}장")

    # 추론 실행
    st.divider()
    st.subheader("3. 추론 실행")
    if st.button("추론 시작", use_container_width=True, type="primary"):
        if not st.session_state.images:
            st.error("이미지 먼저 업로드")
        elif not (st.session_state.classifier_model or st.session_state.segmenter_model):
            st.error("모델 먼저 로드")
        else:
            # 세션 생성 (없으면) — 경로도 같이 저장 → 나중에 세션 선택만으로 완전 복구
            if st.session_state.session_id is None:
                sid = create_session(
                    st.session_state.classifier_name,
                    st.session_state.segmenter_name,
                    note="auto-created",
                    cls_path=st.session_state.get("last_cls_path") or None,
                    seg_path=st.session_state.get("last_seg_path") or None,
                    folder_path=st.session_state.get("last_folder_path") or None,
                )
                st.session_state.session_id = sid
            results = []
            progress = st.progress(0, text="추론 중...")
            t0 = time.time()
            for i, (fn, img) in enumerate(st.session_state.images):
                cls_r = None
                seg_r = None
                if st.session_state.classifier_model is not None:
                    try:
                        cls_r = predict_classifier(st.session_state.classifier_model, img,
                                                   device=st.session_state.device)
                    except Exception as e:
                        cls_r = {"error": str(e)}
                if st.session_state.segmenter_model is not None:
                    try:
                        seg_r = predict_segmenter(st.session_state.segmenter_model, img,
                                                  device=st.session_state.device)
                    except Exception as e:
                        seg_r = {"error": str(e)}
                results.append({"filename": fn, "classifier": cls_r, "segmenter": seg_r})

                # DB: 추론 결과만 저장 (human 정정 기록 보존)
                upsert_inference_result(
                    session_id=st.session_state.session_id,
                    image_filename=fn,
                    image_w=img.size[0], image_h=img.size[1],
                    cls_model_top=(cls_r or {}).get("top_class") if cls_r else None,
                    cls_model_conf=(cls_r or {}).get("top_conf") if cls_r else None,
                    seg_model_polygons=[
                        {"class_id": int(cid), "class_name": cname, "polygon": [[float(x), float(y)] for x, y in poly], "conf": float(c)}
                        for poly, cid, cname, c in zip(
                            (seg_r or {}).get("polygons", []),
                            (seg_r or {}).get("class_ids", []),
                            (seg_r or {}).get("class_names", []),
                            (seg_r or {}).get("confs", []),
                        )
                    ] if seg_r and "error" not in seg_r else None,
                    seg_names_dict={str(k): v for k, v in (seg_r or {}).get("names_dict", {}).items()} if seg_r and "error" not in seg_r else None,
                )

                progress.progress((i + 1) / len(st.session_state.images),
                                  text=f"추론 중 {i+1}/{len(st.session_state.images)}")
            st.session_state.results = results
            st.session_state.inference_done = True
            elapsed = time.time() - t0
            st.success(f"추론 완료 — {len(results)}장, {elapsed:.1f}초 (평균 {elapsed/len(results):.2f}s/img)")
            save_last_config()

    # 보기 모드 + 검수 옵션 — 추론 시작 후에만 노출 (자동 저장은 비활성)
    st.session_state.autosave_enabled = False
    if st.session_state.inference_done:
        st.divider()
        st.subheader("보기 모드")
        # index 동기화 + on_change callback — 사용자 실제 클릭에서만 view_mode 갱신
        # (auto-rerun 시 view_mode 덮어쓰기 방지 — toggle button 으로 set 한 "guide" 안전 보존)
        _mode_keys = ["inspect", "stats", "export"]
        _mode_vals = ["검수", "통계", "내보내기"]
        _cur_idx = (_mode_keys.index(st.session_state.view_mode)
                     if st.session_state.view_mode in _mode_keys else 0)

        def _on_view_mode_change():
            sel = st.session_state.get("view_mode_radio")
            for k, v in zip(_mode_keys, _mode_vals):
                if v == sel:
                    st.session_state.view_mode = k
                    break

        st.radio(
            "보기 모드",
            _mode_vals,
            index=_cur_idx,
            horizontal=True,
            key="view_mode_radio",
            on_change=_on_view_mode_change,
            label_visibility="collapsed",
        )
        mode = st.session_state.view_mode if st.session_state.view_mode in _mode_keys else "inspect"

        if mode == "inspect":
            st.divider()
            st.subheader("검수 옵션")
            prev_sort = st.session_state.sort_mode
            sort_label = st.selectbox(
                "정렬",
                options=list(SORT_OPTIONS.values()),
                index=list(SORT_OPTIONS.keys()).index(st.session_state.sort_mode),
                key="sort_mode_select",
            )
            new_sort = next(k for k, v in SORT_OPTIONS.items() if v == sort_label)
            if new_sort != prev_sort:
                st.session_state.sort_mode = new_sort
                st.session_state.current_idx = 0
                st.rerun()

            st.session_state.show_roi_panel = st.checkbox(
                "폴리곤 ROI 크롭 뷰",
                value=st.session_state.show_roi_panel,
                help="작은 폴리곤을 확대해서 옆에 표시. 미세 결함 검수 시 사용.",
            )


# ============================================================
# Main
# ============================================================
st.markdown(
    f'<h1 style="margin-bottom:2px;">{_bi_icon("patch-check-fill", "8b5cf6", 32)} AI 검수 — Classifier + Segmenter</h1>'
    f'<div style="opacity:0.7;font-size:0.9em;margin-bottom:12px;">'
    f'운영자가 모델 추론 결과를 확인 · 잘못된 부분 정정 → 재학습 데이터로 자동 누적'
    f'</div>',
    unsafe_allow_html=True,
)

# 자동 복구 결과 영구 표시 (사용자가 dismiss 할 때까지)
# Skip flag 으로 자동 복구 건너뛴 경우 안내
if st.session_state.pop("_skip_auto_notice", False):
    st.warning(
        "자동 복구를 건너뛰었습니다 (URL `?skip_auto=1`). "
        "사이드바의 **마지막 설정 자동 로드** 버튼으로 수동 복구 가능합니다. "
        "다시 자동으로 작동하려면 URL 에서 `?skip_auto=1` 을 제거하고 새로고침하세요."
    )

_restore_summary = st.session_state.pop("_last_restore_summary", None)
if _restore_summary:
    s = _restore_summary
    lines = [
        f"• Classifier: **{s['cls'] or '없음'}**",
        f"• Segmenter: **{s['seg'] or '없음'}**",
        f"• 이미지: **{s['imgs']}장**",
        f"• 추론 결과: **{s['labels']}건**",
    ]
    if s["errors"]:
        st.error("자동 복구 — 일부 실패\n\n" + "\n".join(lines) + "\n\n**문제:**\n" + "\n".join(f"• {e}" for e in s["errors"]))
    else:
        st.success("자동 복구 완료\n\n" + "\n".join(lines))

if not st.session_state.inference_done:
    st.info("⬅ 사이드바에서 (1) 모델 로드 → (2) 이미지 로드 → (3) **추론 시작** 버튼")
    if st.session_state.session_id:
        st.markdown(f"현재 세션: **#{st.session_state.session_id}**")
        prev_labels = get_labels_by_session(st.session_state.session_id)
        if prev_labels:
            st.markdown(f"이전 라벨 {len(prev_labels)}건 저장됨 (모델 + 이미지 다시 로드 후 추론 시 자동 매핑)")

    _render_intro_cards()
    st.stop()


# ============================================================
# Mode: guide — 기능·사용 안내 (검수 도중에도 토글 가능)
# ============================================================
if st.session_state.view_mode == "guide":
    _render_intro_cards()
    st.stop()


# ============================================================
# Mode: stats — 검수 통계 대시보드
# ============================================================
if st.session_state.view_mode == "stats":
    st.markdown(f'<h3>{_bi_icon("bar-chart-fill", _icon_default, 22)} 세션 통계</h3>',
                unsafe_allow_html=True)
    if not st.session_state.session_id:
        st.warning("활성 세션 없음")
        st.stop()

    import pandas as pd
    from collections import Counter

    stats = session_stats(st.session_state.session_id)
    all_labels = get_labels_by_session(st.session_state.session_id)

    cols = st.columns(5)
    cols[0].metric("전체", stats.get("total") or 0)
    cols[1].metric("검수 완료", stats.get("inspected") or 0)
    cols[2].metric("정상", stats.get("correct") or 0,
                   delta=f"-{stats.get('wrong') or 0} wrong",
                   delta_color="inverse")
    cols[3].metric("정정 필요", stats.get("wrong") or 0)
    cols[4].metric("애매", stats.get("uncertain") or 0)

    total = (stats.get("total") or 0) or 1
    inspected = stats.get("inspected") or 0
    st.progress(inspected / total, text=f"진행률 {inspected}/{total}")

    if not all_labels:
        st.info("아직 라벨 데이터 없음")
        st.stop()

    # ──────────────────────────────────────────
    # Row 1: 클래스 분포 + 신뢰도 분포
    # ──────────────────────────────────────────
    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("##### 분류기 예측 클래스 분포")
        cls_top_counts = Counter(L.get("cls_model_top") for L in all_labels if L.get("cls_model_top"))
        cls_human_counts = Counter(L.get("cls_human_label") for L in all_labels if L.get("cls_human_label"))
        if cls_top_counts or cls_human_counts:
            df_cls = pd.DataFrame({
                "모델 예측": cls_top_counts,
                "사람 정답": cls_human_counts,
            }).fillna(0).astype(int)
            st.bar_chart(df_cls)
        else:
            st.caption("Classifier 데이터 없음")

    with col_b:
        st.markdown("##### 분류기 신뢰도 분포")
        cls_confs = [L.get("cls_model_conf") for L in all_labels if L.get("cls_model_conf") is not None]
        if cls_confs:
            bins = [0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.001]
            labels_b = ["0-30%", "30-50%", "50-70%", "70-85%", "85-95%", "95-100%"]
            df_conf = pd.DataFrame({"신뢰도": cls_confs})
            df_conf["구간"] = pd.cut(df_conf["신뢰도"], bins=bins, labels=labels_b, include_lowest=True)
            st.bar_chart(df_conf["구간"].value_counts().reindex(labels_b).fillna(0))
            st.caption(
                f"평균 {sum(cls_confs)/len(cls_confs):.3f} · "
                f"최저 {min(cls_confs):.3f} · 0.7 미만 {sum(1 for c in cls_confs if c < 0.7)}장"
            )
        else:
            st.caption("Classifier 신뢰도 없음")

    # ──────────────────────────────────────────
    # Row 2: Confusion matrix + 시간대별 처리량
    # ──────────────────────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("##### 모델 → 사람 정정 매트릭스 (Classifier)")
        rows_cm = []
        for L in all_labels:
            mt = L.get("cls_model_top")
            hu = L.get("cls_human_label") or mt
            v = L.get("cls_verdict")
            if mt and v in ("correct", "wrong", "uncertain"):
                rows_cm.append({"모델": mt, "사람": hu, "판정": v})
        if rows_cm:
            df_cm = pd.DataFrame(rows_cm)
            piv = pd.crosstab(df_cm["모델"], df_cm["사람"])
            st.dataframe(piv, use_container_width=True)
            n_diff = sum(1 for r in rows_cm if r["모델"] != r["사람"])
            st.caption(f"검수 {len(rows_cm)} · 정정 {n_diff} ({n_diff/len(rows_cm)*100:.1f}%)")
        else:
            st.caption("검수 데이터 없음")

    with col_d:
        st.markdown("##### 시간대별 검수 처리량")
        times = []
        for L in all_labels:
            t = L.get("inspected_at")
            if t and len(t) >= 13:
                times.append(t[:13])  # YYYY-MM-DDTHH
        if times:
            df_t = pd.DataFrame({"시간대": times})
            counts = df_t["시간대"].value_counts().sort_index()
            st.bar_chart(counts)
            st.caption(f"마지막: {max(times)} · 시간대 {len(counts)}개")
        else:
            st.caption("검수 시각 없음")

    # ──────────────────────────────────────────
    # Row 3: Segmenter 클래스 + 가중치 + 최근 변경 로그
    # ──────────────────────────────────────────
    col_e, col_f = st.columns(2)

    with col_e:
        st.markdown("##### Segmenter 폴리곤 클래스 분포")
        seg_model_classes = Counter()
        seg_human_classes = Counter()
        for L in all_labels:
            for p in (L.get("seg_model_polygons") or []):
                seg_model_classes[p.get("class_name", "?")] += 1
            for p in (L.get("seg_human_polygons") or []):
                seg_human_classes[p.get("class_name", "?")] += 1
        if seg_model_classes or seg_human_classes:
            df_seg = pd.DataFrame({
                "모델": seg_model_classes,
                "사람": seg_human_classes,
            }).fillna(0).astype(int)
            st.bar_chart(df_seg)
        else:
            st.caption("Segmenter 데이터 없음")

    with col_f:
        st.markdown("##### 정정 가중치 분포")
        weights = []
        for L in all_labels:
            if L.get("cls_verdict") == "wrong":
                weights.append(("cls", float(L.get("cls_correction_weight") or 1.0)))
            if L.get("seg_verdict") == "wrong":
                weights.append(("seg", float(L.get("seg_correction_weight") or 1.0)))
        if weights:
            df_w = pd.DataFrame(weights, columns=["대상", "가중치"])
            st.bar_chart(df_w["가중치"].value_counts().sort_index())
            st.caption(
                f"정정 case {len(weights)} · 평균 ×{df_w['가중치'].mean():.2f} · "
                f"최고 ×{df_w['가중치'].max():.1f}"
            )
        else:
            st.caption("정정 case 없음")

    # ──────────────────────────────────────────
    # 최근 변경 로그
    # ──────────────────────────────────────────
    st.divider()
    st.markdown("##### 최근 검수 변경 로그 (Undo 가능 기록)")
    hist = get_recent_history(st.session_state.session_id, limit=30)
    if hist:
        st.dataframe(hist, use_container_width=True, height=200)
    else:
        st.caption("이력 없음")

    # 전체 표
    st.divider()
    st.markdown("##### 이미지별 상태 표")
    rows = []
    for L in all_labels:
        rows.append({
            "파일": L["image_filename"],
            "상태": status_text(L),
            "분류기 예측": L.get("cls_model_top"),
            "분류기 신뢰도": round(L.get("cls_model_conf") or 0, 3),
            "분류기 정답": L.get("cls_human_label"),
            "분류기 판정": L.get("cls_verdict"),
            "분류기 가중치": L.get("cls_correction_weight"),
            "세그 폴리곤(모델)": len(L.get("seg_model_polygons") or []),
            "세그 폴리곤(사람)": len(L.get("seg_human_polygons") or []),
            "세그 판정": L.get("seg_verdict"),
            "검수 시각": L.get("inspected_at") or "",
        })
    if rows:
        st.dataframe(rows, use_container_width=True, height=400)
    st.stop()


# ============================================================
# Mode: export — Export YOLO + COCO
# ============================================================
if st.session_state.view_mode == "export":
    st.markdown(f'<h3>{_bi_icon("box-seam", _icon_default, 22)} 내보내기 — YOLO / COCO</h3>',
                unsafe_allow_html=True)
    if st.session_state.session_id is None:
        st.warning("세션 없음")
        st.stop()

    formats = st.multiselect(
        "포맷 선택 (복수 가능)",
        options=[
            ("yolo_seg", "YOLO Segmentation (.txt + data.yaml)"),
            ("yolo_cls", "YOLO Classification (폴더/클래스 구조)"),
            ("coco", "COCO (단일 JSON + 이미지)"),
        ],
        default=[("yolo_seg", "YOLO Segmentation (.txt + data.yaml)"),
                  ("yolo_cls", "YOLO Classification (폴더/클래스 구조)"),
                  ("coco", "COCO (단일 JSON + 이미지)")],
        format_func=lambda x: x[1],
    )
    formats_ids = [f[0] for f in formats]

    c1, c2 = st.columns(2)
    only_inspected = c1.checkbox("검수 완료한 것만", value=True)
    only_wrong = c2.checkbox("정정된 것만 (wrong)", value=False)

    # 서버측 저장 폴더 (선택)
    default_save = str(Path.home() / "Downloads" / "hitl_exports")
    save_dir_input = st.text_input(
        "서버측 추가 저장 폴더 (선택, 비워두면 브라우저 다운로드만)",
        value=st.session_state.get("export_save_dir", default_save),
        key="export_save_dir_input",
        help="Zip 만들 때 브라우저로도 다운로드 + 지정 폴더에도 동시 저장. NAS 경로 가능.",
    )
    st.session_state.export_save_dir = save_dir_input

    if st.button("Zip 만들기", type="primary"):
        if not formats_ids:
            st.error("포맷을 선택해주세요")
        elif not st.session_state.images_by_filename:
            st.error("이미지가 메모리에 없습니다. 사이드바에서 이미지 다시 로드 후 진행")
        else:
            with st.spinner("Zip 생성 중..."):
                data = build_export_zip(
                    st.session_state.session_id,
                    st.session_state.images_by_filename,
                    formats_ids,
                    only_inspected=only_inspected,
                    only_wrong=only_wrong,
                )
            zip_name = f"hitl_export_session{st.session_state.session_id}_{time.strftime('%Y%m%d_%H%M%S')}.zip"
            st.success(f"생성 완료 — {len(data)/1024:.1f} KB")

            # 서버측 저장 (지정된 경우)
            if save_dir_input.strip():
                try:
                    save_dir = Path(save_dir_input.strip()).expanduser()
                    save_dir.mkdir(parents=True, exist_ok=True)
                    out_path = save_dir / zip_name
                    out_path.write_bytes(data)
                    st.markdown(
                        f'<div style="background:rgba(45,164,78,0.12);padding:8px 12px;'
                        f'border-left:4px solid #2da44e;border-radius:6px;color:#2da44e;">'
                        f'{_bi_icon("hdd-fill","2da44e",16)} &nbsp;'
                        f'서버에 저장됨: <code>{out_path}</code></div>',
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"서버측 저장 실패: {e}")

            st.download_button(
                "Zip 다운로드 (브라우저)",
                data=data,
                file_name=zip_name,
                mime="application/zip",
                use_container_width=True,
            )
    st.stop()


# ============================================================
# Mode: labelstudio — Label Studio 통합 (풀 polygon vertex 편집)
# ============================================================
if st.session_state.view_mode == "labelstudio":
    st.markdown(
        f'<h3>{_bi_icon("vector-pen", _icon_default, 22)} Label Studio 통합 — Polygon vertex 편집</h3>',
        unsafe_allow_html=True,
    )
    st.caption(
        "streamlit-drawable-canvas 한계로 polygon 모서리 점 단위 편집 불가 → "
        "**Label Studio (Apache 2.0)** 별창 띄워서 검수 진행 → 결과 import."
    )

    # 1) 연결 설정
    st.markdown("##### 1. Label Studio 연결")
    cA, cB = st.columns([2, 3])
    with cA:
        ls_url = st.text_input("Label Studio URL", value=st.session_state.get("ls_url", "http://localhost:8085"),
                                key="ls_url_input")
    with cB:
        ls_token = st.text_input("Access Token", value=st.session_state.get("ls_token", ""),
                                  type="password", key="ls_token_input",
                                  help="Label Studio 우측 상단 프로필 → Account & Settings → Access Token")
    st.session_state.ls_url = ls_url
    st.session_state.ls_token = ls_token

    if ls_token:
        client = LSClient(ls_url, ls_token)
        if client.ping():
            st.markdown(
                f'<div style="background:rgba(45,164,78,0.12);padding:6px 12px;'
                f'border-left:3px solid #2da44e;border-radius:6px;color:#2da44e;font-size:0.9em;">'
                f'{_bi_icon("check-circle-fill","2da44e",14)} &nbsp;연결 OK</div>',
                unsafe_allow_html=True,
            )
        else:
            st.error("Label Studio 연결 실패 — URL/Token 확인")
            st.stop()
    else:
        st.info("Access Token 입력하세요 (Label Studio 우측 상단 프로필 → Account & Settings → Access Token)")
        st.stop()

    if st.session_state.session_id is None or not st.session_state.results:
        st.warning("세션 + 추론 결과 필요 — 사이드바에서 모델 + 이미지 로드 + 추론 먼저 진행")
        st.stop()

    # 2) Project 생성 + Task import
    st.divider()
    st.markdown("##### 2. Label Studio Project 생성 + Task import")
    st.caption(
        f"현 세션 (#{st.session_state.session_id}) 의 추론 결과 **{len(st.session_state.results)}장** 을 "
        "Label Studio project 로 import. 모델 폴리곤은 pre-annotation 으로 표시."
    )

    cI, cJ = st.columns([3, 2])
    with cI:
        proj_title = st.text_input(
            "Project 이름",
            value=f"HITL-S{st.session_state.session_id}-{time.strftime('%Y%m%d_%H%M%S')}",
            key="ls_proj_title",
        )
    with cJ:
        existing_proj_id = st.text_input(
            "또는 기존 Project ID (재import)",
            value=str(st.session_state.get("ls_project_id", "")) if st.session_state.get("ls_project_id") else "",
            key="ls_proj_id_input",
        )

    seg_classes = list(st.session_state.segmenter_classes.values()) if st.session_state.segmenter_classes else ["defect"]
    cls_classes = st.session_state.classifier_classes or []
    label_config = make_label_config(seg_classes, cls_classes)
    with st.expander("Label Studio label_config XML 미리보기"):
        st.code(label_config, language="xml")

    if st.button("Project 만들고 Task import", type="primary", use_container_width=True):
        try:
            # 1) NAS 이미지 → 로컬 카피
            with st.spinner("NAS 이미지를 로컬로 카피 중 (Label Studio docker 접근용, NAS 원본은 안 건드림)..."):
                local_dir = copy_images_to_local(
                    st.session_state.images_by_filename, st.session_state.session_id
                )
            st.success(f"로컬 카피 완료: {local_dir}")

            # 2) Project
            if existing_proj_id.strip():
                project_id = int(existing_proj_id.strip())
                st.info(f"기존 Project #{project_id} 사용")
            else:
                with st.spinner("Label Studio Project 생성 중..."):
                    proj = client.create_project(proj_title, label_config,
                                                  description=f"HITL session #{st.session_state.session_id}")
                project_id = proj["id"]
                st.session_state.ls_project_id = project_id
                st.success(f"Project #{project_id} 생성")

            # 3) Tasks
            tasks = []
            from pathlib import Path as _P
            for r in st.session_state.results:
                fn = r["filename"]
                img = st.session_state.images_by_filename.get(fn)
                if img is None:
                    continue
                img_w, img_h = img.size
                seg_r = r.get("segmenter") or {}
                seg_polys = []
                for poly, cid, cname, c in zip(
                    seg_r.get("polygons", []),
                    seg_r.get("class_ids", []),
                    seg_r.get("class_names", []),
                    seg_r.get("confs", []),
                ):
                    seg_polys.append({"class_id": int(cid), "class_name": cname,
                                      "polygon": poly, "conf": float(c)})
                preds = build_predictions_for_image(seg_polys, img_w, img_h, seg_classes)
                cls_r = r.get("classifier") or {}
                # local_dir 안의 파일 경로
                local_path = local_dir / fn
                tasks.append(build_task_for_image(
                    str(local_path), fn, preds,
                    cls_top=cls_r.get("top_class"),
                    cls_conf=cls_r.get("top_conf"),
                ))

            with st.spinner(f"Tasks {len(tasks)}개 import 중..."):
                resp = client.import_tasks(project_id, tasks)
            st.success(f"Task import OK — {resp}")

            ls_link = f"{ls_url}/projects/{project_id}/data"
            st.markdown(
                f'<div style="background:rgba(31,111,235,0.12);padding:10px 14px;'
                f'border-left:4px solid #1f6feb;border-radius:8px;">'
                f'{_bi_icon("box-arrow-up-right",_link_color_hex,16)} &nbsp;'
                f'<a href="{ls_link}" target="_blank" style="color:{_link_color};font-weight:600;">'
                f'Label Studio 별창 열기 (검수)</a></div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"실패: {type(e).__name__}: {e}")

    # 3) 검수 결과 import
    st.divider()
    st.markdown("##### 3. Label Studio 검수 결과 → 우리 DB import")
    st.caption("Label Studio 에서 검수 완료 후 (또는 일부만), 여기 버튼 누르면 결과를 우리 DB 로 동기화.")

    import_proj_id = st.text_input(
        "Import 대상 Project ID",
        value=str(st.session_state.get("ls_project_id", "")) if st.session_state.get("ls_project_id") else "",
        key="ls_import_proj_id",
    )

    if st.button("Label Studio 결과 → 우리 DB 동기화", type="primary", use_container_width=True):
        if not import_proj_id.strip():
            st.error("Project ID 입력하세요")
        else:
            try:
                with st.spinner("Label Studio export 가져오는 중..."):
                    export = client.export_annotations(int(import_proj_id.strip()), fmt="JSON")
                st.success(f"Export 받음 — {len(export)} task")

                # 클래스 이름 → id 매핑
                seg_id_map = {v: int(k) for k, v in st.session_state.segmenter_classes.items()}
                parsed = parse_export(export, seg_id_map)

                # 우리 DB 로
                n_saved = 0
                for fn, item in parsed.items():
                    img = st.session_state.images_by_filename.get(fn)
                    if img is None:
                        continue
                    # 기존 모델 추론 결과 (segmenter)
                    matching = next((r for r in st.session_state.results if r["filename"] == fn), None)
                    seg_r = (matching or {}).get("segmenter") or {}
                    seg_model_polys = [
                        {"class_id": int(cid), "class_name": cname,
                         "polygon": [[float(x), float(y)] for x, y in poly], "conf": float(c)}
                        for poly, cid, cname, c in zip(
                            seg_r.get("polygons", []),
                            seg_r.get("class_ids", []),
                            seg_r.get("class_names", []),
                            seg_r.get("confs", []),
                        )
                    ] if seg_r and "error" not in seg_r else None
                    cls_r = (matching or {}).get("classifier") or {}
                    # seg_verdict 추정: 사람 폴리곤이 모델 폴리곤 수와 같으면 correct, 다르면 wrong
                    n_human = len(item["seg_polygons"])
                    n_model = len(seg_model_polys or [])
                    if n_human == n_model and n_model > 0:
                        seg_verdict = "correct"
                    elif n_human == 0 and n_model == 0:
                        seg_verdict = "correct"
                    else:
                        seg_verdict = "wrong"
                    save_label_with_history(
                        session_id=st.session_state.session_id,
                        image_filename=fn,
                        image_w=img.size[0], image_h=img.size[1],
                        cls_model_top=cls_r.get("top_class"),
                        cls_model_conf=cls_r.get("top_conf"),
                        cls_human_label=item.get("cls_human_label") or cls_r.get("top_class"),
                        cls_verdict=item.get("cls_verdict") or "correct",
                        cls_correction_weight=5.0 if item.get("cls_verdict") == "wrong" else 1.0,
                        seg_model_polygons=seg_model_polys,
                        seg_human_polygons=item["seg_polygons"],
                        seg_verdict=seg_verdict,
                        seg_correction_weight=5.0 if seg_verdict == "wrong" else 1.0,
                        seg_names_dict={str(k): v for k, v in (seg_r.get("names_dict") or {}).items()} if seg_r else None,
                        inspected_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                        action="labelstudio_import",
                    )
                    n_saved += 1
                st.success(f"✅ {n_saved} 장 우리 DB 로 동기화 완료")
            except Exception as e:
                st.error(f"실패: {type(e).__name__}: {e}")
                import traceback
                st.code(traceback.format_exc())
    st.stop()


# ============================================================
# Mode: inspect — 검수
# ============================================================

# 네비게이션
n_total = len(st.session_state.results)
cur = st.session_state.current_idx

# 진행률 + 정렬된 인덱스
saved_labels_by_fn = {}
if st.session_state.session_id:
    all_lbls = get_labels_by_session(st.session_state.session_id)
    saved_labels_by_fn = {L["image_filename"]: L for L in all_lbls}

ordered_indices = compute_sort_order(
    st.session_state.results, saved_labels_by_fn, st.session_state.sort_mode
)

inspected_count = sum(1 for L in saved_labels_by_fn.values() if L.get("inspected_at"))

cur = min(cur, max(0, n_total - 1))
real_idx = ordered_indices[cur] if ordered_indices else 0
fn_here = st.session_state.results[real_idx]["filename"]
undo_n = get_undo_count(st.session_state.session_id, fn_here) if st.session_state.session_id else 0

# Progress bar — text 없이 시각 게이지만
st.progress(inspected_count / n_total if n_total else 0)

# 진행률 텍스트 — progress 바로 아래 가운데
st.markdown(
    f'<div style="text-align:center;font-weight:600;font-size:0.95em;margin-top:-4px;margin-bottom:8px;opacity:0.85;">'
    f'{cur + 1} / {n_total}</div>',
    unsafe_allow_html=True,
)

# Navigation row: [이전] [jump 작게] [다음]
col_prev, col_jump, col_next = st.columns([2, 1, 2])
with col_prev:
    if ssh.shortcut_button("이전", "ArrowLeft", hint=False,
                            use_container_width=True, disabled=cur == 0):
        st.session_state.current_idx = max(0, cur - 1)
        st.rerun()
with col_jump:
    new_idx = st.number_input("이미지 번호 이동", min_value=1, max_value=n_total,
                                value=cur + 1, step=1, label_visibility="collapsed")
    if new_idx - 1 != cur:
        st.session_state.current_idx = new_idx - 1
        st.rerun()
with col_next:
    if ssh.shortcut_button("다음", "ArrowRight", hint=False,
                            use_container_width=True, disabled=cur >= n_total - 1):
        st.session_state.current_idx = min(n_total - 1, cur + 1)
        st.rerun()

# Undo row (다음 버튼 아래, 별도 행)
if ssh.shortcut_button(f"↶ Undo ({undo_n})", ["ctrl+z", "meta+z"], hint=False,
                        use_container_width=True, disabled=undo_n == 0):
    restored = undo_last(st.session_state.session_id, fn_here)
    if restored:
        st.session_state[f"_panel_revision_{real_idx}"] = (
            st.session_state.get(f"_panel_revision_{real_idx}", 0) + 1
        )
        st.session_state.pop(f"_baseline_fp_{real_idx}", None)
        st.toast(f"Undo: {fn_here}")
        st.rerun()

r = st.session_state.results[real_idx]
img = st.session_state.images[real_idx][1]
saved = get_label(st.session_state.session_id, r["filename"]) if st.session_state.session_id else None

# 이미지 헤더 카드 — 파일명 + 해상도 + 상태 배지
status_str = status_badge(saved)
st.markdown(
    f"""
<div style="background:rgba(128,128,128,0.07);padding:10px 16px;border-radius:8px;
            display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;
            border:1px solid rgba(128,128,128,0.18);">
  <div style="min-width:0;flex:1;">
    <div style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:1.0em;
                color:var(--text-color);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{r['filename']}</div>
    <div style="opacity:0.6;font-size:0.8em;margin-top:2px;">{img.size[0]}×{img.size[1]} px</div>
  </div>
  <div style="flex-shrink:0;margin-left:12px;">{status_str}</div>
</div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────
# 시각화 — Segmenter 결과 있을 때만 상단에 overlay 미리보기 (classifier-only 모드는 패널 안에서 표시)
# ──────────────────────────────────────────────────────
seg_r = r.get("segmenter") or {}
_has_seg_preview = seg_r and "polygons" in seg_r and not seg_r.get("error")
if _has_seg_preview:
    try:
        from inference import draw_segmenter_result
        vis = draw_segmenter_result(img, seg_r)
        st.image(
            vis,
            caption=f"Segmenter 미리보기 — 폴리곤 {len(seg_r['polygons'])}개 (아래 검수 패널에서 정정)",
            use_column_width=True,
        )
    except Exception as e:
        st.error(f"시각화 실패: {e}")

# 모델 종류 확인 → 모드별 UI 분기
has_cls = r.get("classifier") is not None and "error" not in (r.get("classifier") or {})
has_seg = r.get("segmenter") is not None and "error" not in (r.get("segmenter") or {})

# 기본 검수 결과 (검수 안 한 모델은 default)
cls_review = {"human_label": None, "verdict": None, "weight": 1.0, "changed": False}
seg_review = {"human_polygons": None, "verdict": None, "weight": 1.0, "changed": False}

_rev = st.session_state.get(f"_panel_revision_{real_idx}", 0)
_pk_cls = f"cls_{real_idx}_v{_rev}"
_pk_seg = f"seg_{real_idx}_v{_rev}"

if has_cls and has_seg:
    tab_cls, tab_seg = st.tabs(["Classifier 검수", "Segmenter 검수"])
    with tab_cls:
        st.markdown(f'<h4>{_bi_icon("tag-fill", _icon_default, 18)} Classifier 검수</h4>',
                    unsafe_allow_html=True)
        cls_review = classifier_review_panel(
            r.get("classifier"), saved, st.session_state.classifier_classes,
            panel_key=_pk_cls,
        )
    with tab_seg:
        st.markdown(f'<h4>{_bi_icon("scissors", _icon_default, 18)} Segmenter 검수</h4>',
                    unsafe_allow_html=True)
        seg_review = segmenter_review_panel(
            img, r.get("segmenter"), saved, st.session_state.segmenter_classes,
            panel_key=_pk_seg,
            show_roi=st.session_state.show_roi_panel,
        )

elif has_cls:
    col_panel, col_img = st.columns([1, 2])
    with col_img:
        st.image(img, use_column_width=True,
                 caption=f"입력 이미지 ({img.size[0]}×{img.size[1]})")
    with col_panel:
        st.markdown(f'<h4>{_bi_icon("tag-fill", _icon_default, 18)} Classifier 검수</h4>',
                    unsafe_allow_html=True)
        cls_review = classifier_review_panel(
            r.get("classifier"), saved, st.session_state.classifier_classes,
            panel_key=_pk_cls,
        )

elif has_seg:
    st.markdown(f'<h4>{_bi_icon("scissors", _icon_default, 18)} Segmenter 검수</h4>',
                unsafe_allow_html=True)
    seg_review = segmenter_review_panel(
        img, r.get("segmenter"), saved, st.session_state.segmenter_classes,
        panel_key=_pk_seg,
        show_roi=st.session_state.show_roi_panel,
    )

else:
    st.warning("Classifier / Segmenter 둘 다 결과 없음 — 모델 로드 후 추론 다시 진행")

# 마지막 검수 시각
if saved and saved.get("inspected_at"):
    st.caption(f"마지막 검수: {saved.get('inspected_at')}")


def _persist(action: str = "autosave") -> None:
    """현재 review 결과 → save_label_with_history (snapshot 후 upsert)."""
    cls_r = r.get("classifier") or {}
    seg_r_local = r.get("segmenter") or {}
    save_label_with_history(
        session_id=st.session_state.session_id,
        image_filename=r["filename"],
        image_w=img.size[0], image_h=img.size[1],
        cls_model_top=cls_r.get("top_class"),
        cls_model_conf=cls_r.get("top_conf"),
        cls_human_label=cls_review.get("human_label"),
        cls_verdict=cls_review.get("verdict"),
        cls_correction_weight=cls_review.get("weight", 1.0),
        seg_model_polygons=[
            {"class_id": int(cid), "class_name": cname,
             "polygon": [[float(x), float(y)] for x, y in poly], "conf": float(c)}
            for poly, cid, cname, c in zip(
                seg_r_local.get("polygons", []),
                seg_r_local.get("class_ids", []),
                seg_r_local.get("class_names", []),
                seg_r_local.get("confs", []),
            )
        ] if seg_r_local and "error" not in seg_r_local else None,
        seg_human_polygons=seg_review.get("human_polygons"),
        seg_verdict=seg_review.get("verdict"),
        seg_correction_weight=seg_review.get("weight", 1.0),
        seg_names_dict={str(k): v for k, v in (seg_r_local.get("names_dict") or {}).items()} if seg_r_local else None,
        inspected_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        action=action,
    )


# 명시 저장 + 네비
st.divider()
c_save1, c_save2 = st.columns([3, 1])
with c_save1:
    save_and_next = st.checkbox("저장 후 자동으로 다음 이미지로 이동", value=True)
with c_save2:
    if ssh.shortcut_button("저장+다음", ["ctrl+s", "meta+s"], hint=False,
                            use_container_width=True, type="primary"):
        if st.session_state.session_id and (has_cls or has_seg):
            _persist(action="manual_save")
            st.toast(f"저장됨: {r['filename']}")
        if save_and_next and cur < n_total - 1:
            st.session_state.current_idx = cur + 1
            st.rerun()

st.markdown(
    """
<style>
.kbd-tip-table { margin-top:8px; font-size:0.85em; border-collapse:collapse; width:100%; }
.kbd-tip-table th { text-align:left; padding:6px 10px; color:var(--text-color); opacity:0.6;
                     font-weight:500; font-size:0.8em; border-bottom:1px solid rgba(128,128,128,0.2); }
.kbd-tip-table td { padding:6px 10px; vertical-align:middle; color:var(--text-color); }
.kbd-tip-table td.action { opacity:0.75; }
.kbd-tip-table td.keys { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
.kbd-tip-table kbd { background:rgba(128,128,128,0.18); border:1px solid rgba(128,128,128,0.35);
                     border-radius:4px; padding:1px 6px; font-size:0.85em; margin:0 1px; }
.kbd-tip-table tr + tr td { border-top:1px dashed rgba(128,128,128,0.15); }
</style>
<table class="kbd-tip-table">
  <thead>
    <tr><th style="width:24%;">동작</th><th>Windows</th><th>Mac</th></tr>
  </thead>
  <tbody>
    <tr>
      <td class="action">이미지 이동</td>
      <td class="keys"><kbd>←</kbd> <kbd>→</kbd></td>
      <td class="keys"><kbd>←</kbd> <kbd>→</kbd></td>
    </tr>
    <tr>
      <td class="action">저장 + 다음</td>
      <td class="keys"><kbd>Ctrl</kbd>+<kbd>S</kbd></td>
      <td class="keys"><kbd>⌘</kbd>+<kbd>S</kbd></td>
    </tr>
    <tr>
      <td class="action">Undo</td>
      <td class="keys"><kbd>Ctrl</kbd>+<kbd>Z</kbd></td>
      <td class="keys"><kbd>⌘</kbd>+<kbd>Z</kbd></td>
    </tr>
  </tbody>
</table>
    """,
    unsafe_allow_html=True,
)


