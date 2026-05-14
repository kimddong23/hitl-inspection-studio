"""SQLite local DB — 라벨 누적 + 세션 복원."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


import os as _os
# DB 경로: 환경변수 HITL_DB_PATH 우선, 없으면 prototype 폴더 내 기본 파일
DB_PATH = Path(_os.environ.get("HITL_DB_PATH",
                                str(Path(__file__).parent / "hitl_labels.db"))).expanduser()


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    classifier_name TEXT,
    segmenter_name TEXT,
    cls_path TEXT,
    seg_path TEXT,
    folder_path TEXT,
    note TEXT
);

CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    image_filename TEXT NOT NULL,
    image_w INTEGER,
    image_h INTEGER,

    -- Classifier 결과
    cls_model_top TEXT,
    cls_model_conf REAL,
    cls_human_label TEXT,       -- 사람이 정정한 클래스 (NULL = 모델 결과 그대로)
    cls_verdict TEXT,           -- 'correct' / 'wrong' / 'uncertain' / NULL
    cls_correction_weight REAL DEFAULT 1.0,

    -- Segmenter 결과
    seg_model_polygons TEXT,    -- JSON [{class_id, class_name, polygon:[[x,y],...], conf}, ...]
    seg_human_polygons TEXT,    -- JSON 정정 후 polygon (사람 그린/수정한 것)
    seg_verdict TEXT,           -- 'correct' / 'wrong' / 'uncertain' / NULL
    seg_correction_weight REAL DEFAULT 1.0,
    seg_names_dict TEXT,        -- JSON class id → name map

    -- 메타
    inspected_at TEXT,
    note TEXT,
    UNIQUE(session_id, image_filename),
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- 검수 변경 이력 (Undo 용 snapshot stack)
CREATE TABLE IF NOT EXISTS label_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    image_filename TEXT NOT NULL,
    snapshot TEXT NOT NULL,     -- JSON of label row BEFORE change
    changed_at TEXT NOT NULL,
    action TEXT,                -- 'autosave' / 'manual_save' / 'undo'
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_labels_session ON labels(session_id);
CREATE INDEX IF NOT EXISTS idx_labels_filename ON labels(image_filename);
CREATE INDEX IF NOT EXISTS idx_history_session_file ON label_history(session_id, image_filename, id);
"""

# Undo 스택 최대 깊이 (이미지당)
MAX_UNDO_DEPTH = 30


@contextmanager
def get_conn(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path = DB_PATH) -> None:
    with get_conn(db_path) as c:
        c.executescript(SCHEMA)
        # 안전한 ALTER — 기존 DB에 신규 컬럼 추가 (없을 때만)
        existing_cols = {r[1] for r in c.execute("PRAGMA table_info(sessions)").fetchall()}
        for col in ("cls_path", "seg_path", "folder_path"):
            if col not in existing_cols:
                try:
                    c.execute(f"ALTER TABLE sessions ADD COLUMN {col} TEXT")
                except Exception:
                    pass


def create_session(classifier_name: str | None, segmenter_name: str | None,
                   note: str = "",
                   cls_path: str | None = None,
                   seg_path: str | None = None,
                   folder_path: str | None = None) -> int:
    init_db()
    with get_conn() as c:
        cur = c.execute(
            "INSERT INTO sessions (started_at, classifier_name, segmenter_name, "
            "cls_path, seg_path, folder_path, note) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (time.strftime("%Y-%m-%dT%H:%M:%S"),
             classifier_name, segmenter_name,
             cls_path, seg_path, folder_path, note),
        )
        return cur.lastrowid


def get_session(session_id: int) -> dict | None:
    init_db()
    with get_conn() as c:
        row = c.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None


def list_sessions() -> list[dict[str, Any]]:
    init_db()
    with get_conn() as c:
        rows = c.execute(
            "SELECT s.*, "
            "  (SELECT COUNT(*) FROM labels WHERE session_id = s.id) AS n_images, "
            "  (SELECT COUNT(*) FROM labels WHERE session_id = s.id AND inspected_at IS NOT NULL) AS n_inspected "
            "FROM sessions s ORDER BY s.id DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def upsert_inference_result(
    session_id: int,
    image_filename: str,
    image_w: int,
    image_h: int,
    cls_model_top: str | None = None,
    cls_model_conf: float | None = None,
    seg_model_polygons: list | None = None,
    seg_names_dict: dict | None = None,
) -> int:
    """추론 결과만 저장. human 컬럼 (verdict/human_label/inspected_at 등) 절대 건드리지 않음.

    추론 시점에 호출 → 같은 (session_id, filename) 의 사람 정정 기록 보존.
    """
    init_db()
    with get_conn() as c:
        # 기존 row 있으면 model 컬럼만 update, human 컬럼은 그대로
        existing = c.execute(
            "SELECT id FROM labels WHERE session_id = ? AND image_filename = ?",
            (session_id, image_filename),
        ).fetchone()
        if existing:
            cur = c.execute(
                """UPDATE labels SET
                    image_w=?, image_h=?,
                    cls_model_top=?, cls_model_conf=?,
                    seg_model_polygons=?, seg_names_dict=?
                WHERE session_id=? AND image_filename=?""",
                (
                    image_w, image_h, cls_model_top, cls_model_conf,
                    json.dumps(seg_model_polygons) if seg_model_polygons is not None else None,
                    json.dumps(seg_names_dict) if seg_names_dict is not None else None,
                    session_id, image_filename,
                ),
            )
            return existing["id"]
        else:
            cur = c.execute(
                """INSERT INTO labels (
                    session_id, image_filename, image_w, image_h,
                    cls_model_top, cls_model_conf,
                    seg_model_polygons, seg_names_dict
                ) VALUES (?,?,?,?, ?,?, ?,?)""",
                (
                    session_id, image_filename, image_w, image_h,
                    cls_model_top, cls_model_conf,
                    json.dumps(seg_model_polygons) if seg_model_polygons is not None else None,
                    json.dumps(seg_names_dict) if seg_names_dict is not None else None,
                ),
            )
            return cur.lastrowid


def upsert_label(
    session_id: int,
    image_filename: str,
    image_w: int,
    image_h: int,
    cls_model_top: str | None = None,
    cls_model_conf: float | None = None,
    cls_human_label: str | None = None,
    cls_verdict: str | None = None,
    cls_correction_weight: float = 1.0,
    seg_model_polygons: list | None = None,
    seg_human_polygons: list | None = None,
    seg_verdict: str | None = None,
    seg_correction_weight: float = 1.0,
    seg_names_dict: dict | None = None,
    inspected_at: str | None = None,
    note: str = "",
) -> int:
    init_db()
    with get_conn() as c:
        cur = c.execute(
            """
            INSERT INTO labels (
                session_id, image_filename, image_w, image_h,
                cls_model_top, cls_model_conf, cls_human_label, cls_verdict, cls_correction_weight,
                seg_model_polygons, seg_human_polygons, seg_verdict, seg_correction_weight, seg_names_dict,
                inspected_at, note
            ) VALUES (?,?,?,?, ?,?,?,?,?, ?,?,?,?,?, ?,?)
            ON CONFLICT(session_id, image_filename) DO UPDATE SET
                cls_model_top=excluded.cls_model_top,
                cls_model_conf=excluded.cls_model_conf,
                cls_human_label=excluded.cls_human_label,
                cls_verdict=excluded.cls_verdict,
                cls_correction_weight=excluded.cls_correction_weight,
                seg_model_polygons=excluded.seg_model_polygons,
                seg_human_polygons=excluded.seg_human_polygons,
                seg_verdict=excluded.seg_verdict,
                seg_correction_weight=excluded.seg_correction_weight,
                seg_names_dict=excluded.seg_names_dict,
                inspected_at=excluded.inspected_at,
                note=excluded.note
            """,
            (
                session_id, image_filename, image_w, image_h,
                cls_model_top, cls_model_conf, cls_human_label, cls_verdict, cls_correction_weight,
                json.dumps(seg_model_polygons) if seg_model_polygons is not None else None,
                json.dumps(seg_human_polygons) if seg_human_polygons is not None else None,
                seg_verdict, seg_correction_weight,
                json.dumps(seg_names_dict) if seg_names_dict is not None else None,
                inspected_at, note,
            ),
        )
        return cur.lastrowid


def _snapshot_label(c: sqlite3.Connection, session_id: int, image_filename: str,
                    action: str = "autosave") -> bool:
    """변경 전 현재 label row 를 history 에 push. Returns: True if snapshotted."""
    row = c.execute(
        "SELECT * FROM labels WHERE session_id = ? AND image_filename = ?",
        (session_id, image_filename),
    ).fetchone()
    if not row:
        return False
    snap = dict(row)
    c.execute(
        "INSERT INTO label_history (session_id, image_filename, snapshot, changed_at, action) "
        "VALUES (?, ?, ?, ?, ?)",
        (session_id, image_filename, json.dumps(snap),
         time.strftime("%Y-%m-%dT%H:%M:%S"), action),
    )
    # depth 제한 — 오래된 것 삭제
    c.execute(
        """DELETE FROM label_history
            WHERE session_id = ? AND image_filename = ?
              AND id NOT IN (
                SELECT id FROM label_history
                WHERE session_id = ? AND image_filename = ?
                ORDER BY id DESC LIMIT ?
              )""",
        (session_id, image_filename, session_id, image_filename, MAX_UNDO_DEPTH),
    )
    return True


def save_label_with_history(
    session_id: int,
    image_filename: str,
    image_w: int,
    image_h: int,
    cls_model_top: str | None = None,
    cls_model_conf: float | None = None,
    cls_human_label: str | None = None,
    cls_verdict: str | None = None,
    cls_correction_weight: float = 1.0,
    seg_model_polygons: list | None = None,
    seg_human_polygons: list | None = None,
    seg_verdict: str | None = None,
    seg_correction_weight: float = 1.0,
    seg_names_dict: dict | None = None,
    inspected_at: str | None = None,
    note: str = "",
    action: str = "autosave",
) -> int:
    """변경 전 snapshot 을 history 에 저장한 뒤 upsert. (자동저장 / 명시저장 공통 진입점)"""
    init_db()
    with get_conn() as c:
        _snapshot_label(c, session_id, image_filename, action=action)
        c.execute(
            """
            INSERT INTO labels (
                session_id, image_filename, image_w, image_h,
                cls_model_top, cls_model_conf, cls_human_label, cls_verdict, cls_correction_weight,
                seg_model_polygons, seg_human_polygons, seg_verdict, seg_correction_weight, seg_names_dict,
                inspected_at, note
            ) VALUES (?,?,?,?, ?,?,?,?,?, ?,?,?,?,?, ?,?)
            ON CONFLICT(session_id, image_filename) DO UPDATE SET
                cls_model_top=excluded.cls_model_top,
                cls_model_conf=excluded.cls_model_conf,
                cls_human_label=excluded.cls_human_label,
                cls_verdict=excluded.cls_verdict,
                cls_correction_weight=excluded.cls_correction_weight,
                seg_model_polygons=excluded.seg_model_polygons,
                seg_human_polygons=excluded.seg_human_polygons,
                seg_verdict=excluded.seg_verdict,
                seg_correction_weight=excluded.seg_correction_weight,
                seg_names_dict=excluded.seg_names_dict,
                inspected_at=excluded.inspected_at,
                note=excluded.note
            """,
            (
                session_id, image_filename, image_w, image_h,
                cls_model_top, cls_model_conf, cls_human_label, cls_verdict, cls_correction_weight,
                json.dumps(seg_model_polygons) if seg_model_polygons is not None else None,
                json.dumps(seg_human_polygons) if seg_human_polygons is not None else None,
                seg_verdict, seg_correction_weight,
                json.dumps(seg_names_dict) if seg_names_dict is not None else None,
                inspected_at, note,
            ),
        )
        return c.execute(
            "SELECT id FROM labels WHERE session_id=? AND image_filename=?",
            (session_id, image_filename),
        ).fetchone()["id"]


def undo_last(session_id: int, image_filename: str) -> dict | None:
    """마지막 변경 전 상태로 복원. Returns: 복원된 label dict or None (history 없을 시)."""
    init_db()
    with get_conn() as c:
        h = c.execute(
            "SELECT id, snapshot FROM label_history "
            "WHERE session_id = ? AND image_filename = ? "
            "ORDER BY id DESC LIMIT 1",
            (session_id, image_filename),
        ).fetchone()
        if not h:
            return None
        snap = json.loads(h["snapshot"])
        # 현재 상태도 redo 가능성을 위해 push 하고 싶지만, 단순 undo 만 지원 (redo 미지원).
        c.execute(
            """UPDATE labels SET
                image_w=?, image_h=?,
                cls_model_top=?, cls_model_conf=?, cls_human_label=?, cls_verdict=?, cls_correction_weight=?,
                seg_model_polygons=?, seg_human_polygons=?, seg_verdict=?, seg_correction_weight=?, seg_names_dict=?,
                inspected_at=?, note=?
            WHERE session_id=? AND image_filename=?""",
            (
                snap.get("image_w"), snap.get("image_h"),
                snap.get("cls_model_top"), snap.get("cls_model_conf"),
                snap.get("cls_human_label"), snap.get("cls_verdict"),
                snap.get("cls_correction_weight") or 1.0,
                snap.get("seg_model_polygons"), snap.get("seg_human_polygons"),
                snap.get("seg_verdict"), snap.get("seg_correction_weight") or 1.0,
                snap.get("seg_names_dict"),
                snap.get("inspected_at"), snap.get("note") or "",
                session_id, image_filename,
            ),
        )
        c.execute("DELETE FROM label_history WHERE id = ?", (h["id"],))
        # 반환할 row
        row = c.execute(
            "SELECT * FROM labels WHERE session_id = ? AND image_filename = ?",
            (session_id, image_filename),
        ).fetchone()
        d = dict(row) if row else None
        if d:
            for k in ("seg_model_polygons", "seg_human_polygons", "seg_names_dict"):
                if d.get(k):
                    try:
                        d[k] = json.loads(d[k])
                    except Exception:
                        pass
        return d


def get_undo_count(session_id: int, image_filename: str) -> int:
    init_db()
    with get_conn() as c:
        row = c.execute(
            "SELECT COUNT(*) AS n FROM label_history "
            "WHERE session_id = ? AND image_filename = ?",
            (session_id, image_filename),
        ).fetchone()
        return int(row["n"]) if row else 0


def get_recent_history(session_id: int, limit: int = 20) -> list[dict]:
    """세션 전체 최근 변경 로그 (통계/감사용)."""
    init_db()
    with get_conn() as c:
        rows = c.execute(
            "SELECT image_filename, changed_at, action FROM label_history "
            "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_label(session_id: int, image_filename: str) -> dict | None:
    with get_conn() as c:
        row = c.execute(
            "SELECT * FROM labels WHERE session_id = ? AND image_filename = ?",
            (session_id, image_filename),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        for k in ("seg_model_polygons", "seg_human_polygons", "seg_names_dict"):
            if d.get(k):
                try:
                    d[k] = json.loads(d[k])
                except Exception:
                    pass
        return d


def get_labels_by_session(session_id: int) -> list[dict]:
    with get_conn() as c:
        rows = c.execute(
            "SELECT * FROM labels WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in ("seg_model_polygons", "seg_human_polygons", "seg_names_dict"):
                if d.get(k):
                    try:
                        d[k] = json.loads(d[k])
                    except Exception:
                        pass
            out.append(d)
        return out


def session_stats(session_id: int) -> dict[str, Any]:
    with get_conn() as c:
        row = c.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN inspected_at IS NOT NULL THEN 1 ELSE 0 END) AS inspected,
                SUM(CASE WHEN cls_verdict = 'wrong' OR seg_verdict = 'wrong' THEN 1 ELSE 0 END) AS wrong,
                SUM(CASE WHEN cls_verdict = 'correct' AND seg_verdict = 'correct' THEN 1 ELSE 0 END) AS correct,
                SUM(CASE WHEN cls_verdict = 'uncertain' OR seg_verdict = 'uncertain' THEN 1 ELSE 0 END) AS uncertain
            FROM labels WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        return dict(row) if row else {}


def delete_session(session_id: int) -> None:
    with get_conn() as c:
        c.execute("DELETE FROM label_history WHERE session_id = ?", (session_id,))
        c.execute("DELETE FROM labels WHERE session_id = ?", (session_id,))
        c.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


def delete_sessions(session_ids: list[int]) -> int:
    if not session_ids:
        return 0
    placeholders = ",".join("?" * len(session_ids))
    with get_conn() as c:
        c.execute(f"DELETE FROM label_history WHERE session_id IN ({placeholders})", session_ids)
        c.execute(f"DELETE FROM labels WHERE session_id IN ({placeholders})", session_ids)
        cur = c.execute(f"DELETE FROM sessions WHERE id IN ({placeholders})", session_ids)
        return cur.rowcount
