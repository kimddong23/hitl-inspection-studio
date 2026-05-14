import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Streamlit,
  withStreamlitConnection,
  type ComponentProps,
} from 'streamlit-component-lib'
import { Stage, Layer, Image as KImage, Line, Circle, Group, Rect, Text } from 'react-konva'
import type Konva from 'konva'
import './App.css'

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────
type Pt = [number, number]

interface IncomingPolygon {
  class_id: number
  class_name: string
  polygon: Pt[]
  conf?: number
  source?: string
}

interface EditorPolygon {
  id: string            // local id
  class_id: number
  class_name: string
  polygon: Pt[]         // in IMAGE coords (not screen)
  conf?: number
  source: string        // 'model' | 'model_kept' | 'user_added' | 'user_drawn'
}

interface Args {
  image_b64: string
  polygons: IncomingPolygon[]
  class_names: string[]
  image_w: number
  image_h: number
  max_width: number
  theme?: 'light' | 'dark' | string
}

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────
const COLORS_RGB: [number, number, number][] = [
  [231, 76, 60],
  [46, 204, 113],
  [52, 152, 219],
  [243, 156, 18],
  [155, 89, 182],
  [26, 188, 156],
  [241, 196, 15],
]

const colorFor = (cid: number, alpha = 1): string => {
  const [r, g, b] = COLORS_RGB[((cid % COLORS_RGB.length) + COLORS_RGB.length) % COLORS_RGB.length]
  return alpha === 1 ? `rgb(${r},${g},${b})` : `rgba(${r},${g},${b},${alpha})`
}

const newId = (): string =>
  `p_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`

// Distance from point P to segment AB; returns {dist, t (0..1), foot}
const projectToSegment = (
  px: number, py: number,
  ax: number, ay: number,
  bx: number, by: number,
): { dist: number; t: number; foot: Pt } => {
  const dx = bx - ax
  const dy = by - ay
  const lenSq = dx * dx + dy * dy
  let t = 0
  if (lenSq > 1e-9) {
    t = ((px - ax) * dx + (py - ay) * dy) / lenSq
    t = Math.max(0, Math.min(1, t))
  }
  const fx = ax + t * dx
  const fy = ay + t * dy
  const ddx = px - fx
  const ddy = py - fy
  return { dist: Math.sqrt(ddx * ddx + ddy * ddy), t, foot: [fx, fy] }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────────────────
type Mode = 'select' | 'add_polygon' | 'add_rect'

function PolygonEditor({ args }: ComponentProps) {
  const a = args as Args
  const { image_b64, polygons: initialPolys, class_names, image_w, image_h, max_width } = a
  const theme: 'light' | 'dark' = (a.theme === 'light' ? 'light' : 'dark')
  const isLight = theme === 'light'

  // ── State
  const [polys, setPolys] = useState<EditorPolygon[]>(() =>
    (initialPolys || []).map((p) => ({
      id: newId(),
      class_id: p.class_id,
      class_name: p.class_name,
      polygon: (p.polygon || []).map((pt) => [Number(pt[0]), Number(pt[1])] as Pt),
      conf: p.conf,
      source: p.source || 'model',
    })),
  )
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [selectedVertex, setSelectedVertex] = useState<{ polyId: string; idx: number } | null>(null)
  const [mode, setMode] = useState<Mode>('select')
  const [drawingPts, setDrawingPts] = useState<Pt[]>([])      // image-coords
  const [hoverEdge, setHoverEdge] = useState<{ polyId: string; segIdx: number; foot: Pt } | null>(null)
  const [, setDraggingPolyId] = useState<string | null>(null)
  const [rectStart, setRectStart] = useState<Pt | null>(null)
  const [rectEnd, setRectEnd] = useState<Pt | null>(null)
  const [draftClassId, setDraftClassId] = useState<number>(0)

  // zoom / pan: independent transform on container layer (we use Stage scale + offset).
  const [stageScale, setStageScale] = useState<number>(1)
  const [stagePos, setStagePos] = useState<{ x: number; y: number }>({ x: 0, y: 0 })

  // history
  const historyRef = useRef<EditorPolygon[][]>([])
  const futureRef = useRef<EditorPolygon[][]>([])
  const skipHistoryRef = useRef<boolean>(false)

  // image
  const [imgEl, setImgEl] = useState<HTMLImageElement | null>(null)

  // ── Responsive: track iframe inner width
  const [availWidth, setAvailWidth] = useState<number>(
    typeof window !== 'undefined' ? window.innerWidth : max_width
  )
  useEffect(() => {
    if (typeof window === 'undefined') return
    const update = () => setAvailWidth(
      document.documentElement.clientWidth || window.innerWidth
    )
    update()
    window.addEventListener('resize', update)
    const ro = new ResizeObserver(update)
    ro.observe(document.documentElement)
    return () => {
      window.removeEventListener('resize', update)
      ro.disconnect()
    }
  }, [])

  // ── Computed: base scale = shrink image_w to fit min(availWidth, max_width)
  const effectiveMaxW = Math.max(200, Math.min(availWidth - 8, max_width || 99999))
  const baseScale = useMemo(() => {
    if (!image_w || !effectiveMaxW) return 1
    return Math.min(1, effectiveMaxW / image_w)
  }, [image_w, effectiveMaxW])
  const baseW = Math.round(image_w * baseScale)
  const baseH = Math.round(image_h * baseScale)

  // Stage container ref
  const stageRef = useRef<Konva.Stage | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)

  // ─── Load image
  useEffect(() => {
    if (!image_b64) return
    const im = new window.Image()
    im.crossOrigin = 'anonymous'
    im.onload = () => setImgEl(im)
    im.onerror = () => setImgEl(null)
    im.src = image_b64
  }, [image_b64])

  // ─── Set Streamlit frame height to fit
  useEffect(() => {
    // toolbar ~ 60 + 30 padding; stage = baseH; bottom info ~ 40
    const h = baseH + 60 + 50
    Streamlit.setFrameHeight(h)
  }, [baseH, polys.length])

  // ─── Push history before destructive update
  const pushHistory = useCallback((next: EditorPolygon[]) => {
    if (skipHistoryRef.current) {
      skipHistoryRef.current = false
    } else {
      historyRef.current.push(polys)
      if (historyRef.current.length > 100) historyRef.current.shift()
      futureRef.current = []
    }
    setPolys(next)
  }, [polys])

  // ─── Notify Streamlit on changes (debounced — vertex drag 중 매 프레임 rerun 방지)
  useEffect(() => {
    const timer = setTimeout(() => {
      const payload = {
        polygons: polys.map((p) => ({
          class_id: p.class_id,
          class_name: p.class_name,
          polygon: p.polygon.map(([x, y]) => [x, y]),
          source: p.source,
          conf: p.conf ?? null,
        })),
      }
      Streamlit.setComponentValue(payload)
    }, 300) // 300ms 안에 변경 멈춰야 streamlit 으로 전송 — drag 중 끊김 방지
    return () => clearTimeout(timer)
  }, [polys])

  // ─── coordinate transforms
  // Konva stage scale `baseScale * stageScale` maps IMAGE → SCREEN (px in stage)
  // But we will store positions in IMAGE coords and scale the *layer* via Stage.
  // For simplicity, the Stage is fixed at baseW x baseH; image is drawn at baseScale,
  // polygons are drawn at baseScale too (multiply coords).
  // Stage scale & pos handle zoom/pan ON TOP of that.

  const imgToStage = useCallback((p: Pt): Pt => {
    return [p[0] * baseScale, p[1] * baseScale]
  }, [baseScale])

  const stageToImg = useCallback((p: Pt): Pt => {
    return [p[0] / baseScale, p[1] / baseScale]
  }, [baseScale])

  // Get image-coord pointer position considering current stage transform
  const getImagePointer = useCallback((): Pt | null => {
    const stage = stageRef.current
    if (!stage) return null
    const pos = stage.getPointerPosition()
    if (!pos) return null
    // pos is in stage-screen pixels; remove stage scale+pos
    const sx = (pos.x - stagePos.x) / stageScale
    const sy = (pos.y - stagePos.y) / stageScale
    return stageToImg([sx, sy])
  }, [stageScale, stagePos, stageToImg])

  // ─── Mode helpers
  const setModeWithReset = useCallback((m: Mode) => {
    setMode(m)
    setDrawingPts([])
    setRectStart(null)
    setRectEnd(null)
    setSelectedId(null)
    setSelectedVertex(null)
    setHoverEdge(null)
  }, [])

  // ─── Polygon ops
  const deletePoly = useCallback((id: string) => {
    pushHistory(polys.filter((p) => p.id !== id))
    if (selectedId === id) setSelectedId(null)
  }, [polys, pushHistory, selectedId])

  const deleteVertex = useCallback((polyId: string, idx: number) => {
    const next = polys.map((p) => {
      if (p.id !== polyId) return p
      if (p.polygon.length <= 3) return p // keep at least 3 vertices
      return { ...p, polygon: p.polygon.filter((_, i) => i !== idx) }
    })
    pushHistory(next)
    setSelectedVertex(null)
  }, [polys, pushHistory])

  const insertVertex = useCallback((polyId: string, segIdx: number, pt: Pt) => {
    const next = polys.map((p) => {
      if (p.id !== polyId) return p
      const arr = [...p.polygon]
      arr.splice(segIdx + 1, 0, pt)
      return { ...p, polygon: arr }
    })
    pushHistory(next)
  }, [polys, pushHistory])

  const moveVertex = useCallback((polyId: string, idx: number, pt: Pt) => {
    // No history push on intermediate drag; commit on dragend
    setPolys((prev) => prev.map((p) => {
      if (p.id !== polyId) return p
      const arr = p.polygon.map((q, i) => (i === idx ? pt : q))
      return { ...p, polygon: arr }
    }))
  }, [])

  const movePolygonBy = useCallback((polyId: string, dx: number, dy: number) => {
    setPolys((prev) => prev.map((p) => {
      if (p.id !== polyId) return p
      return { ...p, polygon: p.polygon.map(([x, y]) => [x + dx, y + dy] as Pt) }
    }))
  }, [])

  const changeClass = useCallback((polyId: string, classId: number) => {
    const cname = class_names[classId] || `class_${classId}`
    const next = polys.map((p) => p.id === polyId ? { ...p, class_id: classId, class_name: cname } : p)
    pushHistory(next)
  }, [class_names, polys, pushHistory])

  // closeCurrentPolygon — keyboard handler 위에서 정의 필요
  const closeCurrentPolygon = useCallback(() => {
    if (mode !== 'add_polygon' || drawingPts.length < 3) return
    const cname = class_names[draftClassId] || `class_${draftClassId}`
    pushHistory([
      ...polys,
      {
        id: newId(),
        class_id: draftClassId,
        class_name: cname,
        polygon: [...drawingPts],
        source: 'user_drawn',
      },
    ])
    setDrawingPts([])
  }, [mode, drawingPts, polys, pushHistory, class_names, draftClassId])

  // ─── Keyboard
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Ctrl/Cmd + Z
      const meta = e.ctrlKey || e.metaKey
      if (meta && e.key.toLowerCase() === 'z') {
        e.preventDefault()
        if (e.shiftKey) {
          // redo
          const fu = futureRef.current.pop()
          if (fu) {
            historyRef.current.push(polys)
            skipHistoryRef.current = true
            setPolys(fu)
          }
        } else {
          // undo
          const hi = historyRef.current.pop()
          if (hi) {
            futureRef.current.push(polys)
            skipHistoryRef.current = true
            setPolys(hi)
          }
        }
        return
      }
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selectedVertex) {
          e.preventDefault()
          deleteVertex(selectedVertex.polyId, selectedVertex.idx)
        } else if (selectedId) {
          e.preventDefault()
          deletePoly(selectedId)
        }
      }
      if (e.key === 'Escape') {
        if (mode === 'add_polygon' && drawingPts.length > 0) {
          // Esc 는 그리던 폴리곤 취소
          setDrawingPts([])
        } else {
          setModeWithReset('select')
        }
      }
      if (e.key === 'Enter' && mode === 'add_polygon' && drawingPts.length >= 3) {
        e.preventDefault()
        closeCurrentPolygon()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [polys, selectedId, selectedVertex, deletePoly, deleteVertex, setModeWithReset,
       mode, drawingPts, closeCurrentPolygon])

  // ─── Stage event handlers
  const onStageClick = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
    const target = e.target
    const onEmpty = target === stageRef.current || target.getClassName() === 'Image'
    if (mode === 'add_polygon') {
      const p = getImagePointer()
      if (!p) return
      // 자동 close 제거 — 명시적 더블클릭으로만 닫기 (사용자 의도와 일치)
      setDrawingPts((prev) => [...prev, p])
      return
    }
    if (mode === 'select' && onEmpty) {
      setSelectedId(null)
      setSelectedVertex(null)
    }
  }, [mode, drawingPts, baseScale, polys, pushHistory, getImagePointer, class_names, draftClassId])

  const onStageDblClick = useCallback(() => {
    // 더블클릭 close 제거 — 빠른 클릭 오인식 방지.
    // 마감 = (1) 첫 점 (초록 원) 다시 클릭 표준 또는 (2) Enter 키.
  }, [])

  const onStageMouseDown = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
    if (mode === 'add_rect') {
      const p = getImagePointer()
      if (!p) return
      setRectStart(p)
      setRectEnd(p)
      return
    }
    // pan with empty drag (in select mode)
    if (mode === 'select') {
      const target = e.target
      const onEmpty = target === stageRef.current || target.getClassName() === 'Image'
      if (onEmpty && (e.evt.button === 1 || e.evt.shiftKey || e.evt.altKey)) {
        // middle / shift / alt drag = pan; we use Konva stage drag
        const stage = stageRef.current
        if (stage) stage.draggable(true)
      }
    }
  }, [mode, getImagePointer])

  const onStageMouseMove = useCallback(() => {
    if (mode === 'add_rect' && rectStart) {
      const p = getImagePointer()
      if (p) setRectEnd(p)
      return
    }
    if (mode === 'select') {
      // hover-edge detection (image coords)
      const p = getImagePointer()
      if (!p) { setHoverEdge(null); return }
      let best: { polyId: string; segIdx: number; foot: Pt; dist: number } | null = null
      const threshPx = 8 / (baseScale * stageScale) // 8 screen px → image units
      for (const poly of polys) {
        const pts = poly.polygon
        for (let i = 0; i < pts.length; i++) {
          const a = pts[i]
          const b = pts[(i + 1) % pts.length]
          const r = projectToSegment(p[0], p[1], a[0], a[1], b[0], b[1])
          if (r.dist < threshPx && (!best || r.dist < best.dist)) {
            // ignore if foot is too close to existing vertex
            const dToA = Math.hypot(r.foot[0] - a[0], r.foot[1] - a[1])
            const dToB = Math.hypot(r.foot[0] - b[0], r.foot[1] - b[1])
            if (dToA < threshPx * 0.8 || dToB < threshPx * 0.8) continue
            best = { polyId: poly.id, segIdx: i, foot: r.foot, dist: r.dist }
          }
        }
      }
      if (best) setHoverEdge({ polyId: best.polyId, segIdx: best.segIdx, foot: best.foot })
      else setHoverEdge(null)
    }
  }, [mode, rectStart, polys, baseScale, stageScale, getImagePointer])

  const onStageMouseUp = useCallback(() => {
    if (mode === 'add_rect' && rectStart && rectEnd) {
      const [x0, y0] = rectStart
      const [x1, y1] = rectEnd
      const minX = Math.min(x0, x1), maxX = Math.max(x0, x1)
      const minY = Math.min(y0, y1), maxY = Math.max(y0, y1)
      if (maxX - minX > 3 && maxY - minY > 3) {
        const cname = class_names[draftClassId] || `class_${draftClassId}`
        pushHistory([
          ...polys,
          {
            id: newId(),
            class_id: draftClassId,
            class_name: cname,
            polygon: [
              [minX, minY], [maxX, minY], [maxX, maxY], [minX, maxY],
            ],
            source: 'user_drawn',
          },
        ])
      }
      setRectStart(null)
      setRectEnd(null)
      return
    }
    // stop panning
    const stage = stageRef.current
    if (stage && stage.draggable()) {
      setStagePos({ x: stage.x(), y: stage.y() })
      stage.draggable(false)
    }
  }, [mode, rectStart, rectEnd, polys, pushHistory, class_names, draftClassId])

  const onWheel = useCallback((e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault()
    const stage = stageRef.current
    if (!stage) return
    const scaleBy = 1.07
    const oldScale = stageScale
    const pointer = stage.getPointerPosition()
    if (!pointer) return
    const mousePointTo = {
      x: (pointer.x - stagePos.x) / oldScale,
      y: (pointer.y - stagePos.y) / oldScale,
    }
    const direction = e.evt.deltaY > 0 ? -1 : 1
    let newScale = direction > 0 ? oldScale * scaleBy : oldScale / scaleBy
    newScale = Math.max(0.2, Math.min(10, newScale))
    setStageScale(newScale)
    setStagePos({
      x: pointer.x - mousePointTo.x * newScale,
      y: pointer.y - mousePointTo.y * newScale,
    })
  }, [stageScale, stagePos])

  const onContextMenu = useCallback((e: Konva.KonvaEventObject<PointerEvent>) => {
    e.evt.preventDefault()
  }, [])

  // ─── Click on polygon body → select
  const onPolyClick = useCallback((id: string, e: Konva.KonvaEventObject<MouseEvent>) => {
    if (mode !== 'select') return
    e.cancelBubble = true
    setSelectedId(id)
    setSelectedVertex(null)
  }, [mode])

  // ─── Click on edge with hoverEdge → insert vertex
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (!hoverEdge) return
      if (mode !== 'select') return
      // Only handle if click target is the stage container
      const cont = stageRef.current?.container()
      if (!cont || !cont.contains(e.target as Node)) return
      // Skip if user is clicking a vertex
      if ((e.target as HTMLElement).tagName === 'CANVAS') {
        // We do nothing here — the actual click is handled by Konva targets.
        // The PLUS marker captures clicks via its own onClick below.
      }
    }
    window.addEventListener('click', handler)
    return () => window.removeEventListener('click', handler)
  }, [hoverEdge, mode])

  // ─── Reset zoom
  const resetView = useCallback(() => {
    setStageScale(1)
    setStagePos({ x: 0, y: 0 })
  }, [])

  // ─── Selected polygon (for toolbar)
  const selectedPoly = polys.find((p) => p.id === selectedId) || null

  // ─── Render preview line for drawing-mode
  const drawingPreview = useMemo(() => {
    if (mode !== 'add_polygon' || drawingPts.length === 0) return null
    const flat: number[] = []
    for (const p of drawingPts) {
      const [sx, sy] = imgToStage(p)
      flat.push(sx, sy)
    }
    return flat
  }, [drawingPts, imgToStage, mode])

  const rectPreview = useMemo(() => {
    if (mode !== 'add_rect' || !rectStart || !rectEnd) return null
    const [sx0, sy0] = imgToStage(rectStart)
    const [sx1, sy1] = imgToStage(rectEnd)
    return { x: Math.min(sx0, sx1), y: Math.min(sy0, sy1), w: Math.abs(sx1 - sx0), h: Math.abs(sy1 - sy0) }
  }, [mode, rectStart, rectEnd, imgToStage])

  // ─── Cursor
  const cursor = mode === 'add_polygon' || mode === 'add_rect' ? 'crosshair' : 'default'

  return (
    <div ref={containerRef} className="pe-root" data-theme={theme} style={{ width: baseW + 4 }}>
      {/* Toolbar */}
      <div className="pe-toolbar">
        <div className="pe-modes">
          <button
            type="button"
            className={mode === 'select' ? 'pe-btn pe-btn-on' : 'pe-btn'}
            onClick={() => setModeWithReset('select')}
            title="선택 / 편집 (V)"
          >선택</button>
          <button
            type="button"
            className={mode === 'add_polygon' ? 'pe-btn pe-btn-on' : 'pe-btn'}
            onClick={() => setModeWithReset('add_polygon')}
            title="폴리곤 추가 — 클릭으로 점 추가, 더블클릭 또는 첫점 다시 클릭으로 닫기"
          >+ 폴리곤</button>
          <button
            type="button"
            className={mode === 'add_rect' ? 'pe-btn pe-btn-on' : 'pe-btn'}
            onClick={() => setModeWithReset('add_rect')}
            title="사각형 추가 — 클릭+드래그"
          >+ 사각형</button>
        </div>
        <div className="pe-divider" />
        <label className="pe-label">
          새 클래스
          <select
            value={draftClassId}
            onChange={(e) => setDraftClassId(parseInt(e.target.value))}
            className="pe-select"
          >
            {class_names.map((n, i) => (
              <option key={`${n}_${i}`} value={i}>{n}</option>
            ))}
          </select>
        </label>
        <div className="pe-divider" />
        {selectedPoly && (
          <>
            <label className="pe-label">
              선택된 클래스
              <select
                value={selectedPoly.class_id}
                onChange={(e) => changeClass(selectedPoly.id, parseInt(e.target.value))}
                className="pe-select"
              >
                {class_names.map((n, i) => (
                  <option key={`s_${n}_${i}`} value={i}>{n}</option>
                ))}
                {/* fallback if class_id not in list */}
                {!class_names.includes(selectedPoly.class_name) && (
                  <option value={selectedPoly.class_id}>{selectedPoly.class_name}</option>
                )}
              </select>
            </label>
            <button
              type="button"
              className="pe-btn pe-btn-danger"
              onClick={() => deletePoly(selectedPoly.id)}
              title="선택 폴리곤 삭제 (Delete)"
            >삭제</button>
          </>
        )}
        <div className="pe-spacer" />
        <button type="button" className="pe-btn" onClick={resetView} title="줌/팬 초기화">↺ 보기</button>
        <button
          type="button"
          className="pe-btn"
          onClick={() => {
            const hi = historyRef.current.pop()
            if (hi) {
              futureRef.current.push(polys)
              skipHistoryRef.current = true
              setPolys(hi)
            }
          }}
          title="실행 취소 (Ctrl+Z)"
        >↶ Undo</button>
        <button
          type="button"
          className="pe-btn"
          onClick={() => {
            const fu = futureRef.current.pop()
            if (fu) {
              historyRef.current.push(polys)
              skipHistoryRef.current = true
              setPolys(fu)
            }
          }}
          title="다시 실행 (Ctrl+Shift+Z)"
        >↷ Redo</button>
      </div>

      {/* Stage */}
      <div className="pe-stage-wrap" style={{ width: baseW, height: baseH, cursor }}>
        <Stage
          width={baseW}
          height={baseH}
          ref={stageRef}
          scaleX={stageScale}
          scaleY={stageScale}
          x={stagePos.x}
          y={stagePos.y}
          onMouseDown={onStageMouseDown}
          onMouseMove={onStageMouseMove}
          onMouseUp={onStageMouseUp}
          onClick={onStageClick}
          onDblClick={onStageDblClick}
          onWheel={onWheel}
          onContextMenu={onContextMenu}
          onDragEnd={() => {
            const s = stageRef.current
            if (s) setStagePos({ x: s.x(), y: s.y() })
          }}
        >
          <Layer listening>
            {imgEl && (
              <KImage
                image={imgEl}
                x={0}
                y={0}
                width={baseW}
                height={baseH}
                listening={true}
              />
            )}
          </Layer>

          {/* polygons */}
          <Layer>
            {polys.map((poly) => {
              const isSel = poly.id === selectedId
              const flat: number[] = []
              for (const p of poly.polygon) {
                const [sx, sy] = imgToStage(p)
                flat.push(sx, sy)
              }
              const fill = colorFor(poly.class_id, 0.18)
              const stroke = colorFor(poly.class_id, 1)
              return (
                <Group
                  key={poly.id}
                  draggable={mode === 'select'}
                  onClick={(e) => onPolyClick(poly.id, e)}
                  onTap={(e) => onPolyClick(poly.id, e as unknown as Konva.KonvaEventObject<MouseEvent>)}
                  onDragStart={() => {
                    setDraggingPolyId(poly.id)
                    historyRef.current.push(polys)
                    futureRef.current = []
                    skipHistoryRef.current = true
                  }}
                  onDragEnd={(e) => {
                    // Apply final offset only on drag end (avoids accumulation race).
                    const node = e.target
                    const dx = node.x() / baseScale
                    const dy = node.y() / baseScale
                    if (dx !== 0 || dy !== 0) {
                      movePolygonBy(poly.id, dx, dy)
                    }
                    node.position({ x: 0, y: 0 })
                    setDraggingPolyId(null)
                  }}
                >
                  <Line
                    points={flat}
                    closed
                    fill={fill}
                    stroke={stroke}
                    strokeWidth={isSel ? 2.5 : 1.5}
                    hitStrokeWidth={12}
                    listening={mode === 'select'}
                  />
                  {/* class name label */}
                  {poly.polygon.length > 0 && (() => {
                    const [lx, ly] = imgToStage(poly.polygon[0])
                    return (
                      <Text
                        x={lx + 4}
                        y={ly - 16}
                        text={poly.class_name + (poly.conf != null ? ` ${poly.conf.toFixed(2)}` : '')}
                        fontSize={11}
                        fill="#fff"
                        stroke="#000"
                        strokeWidth={0.6}
                        fillAfterStrokeEnabled
                        padding={2}
                        listening={false}
                      />
                    )
                  })()}
                </Group>
              )
            })}
          </Layer>

          {/* vertices on selected polygon */}
          <Layer>
            {selectedPoly && selectedPoly.polygon.map((p, idx) => {
              const [sx, sy] = imgToStage(p)
              const isVSel = selectedVertex?.polyId === selectedPoly.id && selectedVertex.idx === idx
              return (
                <Circle
                  key={`v_${selectedPoly.id}_${idx}`}
                  x={sx}
                  y={sy}
                  radius={isVSel ? 7 : 5}
                  fill={isVSel ? (isLight ? '#000' : '#fff') : colorFor(selectedPoly.class_id, 1)}
                  stroke={isLight ? '#000' : '#fff'}
                  strokeWidth={1.5}
                  draggable
                  onMouseDown={(e) => { e.cancelBubble = true }}
                  onClick={(e) => {
                    e.cancelBubble = true
                    setSelectedVertex({ polyId: selectedPoly.id, idx })
                  }}
                  onContextMenu={(e) => {
                    e.evt.preventDefault()
                    e.cancelBubble = true
                    deleteVertex(selectedPoly.id, idx)
                  }}
                  onDragStart={() => {
                    historyRef.current.push(polys)
                    futureRef.current = []
                    skipHistoryRef.current = true
                  }}
                  onDragMove={(e) => {
                    const node = e.target
                    const ip = stageToImg([node.x(), node.y()])
                    moveVertex(selectedPoly.id, idx, ip)
                  }}
                  onDragEnd={(e) => {
                    const node = e.target
                    const ip = stageToImg([node.x(), node.y()])
                    moveVertex(selectedPoly.id, idx, ip)
                  }}
                />
              )
            })}

            {/* + marker on hovered edge */}
            {hoverEdge && mode === 'select' && (() => {
              const [fx, fy] = imgToStage(hoverEdge.foot)
              return (
                <Group
                  x={fx}
                  y={fy}
                  onClick={(e) => {
                    e.cancelBubble = true
                    insertVertex(hoverEdge.polyId, hoverEdge.segIdx, hoverEdge.foot)
                  }}
                >
                  <Circle radius={8} fill={isLight ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.95)'} stroke="#2da44e" strokeWidth={2} />
                  <Text
                    text="+"
                    fontSize={14}
                    fontStyle="bold"
                    fill="#2da44e"
                    offsetX={4}
                    offsetY={7}
                    listening={false}
                  />
                </Group>
              )
            })()}

            {/* drawing preview line (polygon mode) */}
            {drawingPreview && drawingPreview.length >= 2 && (
              <>
                <Line
                  points={drawingPreview}
                  stroke={colorFor(draftClassId, 1)}
                  strokeWidth={2}
                  dash={[6, 4]}
                  listening={false}
                />
                {drawingPts.map((p, i) => {
                  const [sx, sy] = imgToStage(p)
                  const isFirstClosable = i === 0 && drawingPts.length >= 3
                  if (isFirstClosable) {
                    // 첫 점을 큰 close target 으로 렌더 (표준 UX — LabelMe/CVAT 방식)
                    return (
                      <Group
                        key={`dp_${i}`}
                        x={sx}
                        y={sy}
                        onMouseEnter={(e) => {
                          const stage = e.target.getStage()
                          if (stage) stage.container().style.cursor = 'pointer'
                        }}
                        onMouseLeave={(e) => {
                          const stage = e.target.getStage()
                          if (stage) stage.container().style.cursor = 'crosshair'
                        }}
                        onClick={(e) => {
                          e.cancelBubble = true
                          closeCurrentPolygon()
                        }}
                      >
                        <Circle
                          radius={10}
                          fill="rgba(45,164,78,0.35)"
                          stroke="#2da44e"
                          strokeWidth={2}
                        />
                        <Circle
                          radius={4}
                          fill="#2da44e"
                          stroke={isLight ? '#000' : '#fff'}
                          strokeWidth={1}
                          listening={false}
                        />
                      </Group>
                    )
                  }
                  return (
                    <Circle
                      key={`dp_${i}`}
                      x={sx}
                      y={sy}
                      radius={4}
                      fill={colorFor(draftClassId, 1)}
                      stroke={isLight ? '#000' : '#fff'}
                      strokeWidth={1}
                      listening={false}
                    />
                  )
                })}
              </>
            )}

            {/* drawing preview rect */}
            {rectPreview && (
              <Rect
                x={rectPreview.x}
                y={rectPreview.y}
                width={rectPreview.w}
                height={rectPreview.h}
                fill={colorFor(draftClassId, 0.18)}
                stroke={colorFor(draftClassId, 1)}
                strokeWidth={2}
                dash={[6, 4]}
                listening={false}
              />
            )}
          </Layer>
        </Stage>
      </div>

      {/* status line */}
      <div className="pe-status">
        <span>모드: <b>{mode === 'select' ? '선택/편집' : mode === 'add_polygon' ? '폴리곤 추가' : '사각형 추가'}</b></span>
        <span>· 폴리곤 {polys.length}개</span>
        {selectedPoly && <span>· 선택: {selectedPoly.class_name} ({selectedPoly.polygon.length}점)</span>}
        <span style={{ color: isLight ? '#6e7681' : '#888', fontSize: 11 }}>
          {mode === 'select' && '드래그: 이동 · 정점 우클릭/Delete: 삭제 · 엣지 hover: 정점 추가'}
          {mode === 'add_polygon' && '클릭: 점 추가 · 점 3개 이상 후 첫 점(초록 원) 다시 클릭 또는 Enter: 닫기 · Esc: 취소'}
          {mode === 'add_rect' && '클릭+드래그: 사각형 · Esc: 취소'}
        </span>
      </div>
    </div>
  )
}

export default withStreamlitConnection(PolygonEditor)
