# 서버 배포 시 수정 사항 (체크리스트)

PoC (사용자 Mac local) → 회사 서버 배포 시 변경/추가해야 할 항목 모음.

---

## 1. 저장소 / DB

| 항목 | 현재 (PoC) | 서버 배포 시 |
|---|---|---|
| 라벨 DB | SQLite (`hitl_labels.db`) | PostgreSQL 또는 SQLite + WAL 모드 |
| 이미지 임시 저장 | Streamlit `tempfile` (메모리) | 공유 NFS / S3 / MinIO |
| 모델 파일 | 업로드 또는 path | 공유 storage 또는 미리 등록한 path 만 |
| 라벨 export 결과 | 클라이언트 다운로드 | 서버 storage 동시 저장 + 다운로드 |

### 수정 코드

**`db.py`**:
```python
import os
DB_PATH = Path(os.environ.get("HITL_DB_PATH", "hitl_labels.db"))
# 또는 SQLAlchemy 로 PostgreSQL 지원
```

**`app.py`**:
```python
STORAGE_DIR = os.environ.get("HITL_STORAGE_DIR", "/mnt/hitl_data")
MODEL_DIR = os.environ.get("HITL_MODEL_DIR", "/mnt/hitl_models")
```

**우선순위**: 高

---

## 2. 인증 / 권한 (다중 사용자)

| 항목 | 현재 | 서버 |
|---|---|---|
| 인증 | 없음 (single user) | streamlit-authenticator / OAuth / SSO |
| 라벨러 ID | 없음 | 사용자별 추적 (DB 컬럼 `labeler_id`) |
| 권한 | 모든 사용자 동일 | admin / labeler / viewer 역할 |

### 추가 코드

```python
# requirements.txt
streamlit-authenticator

# app.py 시작 부분
import streamlit_authenticator as stauth
authenticator = stauth.Authenticate(...)
name, auth_status, username = authenticator.login(...)
if not auth_status:
    st.stop()
st.session_state.labeler_id = username
```

**DB 스키마 추가**:
```sql
ALTER TABLE labels ADD COLUMN labeler_id TEXT;
ALTER TABLE labels ADD COLUMN labeler_name TEXT;
```

**우선순위**: 高 (다중 사용자 시 필수)

---

## 3. 다중 사용자 동시 처리

| 항목 | 현재 | 서버 |
|---|---|---|
| Session 분리 | Streamlit 기본 — OK | 그대로 |
| DB 동시 쓰기 | SQLite default = lock 충돌 가능 | WAL 모드 또는 PostgreSQL |
| GPU 추론 | 1명 — 즉시 | 다중 → 큐잉 필요 |
| 모델 로딩 | 매 사용자 별도 로드 (메모리 ↑) | 전역 캐싱 (`@st.cache_resource`) |

### 수정 코드

**`inference.py` 모델 캐싱**:
```python
import streamlit as st

@st.cache_resource
def load_classifier_cached(path: str):
    return YOLO(path)

@st.cache_resource
def load_segmenter_cached(path: str):
    return YOLO(path)
```

**GPU 큐잉** (사용자 ≥ 5명 시):
- 옵션 A: `threading.Lock` 으로 단순 직렬화
- 옵션 B: Celery + Redis worker (정공법)
- 옵션 C: FastAPI 별도 추론 서버 + Streamlit은 client

**우선순위**: 中 (사용자 5명 미만이면 보류)

---

## 4. Docker 컨테이너화

### Dockerfile (예시)

```dockerfile
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HITL_DB_PATH=/data/hitl_labels.db \
    HITL_STORAGE_DIR=/data/storage \
    HITL_MODEL_DIR=/data/models

RUN apt-get update && apt-get install -y \
    python3 python3-pip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY *.py ./

EXPOSE 8501
HEALTHCHECK CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false", \
     "--server.maxUploadSize=500"]
```

### docker-compose.yml

```yaml
version: "3.9"
services:
  hitl_app:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - HITL_DB_PATH=/data/hitl_labels.db
    volumes:
      - ./data:/data
    ports:
      - "8501:8501"
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  # (선택) PostgreSQL
  # postgres:
  #   image: postgres:16
  #   environment:
  #     POSTGRES_DB: hitl
  #     POSTGRES_USER: hitl
  #     POSTGRES_PASSWORD: ...
  #   volumes:
  #     - ./pgdata:/var/lib/postgresql/data

  # (선택) nginx HTTPS
  # nginx:
  #   image: nginx:alpine
  #   ports:
  #     - "443:443"
  #     - "80:80"
  #   volumes:
  #     - ./nginx.conf:/etc/nginx/nginx.conf
  #     - ./certs:/etc/nginx/certs
```

### NVIDIA Container Toolkit 설치 (호스트)

```bash
# Ubuntu 호스트
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**우선순위**: 高 (운영 배포 필수)

---

## 5. 네트워크 / 보안

| 항목 | 변경 |
|---|---|
| HTTPS | nginx/caddy reverse proxy + Let's Encrypt 또는 사내 인증서 |
| 접근 제어 | 인트라넷 only / VPN required / IP allowlist |
| File upload limit | `--server.maxUploadSize=500` (default 200MB) — zip 큰 경우 조정 |
| WebSocket (Streamlit 사용) | nginx 에서 `proxy_set_header Upgrade $http_upgrade;` 필수 |
| CORS | 회사 도메인 only 화이트리스트 |
| Streamlit CSRF | `--server.enableCORS=true --server.enableXsrfProtection=true` |

### nginx.conf 예시 (HTTPS + WebSocket)

```nginx
server {
    listen 443 ssl http2;
    server_name hitl.company.com;
    ssl_certificate /etc/nginx/certs/cert.pem;
    ssl_certificate_key /etc/nginx/certs/key.pem;

    client_max_body_size 500M;

    location / {
        proxy_pass http://hitl_app:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
```

**우선순위**: 高

---

## 6. 로깅 / 모니터링

| 항목 | 변경 |
|---|---|
| 사용자 액션 로그 | DB `audit_log` 테이블 추가 (login/save/export 등) |
| 추론 시간 모니터 | 각 추론 시작/종료 timestamp DB 저장 |
| 에러 로그 | `logging` 모듈 + 파일 출력 |
| GPU 사용량 | `nvidia-smi` 또는 `pynvml` 으로 주기 측정 |
| 사용량 dashboard | Grafana 또는 Streamlit admin 페이지 |

### 추가 코드 (`logger.py`)

```python
import logging
from pathlib import Path
log_dir = Path("/data/logs")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "hitl.log"),
        logging.StreamHandler(),
    ],
)
```

**우선순위**: 中

---

## 7. GPU 최적화

| 항목 | 변경 |
|---|---|
| 모델 캐싱 | `@st.cache_resource` (전역) — 한 번 로드 후 모든 세션 공유 |
| FP16 추론 | NVIDIA GPU 한정. `model.predict(..., half=True)` |
| Batch 추론 | 다중 사용자 큐잉 시 `model.predict([img1, img2, ...])` |
| 모델 미리 워밍업 | 서버 시작 시 dummy 추론 1회 (첫 사용자 대기 최소화) |

**예시 — `inference.py` warmup**:
```python
def warmup(model, device: str = "cuda"):
    from PIL import Image
    dummy = Image.new("RGB", (640, 640), (128, 128, 128))
    _ = model.predict(dummy, verbose=False, device=device)
```

**우선순위**: 中

---

## 8. 백업 / 복구

| 항목 | 변경 |
|---|---|
| DB 정기 백업 | cron 또는 systemd timer: `sqlite3 db .dump | gzip > backup_$(date).sql.gz` |
| 모델 파일 백업 | NAS 동기화 (`rsync --delete /data/models /backup/models`) |
| 라벨 export 자동화 | 매주 새 라벨 자동 export → fine-tune trigger |
| 컨테이너 데이터 | docker volume 별도 백업 |

**우선순위**: 中

---

## 9. 사용자 경험 (UX) 추가

| 항목 | 변경 |
|---|---|
| 단축키 브라우저별 테스트 | Chrome/Firefox/Edge 확인 |
| 파일 크기 알림 | 업로드 전 크기 표시 |
| 처리 중 상태 | "추론 중 — 다른 사용자 대기 중" 메시지 |
| 자동 저장 | 검수 도중 1분마다 자동 DB upsert (진행 중 끊겨도 복원) |
| 검수 history | undo 버튼 / 변경 이력 |

**우선순위**: 中

---

## 10. 운영 / 유지보수

| 항목 | 변경 |
|---|---|
| 모델 버전 관리 | MLflow Model Registry 또는 DVC + Git |
| A/B 테스트 | 신구 모델 동시 추론, 결과 비교 페이지 |
| Fine-tune trigger | 정정 라벨 누적량 → 자동 학습 trigger (별도 worker) |
| Drift 감지 | 입력 이미지 통계 (밝기/색상) 변화 알림 |
| 데이터 보존 | 라벨 보관 기간 정책 (예: 1년) |

**우선순위**: 낮음 (운영 안정 후)

---

## 환경변수 list (요약)

| 변수 | 설명 | 예시 |
|---|---|---|
| `HITL_DB_PATH` | 라벨 DB 경로 | `/data/hitl_labels.db` |
| `HITL_STORAGE_DIR` | 이미지/export 저장 | `/data/storage` |
| `HITL_MODEL_DIR` | 모델 .pt 디렉토리 | `/data/models` |
| `HITL_AUTH_CONFIG` | 인증 설정 파일 | `/etc/hitl/auth.yaml` |
| `STREAMLIT_SERVER_PORT` | 포트 | `8501` |
| `CUDA_VISIBLE_DEVICES` | GPU 선택 | `0` |

---

## 배포 전 최종 체크리스트

- [ ] Dockerfile 빌드 성공
- [ ] GPU passthrough 확인 (`docker run --gpus all nvidia/cuda nvidia-smi`)
- [ ] DB migration (SQLite → PostgreSQL 시)
- [ ] HTTPS 인증서 설정
- [ ] 인증 동작 확인 (다중 사용자 로그인)
- [ ] 모델 캐싱 동작 확인 (`@st.cache_resource`)
- [ ] 파일 업로드 한도 (예: 500MB) 확인
- [ ] 추론 시간 측정 (서버 GPU 기준)
- [ ] DB 백업 cron 등록
- [ ] 로그 rotation 설정 (logrotate)
- [ ] 모니터링 dashboard 확인 (GPU/CPU/메모리)
- [ ] 다중 사용자 동시 접속 테스트 (3~5명)
- [ ] 백업/복구 시뮬레이션 (DB 삭제 후 복원)
- [ ] 운영자 매뉴얼 작성 + 교육

---

## 단계별 배포 (권장)

### Step 1 — Single user staging (1주)
- Docker 컨테이너 빌드
- 회사 서버 1대에 배포
- 운영자 1명만 사용
- 추론 시간 측정 + GPU 사용량 모니터링

### Step 2 — Multi user beta (2~3주)
- 인증 추가
- DB SQLite WAL 또는 PostgreSQL 전환
- 운영자 3~5명 베타 테스트
- 버그 fix + UX 개선

### Step 3 — Production (지속)
- HTTPS + 도메인 연결
- 백업 정책 가동
- 로그/모니터링 dashboard
- Fine-tune 파이프라인 자동화 (별도 작업)

---

## 우선순위 요약

| 우선순위 | 항목 |
|---|---|
| **必** | Dockerfile + GPU passthrough, DB 경로 환경변수화, 모델 캐싱, HTTPS + 인증 |
| **重** | 다중 사용자 (인증/권한/labeler_id), 백업, 로깅 |
| **次** | 모니터링 dashboard, 자동 저장, A/B 테스트 |
| **後** | Drift 감지, MLflow, 자동 fine-tune |
