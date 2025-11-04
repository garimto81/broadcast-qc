# Re:View MVP - 최소 비용 구현 PRD

**Version**: MVP 1.0
**Last Updated**: 2025-11-03
**구현 목표**: 로컬 환경에서 최소 비용으로 핵심 기능 검증

---

## 🎯 MVP 핵심 전략

### 원칙
1. **로컬 우선**: 클라우드 비용 0원
2. **오픈소스 100%**: 라이선스 비용 0원
3. **단일 서버**: Windows PC 1대로 구동
4. **핵심 기능만**: 포커 방송 QC 필수 기능

### 목표
- **개발 비용**: 0원 (개발자 시간 제외)
- **운영 비용**: 0원 (전기료 제외)
- **검증 기간**: 4주
- **타겟 사용자**: 1개 팀 (5명 이하)

---

## 1. 최소 기능 정의 (MVP Scope)

### 1.1 포함 기능 ✅

```yaml
핵심 기능 (MUST HAVE):
  1. 영상 업로드 (로컬 스토리지)
  2. 기본 비디오 QC:
     - 블랙 프레임 감지
     - 씬 전환 로깅
  3. 기본 오디오 QC:
     - 오디오 레벨 시각화
     - 피크 감지
  4. 타임라인 UI:
     - 비디오 플레이어
     - 마커 표시
     - 코멘트 작성
  5. 간단한 리포트:
     - CSV 내보내기
```

### 1.2 제외 기능 ❌

```yaml
나중에 추가 (NICE TO HAVE):
  - AI/ML 고급 분석
  - 실시간 협업
  - 클라우드 스토리지
  - 모바일 앱
  - 사용자 인증/권한
  - API 제공
```

---

## 2. 기술 스택 (100% 무료)

### 2.1 개발 환경

```yaml
운영체제: Windows 11 (기존 PC)
런타임:
  - Python 3.11 (무료)
  - Node.js 20 LTS (무료)

개발 도구:
  - VS Code (무료)
  - Git (무료)
```

### 2.2 백엔드 스택

```yaml
웹 프레임워크: FastAPI (Python)
이유:
  - 빠른 개발
  - 자동 문서화
  - 비동기 지원

비디오 처리:
  - OpenCV (무료)
  - FFmpeg (무료)

오디오 처리:
  - Librosa (무료)
  - PyDub (무료)

데이터베이스: SQLite
이유:
  - 설치 불필요
  - 파일 기반
  - 백업 간편

작업 큐: 없음 (동기 처리)
이유:
  - 단순화
  - 즉시 피드백
```

### 2.3 프론트엔드 스택

```yaml
프레임워크: React (Create React App)
이유:
  - 빠른 시작
  - 풍부한 생태계

UI 라이브러리:
  - Ant Design (무료)
  - Tailwind CSS (무료)

비디오 플레이어: Video.js (무료)

차트: Recharts (무료)

상태 관리: Context API (내장)
```

---

## 3. 시스템 아키텍처 (단순화)

```
[프론트엔드 (React)]
        ↓ HTTP
[백엔드 API (FastAPI)]
        ↓
[로컬 파일 시스템]  [SQLite DB]
    - 영상 파일       - 메타데이터
    - 분석 결과       - 코멘트
```

### 3.1 폴더 구조

```
c:\broadcast-qc-mvp\
├── backend\
│   ├── app\
│   │   ├── main.py           # FastAPI 앱
│   │   ├── models.py         # 데이터 모델
│   │   ├── analysis.py       # 분석 로직
│   │   └── utils.py          # 유틸리티
│   ├── uploads\              # 업로드 영상
│   ├── processed\            # 처리된 파일
│   └── database.db           # SQLite DB
│
├── frontend\
│   ├── src\
│   │   ├── components\       # React 컴포넌트
│   │   ├── pages\           # 페이지
│   │   └── services\        # API 호출
│   └── public\
│
└── docs\                     # 문서
```

---

## 4. 핵심 구현 상세

### 4.1 영상 업로드 및 처리

```python
# backend/app/main.py
from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
import uuid

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    # 1. 파일 저장
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. 메타데이터 추출
    metadata = extract_video_metadata(file_path)

    # 3. DB 저장
    save_to_db(file_id, file.filename, metadata)

    # 4. 분석 시작 (동기)
    analysis_result = analyze_video(file_path)

    return {
        "file_id": file_id,
        "metadata": metadata,
        "analysis": analysis_result
    }
```

### 4.2 비디오 분석 (간단 버전)

```python
# backend/app/analysis.py
import cv2
import numpy as np

def detect_black_frames(video_path, threshold=10):
    """블랙 프레임 감지"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    black_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 평균 밝기 계산
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        if mean_brightness < threshold:
            timecode = frame_count / fps
            black_frames.append({
                "frame": frame_count,
                "timecode": timecode,
                "brightness": mean_brightness
            })

        frame_count += 1

    cap.release()
    return black_frames

def detect_scene_changes(video_path, threshold=30):
    """씬 전환 감지"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    scene_changes = []
    prev_frame = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if prev_frame is not None:
            # 프레임 차이 계산
            diff = cv2.absdiff(frame, prev_frame)
            mean_diff = np.mean(diff)

            if mean_diff > threshold:
                scene_changes.append({
                    "frame": frame_count,
                    "timecode": frame_count / fps,
                    "difference": mean_diff
                })

        prev_frame = frame
        frame_count += 1

    cap.release()
    return scene_changes
```

### 4.3 오디오 분석

```python
# backend/app/audio_analysis.py
import librosa
import numpy as np

def analyze_audio(video_path):
    """오디오 분석"""
    # 비디오에서 오디오 추출
    y, sr = librosa.load(video_path, sr=48000)

    # 오디오 레벨 계산 (1초 단위)
    hop_length = sr  # 1초
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # dB로 변환
    db = librosa.amplitude_to_db(rms)

    # 피크 감지 (-3dB 이상)
    peaks = []
    for i, level in enumerate(db):
        if level > -3:
            peaks.append({
                "time": i,  # 초 단위
                "level": float(level)
            })

    return {
        "levels": db.tolist(),
        "peaks": peaks,
        "duration": len(y) / sr
    }
```

### 4.4 프론트엔드 타임라인

```jsx
// frontend/src/components/Timeline.jsx
import React, { useState, useEffect } from 'react';
import VideoPlayer from './VideoPlayer';
import MarkerTrack from './MarkerTrack';
import CommentPanel from './CommentPanel';

function Timeline({ projectId }) {
    const [markers, setMarkers] = useState([]);
    const [currentTime, setCurrentTime] = useState(0);
    const [comments, setComments] = useState([]);

    useEffect(() => {
        // 분석 결과 로드
        fetchAnalysisResults(projectId).then(setMarkers);
        fetchComments(projectId).then(setComments);
    }, [projectId]);

    const handleTimeUpdate = (time) => {
        setCurrentTime(time);
    };

    const handleMarkerClick = (marker) => {
        // 해당 시간으로 이동
        setCurrentTime(marker.timecode);
    };

    const handleCommentAdd = (comment) => {
        const newComment = {
            ...comment,
            timecode: currentTime,
            timestamp: new Date().toISOString()
        };

        // 로컬 상태 업데이트
        setComments([...comments, newComment]);

        // 서버에 저장
        saveComment(projectId, newComment);
    };

    return (
        <div className="timeline-container">
            <VideoPlayer
                src={`/api/video/${projectId}`}
                onTimeUpdate={handleTimeUpdate}
                currentTime={currentTime}
            />

            <div className="tracks">
                <MarkerTrack
                    title="Black Frames"
                    markers={markers.blackFrames}
                    color="red"
                    onClick={handleMarkerClick}
                />
                <MarkerTrack
                    title="Scene Changes"
                    markers={markers.sceneChanges}
                    color="blue"
                    onClick={handleMarkerClick}
                />
                <MarkerTrack
                    title="Audio Peaks"
                    markers={markers.audioPeaks}
                    color="orange"
                    onClick={handleMarkerClick}
                />
            </div>

            <CommentPanel
                comments={comments}
                currentTime={currentTime}
                onAdd={handleCommentAdd}
            />
        </div>
    );
}
```

### 4.5 데이터베이스 스키마 (SQLite)

```sql
-- 프로젝트
CREATE TABLE projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    video_path TEXT,
    duration REAL,
    fps REAL,
    resolution TEXT
);

-- 분석 마커
CREATE TABLE markers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    type TEXT NOT NULL,  -- 'black_frame', 'scene_change', 'audio_peak'
    timecode REAL NOT NULL,
    severity TEXT,  -- 'critical', 'warning', 'info'
    data TEXT,  -- JSON 형태의 추가 데이터
    FOREIGN KEY (project_id) REFERENCES projects(id)
);

-- 코멘트
CREATE TABLE comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    timecode REAL NOT NULL,
    content TEXT NOT NULL,
    author TEXT DEFAULT 'Anonymous',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tags TEXT,  -- 쉼표로 구분된 태그
    status TEXT DEFAULT 'open',  -- 'open', 'resolved'
    FOREIGN KEY (project_id) REFERENCES projects(id)
);

-- 인덱스
CREATE INDEX idx_markers_project ON markers(project_id);
CREATE INDEX idx_markers_timecode ON markers(project_id, timecode);
CREATE INDEX idx_comments_project ON comments(project_id);
CREATE INDEX idx_comments_timecode ON comments(project_id, timecode);
```

---

## 5. 구현 로드맵 (4주)

### Week 1: 백엔드 기초
```yaml
Day 1-2:
  - FastAPI 프로젝트 설정
  - SQLite 데이터베이스 설정
  - 기본 API 엔드포인트

Day 3-4:
  - 비디오 업로드 기능
  - FFmpeg 통합
  - 메타데이터 추출

Day 5-7:
  - 블랙 프레임 감지
  - 씬 전환 감지
  - 결과 저장
```

### Week 2: 오디오 & 분석
```yaml
Day 8-10:
  - 오디오 추출
  - 레벨 분석
  - 피크 감지

Day 11-14:
  - 분석 결과 API
  - 성능 최적화
  - 에러 처리
```

### Week 3: 프론트엔드
```yaml
Day 15-17:
  - React 프로젝트 설정
  - 비디오 플레이어 통합
  - 기본 레이아웃

Day 18-21:
  - 타임라인 컴포넌트
  - 마커 표시
  - 인터랙션
```

### Week 4: 통합 & 마무리
```yaml
Day 22-24:
  - 코멘트 기능
  - 리포트 생성
  - CSV 내보내기

Day 25-28:
  - 버그 수정
  - 성능 테스트
  - 문서화
```

---

## 6. 개발 환경 설정

### 6.1 필수 설치 프로그램

```bash
# 1. Python 3.11
# https://www.python.org/downloads/

# 2. Node.js 20 LTS
# https://nodejs.org/

# 3. FFmpeg
# https://ffmpeg.org/download.html
# PATH 환경변수에 추가 필요

# 4. Git
# https://git-scm.com/
```

### 6.2 백엔드 설정

```bash
# 프로젝트 폴더 생성
mkdir c:\broadcast-qc-mvp
cd c:\broadcast-qc-mvp

# 가상환경 생성
python -m venv venv
venv\Scripts\activate

# 의존성 설치
pip install fastapi uvicorn
pip install opencv-python-headless
pip install librosa soundfile
pip install python-multipart
pip install sqlalchemy

# 개발 서버 실행
uvicorn app.main:app --reload --port 8000
```

### 6.3 프론트엔드 설정

```bash
# React 앱 생성
npx create-react-app frontend
cd frontend

# 의존성 설치
npm install antd
npm install video.js
npm install recharts
npm install axios

# 개발 서버 실행
npm start
```

---

## 7. 최소 하드웨어 요구사항

```yaml
CPU: Intel i5 이상 (4코어)
RAM: 8GB 이상 (16GB 권장)
저장공간: 100GB 이상 여유 공간
GPU: 불필요 (CPU 처리)
네트워크: 로컬 전용 (인터넷 불필요)
```

---

## 8. 성능 목표 (로컬 환경)

```yaml
영상 업로드: 1GB 파일 < 30초
분석 속도: 1시간 영상 < 10분
동시 사용자: 5명
응답 시간: < 1초
```

---

## 9. 확장 계획 (Phase 2)

### 9.1 단계적 업그레이드

```yaml
Step 1 (Month 2):
  - PostgreSQL 전환
  - Docker 컨테이너화
  - 기본 인증 추가

Step 2 (Month 3):
  - AI 모델 통합 (로컬 실행)
  - WebSocket 실시간 협업
  - 고급 분석 기능

Step 3 (Month 6):
  - 클라우드 마이그레이션 옵션
  - SaaS 전환 준비
  - 엔터프라이즈 기능
```

### 9.2 클라우드 전환 시 예상 비용

```yaml
AWS (최소 구성):
  - EC2 t3.medium: $30/월
  - RDS PostgreSQL: $15/월
  - S3 스토리지: $5/월
  - CloudFront: $10/월
  총: ~$60/월

또는

로컬 서버 유지:
  - 전기료만 부담
  - Cloudflare Tunnel (무료)
  - 외부 접속 가능
```

---

## 10. 리스크 및 제약사항

### 10.1 기술적 제약

```yaml
제약사항:
  - 동시 처리 제한 (순차 처리)
  - 대용량 파일 처리 시간
  - 로컬 스토리지 한계
  - 백업 수동 관리

해결방안:
  - 야간 배치 처리
  - 파일 압축 활용
  - 외장 HDD 추가
  - 일일 백업 스크립트
```

### 10.2 비즈니스 제약

```yaml
제약사항:
  - 원격 접속 불가
  - 확장성 제한
  - 기술 지원 부재

해결방안:
  - VPN 설정
  - 클라우드 마이그레이션 계획
  - 커뮤니티 지원 활용
```

---

## 11. 빠른 시작 가이드

```bash
# 1. 저장소 클론
git clone https://github.com/your-repo/broadcast-qc-mvp.git
cd broadcast-qc-mvp

# 2. 백엔드 실행
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# 3. 프론트엔드 실행 (새 터미널)
cd frontend
npm install
npm start

# 4. 브라우저에서 접속
# http://localhost:3000
```

---

## 12. MVP 성공 기준

```yaml
기능적 성공:
  ✓ 영상 업로드 및 재생
  ✓ 블랙 프레임 90% 이상 감지
  ✓ 오디오 피크 100% 감지
  ✓ 타임코드 정확도 ±1초
  ✓ CSV 리포트 생성

사용자 피드백:
  ✓ 5명 사용자 테스트
  ✓ 주요 버그 0건
  ✓ 사용성 점수 7/10 이상

기술적 검증:
  ✓ 1시간 영상 처리 가능
  ✓ 시스템 안정성 8시간 이상
  ✓ 데이터 손실 0건
```

---

이 MVP 버전은 최소 비용으로 핵심 기능을 검증할 수 있도록 설계되었습니다.
로컬 PC에서 모든 기능이 작동하며, 검증 후 점진적으로 확장 가능합니다.