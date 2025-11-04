# Re:View - 씬 분석 중심 PRD

**Version**: 1.0
**Focus**: Scene-based Video Analysis & Review
**Last Updated**: 2025-11-03

---

## 🎬 제품 재정의

### 새로운 비전
"영상의 모든 씬을 이해하고 리뷰를 자동화하는 AI 플랫폼"

### 핵심 가치
- **씬 단위 분석**: 의미 있는 영상 단위로 분해
- **내용 이해**: 각 씬이 담고 있는 정보 파악
- **리뷰 자동화**: 씬별 품질 평가 및 개선점 제시

---

## 🎯 씬 분석 계층 구조

```
Level 1: 씬 감지 (Detection)
    ↓
Level 2: 씬 분류 (Classification)
    ↓
Level 3: 씬 이해 (Understanding)
    ↓
Level 4: 씬 평가 (Evaluation)
```

---

## 📋 MVP - 씬 감지 및 기본 분석 (4주)

### 목표
"영상을 씬 단위로 자동 분할하고 기본 정보 제공"

### 핵심 기능

#### 1. 씬 경계 감지 (Scene Boundary Detection)

```python
class SceneDetector:
    """씬 전환점 감지"""

    def detect_hard_cuts(self, video_path):
        """급격한 전환 감지"""
        # 알고리즘:
        # 1. 프레임 간 픽셀 차이 계산
        # 2. 히스토그램 차이 분석
        # 3. 엣지 변화율 측정
        # 4. 임계값 초과 시 씬 전환

        thresholds = {
            'pixel_diff': 0.4,
            'histogram_diff': 0.3,
            'edge_diff': 0.35
        }
        return scene_boundaries

    def detect_gradual_transitions(self, video_path):
        """점진적 전환 감지 (Fade, Dissolve)"""
        # 다중 프레임 분석
        # 변화 패턴 인식
        pass
```

#### 2. 씬 메타데이터 추출

```yaml
씬 정보:
  - scene_id: 고유 식별자
  - start_time: 시작 시간 (HH:MM:SS.fff)
  - end_time: 종료 시간
  - duration: 씬 길이
  - frame_count: 프레임 수
  - thumbnail: 대표 이미지
  - transition_type: cut/fade/dissolve
```

#### 3. 기본 씬 특징 분석

```yaml
시각적 특징:
  - dominant_color: 주요 색상
  - brightness: 밝기 평균
  - contrast: 대비 수준
  - motion_level: 움직임 정도 (static/slow/fast)

통계 정보:
  - total_scenes: 전체 씬 개수
  - avg_scene_duration: 평균 씬 길이
  - shortest/longest_scene: 최소/최대 길이
```

### MVP 구현 코드 예시

```python
# backend/app/scene_analyzer.py
import cv2
import numpy as np
from typing import List, Dict

class MVPSceneAnalyzer:
    def __init__(self, threshold=30.0):
        self.threshold = threshold
        self.min_scene_length = 10  # 최소 10프레임

    def analyze_video(self, video_path: str) -> Dict:
        """MVP 씬 분석"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        scenes = []
        prev_frame = None
        scene_start = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if prev_frame is not None:
                # 프레임 차이 계산
                diff = self.calculate_frame_diff(prev_frame, frame)

                if diff > self.threshold:
                    # 씬 전환 감지
                    if frame_idx - scene_start > self.min_scene_length:
                        scenes.append({
                            'scene_id': len(scenes) + 1,
                            'start_frame': scene_start,
                            'end_frame': frame_idx,
                            'start_time': scene_start / fps,
                            'end_time': frame_idx / fps,
                            'duration': (frame_idx - scene_start) / fps,
                            'thumbnail': self.extract_thumbnail(
                                video_path,
                                (scene_start + frame_idx) // 2
                            )
                        })
                        scene_start = frame_idx

            prev_frame = frame
            frame_idx += 1

        cap.release()

        return {
            'scenes': scenes,
            'total_scenes': len(scenes),
            'total_duration': frame_idx / fps,
            'avg_scene_duration': np.mean([s['duration'] for s in scenes])
        }

    def calculate_frame_diff(self, frame1, frame2):
        """프레임 간 차이 계산"""
        # 히스토그램 차이
        hist1 = cv2.calcHist([frame1], [0, 1, 2], None,
                            [32, 32, 32], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([frame2], [0, 1, 2], None,
                            [32, 32, 32], [0, 256, 0, 256, 0, 256])

        hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

        # 엣지 차이
        edges1 = cv2.Canny(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), 50, 150)
        edges2 = cv2.Canny(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), 50, 150)
        edge_diff = np.mean(np.abs(edges1.astype(float) - edges2.astype(float)))

        # 종합 점수
        return hist_diff * 0.7 + edge_diff * 0.3
```

### MVP UI 구성

```jsx
// frontend/src/components/SceneTimeline.jsx
import React from 'react';
import { Timeline, Card, Image } from 'antd';

function SceneTimeline({ scenes }) {
    return (
        <div className="scene-timeline">
            <h2>씬 분석 결과: {scenes.length}개 씬 감지</h2>

            <div className="scene-grid">
                {scenes.map(scene => (
                    <Card
                        key={scene.scene_id}
                        hoverable
                        cover={<Image src={scene.thumbnail} />}
                        onClick={() => seekToScene(scene.start_time)}
                    >
                        <Card.Meta
                            title={`Scene ${scene.scene_id}`}
                            description={`${formatTime(scene.start_time)} - ${formatTime(scene.end_time)}`}
                        />
                        <p>길이: {scene.duration.toFixed(1)}초</p>
                    </Card>
                ))}
            </div>

            <Timeline className="scene-timeline-view">
                {scenes.map(scene => (
                    <Timeline.Item key={scene.scene_id}>
                        <p>Scene {scene.scene_id}</p>
                        <p>{formatTime(scene.start_time)}</p>
                    </Timeline.Item>
                ))}
            </Timeline>
        </div>
    );
}
```

---

## 🚀 Pro - 씬 내용 분석 (8주)

### 추가 기능

#### 1. 샷 타입 분류

```python
class ShotClassifier:
    """샷 크기 및 앵글 분류"""

    shot_types = {
        'EWS': 'Extreme Wide Shot',  # 전경
        'WS': 'Wide Shot',           # 롱샷
        'MS': 'Medium Shot',         # 미디엄샷
        'CU': 'Close Up',            # 클로즈업
        'ECU': 'Extreme Close Up'    # 익스트림 클로즈업
    }

    def classify_shot(self, frame):
        # 얼굴 감지 기반 분류
        faces = self.detect_faces(frame)
        if not faces:
            return 'WS'  # 얼굴 없으면 와이드샷

        # 얼굴 크기로 샷 타입 결정
        face_area_ratio = self.calculate_face_ratio(faces, frame)

        if face_area_ratio > 0.5:
            return 'ECU'
        elif face_area_ratio > 0.3:
            return 'CU'
        elif face_area_ratio > 0.1:
            return 'MS'
        else:
            return 'WS'
```

#### 2. 씬 내용 이해

```python
class SceneContentAnalyzer:
    """씬 내용 AI 분석"""

    def __init__(self):
        self.yolo = YOLO('yolov8x.pt')  # 객체 감지
        self.whisper = whisper.load_model('large')  # 음성 인식

    def analyze_scene_content(self, scene_frames, audio_segment):
        """씬의 내용 분석"""

        # 시각적 요소
        objects = self.detect_objects(scene_frames)
        people_count = self.count_people(scene_frames)
        activities = self.detect_activities(scene_frames)

        # 오디오 요소
        transcript = self.transcribe_audio(audio_segment)
        speaker_count = self.count_speakers(audio_segment)
        music_detected = self.detect_music(audio_segment)

        # 텍스트/그래픽
        on_screen_text = self.extract_text(scene_frames)
        graphics = self.detect_graphics(scene_frames)

        return {
            'visual': {
                'objects': objects,
                'people': people_count,
                'activities': activities
            },
            'audio': {
                'transcript': transcript,
                'speakers': speaker_count,
                'has_music': music_detected
            },
            'graphics': {
                'text': on_screen_text,
                'overlays': graphics
            }
        }
```

#### 3. 씬 분류 및 태깅

```yaml
씬 카테고리:
  - Interview: 인터뷰/대담
  - Action: 액션/스포츠
  - Landscape: 풍경/배경
  - Graphics: 그래픽/타이틀
  - Transition: 전환 효과

자동 태그:
  - #outdoor #daytime #crowded
  - #studio #interview #two-shot
  - #montage #fast-paced #music
```

---

## 🧠 Enterprise - 지능형 씬 분석 (12주)

### 고급 기능

#### 1. 스토리 플로우 분석

```python
class StoryFlowAnalyzer:
    """서사 구조 분석"""

    def analyze_narrative_structure(self, scenes):
        """3막 구조 분석"""

        # 도입부 (Setup)
        setup_scenes = self.identify_setup(scenes[:len(scenes)//3])

        # 전개부 (Confrontation)
        confrontation = self.identify_confrontation(
            scenes[len(scenes)//3:2*len(scenes)//3]
        )

        # 결말부 (Resolution)
        resolution = self.identify_resolution(scenes[2*len(scenes)//3:])

        # 클라이맥스 감지
        climax = self.detect_climax(scenes)

        return {
            'structure': '3-act',
            'setup': setup_scenes,
            'confrontation': confrontation,
            'resolution': resolution,
            'climax': climax,
            'pacing': self.analyze_pacing(scenes)
        }
```

#### 2. 편집 품질 평가

```python
class EditingQualityEvaluator:
    """편집 품질 자동 평가"""

    def evaluate_editing(self, scenes):
        scores = {
            'continuity': self.check_continuity(scenes),  # 연속성
            'rhythm': self.analyze_rhythm(scenes),         # 리듬
            'transitions': self.evaluate_transitions(scenes), # 전환
            'pacing': self.evaluate_pacing(scenes),        # 페이싱
            'coherence': self.check_coherence(scenes)      # 일관성
        }

        overall_score = np.mean(list(scores.values()))

        recommendations = self.generate_recommendations(scores)

        return {
            'scores': scores,
            'overall': overall_score,
            'grade': self.score_to_grade(overall_score),
            'recommendations': recommendations
        }
```

#### 3. AI 기반 하이라이트 생성

```python
class HighlightGenerator:
    """자동 하이라이트 생성"""

    def generate_highlights(self, scenes, target_duration=60):
        """1분 하이라이트 자동 생성"""

        # 씬 중요도 점수 계산
        scene_scores = []
        for scene in scenes:
            score = self.calculate_importance(scene)
            scene_scores.append((scene, score))

        # 상위 씬 선택
        scene_scores.sort(key=lambda x: x[1], reverse=True)

        selected_scenes = []
        total_duration = 0

        for scene, score in scene_scores:
            if total_duration + scene['duration'] <= target_duration:
                selected_scenes.append(scene)
                total_duration += scene['duration']

        # 시간 순서로 정렬
        selected_scenes.sort(key=lambda x: x['start_time'])

        return {
            'scenes': selected_scenes,
            'duration': total_duration,
            'score': np.mean([s[1] for s in scene_scores[:len(selected_scenes)]])
        }
```

---

## 📊 단계별 정확도 및 성능 목표

| 단계 | 씬 감지 정확도 | 처리 속도 | 분석 깊이 |
|------|---------------|-----------|-----------|
| **MVP** | 85% (Hard Cut) | 1x 실시간 | 기본 메타데이터 |
| **Pro** | 95% (모든 전환) | 5x 실시간 (GPU) | 내용 이해 |
| **Enterprise** | 99% | 10x 실시간 | 의미 분석 |

---

## 🛠️ 기술 스택 진화

### MVP (CPU 기반)
```yaml
Core:
  - Python 3.11
  - OpenCV 4.8
  - NumPy
  - FFmpeg

Storage:
  - SQLite
  - Local File System
```

### Pro (GPU 가속)
```yaml
추가:
  - CUDA 11.8
  - PyTorch 2.0
  - YOLO v8
  - Whisper
  - Face Recognition

Optimization:
  - GPU Processing
  - Parallel Analysis
  - Redis Cache
```

### Enterprise (AI 통합)
```yaml
추가:
  - Transformers
  - Video-LLM
  - Multi-modal Models
  - Knowledge Graph

Infrastructure:
  - Kubernetes
  - Distributed Processing
  - Cloud Storage
```

---

## 💡 핵심 차별화 요소

### 방송 리뷰 특화

1. **프로덕션 중심 분석**
   - PD/TD 관점 씬 분류
   - 방송 품질 체크리스트
   - 편집점 자동 마킹

2. **실시간 피드백**
   - 라이브 중 씬 전환 알림
   - 실시간 품질 지표
   - 즉각적 개선 제안

3. **학습 기반 개선**
   - 과거 방송 패턴 학습
   - 팀별 스타일 인식
   - 맞춤형 제안

---

## 🎯 사용 시나리오

### MVP: 기본 씬 분석
```
1. 영상 업로드
2. 자동 씬 분할 (2분 소요/1시간 영상)
3. 씬 타임라인 생성
4. 씬별 썸네일 및 기본 정보 제공
5. CSV 리포트 출력
```

### Pro: 내용 기반 리뷰
```
1. 영상 업로드
2. GPU 가속 분석 (30초 소요/1시간 영상)
3. 씬별 내용 태깅
4. 객체/얼굴/음성 인식
5. 씬 품질 점수 제공
6. 개선 제안 생성
```

### Enterprise: 완전 자동화
```
1. 실시간 스트림 연결
2. 라이브 씬 분석
3. AI 기반 하이라이트 생성
4. 자동 편집 제안
5. 다음 방송 예측 및 추천
```

---

## 📈 ROI 분석

### 시간 절감
- **현재**: 1시간 영상 → 8시간 수동 리뷰
- **MVP**: 1시간 영상 → 2분 처리 + 30분 리뷰
- **Pro**: 1시간 영상 → 30초 처리 + 10분 리뷰
- **Enterprise**: 실시간 처리 + 자동 리포트

### 품질 향상
- **씬 누락**: 100% → 15% → 5% → 1%
- **편집 오류**: 수동 발견 → 85% 자동 → 95% → 99%
- **개선 속도**: 다음 방송 → 즉시 → 실시간

---

## 🚦 구현 우선순위

### Week 1-2: 씬 감지 엔진
```python
# 핵심 알고리즘 구현
- [ ] 프레임 차이 계산
- [ ] 히스토그램 분석
- [ ] 씬 경계 결정
- [ ] 썸네일 추출
```

### Week 3: UI/UX
```javascript
// 프론트엔드 구현
- [ ] 비디오 플레이어
- [ ] 씬 타임라인
- [ ] 썸네일 그리드
- [ ] 씬 점프 네비게이션
```

### Week 4: 통합 및 최적화
```yaml
- [ ] API 연결
- [ ] 성능 최적화
- [ ] 테스트 및 버그 수정
- [ ] 문서화
```

---

## 🎬 결론

씬 분석 중심 MVP는:
1. **기술적으로 실현 가능** (검증된 알고리즘)
2. **즉각적 가치 제공** (수동 작업 자동화)
3. **확장 가능한 구조** (MVP → Pro → Enterprise)
4. **명확한 차별화** (방송 리뷰 특화)

**귀하의 시스템(Ryzen 9 + RTX 3090)에서는 Pro 수준까지 즉시 구현 가능합니다!**

---

*씬 분석이 영상 리뷰의 핵심입니다.*