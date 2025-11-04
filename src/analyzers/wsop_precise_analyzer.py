"""
WSOP 정밀 테이블 분석기
- 5개 트랜지션 정확 감지
- 3개 테이블 타입 구분 (Feature A/B, Virtual)
- 6개 씬 자동 분류
"""

import cv2
import numpy as np
import json
import os
from datetime import timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class TableType(Enum):
    """테이블 타입 정의"""
    FEATURE_A = "feature_table_a"  # 탑뷰
    FEATURE_B = "feature_table_b"  # 사이드뷰
    VIRTUAL = "virtual_table"       # 버추얼(구 아우터)
    TRANSITION = "transition"       # 트랜지션
    UNKNOWN = "unknown"


@dataclass
class TransitionPattern:
    """트랜지션 패턴 정의"""
    name: str
    duration_range: Tuple[float, float]  # 최소, 최대 지속시간(초)
    visual_features: Dict


class WSPOPreciseAnalyzer:
    """WSOP 정밀 분석기"""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        # 비디오 속성
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps

        # 트랜지션 패턴 정의
        self.transition_patterns = [
            TransitionPattern(
                name="wsop_logo",
                duration_range=(1.5, 4.0),
                visual_features={
                    'has_logo': True,
                    'dominant_colors': ['red', 'white', 'black'],
                    'text_regions': True
                }
            ),
            TransitionPattern(
                name="diagonal_stripes",
                duration_range=(1.0, 3.0),
                visual_features={
                    'diagonal_lines': True,
                    'alternating_colors': ['red', 'white'],
                    'motion': 'static'
                }
            )
        ]

        # 분석 결과 저장
        self.transitions = []
        self.scenes = []
        self.frame_cache = {}

    def analyze(self) -> Dict:
        """전체 분석 실행"""
        print("\n" + "="*60)
        print("WSOP 정밀 분석 시작")
        print("="*60)
        print(f"\n파일: {os.path.basename(self.video_path)}")
        print(f"재생시간: {timedelta(seconds=int(self.duration))}")
        print(f"해상도: {self.width}x{self.height}")
        print(f"FPS: {self.fps:.2f}")

        # 1단계: 트랜지션 정밀 감지
        print("\n[1/3] 트랜지션 정밀 감지...")
        self.detect_transitions_precise()

        # 2단계: 씬 구분
        print("\n[2/3] 씬 구분...")
        self.segment_scenes()

        # 3단계: 테이블 타입 분류
        print("\n[3/3] 테이블 타입 분류...")
        self.classify_tables()

        # 리포트 생성
        report = self.generate_report()

        return report

    def detect_transitions_precise(self):
        """정밀 트랜지션 감지"""
        # 프레임 단위 스캔
        potential_transitions = []
        prev_is_transition = False
        transition_start = None

        # 1초 간격으로 샘플링 (빠른 처리)
        sample_interval = max(1, int(self.fps * 1.0))

        for frame_idx in range(0, self.frame_count, sample_interval):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if not ret:
                break

            # 진행률
            if frame_idx % (sample_interval * 50) == 0:
                progress = (frame_idx / self.frame_count) * 100
                print(f"  진행: {progress:.1f}%", end='\r')

            # 트랜지션 특징 체크
            is_transition, confidence = self.check_transition_features(frame, frame_idx)

            if is_transition and not prev_is_transition:
                # 트랜지션 시작
                transition_start = frame_idx
            elif not is_transition and prev_is_transition and transition_start is not None:
                # 트랜지션 종료
                duration = (frame_idx - transition_start) / self.fps

                # 지속시간으로 필터링 (1-5초)
                if 1.0 <= duration <= 5.0:
                    self.transitions.append({
                        'start_frame': transition_start,
                        'end_frame': frame_idx,
                        'start_time': transition_start / self.fps,
                        'end_time': frame_idx / self.fps,
                        'duration': duration,
                        'confidence': confidence
                    })

                transition_start = None

            prev_is_transition = is_transition

            # 프레임 캐싱 제한 (메모리 절약)
            if len(self.frame_cache) < 50 and frame_idx % int(self.fps * 2) == 0:  # 2초마다, 최대 50개
                self.frame_cache[frame_idx] = frame

        print(f"\n  감지된 트랜지션: {len(self.transitions)}개")

        # 트랜지션 병합 (너무 가까운 것들)
        self.merge_close_transitions()

    def check_transition_features(self, frame, frame_idx) -> Tuple[bool, float]:
        """트랜지션 특징 체크"""
        confidence = 0.0

        # 1. WSOP 로고 패턴 체크
        has_logo = self.detect_wsop_logo(frame)
        if has_logo:
            confidence += 0.4

        # 2. 대각선 패턴 체크
        has_diagonal = self.detect_diagonal_pattern(frame)
        if has_diagonal:
            confidence += 0.3

        # 3. 색상 분포 체크
        color_match = self.check_transition_colors(frame)
        if color_match > 0.7:
            confidence += 0.3

        # 4. 텍스트 영역 체크
        has_text = self.detect_text_regions(frame)
        if has_text:
            confidence += 0.2

        # 5. 급격한 변화 체크 (이전 프레임과 비교)
        if frame_idx > 0 and frame_idx - int(self.fps * 0.2) in self.frame_cache:
            prev_frame = self.frame_cache[frame_idx - int(self.fps * 0.2)]
            change_score = self.calculate_frame_difference(prev_frame, frame)
            if change_score > 50:
                confidence += 0.1

        return confidence > 0.5, confidence

    def detect_wsop_logo(self, frame) -> bool:
        """WSOP 로고 감지"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]

        # 중앙 영역 추출
        center_region = hsv[h//3:2*h//3, w//3:2*w//3]

        # 흰색 원형 영역 찾기
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(center_region, white_lower, white_upper)

        # 원형성 체크
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # 최소 크기
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.7:  # 원형에 가까움
                        return True

        return False

    def detect_diagonal_pattern(self, frame) -> bool:
        """대각선 패턴 감지"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 엣지 검출
        edges = cv2.Canny(gray, 50, 150)

        # Hough 변환으로 선 검출
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

        if lines is not None:
            diagonal_count = 0
            for line in lines:
                rho, theta = line[0]
                angle = np.degrees(theta)

                # 대각선 각도 (30-60도 또는 120-150도)
                if (30 < angle < 60) or (120 < angle < 150):
                    diagonal_count += 1

            # 충분한 대각선이 있으면 True
            return diagonal_count >= 5

        return False

    def check_transition_colors(self, frame) -> float:
        """트랜지션 색상 체크"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 빨간색 영역
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)

        # 흰색 영역
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)

        # 검은색 영역
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 255, 30])
        black_mask = cv2.inRange(hsv, black_lower, black_upper)

        total_pixels = frame.shape[0] * frame.shape[1]
        red_ratio = np.count_nonzero(red_mask) / total_pixels
        white_ratio = np.count_nonzero(white_mask) / total_pixels
        black_ratio = np.count_nonzero(black_mask) / total_pixels

        # 트랜지션 색상 비율 (빨강+흰색+검정이 화면의 60% 이상)
        transition_color_ratio = red_ratio + white_ratio + black_ratio

        return transition_color_ratio

    def detect_text_regions(self, frame) -> bool:
        """텍스트 영역 감지"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 하단 1/3 영역 (보통 텍스트가 있는 위치)
        h = gray.shape[0]
        bottom_region = gray[2*h//3:, :]

        # 엣지 밀도로 텍스트 판단
        edges = cv2.Canny(bottom_region, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size

        return edge_density > 0.02

    def calculate_frame_difference(self, frame1, frame2) -> float:
        """프레임 간 차이 계산"""
        if frame1 is None or frame2 is None:
            return 0

        # 히스토그램 비교
        hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()

        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

    def merge_close_transitions(self):
        """가까운 트랜지션 병합"""
        if len(self.transitions) < 2:
            return

        merged = []
        i = 0

        while i < len(self.transitions):
            current = self.transitions[i]

            # 다음 트랜지션과의 간격 체크
            if i < len(self.transitions) - 1:
                next_trans = self.transitions[i + 1]
                gap = next_trans['start_time'] - current['end_time']

                # 2초 이내 간격이면 병합
                if gap < 2.0:
                    merged_trans = {
                        'start_frame': current['start_frame'],
                        'end_frame': next_trans['end_frame'],
                        'start_time': current['start_time'],
                        'end_time': next_trans['end_time'],
                        'duration': next_trans['end_time'] - current['start_time'],
                        'confidence': max(current['confidence'], next_trans['confidence'])
                    }
                    merged.append(merged_trans)
                    i += 2  # 두 개를 처리했으므로
                else:
                    merged.append(current)
                    i += 1
            else:
                merged.append(current)
                i += 1

        self.transitions = merged
        print(f"  병합 후 트랜지션: {len(self.transitions)}개")

    def segment_scenes(self):
        """트랜지션을 기준으로 씬 분할"""
        scenes = []

        # 첫 씬 (시작 ~ 첫 트랜지션)
        if self.transitions:
            if self.transitions[0]['start_time'] > 1.0:
                scenes.append({
                    'scene_id': 1,
                    'start_time': 0,
                    'end_time': self.transitions[0]['start_time'],
                    'duration': self.transitions[0]['start_time']
                })

        # 트랜지션 사이 씬들
        for i in range(len(self.transitions) - 1):
            start_time = self.transitions[i]['end_time']
            end_time = self.transitions[i + 1]['start_time']

            if end_time - start_time > 1.0:  # 1초 이상인 경우만
                scenes.append({
                    'scene_id': len(scenes) + 1,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time
                })

        # 마지막 씬 (마지막 트랜지션 ~ 끝)
        if self.transitions:
            last_trans_end = self.transitions[-1]['end_time']
            if self.duration - last_trans_end > 1.0:
                scenes.append({
                    'scene_id': len(scenes) + 1,
                    'start_time': last_trans_end,
                    'end_time': self.duration,
                    'duration': self.duration - last_trans_end
                })

        self.scenes = scenes
        print(f"  감지된 씬: {len(self.scenes)}개")

    def classify_tables(self):
        """각 씬의 테이블 타입 분류"""
        for scene in self.scenes:
            # 씬 중간 지점 프레임 분석
            middle_time = (scene['start_time'] + scene['end_time']) / 2
            middle_frame_idx = int(middle_time * self.fps)

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
            ret, frame = self.cap.read()

            if ret:
                table_type = self.classify_table_type(frame, scene)
                scene['table_type'] = table_type

                # 추가 샘플링으로 신뢰도 향상
                confidence = self.calculate_classification_confidence(scene)
                scene['confidence'] = confidence
            else:
                scene['table_type'] = TableType.UNKNOWN.value
                scene['confidence'] = 0.0

        print(f"  테이블 분류 완료")

    def classify_table_type(self, frame, scene) -> str:
        """프레임에서 테이블 타입 분류"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]

        # 녹색 테이블 영역 감지
        green_lower = np.array([35, 30, 30])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)

        # 영역별 녹색 분포
        green_ratio = np.count_nonzero(green_mask) / (h * w)

        # 중앙, 상단, 하단 영역 분석
        center_region = green_mask[h//3:2*h//3, w//3:2*w//3]
        top_region = green_mask[:h//3, :]
        bottom_region = green_mask[2*h//3:, :]

        center_ratio = np.count_nonzero(center_region) / center_region.size
        top_ratio = np.count_nonzero(top_region) / top_region.size
        bottom_ratio = np.count_nonzero(bottom_region) / bottom_region.size

        # HUD/오버레이 감지 (버추얼 테이블 특징)
        has_overlay = self.detect_digital_overlay(frame)

        # 카메라 앵글 판단
        camera_angle = self.detect_camera_angle(frame, green_mask)

        # 분류 로직
        if has_overlay:
            return TableType.VIRTUAL.value
        elif green_ratio > 0.3 and center_ratio > 0.5:
            # 중앙에 집중된 녹색 = Feature A (탑뷰)
            return TableType.FEATURE_A.value
        elif green_ratio > 0.2 and (camera_angle == "side" or bottom_ratio > top_ratio * 1.5):
            # 사이드 앵글 또는 하단 집중 = Feature B
            return TableType.FEATURE_B.value
        elif green_ratio > 0.05:
            # 녹색이 있지만 적음 = Virtual
            return TableType.VIRTUAL.value
        else:
            return TableType.UNKNOWN.value

    def detect_digital_overlay(self, frame) -> bool:
        """디지털 오버레이/HUD 감지"""
        # 상단과 하단의 UI 요소 체크
        h, w = frame.shape[:2]

        # 상단 10% 영역
        top_region = frame[:h//10, :]
        # 하단 10% 영역
        bottom_region = frame[9*h//10:, :]

        # 고대비 직선 엣지 찾기 (UI 특징)
        gray_top = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)
        gray_bottom = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)

        edges_top = cv2.Canny(gray_top, 100, 200)
        edges_bottom = cv2.Canny(gray_bottom, 100, 200)

        # 직선 검출
        lines_top = cv2.HoughLinesP(edges_top, 1, np.pi/180, 50, minLineLength=100, maxLineGap=10)
        lines_bottom = cv2.HoughLinesP(edges_bottom, 1, np.pi/180, 50, minLineLength=100, maxLineGap=10)

        # 수평/수직 직선이 많으면 오버레이
        ui_lines = 0

        if lines_top is not None:
            for line in lines_top:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1))
                # 수평 또는 수직선
                if angle < 0.1 or abs(angle - np.pi/2) < 0.1:
                    ui_lines += 1

        if lines_bottom is not None:
            for line in lines_bottom:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1))
                if angle < 0.1 or abs(angle - np.pi/2) < 0.1:
                    ui_lines += 1

        return ui_lines > 5

    def detect_camera_angle(self, frame, green_mask) -> str:
        """카메라 앵글 감지"""
        h, w = green_mask.shape

        # 테이블 컨투어 찾기
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)

            # 컨투어의 모멘트 계산
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])  # 중심 x
                cy = int(M["m01"] / M["m00"])  # 중심 y

                # 중심이 화면 중앙에서 얼마나 벗어났는지
                offset_x = abs(cx - w//2) / (w//2)
                offset_y = abs(cy - h//2) / (h//2)

                if offset_x > 0.3 or offset_y > 0.3:
                    return "side"
                else:
                    return "top"

        return "unknown"

    def calculate_classification_confidence(self, scene) -> float:
        """분류 신뢰도 계산"""
        # 여러 프레임을 샘플링하여 일관성 체크
        sample_count = min(5, int(scene['duration'] * self.fps / 10))  # 최대 5개 샘플

        if sample_count < 2:
            return 0.5

        table_types = []

        for i in range(sample_count):
            sample_time = scene['start_time'] + (scene['duration'] * i / sample_count)
            sample_frame_idx = int(sample_time * self.fps)

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, sample_frame_idx)
            ret, frame = self.cap.read()

            if ret:
                table_type = self.classify_table_type(frame, scene)
                table_types.append(table_type)

        # 가장 많이 나온 타입의 비율이 신뢰도
        if table_types:
            most_common = max(set(table_types), key=table_types.count)
            confidence = table_types.count(most_common) / len(table_types)
            return confidence

        return 0.0

    def generate_report(self) -> Dict:
        """최종 리포트 생성"""
        # 테이블별 통계
        table_stats = {}
        for scene in self.scenes:
            table_type = scene.get('table_type', 'unknown')
            if table_type not in table_stats:
                table_stats[table_type] = {
                    'count': 0,
                    'total_duration': 0,
                    'scenes': []
                }
            table_stats[table_type]['count'] += 1
            table_stats[table_type]['total_duration'] += scene['duration']
            table_stats[table_type]['scenes'].append(scene['scene_id'])

        report = {
            'video_info': {
                'filename': os.path.basename(self.video_path),
                'duration': self.duration,
                'duration_tc': str(timedelta(seconds=int(self.duration))),
                'fps': self.fps,
                'resolution': f"{self.width}x{self.height}",
                'frame_count': self.frame_count
            },
            'transitions': {
                'count': len(self.transitions),
                'list': [
                    {
                        'id': i + 1,
                        'start_tc': str(timedelta(seconds=int(t['start_time']))),
                        'end_tc': str(timedelta(seconds=int(t['end_time']))),
                        'duration': f"{t['duration']:.1f}초",
                        'confidence': f"{t['confidence']:.1%}"
                    }
                    for i, t in enumerate(self.transitions)
                ]
            },
            'scenes': {
                'count': len(self.scenes),
                'list': [
                    {
                        'scene_id': s['scene_id'],
                        'start_tc': str(timedelta(seconds=int(s['start_time']))),
                        'end_tc': str(timedelta(seconds=int(s['end_time']))),
                        'duration': f"{s['duration']:.1f}초",
                        'table_type': s.get('table_type', 'unknown'),
                        'confidence': f"{s.get('confidence', 0):.1%}"
                    }
                    for s in self.scenes
                ]
            },
            'table_analysis': table_stats,
            'summary': {
                'expected_transitions': 5,
                'detected_transitions': len(self.transitions),
                'expected_scenes': 6,
                'detected_scenes': len(self.scenes),
                'table_distribution': {
                    k: f"{v['total_duration'] / self.duration * 100:.1f}%"
                    for k, v in table_stats.items()
                }
            }
        }

        return report

    def save_report(self, report: Dict):
        """리포트 저장"""
        base_name = os.path.splitext(self.video_path)[0]

        # JSON 저장
        json_path = f"{base_name}_precise_analysis.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # CSV 저장
        csv_path = f"{base_name}_precise_scenes.csv"
        self.save_csv(report, csv_path)

        print(f"\n리포트 저장:")
        print(f"  - {json_path}")
        print(f"  - {csv_path}")

    def save_csv(self, report: Dict, csv_path: str):
        """CSV 저장"""
        import csv

        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)

            # 헤더
            writer.writerow(['씬 ID', '시작', '종료', '지속시간', '테이블 타입', '신뢰도'])

            # 씬 정보
            for scene in report['scenes']['list']:
                writer.writerow([
                    scene['scene_id'],
                    scene['start_tc'],
                    scene['end_tc'],
                    scene['duration'],
                    scene['table_type'],
                    scene['confidence']
                ])

    def print_summary(self, report: Dict):
        """요약 출력"""
        print("\n" + "="*60)
        print("분석 결과 요약")
        print("="*60)

        summary = report['summary']
        print(f"\n[트랜지션]")
        print(f"  예상: {summary['expected_transitions']}개")
        print(f"  감지: {summary['detected_transitions']}개")

        if report['transitions']['list']:
            print(f"\n  트랜지션 목록:")
            for trans in report['transitions']['list']:
                print(f"    {trans['id']}. {trans['start_tc']} ~ {trans['end_tc']} ({trans['duration']}, 신뢰도: {trans['confidence']})")

        print(f"\n[씬]")
        print(f"  예상: {summary['expected_scenes']}개")
        print(f"  감지: {summary['detected_scenes']}개")

        if report['scenes']['list']:
            print(f"\n  씬 목록:")
            for scene in report['scenes']['list']:
                print(f"    씬 {scene['scene_id']}: {scene['start_tc']} ~ {scene['end_tc']} "
                      f"({scene['duration']}) - {scene['table_type']} (신뢰도: {scene['confidence']})")

        print(f"\n[테이블 분포]")
        for table_type, percentage in summary['table_distribution'].items():
            print(f"  {table_type}: {percentage}")

    def close(self):
        """리소스 해제"""
        self.cap.release()


def main():
    video_file = "sample_wsop_sc_cy01.mp4"

    if not os.path.exists(video_file):
        print(f"오류: '{video_file}' 파일을 찾을 수 없습니다!")
        return

    # 분석기 초기화
    analyzer = WSPOPreciseAnalyzer(video_file)

    # 분석 실행
    report = analyzer.analyze()

    # 리포트 저장
    analyzer.save_report(report)

    # 요약 출력
    analyzer.print_summary(report)

    # 정리
    analyzer.close()


if __name__ == "__main__":
    main()