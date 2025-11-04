"""
포커 방송 테이블 분류 전문 도구
WSOP 방송 영상에서 피처/아우터 테이블 정확히 구분
"""

import cv2
import numpy as np
import json
import os
from datetime import timedelta
from typing import List, Dict, Tuple
from collections import Counter


class PokerTableClassifier:
    """포커 테이블 씬 정밀 분류기"""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        # 비디오 정보
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps

    def analyze_tables(self) -> Dict:
        """테이블 씬 정밀 분석"""
        print("\n=" * 60)
        print("포커 테이블 정밀 분석")
        print("=" * 60)

        # 1단계: 프레임별 특징 추출
        print("\n[1/3] 프레임 특징 추출 중...")
        frame_features = self.extract_frame_features()

        # 2단계: 씬 경계 감지 및 분류
        print("[2/3] 씬 분류 중...")
        classified_scenes = self.classify_scenes(frame_features)

        # 3단계: 자막 구간 감지
        print("[3/3] 자막 구간 분석 중...")
        subtitle_segments = self.detect_subtitle_segments()

        # 리포트 생성
        report = self.generate_detailed_report(classified_scenes, subtitle_segments)

        return report

    def extract_frame_features(self) -> List[Dict]:
        """각 프레임의 특징 추출"""
        features = []
        sample_rate = max(1, int(self.fps))  # 1초마다 샘플링

        for frame_idx in range(0, self.frame_count, sample_rate):
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_idx % (sample_rate * 10) == 0:
                progress = (frame_idx / self.frame_count) * 100
                print(f"  진행: {progress:.1f}%", end='\r')

            # 특징 추출
            feature = {
                'frame': frame_idx,
                'time': frame_idx / self.fps,
                'table_features': self._analyze_table_features(frame),
                'text_features': self._analyze_text_features(frame),
                'color_features': self._analyze_color_distribution(frame)
            }
            features.append(feature)

        print(f"\n  추출된 프레임: {len(features)}개")
        return features

    def _analyze_table_features(self, frame) -> Dict:
        """테이블 관련 특징 추출"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 포커 테이블 녹색 감지
        green_lower = np.array([35, 30, 30])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)

        # 영역별 녹색 분포
        h, w = frame.shape[:2]
        regions = {
            'center': green_mask[h//4:3*h//4, w//4:3*w//4],
            'top': green_mask[:h//3, :],
            'bottom': green_mask[2*h//3:, :],
            'left': green_mask[:, :w//3],
            'right': green_mask[:, 2*w//3:]
        }

        region_ratios = {}
        for name, region in regions.items():
            region_ratios[name] = np.count_nonzero(region) / region.size if region.size > 0 else 0

        # 전체 녹색 비율
        total_green = np.count_nonzero(green_mask) / green_mask.size

        # 테이블 모양 분석 (원형 vs 타원형)
        if total_green > 0.1:
            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                perimeter = cv2.arcLength(largest, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            else:
                circularity = 0
        else:
            circularity = 0

        return {
            'total_green': total_green,
            'center_green': region_ratios['center'],
            'distribution': region_ratios,
            'circularity': circularity
        }

    def _analyze_text_features(self, frame) -> Dict:
        """텍스트/자막 특징 추출"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 하단 자막 영역
        bottom_region = gray[int(h * 0.7):, :]
        bottom_edges = cv2.Canny(bottom_region, 50, 150)
        bottom_text_density = np.count_nonzero(bottom_edges) / bottom_edges.size

        # 상단 스코어보드 영역
        top_region = gray[:int(h * 0.2), :]
        top_edges = cv2.Canny(top_region, 50, 150)
        top_text_density = np.count_nonzero(top_edges) / top_edges.size

        # 고대비 픽셀 (텍스트 가능성)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        high_contrast = np.count_nonzero(binary) / binary.size

        return {
            'bottom_text': bottom_text_density,
            'top_text': top_text_density,
            'high_contrast': high_contrast,
            'has_subtitle': bottom_text_density > 0.01,
            'has_scoreboard': top_text_density > 0.02
        }

    def _analyze_color_distribution(self, frame) -> Dict:
        """색상 분포 분석"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 주요 색상 히스토그램
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        h_hist = h_hist.flatten() / h_hist.sum()

        # 지배적인 색상 찾기
        dominant_hue = np.argmax(h_hist)

        # 밝기 분석
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        brightness_std = np.std(gray)

        return {
            'dominant_hue': dominant_hue,
            'brightness': brightness,
            'brightness_variance': brightness_std,
            'is_dark': brightness < 50,
            'is_bright': brightness > 200
        }

    def classify_scenes(self, features: List[Dict]) -> List[Dict]:
        """프레임 특징을 기반으로 씬 분류"""
        scenes = []
        current_scene = None
        scene_start_idx = 0

        for i, feature in enumerate(features):
            # 테이블 타입 결정
            table_type = self._determine_table_type(feature)

            # 씬 변경 감지
            if current_scene is None or current_scene['type'] != table_type:
                # 이전 씬 저장
                if current_scene and i - scene_start_idx > 2:  # 최소 2프레임 이상
                    current_scene['end_frame'] = features[i-1]['frame']
                    current_scene['end_time'] = features[i-1]['time']
                    current_scene['duration'] = current_scene['end_time'] - current_scene['start_time']
                    scenes.append(current_scene)

                # 새 씬 시작
                scene_start_idx = i
                current_scene = {
                    'scene_id': len(scenes) + 1,
                    'type': table_type,
                    'start_frame': feature['frame'],
                    'start_time': feature['time'],
                    'start_tc': self._format_timecode(feature['time'])
                }

        # 마지막 씬 저장
        if current_scene:
            current_scene['end_frame'] = features[-1]['frame']
            current_scene['end_time'] = features[-1]['time']
            current_scene['duration'] = current_scene['end_time'] - current_scene['start_time']
            current_scene['end_tc'] = self._format_timecode(current_scene['end_time'])
            scenes.append(current_scene)

        print(f"  분류된 씬: {len(scenes)}개")
        return scenes

    def _determine_table_type(self, feature: Dict) -> str:
        """프레임 특징으로 테이블 타입 결정"""
        table = feature['table_features']
        text = feature['text_features']
        color = feature['color_features']

        # 어두운 화면 = 전환/페이드
        if color['is_dark']:
            return 'transition'

        # 밝은 화면 = 그래픽/로고
        if color['is_bright'] and table['total_green'] < 0.05:
            return 'graphics'

        # 녹색 테이블 감지
        if table['total_green'] > 0.25:
            # 중앙 집중 + 원형 = 피처 테이블 A (탑뷰)
            if table['center_green'] > 0.4 and table['circularity'] > 0.7:
                return 'feature_table_a'
            # 측면 분포 = 피처 테이블 B (사이드뷰)
            elif table['distribution']['left'] > 0.3 or table['distribution']['right'] > 0.3:
                return 'feature_table_b'
            # 균등 분포 = 와이드샷
            else:
                return 'feature_wide'

        # 적은 녹색 = 아우터 테이블 또는 플레이어 클로즈업
        elif table['total_green'] > 0.05:
            # 하단에 녹색 = 아우터 테이블
            if table['distribution']['bottom'] > 0.1:
                return 'outer_table'
            else:
                return 'player_close'

        # 녹색 없음 = 인터뷰 또는 기타
        else:
            # 높은 대비 = 인터뷰
            if color['brightness'] > 100 and color['brightness'] < 180:
                return 'interview'
            else:
                return 'other'

    def detect_subtitle_segments(self) -> List[Dict]:
        """자막이 나타나는 구간 감지"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        subtitle_segments = []
        current_segment = None

        sample_rate = max(1, int(self.fps / 2))  # 0.5초마다

        for frame_idx in range(0, self.frame_count, sample_rate):
            ret, frame = self.cap.read()
            if not ret:
                break

            has_subtitle = self._check_subtitle(frame)

            if has_subtitle and current_segment is None:
                # 자막 시작
                current_segment = {
                    'start_frame': frame_idx,
                    'start_time': frame_idx / self.fps,
                    'start_tc': self._format_timecode(frame_idx / self.fps)
                }
            elif not has_subtitle and current_segment is not None:
                # 자막 종료
                current_segment['end_frame'] = frame_idx
                current_segment['end_time'] = frame_idx / self.fps
                current_segment['end_tc'] = self._format_timecode(frame_idx / self.fps)
                current_segment['duration'] = current_segment['end_time'] - current_segment['start_time']

                if current_segment['duration'] > 0.5:  # 0.5초 이상만
                    subtitle_segments.append(current_segment)
                current_segment = None

        # 마지막 세그먼트 처리
        if current_segment:
            current_segment['end_frame'] = self.frame_count - 1
            current_segment['end_time'] = self.duration
            current_segment['end_tc'] = self._format_timecode(self.duration)
            current_segment['duration'] = current_segment['end_time'] - current_segment['start_time']
            subtitle_segments.append(current_segment)

        print(f"  자막 구간: {len(subtitle_segments)}개")
        return subtitle_segments

    def _check_subtitle(self, frame) -> bool:
        """프레임에 자막이 있는지 확인"""
        h, w = frame.shape[:2]

        # 하단 20% 영역
        bottom_region = frame[int(h * 0.8):, :]
        gray_bottom = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)

        # 엣지 검출
        edges = cv2.Canny(gray_bottom, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size

        # 흰색 픽셀 비율
        _, binary = cv2.threshold(gray_bottom, 200, 255, cv2.THRESH_BINARY)
        white_ratio = np.count_nonzero(binary) / binary.size

        # 자막 판단 (엣지와 흰색 픽셀이 일정 비율 이상)
        return edge_density > 0.005 and white_ratio > 0.005

    def generate_detailed_report(self, scenes: List[Dict], subtitles: List[Dict]) -> Dict:
        """상세 리포트 생성"""
        # 씬 타입별 통계
        scene_types = [s['type'] for s in scenes]
        type_counts = Counter(scene_types)

        # 테이블별 총 시간 계산
        type_durations = {}
        for scene in scenes:
            scene_type = scene['type']
            duration = scene['duration']
            type_durations[scene_type] = type_durations.get(scene_type, 0) + duration

        # 자막 총 시간
        subtitle_duration = sum(s['duration'] for s in subtitles)

        report = {
            'video_info': {
                'filename': os.path.basename(self.video_path),
                'duration': self.duration,
                'duration_tc': self._format_timecode(self.duration),
                'fps': self.fps,
                'resolution': f"{self.width}x{self.height}"
            },
            'scene_statistics': {
                'total_scenes': len(scenes),
                'scene_types': dict(type_counts),
                'type_durations': type_durations,
                'type_percentages': {
                    t: (d / self.duration * 100) for t, d in type_durations.items()
                }
            },
            'table_breakdown': {
                'feature_table_a': [s for s in scenes if s['type'] == 'feature_table_a'],
                'feature_table_b': [s for s in scenes if s['type'] == 'feature_table_b'],
                'feature_wide': [s for s in scenes if s['type'] == 'feature_wide'],
                'outer_table': [s for s in scenes if s['type'] == 'outer_table'],
                'player_close': [s for s in scenes if s['type'] == 'player_close'],
                'interview': [s for s in scenes if s['type'] == 'interview'],
                'other': [s for s in scenes if s['type'] not in [
                    'feature_table_a', 'feature_table_b', 'feature_wide',
                    'outer_table', 'player_close', 'interview', 'graphics', 'transition'
                ]]
            },
            'subtitle_info': {
                'total_segments': len(subtitles),
                'total_duration': subtitle_duration,
                'coverage_percentage': (subtitle_duration / self.duration * 100) if self.duration > 0 else 0,
                'segments': subtitles[:20]  # 처음 20개만
            },
            'all_scenes': scenes
        }

        return report

    def _format_timecode(self, seconds: float) -> str:
        """초를 타임코드로 변환"""
        return str(timedelta(seconds=int(seconds)))

    def save_reports(self, report: Dict):
        """리포트 저장"""
        base_name = os.path.splitext(self.video_path)[0]

        # JSON 저장
        json_path = f"{base_name}_table_analysis.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # CSV 저장
        csv_path = f"{base_name}_table_breakdown.csv"
        self._save_csv(report, csv_path)

        print(f"\n저장된 파일:")
        print(f"  - {json_path}")
        print(f"  - {csv_path}")

    def _save_csv(self, report: Dict, csv_path: str):
        """CSV 형식으로 저장"""
        import csv

        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)

            # 헤더
            writer.writerow(['테이블 타입', '씬 번호', '시작', '종료', '지속시간'])

            # 테이블별 씬 정보
            for table_type, scenes in report['table_breakdown'].items():
                if scenes:
                    for scene in scenes:
                        writer.writerow([
                            table_type,
                            scene['scene_id'],
                            scene['start_tc'],
                            scene.get('end_tc', ''),
                            f"{scene['duration']:.1f}초"
                        ])

    def print_summary(self, report: Dict):
        """요약 출력"""
        print("\n" + "=" * 60)
        print("테이블 분석 결과 요약")
        print("=" * 60)

        stats = report['scene_statistics']
        print(f"\n총 씬 수: {stats['total_scenes']}개")

        print("\n[테이블별 시간 비율]")
        for type_name, percentage in sorted(stats['type_percentages'].items(),
                                           key=lambda x: x[1], reverse=True):
            if percentage > 0:
                bar = '#' * int(percentage / 2)
                print(f"  {type_name:15s}: {bar:50s} {percentage:5.1f}%")

        subtitle_info = report['subtitle_info']
        print(f"\n[자막 분석]")
        print(f"  자막 구간: {subtitle_info['total_segments']}개")
        print(f"  총 시간: {subtitle_info['total_duration']:.1f}초")
        print(f"  커버리지: {subtitle_info['coverage_percentage']:.1f}%")

        # 주요 테이블 정보
        breakdown = report['table_breakdown']
        print(f"\n[테이블 상세]")
        if breakdown['feature_table_a']:
            print(f"  피처 테이블 A (탑뷰): {len(breakdown['feature_table_a'])}개 씬")
        if breakdown['feature_table_b']:
            print(f"  피처 테이블 B (사이드): {len(breakdown['feature_table_b'])}개 씬")
        if breakdown['outer_table']:
            print(f"  아우터 테이블: {len(breakdown['outer_table'])}개 씬")
        if breakdown['player_close']:
            print(f"  플레이어 클로즈업: {len(breakdown['player_close'])}개 씬")

    def close(self):
        """리소스 해제"""
        self.cap.release()


def main():
    video_file = "sample_wsop_sc_cy01.mp4"

    if not os.path.exists(video_file):
        print(f"오류: '{video_file}' 파일을 찾을 수 없습니다!")
        return

    # 분석기 초기화
    classifier = PokerTableClassifier(video_file)

    # 분석 실행
    report = classifier.analyze_tables()

    # 리포트 저장
    classifier.save_reports(report)

    # 요약 출력
    classifier.print_summary(report)

    # 정리
    classifier.close()


if __name__ == "__main__":
    main()