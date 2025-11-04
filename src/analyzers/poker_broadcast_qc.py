"""
포커 방송 전문 QC 분석 도구
- 피처 테이블 A/B 구분
- 아우터 테이블 감지
- 자막/그래픽 검출 및 추출
"""

import cv2
import numpy as np
import json
import os
from datetime import timedelta
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PokerBroadcastQC:
    """포커 방송 전문 QC 분석기"""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        # 비디오 속성
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

        # 테이블 감지 파라미터
        self.green_lower = np.array([35, 40, 40])  # 포커 테이블 녹색 범위
        self.green_upper = np.array([85, 255, 255])

        # 자막 감지 파라미터
        self.subtitle_region = {
            'lower_third': (0, int(self.height * 0.65), self.width, self.height),  # 하단 1/3
            'upper_third': (0, 0, self.width, int(self.height * 0.35)),  # 상단 1/3
            'full_bottom': (0, int(self.height * 0.8), self.width, self.height)  # 최하단 20%
        }

    def analyze_poker_broadcast(self) -> Dict:
        """포커 방송 종합 QC 분석"""
        print("\n[포커 방송 QC 분석 시작]")
        print("="*60)

        print("[1/4] 테이블별 씬 분류 중...")
        table_scenes = self.detect_table_scenes()

        print("[2/4] 자막/그래픽 검출 중...")
        subtitle_frames = self.detect_subtitles()

        print("[3/4] 자막 이미지 추출 중...")
        subtitle_images = self.extract_subtitle_images(subtitle_frames)

        print("[4/4] 리포트 생성 중...")

        # QC 리포트 생성
        report = {
            "video_info": {
                "filename": os.path.basename(self.video_path),
                "resolution": f"{self.width}x{self.height}",
                "fps": f"{self.fps:.2f}",
                "duration": str(timedelta(seconds=int(self.duration))),
                "total_frames": self.frame_count
            },
            "table_analysis": {
                "total_scenes": len(table_scenes),
                "feature_table_a": [],
                "feature_table_b": [],
                "outer_tables": [],
                "other_scenes": [],
                "table_distribution": {}
            },
            "subtitle_analysis": {
                "total_subtitles": len(subtitle_frames),
                "subtitle_frames": subtitle_frames[:100],  # 처음 100개만
                "subtitle_coverage": (len(subtitle_frames) / self.frame_count) * 100 if self.frame_count > 0 else 0,
                "extracted_images": subtitle_images
            },
            "qc_summary": {}
        }

        # 테이블별 씬 분류
        for scene in table_scenes:
            table_type = scene.get('table_type', 'unknown')
            if table_type == 'feature_a':
                report['table_analysis']['feature_table_a'].append(scene)
            elif table_type == 'feature_b':
                report['table_analysis']['feature_table_b'].append(scene)
            elif table_type == 'outer':
                report['table_analysis']['outer_tables'].append(scene)
            else:
                report['table_analysis']['other_scenes'].append(scene)

        # 테이블 분포 계산
        report['table_analysis']['table_distribution'] = {
            'feature_a': len(report['table_analysis']['feature_table_a']),
            'feature_b': len(report['table_analysis']['feature_table_b']),
            'outer': len(report['table_analysis']['outer_tables']),
            'other': len(report['table_analysis']['other_scenes'])
        }

        # QC 요약
        report['qc_summary'] = self._generate_qc_summary(report)

        return report

    def detect_table_scenes(self) -> List[Dict]:
        """테이블별 씬 감지 및 분류"""
        scenes = []
        current_scene = None
        scene_start = 0
        prev_table_type = None

        frame_skip = max(1, int(self.fps / 2))  # 0.5초마다 샘플링

        for frame_idx in range(0, self.frame_count, frame_skip):
            ret, frame = self.cap.read()
            if not ret:
                break

            # 진행률 표시
            if frame_idx % (frame_skip * 10) == 0:
                progress = (frame_idx / self.frame_count) * 100
                print(f"  진행률: {progress:.1f}%", end='\r')

            # 테이블 타입 감지
            table_type = self._detect_table_type(frame)

            # 씬 변경 감지
            if table_type != prev_table_type:
                # 이전 씬 저장
                if current_scene and (frame_idx - scene_start) / self.fps > 1.0:  # 1초 이상인 씬만
                    current_scene['end_frame'] = frame_idx - frame_skip
                    current_scene['end_time'] = (frame_idx - frame_skip) / self.fps
                    current_scene['duration'] = current_scene['end_time'] - current_scene['start_time']
                    current_scene['end_tc'] = self._format_timecode(current_scene['end_time'])
                    scenes.append(current_scene)

                # 새 씬 시작
                scene_start = frame_idx
                current_scene = {
                    'scene_id': len(scenes) + 1,
                    'table_type': table_type,
                    'start_frame': frame_idx,
                    'start_time': frame_idx / self.fps,
                    'start_tc': self._format_timecode(frame_idx / self.fps)
                }

                prev_table_type = table_type

        # 마지막 씬 저장
        if current_scene:
            current_scene['end_frame'] = self.frame_count - 1
            current_scene['end_time'] = self.duration
            current_scene['duration'] = current_scene['end_time'] - current_scene['start_time']
            current_scene['end_tc'] = self._format_timecode(self.duration)
            scenes.append(current_scene)

        print(f"\n  감지된 씬: {len(scenes)}개")
        return scenes

    def _detect_table_type(self, frame) -> str:
        """프레임에서 테이블 타입 감지"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 녹색 테이블 마스크
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        green_ratio = np.count_nonzero(green_mask) / (frame.shape[0] * frame.shape[1])

        # 테이블 위치 분석 (화면 분할)
        height, width = frame.shape[:2]

        # 상단, 중앙, 하단 영역 분석
        upper_region = green_mask[:height//3, :]
        middle_region = green_mask[height//3:2*height//3, :]
        lower_region = green_mask[2*height//3:, :]

        upper_green = np.count_nonzero(upper_region) / upper_region.size
        middle_green = np.count_nonzero(middle_region) / middle_region.size
        lower_green = np.count_nonzero(lower_region) / lower_region.size

        # 좌측, 우측 분석
        left_region = green_mask[:, :width//2]
        right_region = green_mask[:, width//2:]

        left_green = np.count_nonzero(left_region) / left_region.size
        right_green = np.count_nonzero(right_region) / right_region.size

        # 테이블 타입 결정 로직
        if green_ratio > 0.3:
            # 많은 녹색 = 메인 테이블
            if middle_green > upper_green and middle_green > lower_green:
                # 중앙에 집중 = 피처 테이블 A
                return 'feature_a'
            elif abs(left_green - right_green) > 0.1:
                # 좌우 불균형 = 피처 테이블 B (사이드 앵글)
                return 'feature_b'
            else:
                # 균등 분포 = 아우터 테이블
                return 'outer'
        elif green_ratio > 0.05:
            # 적은 녹색 = 아우터 테이블 또는 원거리
            return 'outer'
        else:
            # 녹색 없음 = 인터뷰, 그래픽 등
            # 밝기로 추가 분류
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)

            if mean_brightness > 200:
                return 'graphics'
            elif mean_brightness < 50:
                return 'transition'
            else:
                return 'interview'

    def detect_subtitles(self) -> List[Dict]:
        """자막/그래픽 검출"""
        subtitle_frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_skip = max(1, int(self.fps / 5))  # 초당 5프레임 검사

        for frame_idx in range(0, self.frame_count, frame_skip):
            ret, frame = self.cap.read()
            if not ret:
                break

            # 진행률 표시
            if frame_idx % (frame_skip * 20) == 0:
                progress = (frame_idx / self.frame_count) * 100
                print(f"  진행률: {progress:.1f}%", end='\r')

            # 자막 영역 검사
            has_subtitle, subtitle_info = self._detect_text_regions(frame)

            if has_subtitle:
                subtitle_frames.append({
                    'frame': frame_idx,
                    'timecode': self._format_timecode(frame_idx / self.fps),
                    'time_seconds': frame_idx / self.fps,
                    'regions': subtitle_info
                })

        print(f"\n  감지된 자막: {len(subtitle_frames)}개 프레임")
        return subtitle_frames

    def _detect_text_regions(self, frame) -> Tuple[bool, Dict]:
        """프레임에서 텍스트 영역 감지"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subtitle_info = {}
        has_text = False

        # 하단 자막 영역 검사
        bottom_region = gray[int(self.height * 0.8):, :]

        # 엣지 검출로 텍스트 감지
        edges = cv2.Canny(bottom_region, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size

        # 고대비 영역 검사 (텍스트는 보통 고대비)
        _, binary = cv2.threshold(bottom_region, 200, 255, cv2.THRESH_BINARY)
        white_pixels = np.count_nonzero(binary) / binary.size

        # 텍스트 존재 판단
        if edge_density > 0.01 and white_pixels > 0.01:
            has_text = True
            subtitle_info['bottom'] = {
                'edge_density': edge_density,
                'white_ratio': white_pixels,
                'type': 'subtitle'
            }

        # 상단 그래픽 영역 검사 (스코어보드, 블라인드 정보 등)
        top_region = gray[:int(self.height * 0.2), :]
        edges_top = cv2.Canny(top_region, 50, 150)
        edge_density_top = np.count_nonzero(edges_top) / edges_top.size

        if edge_density_top > 0.02:
            has_text = True
            subtitle_info['top'] = {
                'edge_density': edge_density_top,
                'type': 'scoreboard'
            }

        # 중앙 오버레이 검사 (플레이어 정보 등)
        middle_region = gray[int(self.height * 0.4):int(self.height * 0.6), :]
        _, binary_middle = cv2.threshold(middle_region, 200, 255, cv2.THRESH_BINARY)
        white_middle = np.count_nonzero(binary_middle) / binary_middle.size

        if white_middle > 0.05:
            has_text = True
            subtitle_info['middle'] = {
                'white_ratio': white_middle,
                'type': 'player_info'
            }

        return has_text, subtitle_info

    def extract_subtitle_images(self, subtitle_frames: List[Dict], max_images: int = 20) -> List[str]:
        """자막 프레임 이미지 추출"""
        extracted_images = []

        if not subtitle_frames:
            return extracted_images

        # 균등 간격으로 샘플링
        sample_interval = max(1, len(subtitle_frames) // max_images)
        sampled_frames = subtitle_frames[::sample_interval][:max_images]

        # 출력 디렉토리 생성
        output_dir = os.path.splitext(self.video_path)[0] + "_subtitles"
        os.makedirs(output_dir, exist_ok=True)

        for i, subtitle_info in enumerate(sampled_frames):
            frame_num = subtitle_info['frame']
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()

            if ret:
                # 파일명 생성
                timecode = subtitle_info['timecode'].replace(':', '-')
                filename = f"subtitle_{i+1:03d}_{timecode}.jpg"
                filepath = os.path.join(output_dir, filename)

                # 자막 영역만 크롭 (하단 20%)
                subtitle_crop = frame[int(self.height * 0.8):, :]

                # 이미지 저장
                cv2.imwrite(filepath, subtitle_crop)

                extracted_images.append({
                    'filename': filename,
                    'filepath': filepath,
                    'frame': frame_num,
                    'timecode': subtitle_info['timecode'],
                    'time_seconds': subtitle_info['time_seconds']
                })

                print(f"  자막 이미지 추출: {filename}")

        return extracted_images

    def _format_timecode(self, seconds: float) -> str:
        """초를 타임코드로 변환"""
        return str(timedelta(seconds=int(seconds)))

    def _generate_qc_summary(self, report: Dict) -> Dict:
        """QC 요약 생성"""
        table_dist = report['table_analysis']['table_distribution']
        total_scenes = report['table_analysis']['total_scenes']

        summary = {
            'table_coverage': {
                'feature_a_percentage': (table_dist['feature_a'] / total_scenes * 100) if total_scenes > 0 else 0,
                'feature_b_percentage': (table_dist['feature_b'] / total_scenes * 100) if total_scenes > 0 else 0,
                'outer_percentage': (table_dist['outer'] / total_scenes * 100) if total_scenes > 0 else 0,
                'other_percentage': (table_dist['other'] / total_scenes * 100) if total_scenes > 0 else 0
            },
            'subtitle_coverage': report['subtitle_analysis']['subtitle_coverage'],
            'recommendations': []
        }

        # 권장사항 생성
        if table_dist['feature_a'] == 0 and table_dist['feature_b'] == 0:
            summary['recommendations'].append("경고: 피처 테이블 씬이 없습니다")

        if table_dist['outer'] > total_scenes * 0.5:
            summary['recommendations'].append("아우터 테이블 비중이 높습니다 (50% 이상)")

        if report['subtitle_analysis']['subtitle_coverage'] < 5:
            summary['recommendations'].append("자막 비중이 낮습니다 (5% 미만)")

        if not summary['recommendations']:
            summary['recommendations'].append("모든 QC 항목 정상")

        return summary

    def save_report(self, report: Dict) -> str:
        """리포트 저장"""
        output_path = os.path.splitext(self.video_path)[0] + "_poker_qc_report.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n리포트 저장: {output_path}")
        return output_path

    def generate_csv_report(self, report: Dict) -> str:
        """CSV 형식 리포트 생성"""
        import csv

        csv_path = os.path.splitext(self.video_path)[0] + "_poker_qc.csv"

        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)

            # 헤더
            writer.writerow(['구분', '씬 번호', '테이블 타입', '시작 타임코드', '종료 타임코드', '지속시간(초)'])

            # 피처 테이블 A
            for scene in report['table_analysis']['feature_table_a']:
                writer.writerow(['피처 A', scene['scene_id'], 'feature_a',
                               scene['start_tc'], scene['end_tc'], f"{scene['duration']:.1f}"])

            # 피처 테이블 B
            for scene in report['table_analysis']['feature_table_b']:
                writer.writerow(['피처 B', scene['scene_id'], 'feature_b',
                               scene['start_tc'], scene['end_tc'], f"{scene['duration']:.1f}"])

            # 아우터 테이블
            for scene in report['table_analysis']['outer_tables']:
                writer.writerow(['아우터', scene['scene_id'], 'outer',
                               scene['start_tc'], scene['end_tc'], f"{scene['duration']:.1f}"])

            # 기타
            for scene in report['table_analysis']['other_scenes']:
                writer.writerow(['기타', scene['scene_id'], scene['table_type'],
                               scene['start_tc'], scene['end_tc'], f"{scene['duration']:.1f}"])

            # 자막 정보 추가
            writer.writerow([])
            writer.writerow(['자막 정보'])
            writer.writerow(['프레임', '타임코드', '영역'])

            for subtitle in report['subtitle_analysis']['subtitle_frames'][:50]:  # 처음 50개만
                regions = ', '.join(subtitle['regions'].keys())
                writer.writerow([subtitle['frame'], subtitle['timecode'], regions])

        print(f"CSV 리포트 저장: {csv_path}")
        return csv_path

    def print_summary(self, report: Dict):
        """콘솔에 요약 출력"""
        print("\n" + "="*60)
        print("포커 방송 QC 분석 결과")
        print("="*60)

        # 비디오 정보
        info = report['video_info']
        print(f"\n[비디오 정보]")
        print(f"  파일: {info['filename']}")
        print(f"  해상도: {info['resolution']}")
        print(f"  재생시간: {info['duration']}")
        print(f"  FPS: {info['fps']}")

        # 테이블 분석
        table_dist = report['table_analysis']['table_distribution']
        print(f"\n[테이블별 씬 분포]")
        print(f"  피처 테이블 A: {table_dist['feature_a']}개 씬")
        print(f"  피처 테이블 B: {table_dist['feature_b']}개 씬")
        print(f"  아우터 테이블: {table_dist['outer']}개 씬")
        print(f"  기타 (인터뷰/그래픽): {table_dist['other']}개 씬")

        # 테이블 커버리지
        summary = report['qc_summary']
        print(f"\n[테이블 커버리지]")
        print(f"  피처 A: {summary['table_coverage']['feature_a_percentage']:.1f}%")
        print(f"  피처 B: {summary['table_coverage']['feature_b_percentage']:.1f}%")
        print(f"  아우터: {summary['table_coverage']['outer_percentage']:.1f}%")
        print(f"  기타: {summary['table_coverage']['other_percentage']:.1f}%")

        # 자막 분석
        subtitle_info = report['subtitle_analysis']
        print(f"\n[자막/그래픽 분석]")
        print(f"  감지된 자막 프레임: {subtitle_info['total_subtitles']}개")
        print(f"  자막 커버리지: {subtitle_info['subtitle_coverage']:.1f}%")
        print(f"  추출된 이미지: {len(subtitle_info['extracted_images'])}개")

        # 권장사항
        print(f"\n[QC 권장사항]")
        for i, rec in enumerate(summary['recommendations'], 1):
            print(f"  {i}. {rec}")

        print("\n" + "="*60)

    def close(self):
        """리소스 해제"""
        self.cap.release()


def main():
    """메인 실행 함수"""
    import sys

    video_file = sys.argv[1] if len(sys.argv) > 1 else "sample_wsop_sc_cy01.mp4"

    if not os.path.exists(video_file):
        print(f"오류: 비디오 파일 '{video_file}'을 찾을 수 없습니다!")
        return

    print("="*60)
    print("포커 방송 전문 QC 분석기")
    print("="*60)

    # 분석기 초기화
    analyzer = PokerBroadcastQC(video_file)

    # 분석 실행
    report = analyzer.analyze_poker_broadcast()

    # 리포트 저장
    json_path = analyzer.save_report(report)
    csv_path = analyzer.generate_csv_report(report)

    # 요약 출력
    analyzer.print_summary(report)

    print(f"\n[저장된 파일]")
    print(f"  JSON 리포트: {json_path}")
    print(f"  CSV 리포트: {csv_path}")

    if report['subtitle_analysis']['extracted_images']:
        subtitle_dir = os.path.splitext(video_file)[0] + "_subtitles"
        print(f"  자막 이미지: {subtitle_dir}/")

    # 정리
    analyzer.close()


if __name__ == "__main__":
    main()