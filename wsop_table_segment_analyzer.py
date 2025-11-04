"""
WSOP 방송 테이블 세그먼트 분석기
트랜지션 화면을 감지하여 테이블 전환을 정확히 구분
"""

import cv2
import numpy as np
import json
import os
from datetime import timedelta
from typing import List, Dict, Tuple, Optional


class WSOPTableSegmentAnalyzer:
    """WSOP 트랜지션 기반 테이블 구분 분석기"""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        # 비디오 속성
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps

        # 트랜지션 감지 설정
        self.transition_templates = []
        self.load_transition_templates()

    def load_transition_templates(self):
        """트랜지션 템플릿 로드 또는 정의"""
        # WSOP 트랜지션 특징 정의
        self.transition_features = {
            'wsop_logo': {
                'text_patterns': ['WORLD SERIES', 'POKER', 'WSOP', 'SUPER', 'CIRCUIT'],
                'colors': {
                    'red': ([0, 100, 100], [10, 255, 255]),  # HSV 범위
                    'white': ([0, 0, 200], [180, 30, 255]),
                    'black': ([0, 0, 0], [180, 255, 30])
                }
            },
            'diagonal_pattern': {
                'colors': {
                    'red': ([0, 100, 100], [10, 255, 255]),
                    'white': ([0, 0, 200], [180, 30, 255])
                },
                'pattern': 'diagonal_stripes'
            }
        }

    def analyze_video(self) -> Dict:
        """비디오 전체 분석"""
        print("\n" + "="*60)
        print("WSOP 테이블 세그먼트 분석")
        print("="*60)

        # 1. 트랜지션 감지
        print("\n[1/4] 트랜지션 화면 감지 중...")
        transitions = self.detect_transitions()

        # 2. 세그먼트 생성
        print("[2/4] 테이블 세그먼트 생성 중...")
        segments = self.create_segments(transitions)

        # 3. 각 세그먼트 분석
        print("[3/4] 세그먼트별 테이블 분류 중...")
        classified_segments = self.classify_segments(segments)

        # 4. 자막 감지
        print("[4/4] 자막 분석 중...")
        subtitle_info = self.analyze_subtitles(classified_segments)

        # 리포트 생성
        report = self.generate_report(transitions, classified_segments, subtitle_info)

        return report

    def detect_transitions(self) -> List[Dict]:
        """트랜지션 화면 감지"""
        transitions = []
        frame_skip = max(1, int(self.fps / 5))  # 초당 5프레임 검사

        for frame_idx in range(0, self.frame_count, frame_skip):
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_idx % (frame_skip * 20) == 0:
                progress = (frame_idx / self.frame_count) * 100
                print(f"  진행: {progress:.1f}%", end='\r')

            # 트랜지션 패턴 확인
            is_transition, transition_type = self.check_transition_pattern(frame)

            if is_transition:
                transitions.append({
                    'frame': frame_idx,
                    'time': frame_idx / self.fps,
                    'timecode': self._format_timecode(frame_idx / self.fps),
                    'type': transition_type
                })

        print(f"\n  감지된 트랜지션: {len(transitions)}개")
        return transitions

    def check_transition_pattern(self, frame) -> Tuple[bool, str]:
        """프레임이 트랜지션 패턴인지 확인"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1. WSOP 로고 패턴 체크 (중앙 원형 + 텍스트)
        if self.check_wsop_logo_pattern(frame, hsv):
            return True, 'wsop_logo'

        # 2. 대각선 패턴 체크 (빨강/흰색 줄무늬)
        if self.check_diagonal_pattern(frame, hsv):
            return True, 'diagonal_transition'

        # 3. 검은 화면 체크 (페이드)
        if self.check_black_screen(frame):
            return True, 'black_fade'

        return False, None

    def check_wsop_logo_pattern(self, frame, hsv) -> bool:
        """WSOP 로고 패턴 감지"""
        # 중앙 영역 추출
        h, w = frame.shape[:2]
        center_region = hsv[h//3:2*h//3, w//3:2*w//3]

        # 흰색 영역 검사 (로고 배경)
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(center_region, white_lower, white_upper)
        white_ratio = np.count_nonzero(white_mask) / white_mask.size

        # 빨간색 영역 검사 (포커 칩 색상)
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(center_region, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(center_region, red_lower2, red_upper2)
        red_mask = red_mask1 | red_mask2
        red_ratio = np.count_nonzero(red_mask) / red_mask.size

        # WSOP 로고 특징: 중앙에 흰색 원형 + 빨간색 요소
        if white_ratio > 0.1 and red_ratio > 0.05:
            # 원형 검출 시도
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100,
                                      param1=50, param2=30,
                                      minRadius=50, maxRadius=200)
            if circles is not None:
                return True

        # 텍스트 영역 체크 (엣지 검출)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        text_region = edges[h//2:, :]  # 하단 절반
        edge_density = np.count_nonzero(text_region) / text_region.size

        # WSOP 텍스트가 있으면 엣지 밀도가 높음
        if edge_density > 0.02 and (white_ratio > 0.1 or red_ratio > 0.1):
            return True

        return False

    def check_diagonal_pattern(self, frame, hsv) -> bool:
        """대각선 패턴 감지"""
        h, w = frame.shape[:2]

        # 빨간색 마스크
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = red_mask1 | red_mask2

        # 흰색 마스크
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)

        red_ratio = np.count_nonzero(red_mask) / red_mask.size
        white_ratio = np.count_nonzero(white_mask) / white_mask.size

        # 대각선 패턴: 빨강과 흰색이 비슷한 비율
        if 0.2 < red_ratio < 0.6 and 0.2 < white_ratio < 0.6:
            # 대각선 구조 확인 (Hough 변환)
            edges = cv2.Canny(frame, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

            if lines is not None and len(lines) > 5:
                # 대각선 각도 체크 (30-60도 또는 120-150도)
                angles = []
                for line in lines[:10]:
                    rho, theta = line[0]
                    angle = np.degrees(theta)
                    angles.append(angle)

                diagonal_angles = [a for a in angles if (30 < a < 60) or (120 < a < 150)]
                if len(diagonal_angles) > 2:
                    return True

        return False

    def check_black_screen(self, frame) -> bool:
        """검은 화면 감지"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        return mean_brightness < 20

    def create_segments(self, transitions: List[Dict]) -> List[Dict]:
        """트랜지션을 기준으로 세그먼트 생성"""
        segments = []

        # 시작 세그먼트
        if transitions and transitions[0]['frame'] > 0:
            segments.append({
                'segment_id': 1,
                'start_frame': 0,
                'end_frame': transitions[0]['frame'],
                'start_time': 0,
                'end_time': transitions[0]['time'],
                'start_tc': '0:00:00',
                'end_tc': transitions[0]['timecode']
            })

        # 중간 세그먼트들
        for i in range(len(transitions) - 1):
            segments.append({
                'segment_id': len(segments) + 1,
                'start_frame': transitions[i]['frame'],
                'end_frame': transitions[i + 1]['frame'],
                'start_time': transitions[i]['time'],
                'end_time': transitions[i + 1]['time'],
                'start_tc': transitions[i]['timecode'],
                'end_tc': transitions[i + 1]['timecode']
            })

        # 마지막 세그먼트
        if transitions:
            segments.append({
                'segment_id': len(segments) + 1,
                'start_frame': transitions[-1]['frame'],
                'end_frame': self.frame_count - 1,
                'start_time': transitions[-1]['time'],
                'end_time': self.duration,
                'start_tc': transitions[-1]['timecode'],
                'end_tc': self._format_timecode(self.duration)
            })

        # 세그먼트 길이 계산
        for seg in segments:
            seg['duration'] = seg['end_time'] - seg['start_time']

        print(f"  생성된 세그먼트: {len(segments)}개")
        return segments

    def classify_segments(self, segments: List[Dict]) -> List[Dict]:
        """각 세그먼트의 테이블 타입 분류"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for seg in segments:
            print(f"  세그먼트 {seg['segment_id']} 분석 중...", end='\r')

            # 세그먼트 중간 프레임 샘플링
            sample_frames = []
            start = seg['start_frame']
            end = seg['end_frame']
            step = max(1, (end - start) // 10)  # 10개 샘플

            for frame_idx in range(start, min(end, start + step * 10), step):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.cap.read()
                if ret:
                    sample_frames.append(frame)

            # 테이블 타입 판단
            if sample_frames:
                seg['table_type'] = self.classify_table_type(sample_frames)
                seg['table_confidence'] = self.calculate_confidence(sample_frames, seg['table_type'])
            else:
                seg['table_type'] = 'unknown'
                seg['table_confidence'] = 0

        print(f"\n  분류 완료")
        return segments

    def classify_table_type(self, frames: List[np.ndarray]) -> str:
        """프레임 샘플로 테이블 타입 분류"""
        table_scores = {
            'feature_a': 0,
            'feature_b': 0,
            'outer': 0,
            'interview': 0,
            'graphics': 0
        }

        for frame in frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 녹색 테이블 감지
            green_lower = np.array([35, 40, 40])
            green_upper = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            green_ratio = np.count_nonzero(green_mask) / green_mask.size

            # 영역별 분석
            h, w = frame.shape[:2]
            center_green = np.count_nonzero(green_mask[h//3:2*h//3, w//3:2*w//3]) / (green_mask[h//3:2*h//3, w//3:2*w//3].size)
            bottom_green = np.count_nonzero(green_mask[2*h//3:, :]) / (green_mask[2*h//3:, :].size)

            # 분류 로직
            if green_ratio > 0.25:
                if center_green > 0.4:
                    table_scores['feature_a'] += 1  # 탑뷰
                elif bottom_green > 0.3:
                    table_scores['feature_b'] += 1  # 사이드뷰
                else:
                    table_scores['outer'] += 1
            elif green_ratio > 0.05:
                table_scores['outer'] += 1
            else:
                # 밝기 분석
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                if brightness > 180:
                    table_scores['graphics'] += 1
                else:
                    table_scores['interview'] += 1

        # 가장 높은 점수의 타입 반환
        return max(table_scores, key=table_scores.get)

    def calculate_confidence(self, frames: List[np.ndarray], table_type: str) -> float:
        """분류 신뢰도 계산"""
        # 간단한 신뢰도: 프레임 일관성
        consistent_count = 0
        for frame in frames:
            if self.classify_table_type([frame]) == table_type:
                consistent_count += 1
        return consistent_count / len(frames) if frames else 0

    def analyze_subtitles(self, segments: List[Dict]) -> Dict:
        """세그먼트별 자막 분석"""
        subtitle_info = {
            'total_frames_with_subtitle': 0,
            'segments_with_subtitle': []
        }

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for seg in segments:
            has_subtitle = False
            subtitle_frames = 0

            # 세그먼트 샘플링
            start = seg['start_frame']
            end = seg['end_frame']
            step = max(1, int(self.fps))  # 1초마다

            for frame_idx in range(start, end, step):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.cap.read()
                if ret and self.has_subtitle(frame):
                    subtitle_frames += 1
                    has_subtitle = True

            if has_subtitle:
                seg['has_subtitle'] = True
                seg['subtitle_coverage'] = (subtitle_frames * step) / (end - start)
                subtitle_info['segments_with_subtitle'].append(seg['segment_id'])
                subtitle_info['total_frames_with_subtitle'] += subtitle_frames * step

        return subtitle_info

    def has_subtitle(self, frame) -> bool:
        """프레임에 자막 존재 여부 확인"""
        h, w = frame.shape[:2]
        bottom_region = frame[int(h * 0.8):, :]
        gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)

        # 엣지 검출
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size

        # 흰색 텍스트 검출
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        white_ratio = np.count_nonzero(binary) / binary.size

        return edge_density > 0.005 and white_ratio > 0.005

    def generate_report(self, transitions: List[Dict], segments: List[Dict], subtitle_info: Dict) -> Dict:
        """최종 리포트 생성"""
        # 테이블 타입별 통계
        table_stats = {}
        for seg in segments:
            table_type = seg.get('table_type', 'unknown')
            if table_type not in table_stats:
                table_stats[table_type] = {
                    'count': 0,
                    'total_duration': 0,
                    'segments': []
                }
            table_stats[table_type]['count'] += 1
            table_stats[table_type]['total_duration'] += seg['duration']
            table_stats[table_type]['segments'].append(seg['segment_id'])

        report = {
            'video_info': {
                'filename': os.path.basename(self.video_path),
                'duration': self.duration,
                'duration_tc': self._format_timecode(self.duration),
                'fps': self.fps,
                'resolution': f"{self.width}x{self.height}"
            },
            'transitions': {
                'total': len(transitions),
                'list': transitions
            },
            'segments': {
                'total': len(segments),
                'list': segments
            },
            'table_analysis': table_stats,
            'subtitle_info': subtitle_info,
            'qc_summary': self._generate_qc_summary(table_stats, segments)
        }

        return report

    def _generate_qc_summary(self, table_stats: Dict, segments: List[Dict]) -> Dict:
        """QC 요약 생성"""
        summary = {
            'feature_table_coverage': 0,
            'outer_table_coverage': 0,
            'has_proper_transitions': len(segments) > 1,
            'recommendations': []
        }

        # 피처 테이블 커버리지 계산
        feature_duration = 0
        if 'feature_a' in table_stats:
            feature_duration += table_stats['feature_a']['total_duration']
        if 'feature_b' in table_stats:
            feature_duration += table_stats['feature_b']['total_duration']
        summary['feature_table_coverage'] = (feature_duration / self.duration * 100) if self.duration > 0 else 0

        # 아우터 테이블 커버리지
        if 'outer' in table_stats:
            summary['outer_table_coverage'] = (table_stats['outer']['total_duration'] / self.duration * 100)

        # 권장사항 생성
        if summary['feature_table_coverage'] < 30:
            summary['recommendations'].append("피처 테이블 비중이 낮습니다 (30% 미만)")
        if not summary['has_proper_transitions']:
            summary['recommendations'].append("트랜지션이 감지되지 않았습니다")

        return summary

    def _format_timecode(self, seconds: float) -> str:
        """초를 타임코드로 변환"""
        return str(timedelta(seconds=int(seconds)))

    def save_reports(self, report: Dict):
        """리포트 저장"""
        base_name = os.path.splitext(self.video_path)[0]

        # JSON 저장
        json_path = f"{base_name}_wsop_segments.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # CSV 저장
        csv_path = f"{base_name}_wsop_segments.csv"
        self.save_csv(report, csv_path)

        # 자막 이미지 추출
        self.extract_transition_frames(report['transitions']['list'])

        return json_path, csv_path

    def save_csv(self, report: Dict, csv_path: str):
        """CSV 형식 저장"""
        import csv

        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)

            # 헤더
            writer.writerow(['세그먼트', '테이블 타입', '시작', '종료', '지속시간', '자막', '신뢰도'])

            # 세그먼트 정보
            for seg in report['segments']['list']:
                writer.writerow([
                    seg['segment_id'],
                    seg.get('table_type', 'unknown'),
                    seg['start_tc'],
                    seg['end_tc'],
                    f"{seg['duration']:.1f}초",
                    '있음' if seg.get('has_subtitle') else '없음',
                    f"{seg.get('table_confidence', 0):.1%}"
                ])

    def extract_transition_frames(self, transitions: List[Dict]):
        """트랜지션 프레임 이미지 추출"""
        if not transitions:
            return

        output_dir = os.path.splitext(self.video_path)[0] + "_transitions"
        os.makedirs(output_dir, exist_ok=True)

        for i, trans in enumerate(transitions[:10]):  # 최대 10개
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, trans['frame'])
            ret, frame = self.cap.read()
            if ret:
                filename = f"transition_{i+1:02d}_{trans['timecode'].replace(':', '-')}.jpg"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, frame)
                print(f"  트랜지션 이미지 저장: {filename}")

    def print_summary(self, report: Dict):
        """분석 요약 출력"""
        print("\n" + "="*60)
        print("WSOP 테이블 세그먼트 분석 결과")
        print("="*60)

        print(f"\n[비디오 정보]")
        info = report['video_info']
        print(f"  파일: {info['filename']}")
        print(f"  재생시간: {info['duration_tc']}")

        print(f"\n[트랜지션 분석]")
        print(f"  감지된 트랜지션: {report['transitions']['total']}개")
        if report['transitions']['list']:
            for trans in report['transitions']['list'][:5]:
                print(f"    - {trans['timecode']}: {trans['type']}")

        print(f"\n[세그먼트 분석]")
        print(f"  총 세그먼트: {report['segments']['total']}개")

        print(f"\n[테이블 타입별 분포]")
        for table_type, stats in report['table_analysis'].items():
            percentage = (stats['total_duration'] / self.duration * 100) if self.duration > 0 else 0
            print(f"  {table_type:15s}: {stats['count']:2d}개 세그먼트, {percentage:5.1f}%")

        print(f"\n[자막 정보]")
        subtitle_coverage = report['subtitle_info']['total_frames_with_subtitle'] / self.frame_count * 100
        print(f"  자막 커버리지: {subtitle_coverage:.1f}%")
        print(f"  자막 있는 세그먼트: {len(report['subtitle_info']['segments_with_subtitle'])}개")

        print(f"\n[QC 요약]")
        qc = report['qc_summary']
        print(f"  피처 테이블 커버리지: {qc['feature_table_coverage']:.1f}%")
        print(f"  아우터 테이블 커버리지: {qc['outer_table_coverage']:.1f}%")
        if qc['recommendations']:
            print(f"\n[권장사항]")
            for rec in qc['recommendations']:
                print(f"  - {rec}")

    def close(self):
        """리소스 해제"""
        self.cap.release()


def main():
    video_file = "sample_wsop_sc_cy01.mp4"

    if not os.path.exists(video_file):
        print(f"오류: '{video_file}' 파일을 찾을 수 없습니다!")
        return

    print("WSOP 테이블 세그먼트 분석 시작...")

    # 분석기 초기화
    analyzer = WSOPTableSegmentAnalyzer(video_file)

    # 분석 실행
    report = analyzer.analyze_video()

    # 리포트 저장
    json_path, csv_path = analyzer.save_reports(report)

    # 요약 출력
    analyzer.print_summary(report)

    print(f"\n[저장된 파일]")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")
    print(f"  - 트랜지션 이미지: {os.path.splitext(video_file)[0]}_transitions/")

    # 정리
    analyzer.close()


if __name__ == "__main__":
    main()