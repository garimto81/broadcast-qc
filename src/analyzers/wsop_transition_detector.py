"""
WSOP 트랜지션 감지 도구 (최적화 버전)
빠른 트랜지션 감지로 테이블 세그먼트 구분
"""

import cv2
import numpy as np
import json
import os
from datetime import timedelta


def detect_wsop_transitions(video_path):
    """WSOP 트랜지션 빠른 감지"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print(f"\n비디오 정보:")
    print(f"  파일: {os.path.basename(video_path)}")
    print(f"  재생시간: {timedelta(seconds=int(duration))}")
    print(f"  총 프레임: {frame_count:,}")

    transitions = []
    prev_is_transition = False
    transition_start = None

    # 2초마다 샘플링 (빠른 처리)
    sample_rate = int(fps * 2)

    print("\n트랜지션 감지 중...")
    for frame_idx in range(0, frame_count, sample_rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # 진행률
        if frame_idx % (sample_rate * 10) == 0:
            progress = (frame_idx / frame_count) * 100
            print(f"  진행: {progress:.1f}%", end='\r')

        # 트랜지션 패턴 체크 (간단한 방식)
        is_transition = check_transition_simple(frame)

        if is_transition and not prev_is_transition:
            # 트랜지션 시작
            transition_start = frame_idx
        elif not is_transition and prev_is_transition and transition_start:
            # 트랜지션 종료
            transitions.append({
                'start_frame': transition_start,
                'end_frame': frame_idx,
                'start_time': transition_start / fps,
                'end_time': frame_idx / fps,
                'start_tc': str(timedelta(seconds=int(transition_start / fps))),
                'end_tc': str(timedelta(seconds=int(frame_idx / fps)))
            })
            transition_start = None

        prev_is_transition = is_transition

    cap.release()

    print(f"\n\n감지된 트랜지션: {len(transitions)}개")
    return transitions


def check_transition_simple(frame):
    """간단한 트랜지션 체크"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]

    # 1. 빨간색 비율 체크 (WSOP 로고와 대각선 패턴의 공통 특징)
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 100, 100])
    red_upper2 = np.array([180, 255, 255])

    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = red_mask1 | red_mask2
    red_ratio = np.count_nonzero(red_mask) / red_mask.size

    # 2. 흰색/밝은색 비율 체크
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    white_ratio = np.count_nonzero(white_mask) / white_mask.size

    # 3. 검은색 체크 (페이드)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)

    # 트랜지션 판단
    # - 빨간색과 흰색이 일정 비율 이상 (WSOP 로고 또는 대각선)
    # - 또는 매우 어두운 화면 (페이드)
    if (red_ratio > 0.1 and white_ratio > 0.1) or mean_brightness < 20:
        return True

    # 4. 중앙 원형 체크 (WSOP 로고)
    center_region = frame[h//3:2*h//3, w//3:2*w//3]
    center_gray = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)

    # 원 검출 (간단한 버전)
    _, binary = cv2.threshold(center_gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area > (center_region.shape[0] * center_region.shape[1] * 0.1):  # 중앙 영역의 10% 이상
            return True

    return False


def create_segments_from_transitions(transitions, total_duration, fps):
    """트랜지션으로 세그먼트 생성"""
    segments = []

    # 첫 세그먼트 (영상 시작 ~ 첫 트랜지션)
    if transitions and transitions[0]['start_time'] > 1:
        segments.append({
            'segment_id': 1,
            'start_time': 0,
            'end_time': transitions[0]['start_time'],
            'duration': transitions[0]['start_time'],
            'start_tc': '0:00:00',
            'end_tc': transitions[0]['start_tc']
        })

    # 트랜지션 사이 세그먼트들
    for i in range(len(transitions) - 1):
        segments.append({
            'segment_id': len(segments) + 1,
            'start_time': transitions[i]['end_time'],
            'end_time': transitions[i + 1]['start_time'],
            'duration': transitions[i + 1]['start_time'] - transitions[i]['end_time'],
            'start_tc': transitions[i]['end_tc'],
            'end_tc': transitions[i + 1]['start_tc']
        })

    # 마지막 세그먼트 (마지막 트랜지션 ~ 영상 끝)
    if transitions and transitions[-1]['end_time'] < total_duration - 1:
        segments.append({
            'segment_id': len(segments) + 1,
            'start_time': transitions[-1]['end_time'],
            'end_time': total_duration,
            'duration': total_duration - transitions[-1]['end_time'],
            'start_tc': transitions[-1]['end_tc'],
            'end_tc': str(timedelta(seconds=int(total_duration)))
        })

    return segments


def analyze_segment_content(video_path, segment, fps):
    """세그먼트 내용 간단 분석"""
    cap = cv2.VideoCapture(video_path)

    # 세그먼트 중간 지점 프레임
    middle_frame_idx = int((segment['start_time'] + segment['end_time']) / 2 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
    ret, frame = cap.read()

    if not ret:
        cap.release()
        return 'unknown'

    # 녹색 테이블 감지
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_lower = np.array([35, 40, 40])
    green_upper = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    green_ratio = np.count_nonzero(green_mask) / green_mask.size

    cap.release()

    # 간단한 분류
    if green_ratio > 0.3:
        return 'feature_table'
    elif green_ratio > 0.1:
        return 'outer_table'
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness > 180:
            return 'graphics'
        elif brightness < 50:
            return 'transition'
        else:
            return 'interview'


def main():
    video_file = "sample_wsop_sc_cy01.mp4"

    if not os.path.exists(video_file):
        print(f"오류: '{video_file}' 파일을 찾을 수 없습니다!")
        return

    print("="*60)
    print("WSOP 트랜지션 기반 테이블 분석")
    print("="*60)

    # 비디오 정보 가져오기
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()

    # 트랜지션 감지
    transitions = detect_wsop_transitions(video_file)

    # 세그먼트 생성
    segments = create_segments_from_transitions(transitions, duration, fps)
    print(f"\n생성된 세그먼트: {len(segments)}개")

    # 각 세그먼트 분석
    print("\n세그먼트 내용 분석 중...")
    for seg in segments:
        seg['content_type'] = analyze_segment_content(video_file, seg, fps)

    # 결과 출력
    print("\n" + "="*60)
    print("분석 결과")
    print("="*60)

    print("\n[트랜지션]")
    for i, trans in enumerate(transitions[:5], 1):
        print(f"  {i}. {trans['start_tc']} ~ {trans['end_tc']}")
    if len(transitions) > 5:
        print(f"  ... 외 {len(transitions)-5}개")

    print("\n[세그먼트]")
    for seg in segments[:10]:
        print(f"  세그먼트 {seg['segment_id']:2d}: {seg['start_tc']:>8} ~ {seg['end_tc']:>8} "
              f"({seg['duration']:6.1f}초) - {seg['content_type']}")
    if len(segments) > 10:
        print(f"  ... 외 {len(segments)-10}개")

    # 통계
    content_types = {}
    for seg in segments:
        ct = seg['content_type']
        if ct not in content_types:
            content_types[ct] = {'count': 0, 'duration': 0}
        content_types[ct]['count'] += 1
        content_types[ct]['duration'] += seg['duration']

    print("\n[테이블 타입별 분포]")
    for ct, stats in content_types.items():
        percentage = (stats['duration'] / duration * 100) if duration > 0 else 0
        print(f"  {ct:15s}: {stats['count']:2d}개, {percentage:5.1f}% ({stats['duration']:.1f}초)")

    # CSV 저장
    csv_path = f"{os.path.splitext(video_file)[0]}_segments.csv"
    import csv
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['세그먼트', '시작', '종료', '지속시간', '콘텐츠 타입'])
        for seg in segments:
            writer.writerow([
                seg['segment_id'],
                seg['start_tc'],
                seg['end_tc'],
                f"{seg['duration']:.1f}",
                seg['content_type']
            ])

    print(f"\n결과 저장: {csv_path}")


if __name__ == "__main__":
    main()