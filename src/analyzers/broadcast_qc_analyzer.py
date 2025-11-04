"""
Broadcast QC Analyzer - Professional Video Quality Control Tool
For WSOP and broadcast video content analysis
"""

import cv2
import numpy as np
import json
import os
from datetime import timedelta
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class BroadcastQCAnalyzer:
    """Professional broadcast quality control analyzer"""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        # Video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

        # QC thresholds
        self.black_frame_threshold = 10  # Brightness below this is considered black
        self.freeze_frame_threshold = 0.99  # Similarity above this is freeze
        self.audio_silence_threshold = -50  # dB

        # Scene detection parameters
        self.scene_threshold = 25.0
        self.min_scene_duration = 0.5  # seconds

    def analyze_full_qc(self) -> Dict:
        """Perform comprehensive broadcast QC analysis"""
        print("\n[1/7] Detecting scenes...")
        scenes = self.detect_scenes_advanced()

        print("[2/7] Checking for black frames...")
        black_frames = self.detect_black_frames()

        print("[3/7] Detecting freeze frames...")
        freeze_frames = self.detect_freeze_frames()

        print("[4/7] Analyzing color consistency...")
        color_stats = self.analyze_color_consistency()

        print("[5/7] Checking aspect ratio...")
        aspect_issues = self.check_aspect_ratio()

        print("[6/7] Detecting flash frames...")
        flash_frames = self.detect_flash_frames()

        print("[7/7] Generating QC report...")

        # Compile full QC report
        qc_report = {
            "file_info": {
                "filename": os.path.basename(self.video_path),
                "resolution": f"{self.width}x{self.height}",
                "aspect_ratio": f"{self.width/self.height:.2f}:1",
                "fps": f"{self.fps:.2f}",
                "duration": str(timedelta(seconds=int(self.duration))),
                "total_frames": self.frame_count,
                "bitrate_estimate": self._estimate_bitrate()
            },
            "scene_analysis": {
                "total_scenes": len(scenes),
                "scenes": scenes[:20] if len(scenes) > 20 else scenes,  # First 20 scenes
                "scene_change_rate": len(scenes) / (self.duration / 60) if self.duration > 0 else 0
            },
            "technical_issues": {
                "black_frames": {
                    "count": len(black_frames),
                    "frames": black_frames[:10],  # First 10
                    "percentage": (len(black_frames) / self.frame_count) * 100
                },
                "freeze_frames": {
                    "count": len(freeze_frames),
                    "segments": freeze_frames[:10],
                    "total_duration": sum(f["duration"] for f in freeze_frames)
                },
                "flash_frames": {
                    "count": len(flash_frames),
                    "frames": flash_frames[:10],
                    "severity": "High" if len(flash_frames) > 10 else "Low"
                }
            },
            "color_analysis": color_stats,
            "aspect_ratio_issues": aspect_issues,
            "qc_status": self._determine_qc_status(black_frames, freeze_frames, flash_frames),
            "recommendations": self._generate_recommendations(black_frames, freeze_frames, flash_frames, scenes)
        }

        return qc_report

    def detect_scenes_advanced(self) -> List[Dict]:
        """Advanced scene detection with content classification"""
        scenes = []
        prev_hist = None
        prev_edges = None
        scene_start = 0
        scene_frames = []

        frame_skip = max(1, int(self.fps / 10))  # Sample 10 frames per second

        for frame_idx in range(0, self.frame_count, frame_skip):
            ret, frame = self.cap.read()
            if not ret:
                break

            # Progress
            if frame_idx % 100 == 0:
                progress = (frame_idx / self.frame_count) * 100
                print(f"  Progress: {progress:.1f}%", end='\r')

            # Calculate features
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / edges.size

            if prev_hist is not None:
                # Combined difference metric
                hist_diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
                edge_diff = abs(edge_density - prev_edges)
                combined_diff = hist_diff + edge_diff * 100

                if combined_diff > self.scene_threshold:
                    # Scene change detected
                    if len(scene_frames) > 0:
                        scene_duration = (frame_idx - scene_start) / self.fps
                        if scene_duration >= self.min_scene_duration:
                            scene = self._create_scene_entry(
                                scene_id=len(scenes) + 1,
                                start_frame=scene_start,
                                end_frame=frame_idx - frame_skip,
                                scene_frames=scene_frames
                            )
                            scenes.append(scene)

                    # Start new scene
                    scene_start = frame_idx
                    scene_frames = [frame]
                else:
                    scene_frames.append(frame)
            else:
                scene_frames = [frame]

            prev_hist = hist
            prev_edges = edge_density

        # Add final scene
        if scene_start < self.frame_count - 1:
            scene = self._create_scene_entry(
                scene_id=len(scenes) + 1,
                start_frame=scene_start,
                end_frame=self.frame_count - 1,
                scene_frames=scene_frames
            )
            scenes.append(scene)

        print(f"\n  Detected {len(scenes)} scenes")
        return scenes

    def _create_scene_entry(self, scene_id: int, start_frame: int,
                           end_frame: int, scene_frames: List) -> Dict:
        """Create detailed scene entry with analysis"""
        start_time = start_frame / self.fps
        end_time = end_frame / self.fps
        duration = end_time - start_time

        # Analyze scene content
        content_type = "unknown"
        avg_brightness = 0

        if len(scene_frames) > 0:
            # Sample middle frame
            middle_frame = scene_frames[len(scene_frames) // 2] if len(scene_frames) > 0 else None
            if middle_frame is not None:
                content_type = self._classify_content(middle_frame)
                gray = cv2.cvtColor(middle_frame, cv2.COLOR_BGR2GRAY)
                avg_brightness = np.mean(gray)

        return {
            "scene_id": scene_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "start_tc": self._format_timecode(start_time),
            "end_tc": self._format_timecode(end_time),
            "content_type": content_type,
            "avg_brightness": avg_brightness,
            "qc_notes": []
        }

    def _classify_content(self, frame) -> str:
        """Classify frame content for broadcast"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Brightness analysis
        mean_brightness = np.mean(gray)

        # Color saturation analysis
        saturation = hsv[:, :, 1]
        mean_saturation = np.mean(saturation)

        # Green detection (poker table)
        green_lower = np.array([35, 40, 40])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        green_ratio = np.count_nonzero(green_mask) / (frame.shape[0] * frame.shape[1])

        # Classification logic
        if mean_brightness < 20:
            return "black/fade"
        elif mean_brightness > 240:
            return "white/flash"
        elif green_ratio > 0.3:
            return "table_wide"
        elif green_ratio > 0.1:
            return "table_medium"
        elif mean_saturation < 30:
            return "graphics/text"
        else:
            # Edge detection for detail level
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / edges.size

            if edge_density > 0.1:
                return "player_close"
            else:
                return "crowd/ambient"

    def detect_black_frames(self) -> List[Dict]:
        """Detect black or near-black frames"""
        black_frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_skip = max(1, int(self.fps / 5))  # Check 5 frames per second

        for frame_idx in range(0, self.frame_count, frame_skip):
            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)

            if mean_brightness < self.black_frame_threshold:
                black_frames.append({
                    "frame": frame_idx,
                    "timecode": self._format_timecode(frame_idx / self.fps),
                    "brightness": mean_brightness
                })

        return black_frames

    def detect_freeze_frames(self) -> List[Dict]:
        """Detect frozen/duplicate frames"""
        freeze_segments = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        prev_frame = None
        freeze_start = None
        freeze_count = 0

        frame_skip = max(1, int(self.fps / 10))

        for frame_idx in range(0, self.frame_count, frame_skip):
            ret, frame = self.cap.read()
            if not ret:
                break

            if prev_frame is not None:
                # Compare frames
                diff = cv2.absdiff(frame, prev_frame)
                similarity = 1 - (np.mean(diff) / 255)

                if similarity > self.freeze_frame_threshold:
                    if freeze_start is None:
                        freeze_start = frame_idx - frame_skip
                    freeze_count += 1
                else:
                    if freeze_start is not None and freeze_count > 2:
                        freeze_segments.append({
                            "start_frame": freeze_start,
                            "end_frame": frame_idx - frame_skip,
                            "duration": (freeze_count * frame_skip) / self.fps,
                            "start_tc": self._format_timecode(freeze_start / self.fps)
                        })
                    freeze_start = None
                    freeze_count = 0

            prev_frame = frame.copy()

        return freeze_segments

    def detect_flash_frames(self) -> List[Dict]:
        """Detect sudden brightness changes (flash frames)"""
        flash_frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        prev_brightness = None

        for frame_idx in range(0, self.frame_count, max(1, int(self.fps / 5))):
            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)

            if prev_brightness is not None:
                brightness_change = abs(brightness - prev_brightness)

                if brightness_change > 100:  # Significant change
                    flash_frames.append({
                        "frame": frame_idx,
                        "timecode": self._format_timecode(frame_idx / self.fps),
                        "brightness_change": brightness_change
                    })

            prev_brightness = brightness

        return flash_frames

    def analyze_color_consistency(self) -> Dict:
        """Analyze color consistency throughout video"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        color_samples = []
        frame_skip = max(1, int(self.fps * 2))  # Sample every 2 seconds

        for frame_idx in range(0, self.frame_count, frame_skip):
            ret, frame = self.cap.read()
            if not ret:
                break

            # Calculate average color
            avg_color = np.mean(frame, axis=(0, 1))

            # Calculate color temperature estimate
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            avg_hue = np.mean(hsv[:, :, 0])
            avg_saturation = np.mean(hsv[:, :, 1])
            avg_value = np.mean(hsv[:, :, 2])

            color_samples.append({
                "frame": frame_idx,
                "avg_rgb": avg_color.tolist(),
                "hue": avg_hue,
                "saturation": avg_saturation,
                "brightness": avg_value
            })

        # Calculate statistics
        if color_samples:
            hues = [s["hue"] for s in color_samples]
            saturations = [s["saturation"] for s in color_samples]
            brightnesses = [s["brightness"] for s in color_samples]

            return {
                "samples": len(color_samples),
                "hue_variance": np.std(hues),
                "saturation_variance": np.std(saturations),
                "brightness_variance": np.std(brightnesses),
                "avg_brightness": np.mean(brightnesses),
                "color_consistency": "Good" if np.std(hues) < 30 else "Poor"
            }

        return {}

    def check_aspect_ratio(self) -> Dict:
        """Check for aspect ratio issues"""
        standard_ratios = {
            "16:9": 16/9,
            "4:3": 4/3,
            "21:9": 21/9
        }

        current_ratio = self.width / self.height

        # Find closest standard ratio
        closest_standard = None
        min_diff = float('inf')

        for name, ratio in standard_ratios.items():
            diff = abs(current_ratio - ratio)
            if diff < min_diff:
                min_diff = diff
                closest_standard = name

        return {
            "current": f"{current_ratio:.3f}:1",
            "resolution": f"{self.width}x{self.height}",
            "closest_standard": closest_standard,
            "is_standard": min_diff < 0.01,
            "recommendation": "OK" if min_diff < 0.01 else f"Consider {closest_standard}"
        }

    def _estimate_bitrate(self) -> str:
        """Estimate video bitrate"""
        file_size = os.path.getsize(self.video_path)
        bitrate_mbps = (file_size * 8) / (self.duration * 1000000) if self.duration > 0 else 0
        return f"{bitrate_mbps:.2f} Mbps"

    def _format_timecode(self, seconds: float) -> str:
        """Format seconds as timecode"""
        return str(timedelta(seconds=int(seconds)))

    def _determine_qc_status(self, black_frames, freeze_frames, flash_frames) -> str:
        """Determine overall QC status"""
        issues = 0

        if len(black_frames) > 5:
            issues += 2
        elif len(black_frames) > 0:
            issues += 1

        if len(freeze_frames) > 3:
            issues += 2
        elif len(freeze_frames) > 0:
            issues += 1

        if len(flash_frames) > 10:
            issues += 1

        if issues == 0:
            return "PASS - No issues detected"
        elif issues <= 2:
            return "PASS WITH WARNINGS - Minor issues detected"
        elif issues <= 4:
            return "REVIEW REQUIRED - Multiple issues detected"
        else:
            return "FAIL - Significant issues require attention"

    def _generate_recommendations(self, black_frames, freeze_frames, flash_frames, scenes) -> List[str]:
        """Generate QC recommendations"""
        recommendations = []

        if len(black_frames) > 0:
            recommendations.append(f"Review {len(black_frames)} black frames for potential encoding issues")

        if len(freeze_frames) > 0:
            total_freeze = sum(f["duration"] for f in freeze_frames)
            recommendations.append(f"Check {len(freeze_frames)} freeze segments ({total_freeze:.1f}s total)")

        if len(flash_frames) > 10:
            recommendations.append("High number of flash frames detected - check for strobe effects")

        if len(scenes) < 5:
            recommendations.append("Very few scene changes - verify content is complete")
        elif len(scenes) > 100:
            recommendations.append("Excessive scene changes - possible encoding artifact")

        if not recommendations:
            recommendations.append("Video passes all basic QC checks")

        return recommendations

    def save_qc_report(self, report: Dict, output_path: str = None):
        """Save QC report to JSON"""
        if output_path is None:
            base_name = os.path.splitext(self.video_path)[0]
            output_path = f"{base_name}_qc_report.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return output_path

    def generate_qc_summary(self, report: Dict) -> str:
        """Generate human-readable QC summary"""
        summary = []
        summary.append("=" * 60)
        summary.append("BROADCAST QC REPORT")
        summary.append("=" * 60)

        # File info
        info = report["file_info"]
        summary.append(f"\nFile: {info['filename']}")
        summary.append(f"Resolution: {info['resolution']} ({info['aspect_ratio']})")
        summary.append(f"Duration: {info['duration']} @ {info['fps']} fps")
        summary.append(f"Estimated Bitrate: {info['bitrate_estimate']}")

        # QC Status
        summary.append(f"\nQC STATUS: {report['qc_status']}")
        summary.append("-" * 40)

        # Issues
        issues = report["technical_issues"]
        summary.append("\nTECHNICAL ISSUES:")
        summary.append(f"  Black Frames: {issues['black_frames']['count']} ({issues['black_frames']['percentage']:.2f}%)")
        summary.append(f"  Freeze Frames: {issues['freeze_frames']['count']} segments ({issues['freeze_frames']['total_duration']:.1f}s)")
        summary.append(f"  Flash Frames: {issues['flash_frames']['count']} ({issues['flash_frames']['severity']} severity)")

        # Scene Analysis
        scene_info = report["scene_analysis"]
        summary.append(f"\nSCENE ANALYSIS:")
        summary.append(f"  Total Scenes: {scene_info['total_scenes']}")
        summary.append(f"  Scene Change Rate: {scene_info['scene_change_rate']:.1f} per minute")

        # Color Analysis
        color = report["color_analysis"]
        if color:
            summary.append(f"\nCOLOR ANALYSIS:")
            summary.append(f"  Consistency: {color['color_consistency']}")
            summary.append(f"  Avg Brightness: {color['avg_brightness']:.1f}")
            summary.append(f"  Brightness Variance: {color['brightness_variance']:.1f}")

        # Recommendations
        summary.append("\nRECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            summary.append(f"  {i}. {rec}")

        summary.append("\n" + "=" * 60)

        return "\n".join(summary)

    def close(self):
        """Release resources"""
        self.cap.release()


def main():
    """Main execution"""
    import sys

    video_file = sys.argv[1] if len(sys.argv) > 1 else "sample_wsop_sc_cy01.mp4"

    if not os.path.exists(video_file):
        print(f"Error: Video file '{video_file}' not found!")
        return

    print("\n" + "=" * 60)
    print("PROFESSIONAL BROADCAST QC ANALYZER")
    print("=" * 60)

    # Initialize analyzer
    analyzer = BroadcastQCAnalyzer(video_file)

    # Run full QC analysis
    qc_report = analyzer.analyze_full_qc()

    # Save report
    report_path = analyzer.save_qc_report(qc_report)

    # Generate and print summary
    summary = analyzer.generate_qc_summary(qc_report)
    print("\n" + summary)

    print(f"\nDetailed report saved to: {report_path}")

    # Clean up
    analyzer.close()


if __name__ == "__main__":
    main()