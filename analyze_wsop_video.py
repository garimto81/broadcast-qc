"""
WSOP Video Scene Analyzer - Broadcast QC Tool
Analyzes poker tournament footage for scene boundaries and content
"""

import cv2
import numpy as np
import json
from datetime import timedelta
import os
from typing import List, Dict, Tuple

class WSCPSceneAnalyzer:
    """Scene analyzer optimized for poker broadcast content"""

    def __init__(self, video_path: str, threshold: float = 30.0):
        self.video_path = video_path
        self.threshold = threshold
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps

    def get_video_info(self) -> Dict:
        """Get basic video information"""
        return {
            "filename": os.path.basename(self.video_path),
            "fps": self.fps,
            "frame_count": self.frame_count,
            "resolution": f"{self.width}x{self.height}",
            "duration": str(timedelta(seconds=int(self.duration))),
            "duration_seconds": self.duration
        }

    def detect_scenes(self) -> List[Dict]:
        """Detect scene boundaries using histogram and edge detection"""
        scenes = []
        prev_frame = None
        prev_hist = None
        scene_start_frame = 0
        scene_start_time = 0.0

        print(f"Analyzing {self.frame_count} frames...")

        for frame_idx in range(0, self.frame_count, 2):  # Sample every 2 frames for speed
            ret, frame = self.cap.read()
            if not ret:
                break

            # Progress indicator
            if frame_idx % 100 == 0:
                progress = (frame_idx / self.frame_count) * 100
                print(f"Progress: {progress:.1f}%", end='\r')

            # Calculate histogram
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            if prev_hist is not None:
                # Calculate histogram difference
                hist_diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)

                # Detect scene change
                if hist_diff > self.threshold:
                    scene_end_frame = frame_idx - 2
                    scene_end_time = scene_end_frame / self.fps

                    # Save scene
                    scene = {
                        "scene_id": len(scenes) + 1,
                        "start_frame": scene_start_frame,
                        "end_frame": scene_end_frame,
                        "start_time": scene_start_time,
                        "end_time": scene_end_time,
                        "duration": scene_end_time - scene_start_time,
                        "start_timecode": str(timedelta(seconds=int(scene_start_time))),
                        "end_timecode": str(timedelta(seconds=int(scene_end_time)))
                    }

                    # Analyze scene content
                    scene["content_type"] = self._analyze_content_type(frame)

                    scenes.append(scene)

                    # Reset for new scene
                    scene_start_frame = frame_idx
                    scene_start_time = frame_idx / self.fps

            prev_hist = hist
            prev_frame = frame

        # Add final scene
        if scene_start_frame < self.frame_count - 1:
            scene = {
                "scene_id": len(scenes) + 1,
                "start_frame": scene_start_frame,
                "end_frame": self.frame_count - 1,
                "start_time": scene_start_time,
                "end_time": self.duration,
                "duration": self.duration - scene_start_time,
                "start_timecode": str(timedelta(seconds=int(scene_start_time))),
                "end_timecode": str(timedelta(seconds=int(self.duration)))
            }
            scenes.append(scene)

        print(f"\nDetected {len(scenes)} scenes")
        return scenes

    def _analyze_content_type(self, frame) -> str:
        """Analyze frame to determine poker content type"""
        # Simple heuristic based on dominant colors
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Check for green table (poker table)
        green_lower = np.array([35, 40, 40])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        green_ratio = np.count_nonzero(green_mask) / (frame.shape[0] * frame.shape[1])

        if green_ratio > 0.3:
            return "table_view"
        elif green_ratio > 0.1:
            return "mixed_view"
        else:
            # Check if it's mostly dark (could be graphics/transition)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            if mean_brightness < 50:
                return "transition"
            elif mean_brightness > 200:
                return "graphics"
            else:
                return "player_closeup"

    def generate_report(self, scenes: List[Dict]) -> Dict:
        """Generate comprehensive scene analysis report"""
        report = {
            "video_info": self.get_video_info(),
            "analysis_summary": {
                "total_scenes": len(scenes),
                "average_scene_duration": sum(s["duration"] for s in scenes) / len(scenes) if scenes else 0,
                "shortest_scene": min(scenes, key=lambda x: x["duration"]) if scenes else None,
                "longest_scene": max(scenes, key=lambda x: x["duration"]) if scenes else None
            },
            "content_distribution": {},
            "scenes": scenes
        }

        # Count content types
        content_types = {}
        for scene in scenes:
            content_type = scene.get("content_type", "unknown")
            content_types[content_type] = content_types.get(content_type, 0) + 1

        report["content_distribution"] = content_types

        return report

    def save_report(self, report: Dict, output_path: str = None):
        """Save analysis report to JSON file"""
        if output_path is None:
            base_name = os.path.splitext(self.video_path)[0]
            output_path = f"{base_name}_scene_analysis.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"Report saved to: {output_path}")
        return output_path

    def export_csv(self, scenes: List[Dict], output_path: str = None):
        """Export scenes to CSV for Excel/Sheets"""
        import csv

        if output_path is None:
            base_name = os.path.splitext(self.video_path)[0]
            output_path = f"{base_name}_scenes.csv"

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if scenes:
                fieldnames = ["scene_id", "start_timecode", "end_timecode", "duration", "content_type"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for scene in scenes:
                    writer.writerow({
                        "scene_id": scene["scene_id"],
                        "start_timecode": scene["start_timecode"],
                        "end_timecode": scene["end_timecode"],
                        "duration": f"{scene['duration']:.2f}",
                        "content_type": scene.get("content_type", "")
                    })

        print(f"CSV exported to: {output_path}")
        return output_path

    def close(self):
        """Release video capture"""
        self.cap.release()


def main():
    """Main execution function"""
    video_file = "sample_wsop_sc_cy01.mp4"

    print("=" * 60)
    print("WSOP Video Scene Analyzer - Broadcast QC")
    print("=" * 60)

    # Check if file exists
    if not os.path.exists(video_file):
        print(f"Error: Video file '{video_file}' not found!")
        return

    # Initialize analyzer
    analyzer = WSCPSceneAnalyzer(video_file, threshold=25.0)  # Lower threshold for poker content

    # Get video info
    video_info = analyzer.get_video_info()
    print(f"\nVideo Information:")
    print(f"  File: {video_info['filename']}")
    print(f"  Resolution: {video_info['resolution']}")
    print(f"  Duration: {video_info['duration']}")
    print(f"  FPS: {video_info['fps']:.2f}")
    print(f"  Total Frames: {video_info['frame_count']}")

    print("\nStarting scene detection...")
    print("-" * 40)

    # Detect scenes
    scenes = analyzer.detect_scenes()

    # Generate report
    report = analyzer.generate_report(scenes)

    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Total Scenes Detected: {report['analysis_summary']['total_scenes']}")
    print(f"Average Scene Duration: {report['analysis_summary']['average_scene_duration']:.2f} seconds")

    print("\nContent Type Distribution:")
    for content_type, count in report['content_distribution'].items():
        percentage = (count / len(scenes)) * 100
        print(f"  {content_type}: {count} scenes ({percentage:.1f}%)")

    print("\nFirst 10 Scenes:")
    print("-" * 40)
    for scene in scenes[:10]:
        print(f"Scene {scene['scene_id']:3d}: {scene['start_timecode']} -> {scene['end_timecode']} "
              f"({scene['duration']:.2f}s) - {scene.get('content_type', 'unknown')}")

    if len(scenes) > 10:
        print(f"... and {len(scenes) - 10} more scenes")

    # Save outputs
    print("\nSaving reports...")
    json_path = analyzer.save_report(report)
    csv_path = analyzer.export_csv(scenes)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"✓ JSON Report: {json_path}")
    print(f"✓ CSV Export: {csv_path}")

    # Clean up
    analyzer.close()


if __name__ == "__main__":
    main()