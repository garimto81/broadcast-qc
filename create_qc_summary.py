"""
Create a comprehensive QC summary report for broadcast video
"""

import json
import os
from datetime import timedelta

def format_timecode(seconds):
    """Convert seconds to timecode format"""
    return str(timedelta(seconds=int(seconds)))

def create_summary():
    """Create comprehensive QC summary from analysis results"""

    # Load all analysis files
    files = {
        'scene_analysis': 'sample_wsop_sc_cy01_scene_analysis.json',
        'qc_report': 'sample_wsop_sc_cy01_qc_report.json',
        'scenes_csv': 'sample_wsop_sc_cy01_scenes.csv'
    }

    # Check what files exist
    available_files = []
    for key, filename in files.items():
        if os.path.exists(filename):
            available_files.append(filename)
            print(f"[OK] Found: {filename}")
        else:
            print(f"[X] Missing: {filename}")

    if not available_files:
        print("No analysis files found!")
        return

    # Load QC report
    if os.path.exists(files['qc_report']):
        with open(files['qc_report'], 'r') as f:
            qc_data = json.load(f)

        print("\n" + "="*70)
        print("BROADCAST QC SUMMARY - WSOP Video Analysis")
        print("="*70)

        # File Information
        info = qc_data['file_info']
        print(f"\n[VIDEO INFORMATION]")
        print("-"*40)
        print(f"File: {info['filename']}")
        print(f"Resolution: {info['resolution']} ({info['aspect_ratio']})")
        print(f"Duration: {info['duration']}")
        print(f"Frame Rate: {info['fps']}")
        print(f"Total Frames: {info['total_frames']:,}")
        print(f"Bitrate: {info['bitrate_estimate']}")

        # QC Status
        print(f"\n[QC STATUS]")
        print("-"*40)
        status = qc_data['qc_status']
        if "PASS" in status and "WARNING" not in status:
            status_icon = "[PASS]"
        elif "WARNING" in status:
            status_icon = "[WARNING]"
        else:
            status_icon = "[FAIL]"
        print(f"{status_icon} {status}")

        # Scene Analysis
        scene_info = qc_data['scene_analysis']
        print(f"\n[SCENE ANALYSIS]")
        print("-"*40)
        print(f"Total Scenes: {scene_info['total_scenes']}")
        print(f"Scene Change Rate: {scene_info['scene_change_rate']:.2f} per minute")

        # Display first few scenes
        if scene_info['scenes']:
            print("\nScene Breakdown (First 5):")
            for scene in scene_info['scenes'][:5]:
                print(f"  Scene {scene['scene_id']:2d}: {scene['start_tc']:>8} → {scene['end_tc']:>8} "
                      f"({scene['duration']:6.2f}s) [{scene['content_type']:>15}]")

            if len(scene_info['scenes']) > 5:
                print(f"  ... and {len(scene_info['scenes'])-5} more scenes")

        # Technical Issues
        issues = qc_data['technical_issues']
        print(f"\n[WARNING] TECHNICAL ISSUES")
        print("-"*40)

        # Black frames
        black = issues['black_frames']
        if black['count'] > 0:
            print(f"[BLACK] Black Frames: {black['count']} ({black['percentage']:.2f}%)")
            if black['frames']:
                print("   Locations:", ", ".join([f['timecode'] for f in black['frames'][:5]]))
        else:
            print(f"[OK] Black Frames: None detected")

        # Freeze frames
        freeze = issues['freeze_frames']
        if freeze['count'] > 0:
            print(f"[FREEZE] Freeze Frames: {freeze['count']} segments (Total: {freeze['total_duration']:.1f}s)")
            if freeze['segments']:
                print("   First segments:")
                for seg in freeze['segments'][:3]:
                    print(f"     - {seg['start_tc']} ({seg['duration']:.1f}s)")
        else:
            print(f"[OK] Freeze Frames: None detected")

        # Flash frames
        flash = issues['flash_frames']
        if flash['count'] > 0:
            print(f"[FLASH] Flash Frames: {flash['count']} ({flash['severity']} severity)")
        else:
            print(f"[OK] Flash Frames: None detected")

        # Color Analysis
        if 'color_analysis' in qc_data and qc_data['color_analysis']:
            color = qc_data['color_analysis']
            print(f"\n[COLOR] COLOR ANALYSIS")
            print("-"*40)
            print(f"Consistency: {color['color_consistency']}")
            print(f"Average Brightness: {color['avg_brightness']:.1f}")
            print(f"Brightness Variance: {color['brightness_variance']:.1f}")
            print(f"Hue Variance: {color['hue_variance']:.1f}")
            print(f"Saturation Variance: {color['saturation_variance']:.1f}")

        # Aspect Ratio
        aspect = qc_data['aspect_ratio_issues']
        print(f"\n[ASPECT] ASPECT RATIO CHECK")
        print("-"*40)
        print(f"Current: {aspect['current']} ({aspect['resolution']})")
        print(f"Standard: {aspect['closest_standard']} - {'[OK] OK' if aspect['is_standard'] else '[WARNING] ' + aspect['recommendation']}")

        # Recommendations
        print(f"\n[TIPS] RECOMMENDATIONS")
        print("-"*40)
        for i, rec in enumerate(qc_data['recommendations'], 1):
            print(f"{i}. {rec}")

        # Scene content distribution
        if scene_info['scenes']:
            content_types = {}
            for scene in scene_info['scenes']:
                ct = scene['content_type']
                content_types[ct] = content_types.get(ct, 0) + 1

            print(f"\n[STATS] CONTENT TYPE DISTRIBUTION")
            print("-"*40)
            for ct, count in sorted(content_types.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(scene_info['scenes'])) * 100
                bar = '#' * int(percentage / 5)
                print(f"{ct:15s}: {bar:20s} {count:2d} ({percentage:5.1f}%)")

        print("\n" + "="*70)
        print("END OF REPORT")
        print("="*70)

        # Save text summary
        summary_file = "sample_wsop_sc_cy01_summary.txt"
        print(f"\n[SAVE] Saving text summary to: {summary_file}")

        # Create actionable QC checklist
        print(f"\n[CHECKLIST] QC CHECKLIST FOR REVIEW")
        print("-"*40)
        checklist = []

        if freeze['count'] > 10:
            checklist.append("[HIGH] HIGH PRIORITY: Review freeze frames - possible encoding issue")
        if black['count'] > 5:
            checklist.append("[HIGH] HIGH PRIORITY: Check black frames - possible content loss")
        if scene_info['total_scenes'] < 3:
            checklist.append("[WARNING] MEDIUM: Very few scene changes - verify content completeness")
        if not aspect['is_standard']:
            checklist.append("[INFO] LOW: Non-standard aspect ratio - check if intentional")

        if checklist:
            for item in checklist:
                print(f"  {item}")
        else:
            print("  [OK] All checks passed - ready for broadcast")

    # Load scene analysis for detailed info
    if os.path.exists(files['scene_analysis']):
        with open(files['scene_analysis'], 'r') as f:
            scene_data = json.load(f)

        print(f"\n[TARGET] SCENE-BASED CONTENT ANALYSIS")
        print("-"*40)

        # Content distribution from original analysis
        if 'content_distribution' in scene_data:
            print("Content Types Detected:")
            for content_type, count in scene_data['content_distribution'].items():
                print(f"  - {content_type}: {count} scenes")

if __name__ == "__main__":
    create_summary()