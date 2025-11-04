# Broadcast QC Video Analysis Tool

## Overview
Professional video quality control and scene analysis tools for broadcast content, optimized for poker tournament footage (WSOP) and general broadcast QC workflows.

## Features

### 1. Scene Detection (`analyze_wsop_video.py`)
- **Histogram-based scene detection** with configurable thresholds
- **Content type classification** (table view, player closeup, graphics, transitions)
- **CSV export** for easy integration with spreadsheets
- **JSON reports** for programmatic processing
- Optimized for poker broadcast content

### 2. Comprehensive Broadcast QC (`broadcast_qc_analyzer.py`)
- **Technical issue detection:**
  - Black frames detection
  - Freeze frame segments
  - Flash/strobe detection
- **Color consistency analysis**
- **Aspect ratio verification**
- **Scene change rate analysis**
- **QC status determination** (PASS/WARNING/FAIL)
- **Actionable recommendations**

### 3. QC Summary Generator (`create_qc_summary.py`)
- Consolidates all analysis results
- Generates human-readable reports
- Creates prioritized QC checklists
- Content distribution statistics

## Installation

```bash
# Install required dependencies
pip install opencv-python numpy ffmpeg-python
```

## Usage

### Basic Scene Analysis
```bash
python analyze_wsop_video.py
```
This will analyze `sample_wsop_sc_cy01.mp4` and generate:
- `sample_wsop_sc_cy01_scene_analysis.json` - Detailed scene data
- `sample_wsop_sc_cy01_scenes.csv` - Scene list for spreadsheets

### Full Broadcast QC
```bash
python broadcast_qc_analyzer.py [video_file]
```
Generates comprehensive QC report with technical issue detection.

### Generate Summary Report
```bash
python create_qc_summary.py
```
Creates consolidated summary from all analysis files.

## Output Files

### Scene Analysis Output
- **JSON Report**: Complete scene data with timecodes, durations, and content types
- **CSV Export**: Simplified scene list for review in Excel/Sheets

### QC Report Output
- **Technical Issues**: Black frames, freeze frames, flash detection
- **Scene Analysis**: Scene count, change rate, content distribution
- **Color Analysis**: Consistency metrics, brightness variance
- **Recommendations**: Prioritized list of issues to review

## Analysis Results for WSOP Sample

### Video Information
- **File**: sample_wsop_sc_cy01.mp4
- **Resolution**: 1920x1080 (16:9)
- **Duration**: 9:01 (541 seconds)
- **Frame Rate**: 59.94 fps
- **Bitrate**: 15.84 Mbps

### Key Findings
1. **Scene Detection**:
   - 7 scenes detected (initial analysis)
   - 4 scenes in advanced analysis
   - Low scene change rate (0.44 per minute)

2. **Technical Issues**:
   - ⚠️ **73 freeze frame segments** totaling 247.4 seconds
   - ✅ No black frames detected
   - ✅ No flash/strobe issues

3. **Content Classification**:
   - Primarily player closeup shots (71.4%)
   - One transition detected
   - Content appears stable with few scene changes

### QC Status
**PASS WITH WARNINGS** - Video is broadcastable but requires review of freeze frames

## Recommendations

1. **HIGH PRIORITY**: Review freeze frame segments - may indicate encoding issues or intentional holds
2. **MEDIUM**: Verify low scene count is intentional (could indicate static camera work typical of poker coverage)
3. **LOW**: Content appears consistent with poker broadcast standards

## Technical Details

### Scene Detection Algorithm
- **Combined method**: 70% histogram difference + 30% edge detection
- **Adaptive thresholds**: Configurable based on content type
- **Minimum scene duration**: 0.5 seconds to avoid false positives

### QC Thresholds
- **Black frame**: Mean brightness < 10
- **Freeze detection**: Frame similarity > 99%
- **Flash detection**: Brightness change > 100 units
- **Standard aspect ratios**: 16:9, 4:3, 21:9

## Use Cases

1. **Pre-broadcast QC**: Verify technical quality before transmission
2. **Post-production Review**: Identify issues for correction
3. **Archive Validation**: Ensure content integrity
4. **Automated Workflow**: Integration with broadcast automation systems

## Future Enhancements

- [ ] GPU acceleration for faster processing
- [ ] Audio analysis integration
- [ ] Machine learning-based content classification
- [ ] Real-time processing capability
- [ ] Web-based UI for results visualization
- [ ] Integration with broadcast automation systems

## License
Internal use only - Broadcast QC Division

## Support
For questions or issues, contact the Broadcast QC team.