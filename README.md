# 🛣️ RoadMarkingAI

**Indian Urban Road Marking Inspection System**

An AI-assisted, locally deployable road marking defect detection system for Indian urban and municipal roads. Processes vehicle-mounted video, CCTV footage, and road photographs to identify missing, faded, broken, and incorrectly arranged lane markings.

> **⚠️ Engineering Inspection Assistance Tool**
> This system is intended to support field inspection and engineering review only. It does **not** and cannot automatically determine legal compliance. Final determination of road-marking requirements shall be made by the responsible authority / qualified engineer using the current applicable IRC/MoRTH provisions, sanctioned drawings, and project specifications.

---

## 1. What the Project Does

- **Detects** road marking defects from vehicle video, CCTV video, and individual photographs
- **Associates defects with GPS coordinates** (phone GPS CSV, photo EXIF GPS, or no GPS)
- **Classifies defects** into four categories: Missing, Faded, Broken/Discontinuous, Incorrect Arrangement
- **Generates a professional PDF** inspection report for the responsible road authority
- **Displays an interactive map** of detected defects (when GPS is available)
- **Supports manual review** — every AI result can be confirmed, rejected, or flagged for further review by an inspector
- **Works entirely offline** — no cloud API required

---

## 2. Architecture

```
app.py                          ← Streamlit dashboard (thin UI layer)
│
├── src/
│   ├── input_manager.py        ← File upload, type classification, persistence
│   ├── gps_manager.py          ← GPS CSV parse, EXIF extract, interpolation, Haversine
│   ├── road_roi.py             ← Trapezoidal ROI masking, BEV transform
│   ├── lane_detector.py        ← BaseLaneDetector, ClassicalCVLaneDetector, Seg stub
│   ├── image_processor.py      ← Single-photo processing
│   ├── video_processor.py      ← Frame sampling, detection, GPS association
│   ├── temporal_tracker.py     ← EMA smoothing, spatial segment builder
│   ├── defect_detector.py      ← MISSING / FADED / BROKEN / INCORRECT_ARRANGEMENT
│   ├── road_classification.py  ← Road parameters data model
│   ├── standards_engine.py     ← IRC/MoRTH rules query
│   ├── evidence_manager.py     ← Save annotated evidence images
│   ├── map_generator.py        ← Folium interactive map
│   ├── report_generator.py     ← ReportLab PDF generation
│   └── inspection_pipeline.py  ← End-to-end orchestrator
│
├── config/
│   ├── settings.py             ← All configurable constants
│   └── standards_config.json   ← Standards metadata
│
├── standards/
│   └── irc35_rules.json        ← IRC:35-2015 + MoRTH Section 803 reference rules
│
├── models/
│   └── README.md               ← Instructions for plugging in a trained model
│
└── data/
    ├── uploads/                ← Session upload storage
    ├── processed/              ← Evidence images
    ├── reports/                ← Generated PDF reports
    └── gps/                    ← Optional GPS file storage
```

---

## 3. Installation (Local)

### Prerequisites

- Python 3.11
- pip
- ffmpeg (for video support)

```bash
# Clone the repository
git clone https://github.com/your-org/RoadMarkingAI.git
cd RoadMarkingAI

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Docker Installation (Recommended)

Docker is the primary deployment method. It includes all system dependencies (ffmpeg, OpenCV libs) automatically.

```bash
# Build the Docker image
docker build -t roadmarkingai .

# Run the container
docker run --rm -p 8501:8501 -v "${PWD}/data:/app/data" roadmarkingai
```

Or using Docker Compose:

```bash
docker-compose up --build
```

The application will be available at: **http://localhost:8501**

---

## 5. How to Run (Local, no Docker)

```bash
streamlit run app.py
```

---

## 6. How to Upload Video

1. Open the app at http://localhost:8501
2. In the **"Upload Inspection Data"** section, click the file uploader
3. Select one or more video files (`.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`)
4. Optionally set the **Frame Sampling Rate** in the sidebar (default: 5 FPS)
5. Click **Run Inspection**

Supported camera types:
- Forward-facing vehicle-mounted phone/dashcam
- CCTV footage exported as video files

---

## 7. How to Upload Photographs

1. In the **"Upload Inspection Data"** section, select image files (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`)
2. Multiple images can be uploaded simultaneously
3. GPS coordinates will be read automatically from EXIF data if present
4. Click **Run Inspection**

---

## 8. How to Provide GPS

Three modes are supported:

### Mode 1: Phone GPS CSV
- Upload a CSV file with columns: `timestamp`, `latitude`, `longitude`, `accuracy`
- Also accepted: `time`/`lat`/`lon`/`hdop` and similar variants
- Enter the **Video Start DateTime** (UTC) to synchronise GPS with video timestamps

### Mode 2: Photo EXIF GPS
- If your photographs contain EXIF GPS data (as taken by a smartphone), this is read automatically

### Mode 3: No GPS
- The application works fully without GPS
- Defect positions are reported as time offsets (seconds from video start)
- The PDF and map clearly indicate GPS unavailability

---

## 9. How the Defect Detection Works

### Step-by-step pipeline:

1. **ROI masking** — A configurable trapezoidal region of interest excludes sky, buildings, and roadside objects. Only the road surface ahead is analysed.

2. **HSV thresholding** — The frame is converted to HSV colour space. Both white and yellow marking ranges are thresholded separately. This is more robust than simple brightness thresholding.

3. **Morphological cleaning** — Noise is removed using morphological open/close operations.

4. **Canny + Hough** — Edges are detected and Hough probabilistic line transform finds line segments. Lines are filtered by angle (road markings have characteristic slopes in a forward-camera view).

5. **Temporal tracking (video)** — Results are smoothed with an exponential moving average across frames. This prevents single-frame failures (shadow, vehicle occlusion) from generating false defects.

6. **Spatial segmentation** — The smoothed results are converted to spatial segments (e.g., "0–8 m: detected; 8–23 m: absent; 23–31 m: detected").

7. **Defect classification**:
   - **MISSING**: Sustained absence (≥ 8 frames by default) where marking is expected
   - **FADED**: Marking detected but visibility score consistently below threshold
   - **BROKEN**: Detected → absent → detected pattern with gap too large for normal dash spacing
   - **INCORRECT ARRANGEMENT**: Observed marking geometry inconsistent with expected configuration

8. **Standards lookup** — The road parameters are matched to applicable IRC:35-2015 / MoRTH Section 803 rules.

9. **Evidence saved** — Annotated frames saved for each defect.

10. **PDF report** — Complete inspection report generated with all findings.

---

## 10. Known Limitations

| Limitation | Impact |
|---|---|
| Classical CV only (no trained model) | Higher false positive/negative rate than a trained segmentation model |
| No retroreflectivity measurement | Cannot assess retroreflectivity compliance |
| Wet pavement glare | May cause missed detections or false positives |
| Dark/night conditions | Detection unreliable below ~30 mean brightness |
| Heavy traffic occlusion | Markings hidden by vehicles may be missed |
| GPS accuracy | Smartphone GPS may have ±5–15 m urban error; EXIF GPS has no accuracy field |
| Perspective distance | Without GPS or camera calibration, distances are estimated only |
| No camera calibration | Pixel-to-metre conversion is approximate |

---

## 11. How to Replace the CV Detector with an AI Segmentation Model

1. Place your model weights in `models/` (e.g., `models/lane_seg.pt`)

2. Open `src/lane_detector.py` and implement `SegmentationLaneDetector`:

```python
class SegmentationLaneDetector(BaseLaneDetector):
    def __init__(self, model_path: Path):
        # Load your model here (ultralytics, torch, onnxruntime, etc.)
        from ultralytics import YOLO
        self.model = YOLO(str(model_path))

    def detect_lane_markings(self, frame: np.ndarray) -> LaneDetectionResult:
        results = self.model(frame, verbose=False)
        # Parse results and return LaneDetectionResult
        # (see existing structure for required fields)
        ...
```

3. In `config/settings.py`, add:

```python
MODEL_PATH = BASE_DIR / "models" / "lane_seg.pt"
```

4. In `src/lane_detector.py` → `build_detector()`, the factory will automatically use your model when the file exists:

```python
def build_detector(model_path=None):
    if model_path and model_path.exists():
        return SegmentationLaneDetector(model_path)
    return ClassicalCVLaneDetector()
```

The `BaseLaneDetector` interface ensures zero changes are needed in the pipeline.

---

## 12. How to Modify IRC/MoRTH Rules

Edit `standards/irc35_rules.json`. Each rule follows this schema:

```json
{
  "id": "IRC35-CL-001",
  "standard": "IRC:35-2015",
  "section": "Clause 3 / 4",
  "road_type": ["urban_arterial"],
  "marking_type": "centre_line",
  "divided_undivided": "undivided",
  "applicability_notes": "Plain-language applicability...",
  "engineering_note": "Specific verification requirements...",
  "verification_required": true,
  "source_document": "IRC:35-2015 — Guidelines for Road Markings"
}
```

**Do not add fabricated dimensions.** If the exact dimension is uncertain, add the rule with `applicability_notes` directing the engineer to the relevant clause and set `verification_required: true`.

---

## 13. How to Generate the PDF Report

1. Run an inspection
2. Optionally review each defect (Confirm / Reject / Needs Review)
3. Go to the **"Report"** tab
4. Click **Generate PDF Report**
5. Download the PDF

The PDF includes:
- Cover page with project details
- Executive summary
- Defect location table with GPS coordinates
- Annotated photographic evidence
- Applicable IRC/MoRTH standards references
- Recommended rectification actions
- Engineering disclaimer

---

## 14. Project Structure

```
RoadMarkingAI/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── .dockerignore
├── .gitignore
├── app.py                      ← Streamlit entry point
├── config/
│   ├── settings.py             ← All configuration constants
│   └── standards_config.json
├── src/
│   ├── __init__.py
│   ├── input_manager.py
│   ├── gps_manager.py
│   ├── road_roi.py
│   ├── lane_detector.py
│   ├── image_processor.py
│   ├── video_processor.py
│   ├── temporal_tracker.py
│   ├── defect_detector.py
│   ├── road_classification.py
│   ├── standards_engine.py
│   ├── evidence_manager.py
│   ├── map_generator.py
│   ├── report_generator.py
│   └── inspection_pipeline.py
├── models/
│   └── README.md
├── standards/
│   ├── README.md
│   └── irc35_rules.json
├── data/
│   ├── uploads/
│   ├── processed/
│   ├── reports/
│   └── gps/
└── tests/
    ├── test_gps.py
    ├── test_detector.py
    └── test_report.py
```

---

## Docker Quick Reference

```bash
# Build
docker build -t roadmarkingai .

# Run (with data persistence)
docker run --rm -p 8501:8501 -v "${PWD}/data:/app/data" roadmarkingai

# Docker Compose
docker-compose up --build

# Access
http://localhost:8501
```

---

## Running Tests

```bash
# Install test dependencies (already in requirements.txt via pytest)
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_gps.py -v
```

---

## Licence

MIT — see LICENSE file.

---

*RoadMarkingAI is an engineering inspection assistance tool. It does not constitute a legal compliance engine. All findings require independent engineering verification.*
