# VisionPulse AI | AI Face Detection System 👁️⚡

A full-stack, real-time Computer Vision web application built with **Python**, **OpenCV**, and **Flask**. Designed for computer vision students, software engineers, and developers looking to understand object detection pipelines, real-time video streaming, and modern dashboard architecture.

> **Privacy Notice**: This project implements **Face Detection**, not Facial Recognition or Biometric Identity Matching. It identifies the spatial coordinates of human faces in a frame; it does *not* recognize individual identities, extract biometric fingerprints, or store user biometric profiles.

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [System Architecture](#-system-architecture)
- [Computer Vision Fundamentals (For Students)](#-computer-vision-fundamentals-for-students)
- [Prerequisites](#-prerequisites)
- [Installation & Setup (Windows)](#-installation--setup-windows)
- [Running the Application](#-running-the-application)
- [Using the Features](#-using-the-features)
  - [1. Real-Time Live Webcam Detection](#1-real-time-live-webcam-detection)
  - [2. Image Upload & Analysis](#2-image-upload--analysis)
  - [3. Frame-by-Frame Video Processing](#3-frame-by-frame-video-processing)
  - [4. Real-Time Analytics](#4-real-time-analytics)
- [REST API Reference](#-rest-api-reference)
- [Automated Testing](#-automated-testing)
- [Troubleshooting Guide](#-troubleshooting-guide)
- [Haar Cascade Limitations & Future Upgrades](#-haar-cascade-limitations--future-upgrades)
- [GitHub Setup & Push Guide](#-github-setup--push-guide)

---

## 🌟 Key Features

1. **Live Webcam Face Detection**
   - In-browser camera capture via WebRTC (`navigator.mediaDevices.getUserMedia`).
   - Ultra-low latency canvas bounding boxes drawn directly over video frames.
   - Throttled non-blocking frame streaming preventing network backlogs.
   - Real-time parameter tuning sliders (Scale Factor, Min Neighbors, Stream Rate).
   - Instant camera snapshot tool for inspection.

2. **Image Inspection & Localization**
   - Drag-and-drop or file browser supporting JPG, PNG, WEBP, and BMP.
   - High-precision OpenCV Haar Cascade detection for single portraits and group photos.
   - Detailed detection metrics: face count, execution time in milliseconds, bounding box coordinates (`x`, `y`, `w`, `h`).
   - One-click download of high-resolution annotated image outputs.

3. **Asynchronous Video Processing**
   - Support for MP4, AVI, MOV, and WEBM video uploads.
   - Multi-threaded background frame processor with live progress bar and ETA.
   - Real-time cancellation capability.
   - In-browser video player with annotated video export and download.
   - Explicit distinction between total detection events across frames vs. unique individuals.

4. **Interactive Analytics & Dashboard**
   - Rolling inference latency charts powered by **Chart.js**.
   - Input distribution breakdown (Webcam vs. Image vs. Video).
   - Real-time health check endpoint monitoring detector readiness.
   - Modern dark SaaS aesthetic with high-contrast HUD bounding boxes.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend Framework** | [Python 3.12](https://www.python.org/) & [Flask 3.x](https://flask.palletsprojects.com/) |
| **Computer Vision Engine** | [OpenCV (opencv-python)](https://opencv.org/) (`cv2.CascadeClassifier`, `haarcascade_frontalface_default.xml`) |
| **Numerical Computing** | [NumPy](https://numpy.org/) (array indexing & transformation) |
| **Frontend UI** | HTML5, CSS3, Modern Vanilla JavaScript (ES6+), Canvas API |
| **Design System** | Custom Dark SaaS Theme, [Bootstrap Icons](https://icons.getbootstrap.com/), Google Fonts (*Plus Jakarta Sans*, *JetBrains Mono*) |
| **Data Visualization** | [Chart.js 4.x](https://www.chartjs.org/) |
| **Testing Suite** | [Pytest](https://docs.pytest.org/) (unit tests, synthetic images, API validation) |

---

## 🏛️ System Architecture

```
ai-face-detection/
│
├── app.py                     # Flask server routes, API endpoints, background video worker
├── requirements.txt           # Production dependencies pinned for stability
├── README.md                  # Comprehensive documentation and student guide
├── .gitignore                 # Excludes venv, pycache, temporary uploads and outputs
│
├── detector/                  # Modular Computer Vision Engine
│   ├── __init__.py            # Exposes FaceDetector and DetectionResult
│   ├── face_detector.py       # Core OpenCV Haar Cascade detection logic & codecs
│   └── models/                # Pre-trained classifier models
│       └── haarcascade_frontalface_default.xml
│
├── templates/
│   └── index.html             # Single-page dashboard application template
│
├── static/
│   ├── css/
│   │   └── style.css          # Modern dark SaaS design system & HUD styling
│   └── js/
│       └── script.js          # WebRTC webcam loop, canvas overlays, polling & Chart.js
│
├── uploads/                   # Temporary folder for raw uploaded files (.gitkeep)
├── outputs/                   # Export directory for annotated images and videos (.gitkeep)
│
└── tests/
    ├── __init__.py
    └── test_app.py            # Automated test suite (19 test cases)
```

---

## 🧠 Computer Vision Fundamentals (For Students)

If you are a student learning computer vision, here is how the Viola-Jones Haar Cascade algorithm works:

```
[ Input Frame ] ──> [ Grayscale Conversion ] ──> [ Integral Image Computation ]
                                                          │
                                                          ▼
[ Detection Results ] <── [ Final Stage Pass ] <── [ AdaBoost Cascade Stages ]
  (Bounding Boxes)                                   (Rejects 95%+ background)
```

1. **Haar-like Features**: Rectangular patterns representing differences in brightness. For human faces, the bridge of the nose is typically lighter than the eyes, and forehead regions are lighter than eye sockets.
2. **Integral Images**: Allows computing the sum of all pixel values inside any arbitrary rectangle in constant \(O(1)\) time using four array lookups, making real-time multi-scale detection possible.
3. **AdaBoost**: Evaluates over 160,000 potential feature rectangles and selects only the most discriminative features to train weak classifiers into a strong classifier.
4. **Attentional Cascade**: Arranges classifiers in stages. Sub-windows that fail early stages are discarded immediately, saving computational cycles.

### Understanding Detection Parameters
- **`scaleFactor`** *(Default: 1.10)*: How much the image size is reduced at each image scale pass (1.10 = 10% reduction). Smaller values (e.g. 1.05) increase accuracy for different face sizes at the expense of processing speed.
- **`minNeighbors`** *(Default: 5)*: Number of neighboring candidate rectangles required to retain a detected face. Higher values eliminate false positives; lower values detect fainter or partially obscured faces.
- **`minSize`** *(Default: 30x30)*: The smallest bounding box dimensions considered a candidate face.

---

## 📋 Prerequisites

Before setting up the project, verify you have:
- **Windows 10 or 11**
- **Python 3.10 to 3.12** installed and added to your `PATH` (Verify in terminal: `python --version`)
- A modern web browser (**Chrome**, **Edge**, or **Firefox**)
- An integrated or USB webcam (for live camera testing)

---

## 💻 Installation & Setup (Windows)

Open **PowerShell** in the project directory:

### 1. Create a Python Virtual Environment
```powershell
python -m venv .venv
```

### 2. Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell displays an execution policy error, enable script execution: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### 3. Upgrade pip and Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Application

### 1. Start the Flask Server
```powershell
python app.py
```

You will see output similar to:
```
[INFO] FaceDetector initialized successfully with Haar Cascade model.
[INFO] Starting Face Detection Web Application on http://127.0.0.1:5000
 * Serving Flask app 'app'
 * Running on http://127.0.0.1:5000
```

### 2. Open the Dashboard in your Browser
Navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🎯 Using the Features

### 1. Real-Time Live Webcam Detection
1. Click **"Live Webcam"** in the sidebar.
2. Click the **"Start Camera"** button.
3. When prompted by your browser, click **"Allow"** to grant camera permission.
4. The application captures video frames, sends them to the local Flask backend, and renders futuristic cyan HUD bounding boxes directly over detected faces.
5. Adjust the **Scale Factor**, **Min Neighbors**, or **Stream Rate (FPS)** sliders to observe how sensitivity changes in real-time.
6. Click **"Snapshot"** to capture the current frame and open it directly in the Image Inspection tab.
7. Click **"Stop Camera"** when finished to release all hardware camera resources.

### 2. Image Upload & Analysis
1. Select **"Image Detection"** in the sidebar.
2. Drag and drop any photo (`.jpg`, `.png`, `.webp`, `.bmp`) or click **"Browse Files"**.
3. The engine instantly detects all faces, displays the bounding box table with exact pixel coordinates (`X`, `Y`, `Width`, `Height`), and highlights the face count.
4. Click **"Download Annotated Image"** to save the processed image with drawn boxes.

### 3. Frame-by-Frame Video Processing
1. Select **"Video Processing"** in the sidebar.
2. Upload any short video (`.mp4`, `.avi`, `.mov`, `.webm`).
3. An asynchronous worker thread processes each frame, tracking detection events and updating a progress bar.
4. When complete, play back the annotated video directly in the browser player or click **"Download Processed Video"**.

### 4. Real-Time Analytics
1. Select **"Analytics"** in the sidebar.
2. Review the **Inference Latency History** line chart showing execution time in milliseconds.
3. Review the **Workload Distribution** doughnut chart showing requests by source.

---

## 📡 REST API Reference

All API responses follow a structured, predictable JSON format.

### `GET /api/health`
Checks application health and model readiness.
```json
{
  "status": "healthy",
  "detector_ready": true,
  "detector_model": "Haar Cascade (haarcascade_frontalface_default.xml)",
  "service": "AI Face Detection Studio",
  "version": "1.0.0",
  "timestamp": 1729579200.0
}
```

### `POST /api/detect-image`
Detects faces in an uploaded image file (multipart/form-data) or JSON base64.
- **Request (Multipart)**: `image` (binary file), optional `scale_factor` (float), `min_neighbors` (int).
- **Request (JSON)**: `{ "image_base64": "data:image/jpeg;base64,...", "scale_factor": 1.1, "min_neighbors": 5 }`
- **Response**:
```json
{
  "success": true,
  "face_count": 2,
  "processing_time_ms": 32.4,
  "image_width": 1280,
  "image_height": 720,
  "faces": [
    { "x": 340, "y": 180, "width": 150, "height": 150 },
    { "x": 620, "y": 210, "width": 140, "height": 140 }
  ],
  "download_url": "/api/download/detected_abc123.jpg",
  "message": "Detected 2 faces successfully."
}
```

### `POST /api/detect-frame`
High-speed endpoint tailored for live webcam frame streaming.
- **Request**:
```json
{
  "frame": "data:image/jpeg;base64,...",
  "scale_factor": 1.1,
  "min_neighbors": 5
}
```
- **Response**:
```json
{
  "success": true,
  "face_count": 1,
  "processing_time_ms": 18.2,
  "faces": [
    { "x": 210, "y": 95, "width": 115, "height": 115 }
  ]
}
```

### `POST /api/detect-video`
Uploads a video file and initiates a background processing job.
- **Request**: `video` (multipart video file).
- **Response**:
```json
{
  "success": true,
  "job_id": "c1f7a0...",
  "message": "Video accepted and processing started."
}
```

### `GET /api/video-status/<job_id>`
Polls the progress of an active video processing job.
```json
{
  "success": true,
  "job_id": "c1f7a0...",
  "status": "processing",
  "progress": 65.5,
  "current_frame": 131,
  "total_frames": 200,
  "total_detection_events": 240,
  "download_url": null
}
```

### `GET /api/stats`
Returns aggregated backend performance metrics.

---

## 🧪 Automated Testing

The project includes an automated test suite verifying all layers:

To run all tests:
```powershell
.\.venv\Scripts\pytest -v
```

Expected result:
```
tests/test_app.py::test_detector_initialization PASSED
tests/test_app.py::test_detect_blank_image PASSED
tests/test_app.py::test_detect_grayscale_input PASSED
tests/test_app.py::test_detect_invalid_inputs PASSED
tests/test_app.py::test_image_bytes_codec PASSED
tests/test_app.py::test_decode_corrupted_image_bytes PASSED
tests/test_app.py::test_dashboard_route PASSED
tests/test_app.py::test_health_endpoint PASSED
tests/test_app.py::test_stats_endpoint PASSED
tests/test_app.py::test_detect_image_file_upload PASSED
tests/test_app.py::test_detect_image_base64_json PASSED
tests/test_app.py::test_detect_image_unsupported_file PASSED
tests/test_app.py::test_detect_image_missing_file PASSED
tests/test_app.py::test_detect_frame_endpoint PASSED
tests/test_app.py::test_detect_frame_missing_frame PASSED
tests/test_app.py::test_detect_video_unsupported_format PASSED
tests/test_app.py::test_video_status_nonexistent_job PASSED
tests/test_app.py::test_download_security_path_traversal PASSED
tests/test_app.py::test_full_video_processing_workflow PASSED

============================= 19 passed in 41.46s =============================
```

---

## 🔧 Troubleshooting Guide

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **"Cannot access camera / Permission Denied"** | Browser blocked camera access. | Click the camera icon or padlock in your browser's address bar and set Camera to **Allow**, then refresh the page. |
| **"Failed to load OpenCV Cascade Classifier"** | XML model file missing. | Ensure `detector/models/haarcascade_frontalface_default.xml` exists. The application will automatically fall back to OpenCV's bundled assets. |
| **"No faces detected in a clear photo"** | Face is angled, poorly lit, or too small. | Lower the `minNeighbors` slider to `3` or `4` and set `scaleFactor` to `1.05`. |
| **"Script execution disabled error" in PowerShell** | Windows ExecutionPolicy restriction. | Run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in your PowerShell terminal before activating `.venv`. |
| **"Port 5000 in use"** | Another process is occupying port 5000. | Start on an alternate port: `$env:PORT="5050"; python app.py` |

---

## ⚖️ Haar Cascade Limitations & Future Upgrades

### Limitations of Classical Haar Cascades
- **Pose Sensitivity**: Standard frontal cascades perform poorly when faces are tilted beyond ~30 degrees or turned in profile.
- **Lighting Sensitivity**: Uneven lighting, backlighting, or severe shadows can hide key Haar gradients.
- **Occlusions**: Sunglasses, surgical masks, or heavy scarves can prevent cascade triggers.

### Recommended Future Upgrades
1. **OpenCV DNN Face Detector**: Integrate OpenCV's Caffe or ONNX SSD ResNet-10 model for high resistance to tilted faces and partial occlusions.
2. **YuNet (libfacedetection)**: Ultra-fast lightweight deep learning model designed for mobile and embedded devices.
3. **MediaPipe Face Mesh**: Detect 468 3D facial landmarks for head pose estimation and expression tracking.

---

## 🐙 GitHub Setup & Push Guide

To upload this project to your GitHub account:

1. **Initialize Git Repository**:
   ```powershell
   git init
   git add .
   git commit -m "Initial commit: AI Face Detection Web Application"
   ```

2. **Create a New Repository on GitHub**:
   - Go to [github.com/new](https://github.com/new).
   - Enter repository name: `ai-face-detection`.
   - Choose **Public** or **Private**.
   - Do **NOT** initialize with a README (you already have one).

3. **Link and Push**:
   ```powershell
   git branch -M main
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/ai-face-detection.git
   git push -u origin main
   ```

---

## 📜 License & Credits

Built for educational purposes as an applied computer vision and full-stack software engineering project.
- Haar Cascade Model by Paul Viola & Michael Jones (OpenCV distribution).
- Icons provided by [Bootstrap Icons](https://icons.getbootstrap.com/).
- Chart visualizations by [Chart.js](https://www.chartjs.org/).
