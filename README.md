# 🤟 Avatar Generator — Sign Language to HamNoSys Pipeline

> **Final Year Project (Group 14)** — An end-to-end system that converts sign language videos into HamNoSys notation, then to SiGML for 3D avatar animation.

---

## 📌 What It Does

1. **Input**: A video of a person performing a sign language gesture
2. **Processing**: 10 ML modules analyze the video using MediaPipe landmarks
3. **Output**: HamNoSys notation → SiGML XML → 3D avatar animation

### Pipeline Modules

| # | Module | What it detects |
|---|--------|----------------|
| 1 | Handshape | Hand shape classification (flat, fist, bird, etc.) |
| 2 | Orientation | Extended finger direction + palm orientation |
| 3 | Upper Body Location | Shoulder/chest level positions |
| 4 | Head & Face Location | Head, forehead, chin, ear regions |
| 5 | Hand Location | Spatial position of hands |
| 6 | Finger Location | Finger-level positioning |
| 7 | Contact Type | Touch, brush, close contact detection |
| 8 | Arm & Space | Arm positioning in signing space |
| 9 | Movement 1 | Primary movement patterns |
| 10 | Movement 2 | Secondary/complex movements |

---

## ⚠️ Prerequisites

Before cloning, make sure you have:

- **Python 3.10+**
- **Git** ([Download](https://git-scm.com/downloads))
- **Git LFS** ([Download](https://git-lfs.com/)) — **required** for model files

### Install Git LFS

```bash
# Windows (if you have Git for Windows, LFS is often included)
git lfs install

# macOS
brew install git-lfs
git lfs install

# Ubuntu/Debian
sudo apt install git-lfs
git lfs install
```

> **Why Git LFS?** The trained classifier models (`.pkl` files) are large (~754 MB total). Git LFS stores them efficiently without bloating the repository.

---

## 🚀 Cloning the Repository

```bash
# Step 1: Make sure Git LFS is installed
git lfs install

# Step 2: Clone (LFS files download automatically)
git clone https://github.com/SaiSriSatvic2005/Avatar-Generator-FYP-.git

# Step 3: Navigate into the project
cd Avatar-Generator-FYP-

# Step 4: Verify LFS files were downloaded
git lfs ls-files
```

You should see output like:
```
* clf_finger_bird.pkl
* clf_finger_right.pkl
* clf_finger_signer.pkl
* clf_palm.pkl
* clf_view.pkl
* enc_finger_bird.pkl
* enc_finger_right.pkl
* enc_finger_signer.pkl
* enc_palm.pkl
* enc_view.pkl
```

> **⚠️ If you cloned WITHOUT Git LFS installed**, the `.pkl` files will just be pointer files (tiny text). Fix it with:
> ```bash
> git lfs install
> git lfs pull
> ```

---

## 📦 Installation

```bash
# Install Python dependencies
cd Integration-20260706T062240Z-3-001/Integration
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `mediapipe` (≥0.10.20) | Hand/body landmark detection |
| `opencv-python` | Video processing |
| `numpy` | Numerical operations |
| `scikit-learn` | ML classifiers (SVM, etc.) |
| `joblib` | Model serialization |

For the **web app**, additionally install:
```bash
pip install flask
```

---

## ▶️ Usage

### CLI — Run on a single video

```bash
cd Integration-20260706T062240Z-3-001/Integration

# Default video (Prompt_1.mp4)
python run_local.py

# Custom video
python run_local.py path/to/your_video.mp4

# Custom output path
python run_local.py input.mp4 -o output.mp4
```

### Web App — Flask UI

```bash
cd webapp
python app.py
```
Then open http://localhost:5000 in your browser, upload a video, and get:
- HamNoSys tags
- HamNoSys Unicode
- SiGML output for avatar rendering

---

## 📂 Project Structure

```
Avatar-Generator-FYP-/
│
├── Integration-20260706T062240Z-3-001/
│   └── Integration/                    # 🧠 Core ML pipeline
│       ├── integration_pipeline.py     # Main pipeline (combines all 10 modules)
│       ├── run_local.py                # CLI runner
│       ├── generate_sigml_flow.py      # HamNoSys → SiGML converter
│       ├── Handshape_Model.py          # Module 1: Handshape classification
│       ├── ori_model2.py               # Module 2: Orientation detection
│       ├── upper_body_locations_video.py # Module 3: Upper body location
│       ├── Head_and_face_location.py   # Module 4: Head & face location
│       ├── hand_location_video_P.py    # Module 5: Hand location
│       ├── fing_locations_d.py         # Module 6: Finger location
│       ├── contact_types_e.py          # Module 7: Contact type
│       ├── Arm_and_Space_positions.py  # Module 8: Arm & space position
│       ├── movement1_prava.py          # Module 9: Movement type 1
│       ├── Movement_2.py              # Module 10: Movement type 2
│       ├── clf_*.pkl                   # Trained classifiers (Git LFS)
│       ├── enc_*.pkl                   # Label encoders (Git LFS)
│       └── requirements.txt
│
├── webapp/                             # 🌐 Flask web application
│   ├── app.py                          # Flask server
│   ├── static/css/styles.css
│   ├── static/js/app.js
│   └── templates/index.html
│
├── Senior Code/                        # 📚 Reference: HamNoSys2SiGML converter
│   └── HamNoSys2SiGML-master/
│
├── UI/                                 # 🎨 HamNoSys hex lookup tool (React/Vite)
│   └── hamnosys_project/
│
├── References/                         # 📄 Senior batch reports & docs
│
├── .gitattributes                      # Git LFS tracking config
├── .gitignore
└── README.md
```

---

## 🔬 Model Files (Git LFS)

| File | Size | Description |
|------|------|-------------|
| `clf_palm.pkl` | ~538 MB | Palm orientation classifier |
| `clf_finger_signer.pkl` | ~110 MB | Finger signer classifier |
| `clf_finger_bird.pkl` | ~40 MB | Finger bird classifier |
| `clf_finger_right.pkl` | ~30 MB | Finger right classifier |
| `clf_view.pkl` | ~1.2 MB | View classifier |
| `enc_*.pkl` | < 1 KB each | Label encoders for each classifier |

---

## 🛠️ Tech Stack

- **Computer Vision**: MediaPipe, OpenCV
- **ML/Classification**: scikit-learn (SVM classifiers)
- **Notation System**: HamNoSys → SiGML
- **Web Framework**: Flask
- **Frontend**: HTML/CSS/JS, React (Vite) for hex lookup tool

---

## 👥 Team — Group 14

Final Year Project — Sign Language Avatar Generator

---

## 📄 License

This project is part of an academic Final Year Project.
