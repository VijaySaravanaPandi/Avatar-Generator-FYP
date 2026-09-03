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

- **Python 3.10 – 3.12** (Python 3.12 recommended for MediaPipe compatibility)
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
git clone https://github.com/VijaySaravanaPandi/Avatar-Generator-FYP.git

# Step 3: Navigate into the project
cd Avatar-Generator-FYP

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
| `mediapipe` (≤0.10.14) | Hand/body landmark detection |
| `opencv-python` | Video processing |
| `numpy` | Numerical operations |
| `scikit-learn` | ML classifiers (SVM, etc.) |
| `joblib` | Model serialization |
| `flask` | Web dashboard server |

---

## ▶️ Usage

### Web App — Flask UI & 3D Avatar Player (Recommended)

From the project root:
```bash
py -3.12 webapp\app.py
```
*Or navigate into `webapp` first:*
```bash
cd webapp
py -3.12 app.py
```
*(On macOS / Linux: `python3 webapp/app.py`)*

Then open **http://localhost:5000** in your browser to access the full UI dashboard and 3D avatar.

---

### CLI — Run on a single video

From `Integration-20260706T062240Z-3-001/Integration`:
```bash
cd Integration-20260706T062240Z-3-001/Integration

# Default video (Prompt_1.mp4)
py -3.12 run_local.py

# Custom video
py -3.12 run_local.py path/to/your_video.mp4

# Custom output path
py -3.12 run_local.py input.mp4 -o output.mp4
```

Then open **http://localhost:5000** in your browser to access:
- **Video Upload & Analysis**: Upload sign language video recordings (.mp4, .avi, .mov)
- **Generated HamNoSys Phonetic Sequence**: Token tags and native HamNoSys font glyph rendering
- **HamKeyboard Symbol Breakdown**: Interactive categorized visual pills for handshapes, orientations, and locations
- **SiGML XML Inspector**: Inspect and copy raw Signing Gesture Markup Language XML code
- **3D Signing Avatar Player**: Live SiGML synthesis with JASigning WebGL Avatar, frame controls, and speed adjustments

### Avatar playback

After processing, the generated SiGML is sent directly to the JASigning WebGL avatar. The player waits briefly for the WebGL engine to become ready, so the avatar performs the generated signing movement instead of remaining in its neutral upright stance.

If the avatar does not start immediately:

1. Wait a few seconds after opening the page for the 3D engine to load.
2. Use **Replay From Start** or **Play Both (Sync)**.
3. Refresh the browser with `Ctrl+F5` after updating the project.
4. Confirm that the browser can access the JASigning WebGL resources; an internet connection is required for those external assets.

The motion shown by the avatar is determined by the HamNoSys/SiGML sequence predicted from the uploaded video. A low-confidence or incomplete prediction can therefore produce a simplified gesture.

### Accuracy and training truthfulness

The bundled BSLDict metadata labels clips with English glosses; it does **not**
provide HamNoSys or SiGML ground truth.  An English gloss cannot be converted to
an exact BSL articulation by rules alone.  Consequently, `training/hamnosys_annotations.csv`
is the required source for the production video-to-avatar model: every row must
contain a human-verified video, lexical sign ID, and HamNoSys sequence.  Train it
with:

```bash
py -3.12 training/train_annotated_sign_matcher.py --manifest training/hamnosys_annotations.csv
```

The trainer holds out one recording per repeated sign and saves a model only as
`release_ready` when held-out top-1 sign accuracy is at least 80%.  Include at
least two recordings per sign from different signers and reserve an additional
test set before claiming performance.  Until that labelled corpus exists, the
application will use its visual fallback and must not claim exact avatar
reproduction for arbitrary BSL uploads.

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
├── webapp/                             # 🌐 Flask web application & 3D Avatar UI
│   ├── app.py                          # Flask server
│   ├── static/
│   │   ├── css/styles.css              # Custom styled responsive dashboard UI
│   │   ├── js/app.js                   # Client logic, token parsing & SiGML dispatch
│   │   ├── HamNoSys.ttf                # Bundled HamNoSys font glyphs
│   │   └── fonts/                      # Font assets
│   └── templates/index.html            # Web app dashboard UI
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
- **Frontend**: Modern HTML5 / CSS3 / Vanilla JS, JASigning 3D WebGL Avatar
- **Supporting Tools**: React (Vite) for hex lookup tool

---

## 👥 Team — Group 14

Final Year Project — Sign Language Avatar Generator

---

## 📄 License

This project is part of an academic Final Year Project.
