import os
import sys
import uuid
import subprocess
import xml.etree.ElementTree as ET
from flask import Flask, request, jsonify, render_template, send_from_directory

# Suppress C++ / MediaPipe / TensorFlow Lite stderr noise completely
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["ABSL_LOG_LEVEL"] = "error"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
import warnings
warnings.filterwarnings("ignore")

try:
    _null_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_null_fd, 2)
    os.close(_null_fd)
    sys.stderr = sys.stdout
except Exception:
    pass

# Set up paths for the integration pipeline
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_BASE_DIR, ".."))

INTEGRATION_DIR = os.path.join(PROJECT_ROOT, "Integration-20260706T062240Z-3-001", "Integration")
if INTEGRATION_DIR not in sys.path:
    sys.path.insert(0, INTEGRATION_DIR)

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(_BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

SPREADSHEET_PATH = os.path.join(PROJECT_ROOT, "Senior Code", "HamNoSys2SiGML-master", "HamNoSys2SiGML-master", "Original", "conversionSpreadSheet.txt")
HAM2SIGML_SCRIPT = os.path.join(PROJECT_ROOT, "Senior Code", "HamNoSys2SiGML-master", "HamNoSys2SiGML-master", "Original", "HamNoSys2SiGML.py")

def load_reverse_mapping(spreadsheet_path):
    mapping = {}
    with open(spreadsheet_path, "r", encoding="utf-8") as f:
        for line in f:
            if "," in line:
                parts = line.strip().split(",")
                tag = parts[0].strip()
                code = parts[1].strip().split()[0].strip()
                mapping[tag] = code
    return mapping

def hamnosys_to_sigml(hamnosys_tags):
    """Map the model notation to the SiGML XML played by the browser avatar."""
    mapping = load_reverse_mapping(SPREADSHEET_PATH)
    tokens = hamnosys_tags.split()
    missing = [token for token in tokens if token not in mapping]
    if missing:
        # Omitting a token silently changes the movement the avatar performs.
        raise ValueError("HamNoSys tokens have no SiGML mapping: " + ", ".join(sorted(set(missing))))

    unicode_str = "".join(chr(int(mapping[token], 16)) for token in tokens)
    completed = subprocess.run(
        [sys.executable, HAM2SIGML_SCRIPT, unicode_str],
        capture_output=True, text=True, cwd=os.path.dirname(HAM2SIGML_SCRIPT), timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "HamNoSys-to-SiGML conversion failed")
    if "<sigml" not in completed.stdout:
        raise RuntimeError("HamNoSys-to-SiGML converter did not return SiGML XML")
    return unicode_str, completed.stdout


def hamnosys_sequence_to_sigml(hamnosys_sequence):
    """Convert action phases into one multi-sign SiGML document for CWASA."""
    phases = [phase.strip() for phase in (hamnosys_sequence or []) if phase and phase.strip()]
    if not phases:
        raise ValueError("No HamNoSys action phases were produced")
    converted = [hamnosys_to_sigml(phase) for phase in phases]
    if len(converted) == 1:
        return converted[0]

    root = ET.Element("sigml")
    for phase_index, (_, sigml_text) in enumerate(converted, start=1):
        document = ET.fromstring(sigml_text)
        sign = document.find("hns_sign")
        if sign is None:
            raise RuntimeError("HamNoSys-to-SiGML conversion did not contain an hns_sign element")
        sign.set("gloss", f"video-action-{phase_index}")
        root.append(sign)
    return "".join(unicode_text for unicode_text, _ in converted), ET.tostring(root, encoding="unicode")

def build_avatar_result(video_path, original_filename=None):
    """Run trained video inference, then produce the avatar-ready SiGML."""
    try:
        from integration_pipeline import process_video
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "The video model could not load. Install the project's compatible "
            "MediaPipe release (pip install 'mediapipe==0.10.14') and restart the app."
        ) from exc
    # The browser only needs the SiGML result.  Skipping the optional annotated
    # MP4 avoids a second full video pass and keeps uploads responsive.
    result = process_video(video_path, original_filename=original_filename, annotate=False)
    hamnosys_tags = result.get("hamnosys", "").strip()
    if not hamnosys_tags:
        raise RuntimeError("The video model did not produce HamNoSys notation")
    unicode_str, sigml_output = hamnosys_sequence_to_sigml(result.get("avatar_sequence") or [hamnosys_tags])
    return result, hamnosys_tags, unicode_str, sigml_output

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/samples/<path:filename>')
def get_sample_video(filename):
    return send_from_directory(INTEGRATION_DIR, filename)

@app.route('/process-sample/<path:filename>', methods=['POST'])
def process_sample_video(filename):
    video_path = os.path.join(INTEGRATION_DIR, filename)
    if not os.path.exists(video_path):
        return jsonify({"error": f"Sample video '{filename}' not found"}), 404

    try:
        result, hamnosys_tags, unicode_str, sigml_output = build_avatar_result(video_path)

        return jsonify({
            "hamnosys_tags": hamnosys_tags,
            "hamnosys_unicode": unicode_str,
            "sigml": sigml_output,
            "video_url": f"/samples/{filename}",
            "predicted_gloss": result.get("predicted_gloss"),
            "gloss_confidence": result.get("gloss_confidence", 0),
            "gloss_match_reason": result.get("gloss_match_reason"),
            "notation_source": result.get("notation_source", "visual_modules"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    original_filename = file.filename
    filename = str(uuid.uuid4()) + "_" + original_filename
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(video_path)

    try:
        # Run integration pipeline — pass original_filename so lexicon can match BSLDict glosses
        result, hamnosys_tags, unicode_str, sigml_output = build_avatar_result(
            video_path, original_filename=original_filename
        )
        
        # Optionally clean up the uploaded file to save space
        if os.path.exists(video_path):
            os.remove(video_path)
        
        return jsonify({
            "hamnosys_tags": hamnosys_tags,
            "hamnosys_unicode": unicode_str,
            "sigml": sigml_output,
            "predicted_gloss": result.get("predicted_gloss"),
            "gloss_confidence": result.get("gloss_confidence", 0),
            "gloss_match_reason": result.get("gloss_match_reason"),
            "notation_source": result.get("notation_source", "visual_modules"),
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=False)
