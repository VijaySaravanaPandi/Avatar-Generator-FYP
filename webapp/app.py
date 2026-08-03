import os
import sys
import uuid
import subprocess
from flask import Flask, request, jsonify, render_template

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
import warnings
warnings.filterwarnings("ignore")

# Set up paths for the integration pipeline
INTEGRATION_DIR = r"d:\academics\HamNoSys_Group14\Integration-20260706T062240Z-3-001\Integration"
if INTEGRATION_DIR not in sys.path:
    sys.path.insert(0, INTEGRATION_DIR)

from integration_pipeline import process_video

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

SPREADSHEET_PATH = r"d:\academics\HamNoSys_Group14\Senior Code\HamNoSys2SiGML-master\HamNoSys2SiGML-master\Original\conversionSpreadSheet.txt"
HAM2SIGML_SCRIPT = r"d:\academics\HamNoSys_Group14\Senior Code\HamNoSys2SiGML-master\HamNoSys2SiGML-master\Original\HamNoSys2SiGML.py"

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
        
    filename = str(uuid.uuid4()) + "_" + file.filename
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(video_path)
    
    try:
        # Run integration pipeline
        result = process_video(video_path)
        hamnosys_tags = result.get('hamnosys', '')
        
        # Load mapping and convert to unicode
        mapping = load_reverse_mapping(SPREADSHEET_PATH)
        unicode_chars = []
        for tag in hamnosys_tags.split():
            if tag in mapping:
                unicode_chars.append(chr(int(mapping[tag], 16)))
        unicode_str = "".join(unicode_chars)
        
        # Convert to SiGML
        cmd = [sys.executable, HAM2SIGML_SCRIPT, unicode_str]
        process = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(HAM2SIGML_SCRIPT))
        
        if process.returncode != 0:
            return jsonify({"error": "Error converting to SiGML", "details": process.stderr}), 500
            
        sigml_output = process.stdout
        
        # Optionally clean up the uploaded file to save space
        if os.path.exists(video_path):
            os.remove(video_path)
        
        return jsonify({
            "hamnosys_tags": hamnosys_tags,
            "hamnosys_unicode": unicode_str,
            "sigml": sigml_output
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=False)
