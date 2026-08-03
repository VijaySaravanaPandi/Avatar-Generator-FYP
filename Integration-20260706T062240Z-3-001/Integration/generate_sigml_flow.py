import os
import sys
import subprocess

# Set up paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

SPREADSHEET_PATH = r"d:\academics\HamNoSys_Group14\Senior Code\HamNoSys2SiGML-master\HamNoSys2SiGML-master\Original\conversionSpreadSheet.txt"
HAM2SIGML_SCRIPT = r"d:\academics\HamNoSys_Group14\Senior Code\HamNoSys2SiGML-master\HamNoSys2SiGML-master\Original\HamNoSys2SiGML.py"

from integration_pipeline import process_video

def load_reverse_mapping(spreadsheet_path):
    mapping = {}
    with open(spreadsheet_path, "r", encoding="utf-8") as f:
        for line in f:
            if "," in line:
                parts = line.strip().split(",")
                tag = parts[0].strip()
                # The second part might contain extra text like "E054    (Also 00EE)"
                code = parts[1].strip().split()[0].strip()
                mapping[tag] = code
    return mapping

def main(video_path):
    print("Running integration pipeline...")
    result = process_video(video_path)
    hamnosys_tags = result['hamnosys']
    print(f"Obtained tags: {hamnosys_tags}")

    print("Loading conversion mapping...")
    mapping = load_reverse_mapping(SPREADSHEET_PATH)

    unicode_chars = []
    for tag in hamnosys_tags.split():
        if tag in mapping:
            # Create actual unicode characters (e.g. U+E001) rather than string literals
            unicode_chars.append(chr(int(mapping[tag], 16)))
        else:
            print(f"Warning: Tag {tag} not found in conversion spreadsheet.")

    unicode_str = "".join(unicode_chars)
    
    txt_output = os.path.join(_SCRIPT_DIR, "hamnosys.txt")
    print(f"Writing to {txt_output}...")
    with open(txt_output, "w", encoding="utf-8") as f:
        f.write(unicode_str)
        
    sigml_output = os.path.join(_SCRIPT_DIR, "output.sigml")
    print("Running HamNoSys2SiGML.py...")
    
    # We run it and capture output
    cmd = [sys.executable, HAM2SIGML_SCRIPT, unicode_str]
    process = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(HAM2SIGML_SCRIPT))

    
    if process.returncode == 0:
        with open(sigml_output, "w", encoding="utf-8") as f:
            f.write(process.stdout)
        print(f"Successfully generated {sigml_output}")
    else:
        print("Error running HamNoSys2SiGML.py")
        print(process.stderr)


if __name__ == "__main__":
    # Suppress C++ level warnings from MediaPipe / TensorFlow Lite
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["GLOG_minloglevel"] = "3"
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

    import warnings
    warnings.filterwarnings("ignore")

    video = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_SCRIPT_DIR, "hello_hi.mp4")
    main(video)
