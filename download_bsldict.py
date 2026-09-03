"""
BSLDict Root Downloader for Avatar-Generator Project
Run directly from your project root: python download_bsldict.py
"""

import sys
from pathlib import Path

# Add bsldict/bsldict directory to sys.path so we can import or run directly
script_dir = Path(__file__).resolve().parent
bsldict_inner_dir = script_dir / "bsldict" / "bsldict"

if not (bsldict_inner_dir / "download_videos_windows.py").exists():
    print(f"[Error] Could not find {bsldict_inner_dir / 'download_videos_windows.py'}")
    sys.exit(1)

sys.path.insert(0, str(bsldict_inner_dir))
# pyrefly: ignore [missing-import]
import download_videos_windows

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download all ~14,000 BSLDict videos directly from project root.")
    parser.add_argument(
        "--data_path",
        type=Path,
        default=bsldict_inner_dir / "videos_original",
        help="Directory to save downloaded videos (default: bsldict/bsldict/videos_original)",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Optional: Limit number of videos to download (leave blank for all ~14,000)",
    )
    args = parser.parse_args()
    print("=" * 60)
    print(" BSLDict 14,000 Video Downloader Started")
    print(" Destination:", args.data_path.resolve())
    print("=" * 60)
    download_videos_windows.main(args.data_path, args.max_videos)
