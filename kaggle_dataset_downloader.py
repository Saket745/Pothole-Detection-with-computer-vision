"""
Kaggle Dataset Downloader & Hub Integration
Allows automatic fetching of road damage and pothole datasets from Kaggle.
"""

import sys
from pathlib import Path

def download_pothole_dataset(dataset_name="chitholian/annotated-potholes-dataset"):
    try:
        import kagglehub
    except ImportError:
        print("[!] kagglehub is not installed. Please install it via: pip install kagglehub")
        sys.exit(1)

    print(f"[*] Downloading dataset '{dataset_name}' from Kaggle...")
    try:
        path = kagglehub.dataset_download(dataset_name)
        print(f"[+] Successfully downloaded dataset to: {path}")
        return path
    except Exception as e:
        print(f"[-] Failed to download dataset: {e}")
        print("\nTip: Ensure your Kaggle API key is configured at ~/.kaggle/kaggle.json")
        return None

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "chitholian/annotated-potholes-dataset"
    download_pothole_dataset(target)
