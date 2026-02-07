"""Install helper for torch builds.

This script attempts to detect whether the host has CUDA 13.0 available and
installs an appropriate torch build:

- If CUDA 13.0 is detected -> installs torch using the PyTorch CUDA 13.0 index
  (pip install torch --extra-index-url https://download.pytorch.org/whl/cu130)
- Otherwise -> installs CPU-only torch (pip install torch)

Usage:
    python scripts/install_torch.py [--force-cpu | --force-cuda130] [--dry-run]

The script is conservative and prints the chosen command before executing it.
"""

from __future__ import annotations
import argparse
import subprocess
import sys
import shutil
import re
from typing import Optional


def run(cmd, dry_run=False):
    print("Running:", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd)


def detect_cuda_version() -> Optional[str]:
    """Try to detect CUDA version (returns e.g. '13.0' or None)."""
    # Try nvidia-smi
    nvs = shutil.which("nvidia-smi")
    if nvs:
        try:
            out = subprocess.check_output([nvs, "--query-gpu=driver_version,compute_cap", "--format=csv,noheader"], stderr=subprocess.STDOUT, text=True)
        except Exception:
            try:
                out = subprocess.check_output([nvs], stderr=subprocess.STDOUT, text=True)
            except Exception:
                out = ""
        # Look for common patterns
        m = re.search(r"CUDA\s*Version\s*:?\s*(\d+\.\d+)", out)
        if m:
            return m.group(1)
    # Try nvcc
    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            out = subprocess.check_output([nvcc, "--version"], stderr=subprocess.STDOUT, text=True)
            m = re.search(r"release\s*(\d+\.\d+)", out)
            if m:
                return m.group(1)
        except Exception:
            pass
    # As a last resort, check environment variable
    cuda_home = (sys.environ.get("CUDA_HOME") or sys.environ.get("CUDA_PATH"))
    if cuda_home:
        # Try to glean version from path
        m = re.search(r"(\d+\.\d+)", cuda_home)
        if m:
            return m.group(1)
    return None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU installation")
    parser.add_argument("--force-cuda130", action="store_true", help="Force CUDA 13.0 installation")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    args = parser.parse_args(argv)

    if args.force_cpu and args.force_cuda130:
        print("Cannot force both cpu and cuda130 at the same time.")
        return 2

    chosen = None

    if args.force_cpu:
        chosen = "cpu"
    elif args.force_cuda130:
        chosen = "cuda130"
    else:
        ver = detect_cuda_version()
        print("Detected CUDA version:", ver)
        if ver and ver.startswith("13"):
            chosen = "cuda130"
        else:
            chosen = "cpu"

    # Construct command for torch
    pip_cmd = [sys.executable, "-m", "pip", "install", "-U"]
    if chosen == "cpu":
        pip_cmd += ["torch"]
        note = "CPU-only wheel from PyPI"
    else:  # cuda130
        # Use the PyTorch CUDA 13.0 index
        pip_cmd += ["torch", "--extra-index-url", "https://download.pytorch.org/whl/cu130"]
        note = "CUDA 13.0 build from PyTorch index"

    print(f"Chosen installation: {chosen} ({note})")
    print("Command:", " ".join(pip_cmd))

    rc = run(pip_cmd, dry_run=args.dry_run)
    if rc != 0:
        print("Installation command failed with code", rc, "— you may need to run it manually or check driver install.")
        return rc

    print("Torch installation finished.")
    
    # Install spacy
    print("\n--- Installing spaCy ---")
    spacy_cmd = [sys.executable, "-m", "pip", "install", "-U"]
    if chosen == "cuda130":
        spacy_cmd += ["spacy[cuda130]"]
        spacy_note = "spaCy with CUDA 13.0 support"
    else:
        spacy_cmd += ["spacy"]
        spacy_note = "spaCy (CPU)"
    
    print(f"Installing: {spacy_note}")
    print("Command:", " ".join(spacy_cmd))
    
    rc = run(spacy_cmd, dry_run=args.dry_run)
    if rc != 0:
        print("spaCy installation failed with code", rc)
        return rc
    
    print("spaCy installation finished.")
    
    # Download spacy models
    print("\n--- Downloading spaCy models ---")
    if chosen == "cuda130":
        models = ["en_core_web_trf", "it_core_news_lg"]
    else:
        models = ["en_core_web_sm", "it_core_news_sm"]
    
    for model in models:
        print(f"\nDownloading {model}...")
        download_cmd = [sys.executable, "-m", "spacy", "download", model]
        print("Command:", " ".join(download_cmd))
        
        rc = run(download_cmd, dry_run=args.dry_run)
        if rc != 0:
            print(f"Warning: Download of {model} failed with code {rc}")
            # Continue with other models even if one fails
    
    print("\n=== All installations finished ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())