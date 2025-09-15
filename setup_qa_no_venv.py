#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REQ_FILE = Path(__file__).with_name("requirements_qa.txt")


def ensure_json(path: Path, default_obj):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default_obj, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/wan22_system")
    args = parser.parse_args()

    root = Path(args.root)
    logs_dir = root / "auto_state" / "logs_improved"
    state_dir = root / "auto_state"

    logs_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    # Install dependencies into the CURRENT Python (no venv)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"]) 
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", str(REQ_FILE)])

    # Ensure state jsons
    ensure_json(root / "knowledge.json", {"history": []})
    ensure_json(root / "review_queue.json", {"pending": []})
    ensure_json(root / "manual_ratings.json", {})
    ensure_json(root / "bandit_state.json", {"combo_stats": {}, "t": 0, "banned_combos": []})

    print("Готово ✅ Залежності встановлено і стейт-файли підготовлено.")
    print(f"ROOT:       {root}")
    print(f"STATE DIR:  {state_dir}")
    print(f"LOGS DIR:   {logs_dir}")


if __name__ == "__main__":
    main()
