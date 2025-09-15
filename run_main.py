#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple launcher for two-stage T2I→I2V with sensible defaults.

Usage:
  python run_main.py --seconds 6 --iterations 2 --use-enhanced-analysis

Defaults (can be overridden by passing flags accepted by qa.cli):
  - --two-stage
  - --t2i2v-root /workspace/Agent_T2I2V
  - --image-workflow /workspace/Agent_T2I2V/workflows/flux_dev_full_text_to_image.json
  - --i2v-workflow /workspace/Agent_T2I2V/workflows/video_wan2_2_14B_i2v.json
  - --reference-file /workspace/Agent_T2I2V/auto_state/reference_params.json
  - --i2v-widths 960x540,1280x720,768x432
  - --randomize-sizes
  - --randomize-fps --fps-min 20 --fps-max 35
"""

import sys

from qa.cli import main as qa_main


def main():
    # User-facing short flags
    user_args = sys.argv[1:]

    # Base defaults for two-stage mode
    defaults = [
        "--two-stage",
        "--t2i2v-root", "/workspace/Agent_T2I2V",
        "--image-workflow", "/workspace/Agent_T2I2V/workflows/flux_dev_full_text_to_image.json",
        "--i2v-workflow", "/workspace/Agent_T2I2V/workflows/video_wan2_2_14B_i2v.json",
        "--reference-file", "/workspace/Agent_T2I2V/auto_state/reference_params.json",
        "--i2v-widths", "960x540,1280x720,768x432",
        "--randomize-sizes",
        "--randomize-fps", "--fps-min", "20", "--fps-max", "35",
    ]

    # Merge: put defaults first, then user args to allow overrides
    sys.argv = [sys.argv[0]] + defaults + user_args
    qa_main()


if __name__ == "__main__":
    main()


