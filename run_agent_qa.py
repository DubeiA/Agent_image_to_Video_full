#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QA wrapper to run enhanced_video_agent_v4_merged_full.py without modifying originals.
- Forces logs to /workspace/wan22_system/auto_state/logs_improved
- Enforces ban rule: after >=3 attempts and all last 3 scores < 0.55 -> ban immediately
"""

import os
import json
import argparse


# Pre-patch sklearn import to provide SVM symbol if missing (map to SVC)
try:
    import sklearn.svm as _sksvm  # type: ignore
    if not hasattr(_sksvm, 'SVM'):
        from sklearn.svm import SVC  # type: ignore
        setattr(_sksvm, 'SVM', SVC)
except Exception:
    pass

from qa.agent_namespace import agent_mod
from qa.patches import (
    patch_logging_to_auto_state,
    patch_bandit_ban_rule,
    patch_video_processor_init,
    patch_openrouter_quality_guard,
    patch_enrich_run_iteration_metrics,
)


from qa.patches import patch_logging_to_auto_state


from qa.patches import patch_bandit_ban_rule


from qa.patches import patch_video_processor_init


from qa.patches import patch_openrouter_quality_guard


from qa.patches import patch_enrich_run_iteration_metrics


from qa.cli import main


if __name__ == "__main__":
    main()


