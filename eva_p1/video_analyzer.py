# Copied from eva_p1_comfy_video_bandit.py
import os, math
from typing import Dict, Any, List
import numpy as np
import cv2
import statistics as stats
from eva_env_base import log

class VideoAnalyzer:
    """Enhanced video quality analyzer"""

    def __init__(self, sample_every: int = 2, max_frames: int = 80):
        self.sample_every = max(1, sample_every)
        self.max_frames = max_frames

    def _read_frames(self, path: str) -> List[np.ndarray]:
        """Read video frames for analysis"""
        cap = cv2.VideoCapture(path)
        frames, idx = [], 0

        while True:
            ret, frame = cap.read()
            if not ret: 
                break
            if idx % self.sample_every == 0:
                frames.append(frame)
                if len(frames) >= self.max_frames: 
                    break
            idx += 1

        cap.release()
        return frames

    @staticmethod
    def _blur(gray: np.ndarray) -> float:
        """Calculate blur metric"""
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _exposure(gray: np.ndarray, lo: int = 15, hi: int = 240) -> float:
        """Calculate exposure quality"""
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        good = hist[lo:hi].sum()
        total = hist.sum() + 1e-6
        return float(good / total)

    @staticmethod
    def _blockiness(gray: np.ndarray, block: int = 8) -> float:
        """Calculate compression artifacts"""
        h, w = gray.shape
        v_edges = gray[:, block::block] - gray[:, block-1:w-1:block]
        h_edges = gray[block::block, :] - gray[block-1:h-1:block, :]
        return float(np.mean(np.abs(v_edges)) + np.mean(np.abs(h_edges)))

    @staticmethod
    def _flicker(prev_gray: np.ndarray, gray: np.ndarray) -> float:
        """Calculate temporal flicker"""
        diff = cv2.absdiff(prev_gray, gray)
        return float(np.mean(diff))

    def analyze(self, path: str) -> Dict[str, float]:
        """Comprehensive video analysis"""
        if not os.path.exists(path):
            log.error(f"Video file not found: {path}")
            return {"overall": 0.0, "blur": 0.0, "exposure": 0.0, "flicker": 1.0, "blockiness": 1.0}

        frames = self._read_frames(path)
        if len(frames) < 2:
            log.warning(f"Insufficient frames in {path}")
            return {"overall": 0.0, "blur": 0.0, "exposure": 0.0, "flicker": 1.0, "blockiness": 1.0}

        blurs, expos, blocks, flicks = [], [], [], []
        prev_gray = None

        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            blurs.append(self._blur(gray))
            expos.append(self._exposure(gray))
            blocks.append(self._blockiness(gray))

            if prev_gray is not None:
                flicks.append(self._flicker(prev_gray, gray))
            prev_gray = gray

        # Розрахунок нормалізованих метрик
        blur_med = max(1e-6, stats.median(blurs))
        block_med = max(1e-6, stats.median(blocks))
        flick_med = max(1e-6, stats.median(flicks) if flicks else 1.0)

        blur_norm = min(1.0, math.log1p(blur_med) / 8.0)
        expos_norm = float(np.clip(stats.median(expos), 0.0, 1.0))
        block_norm = float(np.clip(1.0 - (block_med / (block_med + 25.0)), 0.0, 1.0))
        flick_norm = float(np.clip(1.0 - (flick_med / (flick_med + 8.0)), 0.0, 1.0))

        # Зважений загальний скор
        w_blur, w_expo, w_block, w_flick = 0.3, 0.2, 0.2, 0.3
        overall = (
            w_blur * blur_norm +
            w_expo * expos_norm +
            w_block * block_norm +
            w_flick * flick_norm
        )

        return {
            "overall": float(overall),
            "blur": float(blur_norm),
            "exposure": float(expos_norm),
            "blockiness": float(block_norm),
            "flicker": float(flick_norm),
        }

