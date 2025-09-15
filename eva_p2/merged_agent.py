import os, time
from dataclasses import asdict
from eva_env_base import log
from eva_p1.analysis_config import AnalysisConfig
from eva_p1.agent_base import EnhancedVideoAgentV4
from eva_p3.logger import EnhancedLogger
from eva_p3.video_processor import VideoProcessor
try:
    from eva_p3.training import EnhancedTrainingSystem
except Exception:
    EnhancedTrainingSystem = None

# 1:1 copy from eva_p2_merged_agent.py
class EnhancedVideoAgentV4Merged(EnhancedVideoAgentV4):
    def __init__(self, api: str, base_workflow: str, state_dir: str = None, seconds: float = 5.0, openrouter_key: str = None,
                 use_enhanced_analysis: bool = True, train_improved: bool = False, logger=None,
                 reference_only: bool = False, reference_file: str | None = None):
        super().__init__(api=api, base_workflow=base_workflow, state_dir=state_dir, seconds=seconds, openrouter_key=openrouter_key,
                         reference_only=reference_only, reference_file=reference_file)

        self.use_enhanced_analysis = bool(use_enhanced_analysis)
        self.enhanced_logger = None
        self.video_processor = None
        self.training_system = None

        # Initialize improved logger & analyzer
        try:
            cfg = AnalysisConfig()  # defaults are fine
            # Use provided logger if any, else create default
            self.enhanced_logger = logger if logger is not None else EnhancedLogger(config=cfg)
            self.video_processor = VideoProcessor(config=cfg, logger=self.enhanced_logger)
            log.info("🧪 Enhanced analyzer & logger are ready")
        except Exception as e:
            log.warning(f"Enhanced analyzer init failed: {e}")

        # Optional: run improved training system once
        if train_improved:
            if EnhancedTrainingSystem is None:
                log.warning("Improved training system not available; skipping training phase")
            else:
                try:
                    self.training_system = EnhancedTrainingSystem(logger=self.enhanced_logger)
                    self.training_system.train_all_models(epochs=3)  # keep it short
                    log.info("🧪 Improved training system finished")
                except Exception as e:
                    log.warning(f"Improved training system failed: {e}")

    def run_iteration_v4(self, params):
        # Generate with the original pipeline first
        base_score, metrics, video_path, wf = super().run_iteration_v4(params)

        # Optionally enrich analysis with improved analyzer
        if self.use_enhanced_analysis and self.video_processor and video_path and os.path.exists(video_path):
            try:
                det = self.video_processor.analyze_video(video_path)
                # Convert to plain dict if possible
                det_dict = det.__dict__ if hasattr(det, "__dict__") else (asdict(det) if asdict else {})
                # Derive a "deep_quality" (higher is better) from improved detector (lower deepfake score => higher quality)
                deep_penalty = float(getattr(det, "overall_deepfake_score", 0.5))
                deep_quality = max(0.0, min(1.0, 1.0 - deep_penalty))

                # Blend original & improved (clamp to [0,1])
                blended = max(0.0, min(1.0, 0.6 * float(base_score) + 0.4 * deep_quality))

                # Augment metrics
                metrics = dict(metrics) if isinstance(metrics, dict) else {}
                metrics.update({
                    "overall_simple": float(base_score),
                    "deep_quality": float(deep_quality),
                    "anatomy_score": float(getattr(det, "anatomical_score", 0.0)),
                    "face_quality_score": float(getattr(det, "face_quality_score", 0.0)),
                    "artifact_score": float(getattr(det, "artifact_score", 0.0)),
                    "temporal_consistency": float(getattr(det, "temporal_consistency", 0.0)),
                    # Higher is worse for artifacts; ensure field is present
                    "frequency_artifact_score": float(getattr(det, "frequency_artifact_score", 0.0)),
                })
                # Keep the key that search_v4 expects
                metrics["overall"] = float(blended)

                # Log a compact summary into the main logger
                log.info(f"🔬 Enhanced analysis: deep_quality={deep_quality:.3f} -> blended={blended:.3f}")
                if self.enhanced_logger:
                    # Also store a JSON line
                    try:
                        payload = {
                            "video": os.path.basename(video_path),
                            "base_overall": float(base_score),
                            "deep_penalty": deep_penalty,
                            "deep_quality": deep_quality,
                            "blended_overall": float(blended),
                            "detector": det_dict,
                            "timestamp": time.time(),
                        }
                        self.enhanced_logger._write_jsonl(os.path.join(self.enhanced_logger.log_dir, "merged_analysis.jsonl"), payload)
                    except Exception as le:
                        log.warning(f"Failed to write merged analysis log: {le}")

                return float(blended), metrics, video_path, wf
            except Exception as e:
                log.warning(f"Enhanced analyzer failed: {e}")

        # Fallback to original metrics if enhanced failed/disabled
        return float(base_score), metrics, video_path, wf


# --- Patch CLI to instantiate the merged agent ---
