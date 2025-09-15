import os, sys, json, datetime
from typing import Dict, Any
import logging
from eva_p1.analysis_config import AnalysisConfig

class EnhancedLogger:
    """Comprehensive logging system for all processes."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.setup_loggers()
        self.training_logs = []
        self.analysis_logs = []

    def setup_loggers(self):
        """Setup multiple loggers for different purposes."""

        # Create logs directory
        os.makedirs("logs", exist_ok=True)

        # Main logger
        self.main_logger = logging.getLogger("enhanced_video_agent")
        self.main_logger.setLevel(getattr(logging, self.config.log_level))

        # File handlers
        main_handler = logging.FileHandler("logs/main.log")
        training_handler = logging.FileHandler("logs/training.log")
        analysis_handler = logging.FileHandler("logs/analysis.log")

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)

        # Formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )

        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )

        # Set formatters
        main_handler.setFormatter(detailed_formatter)
        training_handler.setFormatter(detailed_formatter)
        analysis_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)

        # Add handlers
        self.main_logger.addHandler(main_handler)
        self.main_logger.addHandler(console_handler)

        # Training logger
        self.training_logger = logging.getLogger("training")
        self.training_logger.setLevel(logging.DEBUG)
        self.training_logger.addHandler(training_handler)

        # Analysis logger
        self.analysis_logger = logging.getLogger("analysis")
        self.analysis_logger.setLevel(logging.DEBUG)
        self.analysis_logger.addHandler(analysis_handler)

    def log_training_step(self, epoch: int, step: int, loss: float, 
                         accuracy: float, additional_metrics: Dict[str, Any] = None):
        """Log detailed training step information."""

        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "accuracy": accuracy,
            "additional_metrics": additional_metrics or {}
        }

        self.training_logs.append(log_entry)

        # Log to file
        self.training_logger.info(
            f"Epoch {epoch}, Step {step}: Loss={loss:.6f}, Accuracy={accuracy:.4f}"
        )

        if additional_metrics:
            self.training_logger.debug(f"Additional metrics: {additional_metrics}")

    def log_analysis_result(self, result):
        """Log comprehensive analysis results."""

        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "video_path": result.video_path,
            "is_deepfake": result.is_deepfake,
            "confidence": result.confidence,
            "processing_time": result.processing_time,
            "anatomical_score": result.anatomical_score,
            "face_quality_score": result.face_quality_score,
            "artifact_score": result.artifact_score,
            "frames_analyzed": result.frames_analyzed,
            "faces_detected": result.faces_detected
        }

        self.analysis_logs.append(log_entry)

        self.analysis_logger.info(
            f"Analysis complete: {result.video_path} - "
            f"Deepfake: {result.is_deepfake}, Confidence: {result.confidence:.4f}"
        )

    def save_logs_to_json(self):
        """Save all logs to JSON files."""

        if self.config.save_detailed_logs:
            # Save training logs
            with open("logs/training_logs.json", "w") as f:
                json.dump(self.training_logs, f, indent=2)

            # Save analysis logs
            with open("logs/analysis_logs.json", "w") as f:
                json.dump(self.analysis_logs, f, indent=2)

            self.main_logger.info("Logs saved to JSON files")

    def create_summary_report(self):
        """Create a comprehensive summary report."""

        summary = {
            "generation_timestamp": datetime.datetime.now().isoformat(),
            "total_training_steps": len(self.training_logs),
            "total_analyses": len(self.analysis_logs),
            "training_summary": self._summarize_training(),
            "analysis_summary": self._summarize_analysis()
        }

        # Save as JSON
        with open("logs/summary_report.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save as human-readable text
        self._create_text_summary(summary)

        return summary

    def _summarize_training(self):
        """Summarize training logs."""
        if not self.training_logs:
            return {}

        losses = [log['loss'] for log in self.training_logs]
        accuracies = [log['accuracy'] for log in self.training_logs]

        return {
            "total_epochs": max([log['epoch'] for log in self.training_logs]) if self.training_logs else 0,
            "final_loss": losses[-1] if losses else 0,
            "final_accuracy": accuracies[-1] if accuracies else 0,
            "best_loss": min(losses) if losses else 0,
            "best_accuracy": max(accuracies) if accuracies else 0,
            "loss_improvement": (losses[0] - losses[-1]) if len(losses) > 1 else 0
        }

    def _summarize_analysis(self):
        """Summarize analysis logs."""
        if not self.analysis_logs:
            return {}

        deepfakes_detected = sum(1 for log in self.analysis_logs if log['is_deepfake'])
        total_analyses = len(self.analysis_logs)

        confidences = [log['confidence'] for log in self.analysis_logs]
        processing_times = [log['processing_time'] for log in self.analysis_logs]

        return {
            "total_videos_analyzed": total_analyses,
            "deepfakes_detected": deepfakes_detected,
            "detection_rate": deepfakes_detected / total_analyses if total_analyses > 0 else 0,
            "average_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "average_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0,
            "confidence_distribution": {
                "min": min(confidences) if confidences else 0,
                "max": max(confidences) if confidences else 0,
                "std": (sum((c - (sum(confidences)/len(confidences)))**2 for c in confidences)/len(confidences))**0.5 if confidences else 0
            }
        }

    def _create_text_summary(self, summary: Dict[str, Any]):
        """Create human-readable text summary."""

        text_summary = f"""
Enhanced Video Agent v4 - Analysis Summary Report
Generated: {summary['generation_timestamp']}
==================================================

TRAINING SUMMARY:
-----------------
Total Training Steps: {summary['training_summary'].get('total_epochs', 0)}
Final Loss: {summary['training_summary'].get('final_loss', 0):.6f}
Final Accuracy: {summary['training_summary'].get('final_accuracy', 0):.4f}
Best Loss: {summary['training_summary'].get('best_loss', 0):.6f}
Best Accuracy: {summary['training_summary'].get('best_accuracy', 0):.4f}
Loss Improvement: {summary['training_summary'].get('loss_improvement', 0):.6f}

ANALYSIS SUMMARY:
-----------------
Total Videos Analyzed: {summary['analysis_summary'].get('total_videos_analyzed', 0)}
Deepfakes Detected: {summary['analysis_summary'].get('deepfakes_detected', 0)}
Detection Rate: {summary['analysis_summary'].get('detection_rate', 0):.2%}
Average Confidence: {summary['analysis_summary'].get('average_confidence', 0):.4f}
Average Processing Time: {summary['analysis_summary'].get('average_processing_time', 0):.2f}s

CONFIDENCE STATISTICS:
----------------------
Min Confidence: {summary['analysis_summary'].get('confidence_distribution', {}).get('min', 0):.4f}
Max Confidence: {summary['analysis_summary'].get('confidence_distribution', {}).get('max', 0):.4f}
Std Deviation: {summary['analysis_summary'].get('confidence_distribution', {}).get('std', 0):.4f}
"""

        with open("logs/summary_report.txt", "w") as f:
            f.write(text_summary)
