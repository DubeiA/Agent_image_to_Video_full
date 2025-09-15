import os
import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Any

from eva_p3.detection_types import DetectionResult
from eva_p1.analysis_config import AnalysisConfig


class EnhancedVideoAgent:
    """Main Enhanced Video Agent v4 class with all improvements."""

    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
        # Lazy/simple processor stub; real processor may be provided elsewhere
        self.processor = VideoProcessor(self.config)
        self.training_system = None
        self.logger = self.processor.analyzer.logger

        # Initialize performance metrics
        self.performance_metrics = {
            'total_videos_processed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'deepfakes_detected': 0,
            'detection_accuracy': []
        }

    def analyze_video(self, video_path: str) -> DetectionResult:
        """Analyze video with enhanced capabilities."""

        try:
            # Update metrics
            self.performance_metrics['total_videos_processed'] += 1

            # Perform analysis
            result = self.processor.analyze_video(video_path)

            # Update performance metrics
            self.performance_metrics['total_processing_time'] += result.processing_time
            self.performance_metrics['average_processing_time'] = (
                self.performance_metrics['total_processing_time'] / 
                self.performance_metrics['total_videos_processed']
            )

            if result.is_deepfake:
                self.performance_metrics['deepfakes_detected'] += 1

            # Save result if configured
            if self.config.create_summary:
                self.save_analysis_result(result)

            return result

        except Exception as e:
            self.logger.main_logger.error(f"Error in video analysis: {str(e)}")
            raise

    def train_models(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train all detection models."""

        self.logger.main_logger.info("Starting model training...")
        return self.training_system.train_all_models(training_data)

    def batch_analyze(self, video_paths: List[str]) -> List[DetectionResult]:
        """Analyze multiple videos in batch."""

        self.logger.main_logger.info(f"Starting batch analysis of {len(video_paths)} videos")
        results = []

        for i, video_path in enumerate(video_paths):
            try:
                self.logger.main_logger.info(f"Processing video {i+1}/{len(video_paths)}: {video_path}")
                result = self.analyze_video(video_path)
                results.append(result)
            except Exception as e:
                self.logger.main_logger.error(f"Error processing {video_path}: {str(e)}")
                # Create error result
                error_result = DetectionResult(
                    is_deepfake=False,
                    confidence=0.0,
                    processing_time=0.0,
                    anatomical_score=0.0,
                    face_quality_score=0.0,
                    artifact_score=0.0,
                    temporal_consistency=0.0,
                    frame_scores=[],
                    face_landmarks_consistency=0.0,
                    eye_blink_naturalness=0.0,
                    micro_expression_analysis=0.0,
                    frames_analyzed=0,
                    faces_detected=0,
                    analysis_methods=['error'],
                    detected_errors=[{'batch_error': str(e)}],
                    confidence_breakdown={'error': 1.0},
                    video_path=video_path,
                    analysis_timestamp=datetime.datetime.now().isoformat(),
                    model_version="4.0-improved"
                )
                results.append(error_result)

        self.logger.main_logger.info(f"Batch analysis completed. Processed {len(results)} videos")
        return results

    def save_analysis_result(self, result: DetectionResult):
        """Save analysis result to file."""

        try:
            os.makedirs('results', exist_ok=True)

            # Generate filename from video path and timestamp
            video_name = Path(result.video_path).stem
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_{video_name}_{timestamp}.json"
            filepath = os.path.join('results', filename)

            # Convert result to dictionary
            result_dict = asdict(result)

            # Save as JSON
            with open(filepath, 'w') as f:
                json.dump(result_dict, f, indent=2)

            self.logger.main_logger.info(f"Analysis result saved: {filepath}")

        except Exception as e:
            self.logger.main_logger.error(f"Error saving analysis result: {str(e)}")

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""

        report = {
            'performance_metrics': self.performance_metrics,
            'configuration': asdict(self.config),
            'system_info': {
                'version': '4.0-improved',
                'timestamp': datetime.datetime.now().isoformat(),
                'device': str(self.processor.analyzer.device)
            }
        }

        # Save performance report
        try:
            with open('logs/performance_report.json', 'w') as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            self.logger.main_logger.error(f"Error saving performance report: {str(e)}")

        return report

    def cleanup_resources(self):
        """Cleanup resources and save logs."""

        try:
            # Save all logs
            self.logger.save_logs_to_json()

            # Create final summary report
            self.logger.create_summary_report()

            # Generate performance report
            self.generate_performance_report()

            self.logger.main_logger.info("Enhanced Video Agent v4 shutdown completed")

        except Exception as e:
            self.logger.main_logger.error(f"Error during cleanup: {str(e)}")


class VideoProcessor:
    """Lightweight processor stub providing analyze_video interface expected by merged agent.

    Returns an object with fields used downstream (overall_deepfake_score, anatomy/face/artifact/temporal scores).
    """

    def __init__(self, config: AnalysisConfig, logger=None):
        self.config = config
        self.analyzer = logger  # keep attr name used by some loggers

    def analyze_video(self, video_path: str):
        class _Result:
            # Lower penalty => higher quality. Keep neutral defaults.
            overall_deepfake_score = 0.5
            anatomical_score = 0.0
            face_quality_score = 0.0
            artifact_score = 0.0
            temporal_consistency = 0.0
            frequency_artifact_score = 0.0
            frames_analyzed = 0
            faces_detected = 0
            analysis_methods = []
            detected_errors = []
        return _Result()
