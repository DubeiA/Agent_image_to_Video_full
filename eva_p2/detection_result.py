from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class DetectionResult:
    """Enhanced detection result with comprehensive analysis."""

    # Basic detection results
    is_deepfake: bool
    confidence: float
    processing_time: float

    # Enhanced analysis results
    anatomical_score: float
    face_quality_score: float
    artifact_score: float
    temporal_consistency: float
    # Frequency-domain artifact indicator (higher = more artifacts)
    frequency_artifact_score: float

    # Detailed metrics
    frame_scores: List[float]
    face_landmarks_consistency: float
    eye_blink_naturalness: float
    micro_expression_analysis: float

    # Technical details
    frames_analyzed: int
    faces_detected: int
    analysis_methods: List[str]

    # Error analysis
    detected_errors: List[Dict[str, Any]]
    confidence_breakdown: Dict[str, float]

    # Metadata
    video_path: str
    analysis_timestamp: str
    model_version: str
