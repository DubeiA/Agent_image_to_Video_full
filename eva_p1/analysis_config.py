# Copied from eva_p1_config_gpt.py
class AnalysisConfig:
    """Configuration class for video analysis parameters."""
    
    # Basic parameters
    frame_skip: int = 5
    max_frames: int = 300
    face_confidence_threshold: float = 0.7

    # Enhanced detection parameters
    anatomical_threshold: float = 0.6
    face_quality_threshold: float = 0.8
    artifact_threshold: float = 0.5

    # Analysis modes
    enable_temporal_analysis: bool = True
    enable_frequency_analysis: bool = True
    enable_optical_flow: bool = True
    enable_face_landmarks: bool = True
    enable_eye_blink_analysis: bool = True

    # Performance parameters
    batch_size: int = 32
    num_workers: int = 4
    gpu_enabled: bool = True

    # Logging parameters
    log_level: str = "INFO"
    save_detailed_logs: bool = True
    log_training_process: bool = True

    # Output parameters
    output_format: str = "json"
    save_frames: bool = False
    create_summary: bool = True
    
    def __post_init__(self):
        """Валідація параметрів після ініціалізації."""
        if self.frame_skip < 1:
            self.frame_skip = 1
        if self.max_frames < 10:
            self.max_frames = 10
        if not 0.0 <= self.face_confidence_threshold <= 1.0:
            self.face_confidence_threshold = 0.7


# --- Global option sets used by bandit / workflow param generation ---
# Kept conservative to balance stability and speed.
FPS_OPTIONS = [16, 20, 24, 25, 30]
SECONDS_OPTIONS = [5.0, 6.0, 7.0, 8.0, 10.0]
CFG_SCALES = [6.0, 7.0, 7.5, 8.0, 9.0]
STEPS_OPTIONS = [20, 25, 30, 35, 40]
# Common 16:9 resolutions (width, height)
RESOLUTION_OPTIONS = [
    (768, 432),
    (960, 540),
    (1024, 576),
    (1280, 720),
]
