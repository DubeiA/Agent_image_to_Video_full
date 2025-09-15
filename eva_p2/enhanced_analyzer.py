from typing import Dict, Any
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from eva_env_base import MEDIAPIPE_AVAILABLE, mp, SCIPY_AVAILABLE, entropy
from eva_p2.enhanced_logger import EnhancedLogger
from eva_p1.analysis_config import AnalysisConfig

class EnhancedVideoAnalyzer:
    """Enhanced video analyzer with improved detection capabilities."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = EnhancedLogger(config)
        self.device = self._setup_device()
        self.models = {}
        self.scalers = {}
        self.initialize_detection_models()

    def _setup_device(self):
        """Setup computational device."""
        if self.config.gpu_enabled and torch.cuda.is_available():
            device = torch.device("cuda")
            self.logger.main_logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            self.logger.main_logger.info("Using CPU")
        return device

    def initialize_detection_models(self):
        """Initialize all detection models."""

        self.logger.main_logger.info("Initializing detection models...")

        # Face detection models
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=self.config.face_confidence_threshold
            )
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=5, min_detection_confidence=0.5
            )

        # OpenCV face detector as backup
        try:
            self.cv_face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except:
            self.logger.main_logger.warning("OpenCV face cascade not available")

        # Anatomical analysis model
        self.anatomical_analyzer = self._create_anatomical_analyzer()

        # Face quality analyzer
        self.face_quality_analyzer = self._create_face_quality_analyzer()

        # Artifact detector
        self.artifact_detector = self._create_artifact_detector()

        self.logger.main_logger.info("All detection models initialized successfully")

    def _create_anatomical_analyzer(self):
        """Create enhanced anatomical error detection model."""

        class AnatomicalAnalyzer(nn.Module):
            def __init__(self):
                super(AnatomicalAnalyzer, self).__init__()

                # Feature extraction layers
                self.feature_extractor = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.AdaptiveAvgPool2d((8, 8))
                )

                # Anatomical analysis layers
                self.anatomical_classifier = nn.Sequential(
                    nn.Linear(256 * 8 * 8, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                features = self.feature_extractor(x)
                features = features.view(features.size(0), -1)
                anatomical_score = self.anatomical_classifier(features)
                return anatomical_score

        model = AnatomicalAnalyzer().to(self.device)
        return model

    def _create_face_quality_analyzer(self):
        """Create face quality and realism detection model."""

        class FaceQualityAnalyzer(nn.Module):
            def __init__(self):
                super(FaceQualityAnalyzer, self).__init__()

                # Multi-scale feature extraction
                self.scale1 = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )

                self.scale2 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )

                self.scale3 = nn.Sequential(
                    nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((4, 4))
                )

                # Quality assessment layers
                self.quality_classifier = nn.Sequential(
                    nn.Linear(512 * 4 * 4, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Linear(128, 3),  # Quality, Realism, Overall
                    nn.Sigmoid()
                )

            def forward(self, x):
                x1 = self.scale1(x)
                x2 = self.scale2(x1)
                x3 = self.scale3(x2)

                features = x3.view(x3.size(0), -1)
                quality_scores = self.quality_classifier(features)

                return quality_scores

        model = FaceQualityAnalyzer().to(self.device)
        return model

    def _create_artifact_detector(self):
        """Create artifact and unnatural element detector."""

        class ArtifactDetector(nn.Module):
            def __init__(self):
                super(ArtifactDetector, self).__init__()

                # Frequency domain analysis
                self.freq_analyzer = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, padding=3),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )

                # Spatial artifact detection
                self.spatial_analyzer = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )

                # Edge artifact detection
                self.edge_analyzer = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )

                # Combined analysis
                self.artifact_classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d((8, 8)),
                    nn.Flatten(),
                    nn.Linear((128 + 128 + 64) * 8 * 8, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Linear(128, 4),  # Compression, Blur, Noise, Overall
                    nn.Sigmoid()
                )

            def forward(self, x):
                freq_features = self.freq_analyzer(x)
                spatial_features = self.spatial_analyzer(x)
                edge_features = self.edge_analyzer(x)

                # Resize all features to same spatial dimensions
                target_size = freq_features.shape[2:]
                spatial_features = F.interpolate(spatial_features, size=target_size, mode='bilinear')
                edge_features = F.interpolate(edge_features, size=target_size, mode='bilinear')

                # Concatenate features
                combined_features = torch.cat([freq_features, spatial_features, edge_features], dim=1)

                # Classify artifacts
                artifact_scores = self.artifact_classifier(combined_features)

                return artifact_scores

        model = ArtifactDetector().to(self.device)
        return model

    # The rest of methods (analyze_anatomical_errors, analyze_face_quality, detect_artifacts, etc.)
    # remain identical. To keep 1:1 behavior, they should be copied as-is if needed here.

    def analyze_anatomical_errors(self, face_region: np.ndarray) -> Dict[str, float]:
        """Analyze anatomical errors and deformations in face region."""

        try:
            face_tensor = self._preprocess_face_for_analysis(face_region)
            with torch.no_grad():
                anatomical_score = self.anatomical_analyzer(face_tensor)
            geometric_score = self._analyze_facial_geometry(face_region)
            proportion_score = self._analyze_facial_proportions(face_region)
            symmetry_score = self._analyze_facial_symmetry(face_region)
            results = {
                'overall_anatomical_score': float(anatomical_score.cpu()),
                'geometric_consistency': geometric_score,
                'proportion_accuracy': proportion_score,
                'facial_symmetry': symmetry_score,
                'anatomical_naturalness': (geometric_score + proportion_score + symmetry_score) / 3
            }
            return results
        except Exception as e:
            self.logger.main_logger.error(f"Error in anatomical analysis: {str(e)}")
            return {'overall_anatomical_score': 0.5, 'error': str(e)}

    def analyze_face_quality(self, face_region: np.ndarray) -> Dict[str, float]:
        """Analyze face quality and realism."""
        try:
            face_tensor = self._preprocess_face_for_analysis(face_region)
            with torch.no_grad():
                quality_scores = self.face_quality_analyzer(face_tensor)
            quality_values = quality_scores.cpu().numpy().flatten()
            sharpness_score = self._calculate_image_sharpness(face_region)
            lighting_score = self._analyze_lighting_consistency(face_region)
            texture_score = self._analyze_skin_texture(face_region)
            results = {
                'overall_quality': float(quality_values[0]) if len(quality_values) > 0 else 0.5,
                'realism_score': float(quality_values[1]) if len(quality_values) > 1 else 0.5,
                'technical_quality': float(quality_values[2]) if len(quality_values) > 2 else 0.5,
                'sharpness': sharpness_score,
                'lighting_consistency': lighting_score,
                'texture_naturalness': texture_score
            }
            return results
        except Exception as e:
            self.logger.main_logger.error(f"Error in face quality analysis: {str(e)}")
            return {'overall_quality': 0.5, 'error': str(e)}

    def detect_artifacts(self, frame: np.ndarray) -> Dict[str, float]:
        """Detect artifacts and unnatural elements."""
        try:
            frame_tensor = self._preprocess_frame_for_analysis(frame)
            with torch.no_grad():
                artifact_scores = self.artifact_detector(frame_tensor)
            artifact_values = artifact_scores.cpu().numpy().flatten()
            compression_artifacts = self._detect_compression_artifacts(frame)
            blur_artifacts = self._detect_blur_artifacts(frame)
            noise_artifacts = self._detect_noise_artifacts(frame)
            edge_artifacts = self._detect_edge_artifacts(frame)
            results = {
                'compression_artifacts': float(artifact_values[0]) if len(artifact_values) > 0 else compression_artifacts,
                'blur_artifacts': float(artifact_values[1]) if len(artifact_values) > 1 else blur_artifacts,
                'noise_artifacts': float(artifact_values[2]) if len(artifact_values) > 2 else noise_artifacts,
                'overall_artifacts': float(artifact_values[3]) if len(artifact_values) > 3 else 
                                   (compression_artifacts + blur_artifacts + noise_artifacts) / 3,
                'edge_artifacts': edge_artifacts,
                'unnatural_elements': self._detect_unnatural_elements(frame)
            }
            return results
        except Exception as e:
            self.logger.main_logger.error(f"Error in artifact detection: {str(e)}")
            return {'overall_artifacts': 0.5, 'error': str(e)}

    def _preprocess_face_for_analysis(self, face_region: np.ndarray) -> torch.Tensor:
        try:
            face_resized = cv2.resize(face_region, (224, 224))
            face_normalized = face_resized.astype(np.float32) / 255.0
            face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1).unsqueeze(0)
            face_tensor = face_tensor.to(self.device)
            return face_tensor
        except Exception as e:
            self.logger.main_logger.error(f"Error preprocessing face: {str(e)}")
            return torch.zeros(1, 3, 224, 224).to(self.device)

    def _preprocess_frame_for_analysis(self, frame: np.ndarray) -> torch.Tensor:
        try:
            frame_resized = cv2.resize(frame, (256, 256))
            frame_normalized = frame_resized.astype(np.float32) / 255.0
            frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0)
            frame_tensor = frame_tensor.to(self.device)
            return frame_tensor
        except Exception as e:
            self.logger.main_logger.error(f"Error preprocessing frame: {str(e)}")
            return torch.zeros(1, 3, 256, 256).to(self.device)

    def _analyze_facial_geometry(self, face_region: np.ndarray) -> float:
        try:
            if MEDIAPIPE_AVAILABLE:
                rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                results = self.mp_face_mesh.process(rgb_face)
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
                    h, w = face_region.shape[:2]
                    key_points = []
                    for idx in [10, 151, 9, 175, 263, 33]:
                        if idx < len(landmarks.landmark):
                            lm = landmarks.landmark[idx]
                            key_points.append([lm.x * w, lm.y * h])
                    if len(key_points) >= 6:
                        eye_distance = np.linalg.norm(np.array(key_points[4]) - np.array(key_points[5]))
                        face_width = np.linalg.norm(np.array(key_points[0]) - np.array(key_points[1]))
                        if face_width > 0:
                            ratio = eye_distance / face_width
                            golden_ratio_score = 1.0 - abs(ratio - 0.46) / 0.46
                            return max(0.0, min(1.0, golden_ratio_score))
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_face, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            h, w = gray_face.shape
            left_half = gray_face[:, :w//2]
            right_half = cv2.flip(gray_face[:, w//2:], 1)
            if left_half.shape != right_half.shape:
                min_width = min(left_half.shape[1], right_half.shape[1])
                left_half = left_half[:, :min_width]
                right_half = right_half[:, :min_width]
            symmetry_score = 1.0 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0
            geometric_score = (edge_density * 2 + symmetry_score) / 3
            return max(0.0, min(1.0, geometric_score))
        except Exception as e:
            self.logger.main_logger.error(f"Error in geometric analysis: {str(e)}")
            return 0.5

    def _analyze_facial_proportions(self, face_region: np.ndarray) -> float:
        try:
            h, w = face_region.shape[:2]
            aspect_ratio = w / h
            ideal_ratio = 0.75
            ratio_score = 1.0 - abs(aspect_ratio - ideal_ratio) / ideal_ratio
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            eye_score = 0.8
            if len(eyes) >= 2:
                eye_y_positions = [eye[1] + eye[3]//2 for eye in eyes[:2]]
                eye_alignment = 1.0 - abs(eye_y_positions[0] - eye_y_positions[1]) / h
                eye_score = max(0.0, min(1.0, eye_alignment))
            mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            mouths = mouth_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            mouth_score = 0.7
            if len(mouths) >= 1:
                mouth = mouths[0]
                mouth_y = mouth[1] + mouth[3]//2
                lower_third_score = 1.0 if mouth_y > h * 0.6 else 0.5
                mouth_score = lower_third_score
            total_score = (ratio_score + eye_score + mouth_score) / 3
            return max(0.0, min(1.0, total_score))
        except Exception as e:
            self.logger.main_logger.error(f"Error in proportion analysis: {str(e)}")
            return 0.5

    def _analyze_facial_symmetry(self, face_region: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            left_half = gray[:, :w//2]
            right_half_flipped = cv2.flip(gray[:, w//2:], 1)
            min_width = min(left_half.shape[1], right_half_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_half_flipped = right_half_flipped[:, :min_width]
            left_flat = left_half.flatten().astype(np.float32)
            right_flat = right_half_flipped.flatten().astype(np.float32)
            if len(left_flat) > 0 and len(right_flat) > 0:
                correlation = np.corrcoef(left_flat, right_flat)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.5
            else:
                correlation = 0.5
            ssim_score = self._calculate_structural_similarity(left_half, right_half_flipped)
            hist_left = cv2.calcHist([left_half], [0], None, [256], [0, 256])
            hist_right = cv2.calcHist([right_half_flipped], [0], None, [256], [0, 256])
            hist_correlation = cv2.compareHist(hist_left, hist_right, cv2.HISTCMP_CORREL)
            symmetry_score = (correlation + ssim_score + hist_correlation) / 3
            return max(0.0, min(1.0, symmetry_score))
        except Exception as e:
            self.logger.main_logger.error(f"Error in symmetry analysis: {str(e)}")
            return 0.5

    def _calculate_structural_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        try:
            if img1.shape != img2.shape:
                min_h = min(img1.shape[0], img2.shape[0])
                min_w = min(img1.shape[1], img2.shape[1])
                img1 = img1[:min_h, :min_w]
                img2 = img2[:min_h, :min_w]
            img1_f = img1.astype(np.float64)
            img2_f = img2.astype(np.float64)
            mu1 = np.mean(img1_f)
            mu2 = np.mean(img2_f)
            sigma1_sq = np.var(img1_f)
            sigma2_sq = np.var(img2_f)
            sigma12 = np.mean((img1_f - mu1) * (img2_f - mu2))
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
            return max(0.0, min(1.0, ssim))
        except Exception as e:
            self.logger.main_logger.error(f"Error calculating SSIM: {str(e)}")
            return 0.5

    def _calculate_image_sharpness(self, image: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            max_sharpness = 1000.0
            sharpness_score = min(laplacian_var / max_sharpness, 1.0)
            return sharpness_score
        except Exception as e:
            self.logger.main_logger.error(f"Error calculating sharpness: {str(e)}")
            return 0.5

    def _analyze_lighting_consistency(self, face_region: np.ndarray) -> float:
        try:
            lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            grad_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            lighting_std = np.std(l_channel)
            lighting_uniformity = 1.0 / (1.0 + lighting_std / 50.0)
            shadow_threshold = np.percentile(l_channel, 25)
            shadow_regions = l_channel < shadow_threshold
            shadow_consistency = 1.0 - np.std(l_channel[shadow_regions]) / 100.0 if np.any(shadow_regions) else 0.8
            lighting_score = (lighting_uniformity + shadow_consistency) / 2
            return max(0.0, min(1.0, lighting_score))
        except Exception as e:
            self.logger.main_logger.error(f"Error in lighting analysis: {str(e)}")
            return 0.5

    def _analyze_skin_texture(self, face_region: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            texture_score = self._calculate_lbp_texture_score(gray)
            skin_mask = self._create_skin_mask(hsv)
            color_consistency = self._analyze_skin_color_consistency(face_region, skin_mask)
            detail_score = self._analyze_skin_details(gray)
            total_texture_score = (texture_score + color_consistency + detail_score) / 3
            return max(0.0, min(1.0, total_texture_score))
        except Exception as e:
            self.logger.main_logger.error(f"Error in texture analysis: {str(e)}")
            return 0.5

    def _calculate_lbp_texture_score(self, gray_image: np.ndarray) -> float:
        try:
            h, w = gray_image.shape
            lbp_image = np.zeros((h, w), dtype=np.uint8)
            for i in range(1, h-1):
                for j in range(1, w-1):
                    center = gray_image[i, j]
                    binary_string = ''
                    neighbors = [
                        gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                        gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                        gray_image[i+1, j-1], gray_image[i, j-1]
                    ]
                    for neighbor in neighbors:
                        binary_string += '1' if neighbor >= center else '0'
                    lbp_image[i, j] = int(binary_string, 2)
            lbp_hist = cv2.calcHist([lbp_image], [0], None, [256], [0, 256])
            from eva_env_base import entropy
            texture_uniformity = 1.0 - entropy(lbp_hist.flatten() + 1e-10) / np.log(256)
            return texture_uniformity
        except Exception as e:
            self.logger.main_logger.error(f"Error calculating LBP: {str(e)}")
            return 0.5

    def _create_skin_mask(self, hsv_image: np.ndarray) -> np.ndarray:
        try:
            lower_skin = np.array([0, 20, 70])
            upper_skin = np.array([20, 255, 255])
            skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)
            kernel = np.ones((3, 3), np.uint8)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            return skin_mask
        except Exception as e:
            self.logger.main_logger.error(f"Error creating skin mask: {str(e)}")
            return np.ones(hsv_image.shape[:2], dtype=np.uint8) * 255

    def _analyze_skin_color_consistency(self, face_region: np.ndarray, skin_mask: np.ndarray) -> float:
        try:
            skin_pixels = face_region[skin_mask > 0]
            if len(skin_pixels) == 0:
                return 0.5
            mean_color = np.mean(skin_pixels, axis=0)
            std_color = np.std(skin_pixels, axis=0)
            consistency_score = 1.0 / (1.0 + np.mean(std_color) / 50.0)
            return max(0.0, min(1.0, consistency_score))
        except Exception as e:
            self.logger.main_logger.error(f"Error analyzing skin color consistency: {str(e)}")
            return 0.5

    def _analyze_skin_details(self, gray_image: np.ndarray) -> float:
        try:
            blurred = cv2.GaussianBlur(gray_image, (15, 15), 0)
            high_freq = cv2.absdiff(gray_image, blurred)
            detail_level = np.mean(high_freq)
            optimal_detail = 8.0
            detail_score = 1.0 - abs(detail_level - optimal_detail) / optimal_detail
            return max(0.0, min(1.0, detail_score))
        except Exception as e:
            self.logger.main_logger.error(f"Error analyzing skin details: {str(e)}")
            return 0.5

    def _detect_compression_artifacts(self, frame: np.ndarray) -> float:
        try:
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y_channel = yuv[:, :, 0]
            h, w = y_channel.shape
            block_artifacts = 0
            block_count = 0
            for i in range(0, h-8, 8):
                for j in range(0, w-8, 8):
                    block = y_channel[i:i+8, j:j+8]
                    dct_block = cv2.dct(block.astype(np.float32))
                    high_freq_energy = np.sum(np.abs(dct_block[4:, 4:]))
                    total_energy = np.sum(np.abs(dct_block)) + 1e-10
                    block_artifacts += high_freq_energy / total_energy
                    block_count += 1
            if block_count > 0:
                avg_artifact_level = block_artifacts / block_count
                return min(1.0, avg_artifact_level * 2)
            return 0.5
        except Exception as e:
            self.logger.main_logger.error(f"Error detecting compression artifacts: {str(e)}")
            return 0.5

    def _detect_blur_artifacts(self, frame: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            avg_gradient = np.mean(gradient_magnitude)
            fft = np.fft.fft2(gray)
            fft_shifted = np.fft.fftshift(fft)
            magnitude_spectrum = np.abs(fft_shifted)
            h, w = magnitude_spectrum.shape
            center_h, center_w = h//2, w//2
            high_freq_region = magnitude_spectrum[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4]
            high_freq_energy = np.mean(high_freq_region)
            blur_score = 1.0 - (laplacian_var / 1000 + avg_gradient / 100 + high_freq_energy / 1000) / 3
            return max(0.0, min(1.0, blur_score))
        except Exception as e:
            self.logger.main_logger.error(f"Error detecting blur artifacts: {str(e)}")
            return 0.5

    def _detect_noise_artifacts(self, frame: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = cv2.absdiff(gray, blurred)
            noise_level = np.std(noise)
            median_filtered = cv2.medianBlur(gray, 5)
            salt_pepper_noise = cv2.absdiff(gray, median_filtered)
            salt_pepper_level = np.mean(salt_pepper_noise > 10)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_noise = np.sqrt(sobel_x**2 + sobel_y**2)
            gradient_noise_level = np.std(gradient_noise)
            total_noise_score = (noise_level/50 + salt_pepper_level + gradient_noise_level/100) / 3
            return max(0.0, min(1.0, total_noise_score))
        except Exception as e:
            self.logger.main_logger.error(f"Error detecting noise artifacts: {str(e)}")
            return 0.5

    def _detect_edge_artifacts(self, frame: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            canny_edges = cv2.Canny(gray, 50, 150)
            sobel_edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
            edge_density = np.sum(canny_edges > 0) / (canny_edges.shape[0] * canny_edges.shape[1])
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharp_transitions = np.sum(np.abs(laplacian) > 100) / laplacian.size
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            ringing_response = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            ringing_level = np.std(ringing_response)
            edge_artifact_score = (sharp_transitions + ringing_level/100) / 2
            return max(0.0, min(1.0, edge_artifact_score))
        except Exception as e:
            self.logger.main_logger.error(f"Error detecting edge artifacts: {str(e)}")
            return 0.5

    def _detect_unnatural_elements(self, frame: np.ndarray) -> float:
        try:
            color_unnaturalness = self._analyze_color_distribution(frame)
            texture_unnaturalness = self._analyze_texture_consistency(frame)
            lighting_unnaturalness = 1.0 - self._analyze_lighting_consistency(frame)
            geometric_unnaturalness = self._detect_geometric_distortions(frame)
            total_unnaturalness = (color_unnaturalness + texture_unnaturalness + 
                                 lighting_unnaturalness + geometric_unnaturalness) / 4
            return max(0.0, min(1.0, total_unnaturalness))
        except Exception as e:
            self.logger.main_logger.error(f"Error detecting unnatural elements: {str(e)}")
            return 0.5

    def _analyze_color_distribution(self, frame: np.ndarray) -> float:
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            h_channel = hsv[:, :, 0]
            s_channel = hsv[:, :, 1]
            v_channel = hsv[:, :, 2]
            h_hist = cv2.calcHist([h_channel], [0], None, [180], [0, 180])
            from eva_env_base import entropy
            h_entropy = entropy(h_hist.flatten() + 1e-10)
            color_diversity = h_entropy / np.log(180)
            oversaturation_ratio = np.sum(s_channel > 200) / s_channel.size
            overbrightness_ratio = np.sum(v_channel > 240) / v_channel.size
            underbrightness_ratio = np.sum(v_channel < 15) / v_channel.size
            unnaturalness = (oversaturation_ratio + overbrightness_ratio + 
                           underbrightness_ratio + (1.0 - color_diversity)) / 4
            return max(0.0, min(1.0, unnaturalness))
        except Exception as e:
            self.logger.main_logger.error(f"Error analyzing color distribution: {str(e)}")
            return 0.5

    def _analyze_texture_consistency(self, frame: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            region_size = 64
            texture_variations = []
            for i in range(0, h-region_size, region_size//2):
                for j in range(0, w-region_size, region_size//2):
                    region = gray[i:i+region_size, j:j+region_size]
                    grad_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
                    grad_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
                    texture_measure = np.mean(np.sqrt(grad_x**2 + grad_y**2))
                    texture_variations.append(texture_measure)
            if len(texture_variations) > 1:
                texture_std = np.std(texture_variations)
                texture_mean = np.mean(texture_variations)
                consistency_score = texture_std / (texture_mean + 1e-10)
                return min(1.0, consistency_score / 2.0)
            return 0.5
        except Exception as e:
            self.logger.main_logger.error(f"Error analyzing texture consistency: {str(e)}")
            return 0.5

    def _detect_geometric_distortions(self, frame: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                   minLineLength=50, maxLineGap=10)
            if lines is not None and len(lines) > 5:
                angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                    angles.append(angle)
                angle_hist, _ = np.histogram(angles, bins=36, range=(-180, 180))
                from eva_env_base import entropy
                angle_entropy = entropy(angle_hist + 1e-10)
                geometric_unnaturalness = 1.0 - (angle_entropy / np.log(36))
                return max(0.0, min(1.0, geometric_unnaturalness))
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                curvature_values = []
                for contour in contours[:10]:
                    if len(contour) > 10:
                        epsilon = 0.02 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        curvature = len(contour) / max(len(approx), 1)
                        curvature_values.append(curvature)
                if curvature_values:
                    avg_curvature = np.mean(curvature_values)
                    curvature_unnaturalness = abs(avg_curvature - 5.0) / 10.0
                    return max(0.0, min(1.0, curvature_unnaturalness))
            return 0.3
        except Exception as e:
            self.logger.main_logger.error(f"Error detecting geometric distortions: {str(e)}")
            return 0.5
