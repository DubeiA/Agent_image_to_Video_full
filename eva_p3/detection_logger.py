import os
import cv2
import numpy as np
import datetime
import time
from typing import List, Dict, Any

from eva_p3.eva_p3_detection_types import DetectionResult
from eva_p3.eva_p3_logger import EnhancedLogger
from eva_p3_enhanced_analyzer import EnhancedVideoAnalyzer
from eva_p1.analysis_config import AnalysisConfig

MEDIAPIPE_AVAILABLE = cv2.getBuildInformation()['modules']['video'].get('MediaPipe') == 'ON'

class VideoProcessor:
    """Main video processing class with enhanced analysis capabilities."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.analyzer = EnhancedVideoAnalyzer(config)
        self.frame_cache = {}
        self.processing_stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'faces_detected': 0,
            'processing_time': 0.0
        }

    def analyze_video(self, video_path: str) -> DetectionResult:
        """Analyze video file for deepfake detection with enhanced capabilities."""

        start_time = time.time()
        self.analyzer.logger.main_logger.info(f"Starting analysis of: {video_path}")

        try:
            # Validate video file
            if not self._validate_video_file(video_path):
                raise ValueError(f"Invalid video file: {video_path}")

            # Extract frames
            frames = self._extract_frames(video_path)
            if not frames:
                raise ValueError("No frames could be extracted from video")

            # Process frames
            frame_results = []
            face_regions = []
            temporal_data = []

            self.analyzer.logger.main_logger.info(f"Processing {len(frames)} frames")

            for i, frame in enumerate(frames):
                frame_result = self._analyze_frame(frame, i)
                frame_results.append(frame_result)

                # Extract face regions for detailed analysis
                faces = self._detect_faces(frame)
                if faces:
                    face_regions.extend(faces)
                    self.processing_stats['faces_detected'] += len(faces)

                # Collect temporal data
                temporal_data.append({
                    'frame_index': i,
                    'timestamp': i / 30.0,  # Assume 30fps
                    'frame_score': frame_result.get('overall_score', 0.5)
                })

                # Log progress
                if (i + 1) % 50 == 0:
                    self.analyzer.logger.main_logger.info(f"Processed {i + 1}/{len(frames)} frames")

            # Enhanced analysis on collected data
            enhanced_results = self._perform_enhanced_analysis(
                frame_results, face_regions, temporal_data
            )

            # Create final detection result
            result = self._create_detection_result(
                video_path, frame_results, enhanced_results, start_time
            )

            # Log analysis result
            self.analyzer.logger.log_analysis_result(result)

            processing_time = time.time() - start_time
            self.analyzer.logger.main_logger.info(
                f"Analysis completed in {processing_time:.2f}s - "
                f"Result: {'Deepfake' if result.is_deepfake else 'Authentic'} "
                f"(Confidence: {result.confidence:.4f})"
            )

            return result

        except Exception as e:
            self.analyzer.logger.main_logger.error(f"Error analyzing video: {str(e)}")
            # Return default result with error information
            return DetectionResult(
                is_deepfake=False,
                confidence=0.0,
                processing_time=time.time() - start_time,
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
                detected_errors=[{'error': str(e)}],
                confidence_breakdown={'error': 1.0},
                video_path=video_path,
                analysis_timestamp=datetime.datetime.now().isoformat(),
                model_version="4.0-improved"
            )

    def _validate_video_file(self, video_path: str) -> bool:
        """Validate video file exists and is readable."""

        if not os.path.exists(video_path):
            self.analyzer.logger.main_logger.error(f"Video file not found: {video_path}")
            return False

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.analyzer.logger.main_logger.error(f"Cannot open video file: {video_path}")
                return False

            # Check if we can read at least one frame
            ret, frame = cap.read()
            cap.release()

            if not ret:
                self.analyzer.logger.main_logger.error(f"Cannot read frames from video: {video_path}")
                return False

            return True

        except Exception as e:
            self.analyzer.logger.main_logger.error(f"Error validating video file: {str(e)}")
            return False

    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from video with smart sampling."""

        frames = []
        try:
            cap = cv2.VideoCapture(video_path)

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            self.processing_stats['total_frames'] = total_frames

            # Calculate frame sampling
            if total_frames <= self.config.max_frames:
                frame_skip = max(1, self.config.frame_skip)
            else:
                frame_skip = max(1, total_frames // self.config.max_frames)

            frame_count = 0
            extracted_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    frames.append(frame.copy())
                    extracted_count += 1

                    if extracted_count >= self.config.max_frames:
                        break

                frame_count += 1

            cap.release()
            self.processing_stats['processed_frames'] = len(frames)

            self.analyzer.logger.main_logger.info(
                f"Extracted {len(frames)} frames from {total_frames} total frames "
                f"(skip={frame_skip}, fps={fps:.2f})"
            )

            return frames

        except Exception as e:
            self.analyzer.logger.main_logger.error(f"Error extracting frames: {str(e)}")
            return []

    def _analyze_frame(self, frame: np.ndarray, frame_index: int) -> Dict[str, Any]:
        """Analyze individual frame with enhanced detection."""

        try:
            # Basic quality checks
            frame_quality = self._assess_frame_quality(frame)

            # Detect artifacts
            artifacts = self.analyzer.detect_artifacts(frame)

            # Detect faces for detailed analysis
            faces = self._detect_faces(frame)

            face_analysis_results = []
            if faces:
                for face_region in faces:
                    # Enhanced face analysis
                    anatomical_result = self.analyzer.analyze_anatomical_errors(face_region)
                    quality_result = self.analyzer.analyze_face_quality(face_region)

                    face_analysis_results.append({
                        'anatomical_analysis': anatomical_result,
                        'quality_analysis': quality_result
                    })

            # Temporal analysis (if previous frames available)
            temporal_score = self._analyze_temporal_consistency(frame, frame_index)

            # Frequency domain analysis
            frequency_score = self._analyze_frequency_domain(frame)

            # Combine all analysis results
            overall_score = self._calculate_frame_score(
                frame_quality, artifacts, face_analysis_results, 
                temporal_score, frequency_score
            )

            return {
                'frame_index': frame_index,
                'overall_score': overall_score,
                'frame_quality': frame_quality,
                'artifacts': artifacts,
                'face_analysis': face_analysis_results,
                'temporal_score': temporal_score,
                'frequency_score': frequency_score,
                'faces_detected': len(faces)
            }

        except Exception as e:
            self.analyzer.logger.main_logger.error(f"Error analyzing frame {frame_index}: {str(e)}")
            return {
                'frame_index': frame_index,
                'overall_score': 0.5,
                'error': str(e)
            }

    def _detect_faces(self, frame: np.ndarray) -> List[np.ndarray]:
        """Detect faces in frame using multiple methods."""

        faces = []
        try:
            # Primary detection using MediaPipe
            if MEDIAPIPE_AVAILABLE:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.analyzer.mp_face_detection.process(rgb_frame)

                if results.detections:
                    h, w = frame.shape[:2]
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box

                        # Convert relative coordinates to absolute
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)

                        # Extract face region with padding
                        padding = 20
                        x1 = max(0, x - padding)
                        y1 = max(0, y - padding)
                        x2 = min(w, x + width + padding)
                        y2 = min(h, y + height + padding)

                        face_region = frame[y1:y2, x1:x2]
                        if face_region.size > 0:
                            faces.append(face_region)

            # Fallback detection using OpenCV
            if not faces and hasattr(self.analyzer, 'cv_face_cascade'):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_rects = self.analyzer.cv_face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )

                for (x, y, w, h) in face_rects:
                    # Extract face region with padding
                    padding = 20
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    y2 = min(frame.shape[0], y + h + padding)

                    face_region = frame[y1:y2, x1:x2]
                    if face_region.size > 0:
                        faces.append(face_region)

            return faces

        except Exception as e:
            self.analyzer.logger.main_logger.error(f"Error detecting faces: {str(e)}")
            return []

    def _assess_frame_quality(self, frame: np.ndarray) -> Dict[str, float]:
        """Assess basic frame quality metrics."""

        try:
            # Sharpness assessment
            sharpness = self.analyzer._calculate_image_sharpness(frame)

            # Brightness assessment
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray) / 255.0

            # Contrast assessment
            contrast = np.std(gray) / 255.0

            # Color balance assessment
            b, g, r = cv2.split(frame)
            color_balance = 1.0 - (np.std([np.mean(b), np.mean(g), np.mean(r)]) / 255.0)

            # Resolution quality (effective resolution based on detail)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            resolution_quality = min(1.0, np.var(laplacian) / 1000.0)

            return {
                'sharpness': sharpness,
                'brightness': brightness,
                'contrast': contrast,
                'color_balance': color_balance,
                'resolution_quality': resolution_quality,
                'overall_quality': (sharpness + brightness + contrast + 
                                  color_balance + resolution_quality) / 5
            }

        except Exception as e:
            self.analyzer.logger.main_logger.error(f"Error assessing frame quality: {str(e)}")
            return {'overall_quality': 0.5}

    def _analyze_temporal_consistency(self, frame: np.ndarray, frame_index: int) -> float:
        """Analyze temporal consistency with previous frames."""

        try:
            if frame_index == 0 or frame_index not in self.frame_cache:
                # Store frame for future comparisons
                self.frame_cache[frame_index] = frame.copy()
                return 1.0  # No previous frame to compare

            # Get previous frame
            prev_index = frame_index - 1
            if prev_index in self.frame_cache:
                prev_frame = self.frame_cache[prev_index]

                # Calculate optical flow
                if self.config.enable_optical_flow:
                    flow_consistency = self._calculate_optical_flow_consistency(prev_frame, frame)
                else:
                    flow_consistency = 0.8

                # Calculate frame difference consistency
                diff_consistency = self._calculate_frame_difference_consistency(prev_frame, frame)

                # Calculate feature consistency
                feature_consistency = self._calculate_feature_consistency(prev_frame, frame)

                # Combine temporal metrics
                temporal_score = (flow_consistency + diff_consistency + feature_consistency) / 3

                # Update frame cache (keep only recent frames)
                if len(self.frame_cache) > 10:
                    oldest_key = min(self.frame_cache.keys())
                    del self.frame_cache[oldest_key]

                self.frame_cache[frame_index] = frame.copy()

                return temporal_score

            return 0.8  # Default score when no previous frame

        except Exception as e:
            self.analyzer.logger.main_logger.error(f"Error in temporal analysis: {str(e)}")
            return 0.5

    def _calculate_optical_flow_consistency(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """Calculate optical flow consistency between frames."""

        try:
            # Convert to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # Try dense optical flow (Farneback) first
            try:
                fb_flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None,
                    0.5, 3, 21, 3, 5, 1.2, 0
                )  # (H,W,2)
                if fb_flow is not None and fb_flow.ndim == 3 and fb_flow.shape[2] == 2:
                    magnitude = np.sqrt(fb_flow[..., 0]**2 + fb_flow[..., 1]**2)
                    avg_magnitude = float(np.mean(magnitude))
                else:
                    avg_magnitude = None
            except Exception:
                avg_magnitude = None

            # If Farneback failed, fallback to LK with detected features
            if avg_magnitude is None:
                pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=500, qualityLevel=0.01, minDistance=7, blockSize=7)
                if pts is None or len(pts) == 0:
                    return 0.7
                # Ensure correct dtype, shape and contiguity (Nx1x2, float32)
                pts = np.ascontiguousarray(pts.reshape(-1, 1, 2).astype(np.float32))
                prev_gray_c = np.ascontiguousarray(prev_gray)
                curr_gray_c = np.ascontiguousarray(curr_gray)
                try:
                    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                        prev_gray_c, curr_gray_c, pts, None,
                        winSize=(21, 21), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
                    )
                except Exception as lk_err:
                    # Handle gracefully instead of crashing; log and fallback
                    self.analyzer.logger.main_logger.warning(
                        f"PyrLK flow failed with {lk_err}; falling back to default temporal score")
                    return 0.7
                if next_pts is None or status is None:
                    return 0.7
                status = status.reshape(-1)
                valid_prev = pts.reshape(-1, 2)[status == 1]
                valid_next = next_pts.reshape(-1, 2)[status == 1]
                if valid_prev.size == 0 or valid_next.size == 0:
                    return 0.7
                vec = valid_next - valid_prev
                magnitude = np.sqrt(vec[:, 0]**2 + vec[:, 1]**2)
                if magnitude.size == 0:
                    return 0.7
                avg_magnitude = float(np.mean(magnitude))

            # Score based on motion naturalness (not too static, not too chaotic)
            if avg_magnitude < 0.5:
                consistency_score = 0.9  # very little motion but stable
            elif avg_magnitude < 5.0:
                consistency_score = 1.0  # normal motion
            elif avg_magnitude < 20.0:
                consistency_score = 0.8  # high but acceptable
            else:
                consistency_score = 0.3  # excessive motion

            return consistency_score

        except Exception as e:
            self.analyzer.logger.main_logger.error(f"Error calculating optical flow: {str(e)}")
            return 0.5

    def _calculate_frame_difference_consistency(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """Calculate frame difference consistency."""

        try:
            # Ensure same dimensions
            if prev_frame.shape != curr_frame.shape:
                curr_frame = cv2.resize(curr_frame, (prev_frame.shape[1], prev_frame.shape[0]))

            # Calculate absolute difference
            diff = cv2.absdiff(prev_frame, curr_frame)

            # Convert to grayscale for analysis
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            # Calculate statistics
            mean_diff = np.mean(gray_diff)
            std_diff = np.std(gray_diff)

            # Natural videos have moderate, consistent changes
            if mean_diff < 5:
                # Very similar frames (possibly suspicious)
                consistency_score = 0.6
            elif mean_diff < 30:
                # Normal frame differences
                consistency_score = 1.0
            elif mean_diff < 80:
                # Larger changes but acceptable
                consistency_score = 0.8
            else:
                # Excessive changes (suspicious)
                consistency_score = 0.3

            # Adjust based on standard deviation
            if std_diff > mean_diff:
                consistency_score *= 0.8  # High variation is suspicious

            return max(0.0, min(1.0, consistency_score))

        except Exception as e:
            self.analyzer.logger.main_logger.error(f"Error calculating frame difference: {str(e)}")
            return 0.5

    def _calculate_feature_consistency(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """Calculate feature consistency between frames."""

        try:
            # Convert to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # Detect keypoints and compute descriptors
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(prev_gray, None)
            kp2, des2 = sift.detectAndCompute(curr_gray, None)

            if des1 is not None and des2 is not None and len(des1) > 10 and len(des2) > 10:
                # Match features
                matcher = cv2.BFMatcher()
                matches = matcher.knnMatch(des1, des2, k=2)

                # Apply ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)

                # Calculate consistency based on matches
                match_ratio = len(good_matches) / min(len(kp1), len(kp2))

                # Natural videos should have reasonable feature consistency
                if match_ratio > 0.3:
                    consistency_score = 1.0
                elif match_ratio > 0.1:
                    consistency_score = 0.8
                else:
                    consistency_score = 0.5

                return consistency_score

            return 0.7  # Default when feature detection fails

        except Exception as e:
            self.analyzer.logger.main_logger.error(f"Error calculating feature consistency: {str(e)}")
            return 0.5

    def _analyze_frequency_domain(self, frame: np.ndarray) -> float:
        """Analyze frame in frequency domain."""

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply FFT
            fft = np.fft.fft2(gray)
            fft_shifted = np.fft.fftshift(fft)
            magnitude_spectrum = np.abs(fft_shifted)

            # Analyze frequency distribution
            h, w = magnitude_spectrum.shape
            center_h, center_w = h//2, w//2

            # Low frequency content (center region)
            low_freq_region = magnitude_spectrum[center_h-h//8:center_h+h//8, 
                                              center_w-w//8:center_w+w//8]
            low_freq_energy = np.mean(low_freq_region)

            # High frequency content (outer regions)
            high_freq_mask = np.ones_like(magnitude_spectrum)
            high_freq_mask[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4] = 0
            high_freq_energy = np.mean(magnitude_spectrum * high_freq_mask)

            # Calculate frequency ratio
            freq_ratio = high_freq_energy / (low_freq_energy + 1e-10)

            # Natural images have balanced frequency content
            if 0.1 <= freq_ratio <= 0.8:
                frequency_score = 1.0
            elif 0.05 <= freq_ratio <= 1.2:
                frequency_score = 0.8
            else:
                frequency_score = 0.5

            return frequency_score

        except Exception as e:
            self.analyzer.logger.main_logger.error(f"Error in frequency domain analysis: {str(e)}")
            return 0.5

    def _calculate_frame_score(self, frame_quality: Dict[str, float], artifacts: Dict[str, float],
                             face_analysis: List[Dict[str, Any]], temporal_score: float,
                             frequency_score: float) -> float:
        """Calculate overall frame score."""

        try:
            # Frame quality component
            quality_score = frame_quality.get('overall_quality', 0.5)

            # Artifact component (lower artifacts = higher score)
            artifact_score = 1.0 - artifacts.get('overall_artifacts', 0.5)

            # Face analysis component
            face_score = 0.7  # Default when no faces
            if face_analysis:
                face_scores = []
                for face_result in face_analysis:
                    anatomical = face_result.get('anatomical_analysis', {})
                    quality = face_result.get('quality_analysis', {})

                    face_anatomical_score = anatomical.get('overall_anatomical_score', 0.5)
                    face_quality_score = quality.get('overall_quality', 0.5)

                    combined_face_score = (face_anatomical_score + face_quality_score) / 2
                    face_scores.append(combined_face_score)

                face_score = np.mean(face_scores)

            # Combine all components
            weights = {
                'quality': 0.2,
                'artifacts': 0.25,
                'faces': 0.3,
                'temporal': 0.15,
                'frequency': 0.1
            }

            overall_score = (
                weights['quality'] * quality_score +
                weights['artifacts'] * artifact_score +
                weights['faces'] * face_score +
                weights['temporal'] * temporal_score +
                weights['frequency'] * frequency_score
            )

            return max(0.0, min(1.0, overall_score))

        except Exception as e:
            self.analyzer.logger.main_logger.error(f"Error calculating frame score: {str(e)}")
            return 0.5


    def _perform_enhanced_analysis(self, frame_results: List[Dict[str, Any]], 
                                 face_regions: List[np.ndarray], 
                                 temporal_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform enhanced analysis on collected video data."""

        try:
            # Global temporal consistency analysis
            temporal_consistency = self._analyze_global_temporal_consistency(temporal_data)

            # Face consistency analysis across frames
            face_consistency = self._analyze_face_consistency(face_regions)

            # Eye blink analysis
            eye_blink_naturalness = self._analyze_eye_blink_patterns(face_regions)

            # Micro-expression analysis
            micro_expression_score = self._analyze_micro_expressions(face_regions)

            # Statistical analysis of frame scores
            statistical_analysis = self._perform_statistical_analysis(frame_results)

            # Artifact pattern analysis
            artifact_patterns = self._analyze_artifact_patterns(frame_results)

            return {
                'temporal_consistency': temporal_consistency,
                'face_consistency': face_consistency,
                'eye_blink_naturalness': eye_blink_naturalness,
                'micro_expression_score': micro_expression_score,
                'statistical_analysis': statistical_analysis,
                'artifact_patterns': artifact_patterns
            }

        except Exception as e:
            self.analyzer.logger.main_logger.error(f"Error in enhanced analysis: {str(e)}")
            return {
                'temporal_consistency': 0.5,
                'face_consistency': 0.5,
                'eye_blink_naturalness': 0.5,
                'micro_expression_score': 0.5,
                'statistical_analysis': {},
                'artifact_patterns': {}
            }

    def _analyze_global_temporal_consistency(self, temporal_data: List[Dict[str, Any]]) -> float:
        """Analyze global temporal consistency across all frames."""

        try:
            if len(temporal_data) < 2:
                return 1.0

            # Extract frame scores
            frame_scores = [data['frame_score'] for data in temporal_data]

            # Calculate temporal smoothness
            score_differences = []
            for i in range(1, len(frame_scores)):
                diff = abs(frame_scores[i] - frame_scores[i-1])
                score_differences.append(diff)

            # Smoothness metric (lower differences = higher consistency)
            avg_difference = np.mean(score_differences)
            smoothness_score = 1.0 / (1.0 + avg_difference * 5)  # Scale factor

            # Trend analysis
            # Natural videos should not have sudden quality jumps
            trend_consistency = self._analyze_score_trends(frame_scores)

            # Combine temporal metrics
            temporal_consistency = (smoothness_score + trend_consistency) / 2

            return max(0.0, min(1.0, temporal_consistency))

        except Exception as e:
            self.analyzer.logger.main_logger.error(f"Error in temporal consistency analysis: {str(e)}")
            return 0.5

    def _analyze_score_trends(self, frame_scores: List[float]) -> float:
        """Analyze trends in frame scores for unnaturalness."""

        try:
            if len(frame_scores) < 10:
                return 0.8

            # Calculate moving average
            window_size = min(10, len(frame_scores) // 3)
            moving_avg = []

            for i in range(len(frame_scores) - window_size + 1):
                avg = np.mean(frame_scores[i:i + window_size])
                moving_avg.append(avg)

            # Analyze trend stability
            trend_variance = np.var(moving_avg)

            # Natural videos should have relatively stable quality
            if trend_variance < 0.01:
                trend_score = 1.0
            elif trend_variance < 0.05:
                trend_score = 0.8
            else:
                trend_score = 0.5

            return trend_score

        except Exception as e:
            self.analyzer.logger.main_logger.error(f"Error analyzing score trends: {str(e)}")
            return 0.5

    def _analyze_face_consistency(self, face_regions: List[np.ndarray]) -> float:
        """Analyze consistency across detected face regions."""

        try:
            if len(face_regions) < 2:
                return 0.8  # Default when insufficient data

            # Sample faces for analysis (limit computation)
            sample_size = min(20, len(face_regions))
            sampled_faces = face_regions[:sample_size]

            consistency_scores = []

            # Compare consecutive faces
            for i in range(1, len(sampled_faces)):
                face1 = sampled_faces[i-1]
                face2 = sampled_faces[i]

                # Resize faces to same size for comparison
                target_size = (128, 128)
                face1_resized = cv2.resize(face1, target_size)
                face2_resized = cv2.resize(face2, target_size)

                # Calculate consistency metrics
                color_consistency = self._calculate_face_color_consistency(face1_resized, face2_resized)
                texture_consistency = self._calculate_face_texture_consistency(face1_resized, face2_resized)
                structure_consistency = self._calculate_face_structure_consistency(face1_resized, face2_resized)

                combined_consistency = (color_consistency + texture_consistency + structure_consistency) / 3
                consistency_scores.append(combined_consistency)

            # Calculate overall consistency
            overall_consistency = np.mean(consistency_scores) if consistency_scores else 0.5

            return max(0.0, min(1.0, overall_consistency))

        except Exception as e:
            self.analyzer.logger.main_logger.error(f"Error analyzing face consistency: {str(e)}")
            return 0.5

    def _calculate_face_color_consistency(self, face1: np.ndarray, face2: np.ndarray) -> float:
        """Calculate color consistency between two faces."""

        try:
            # Calculate color histograms
            hist1 = cv2.calcHist([face1], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([face2], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])

            # Compare histograms
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

            return max(0.0, min(1.0, correlation))

        except Exception as e:
            self.analyzer.logger.main_logger.error(f"Error calculating face color consistency: {str(e)}")
            return 0.5

    def _calculate_face_texture_consistency(self, face1: np.ndarray, face2: np.ndarray) -> float:
        """Calculate texture consistency between two faces."""

        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)

            # Calculate LBP features
            lbp1 = self.analyzer._calculate_lbp_texture_score(gray1)
            lbp2 = self.analyzer._calculate_lbp_texture_score(gray2)

            # Calculate texture similarity
            texture_similarity = 1.0 - abs(lbp1 - lbp2)

            return max(0.0, min(1.0, texture_similarity))

        except Exception as e:
            self.analyzer.logger.main_logger.error(f"Error calculating face texture consistency: {str(e)}")
            return 0.5

    def _calculate_face_structure_consistency(self, face1: np.ndarray, face2: np.ndarray) -> float:
        """Calculate structural consistency between two faces."""

        try:
            # Calculate structural similarity using gradients
            gray1 = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)

            # Calculate gradients
            grad1_x = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
            grad1_y = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
            grad2_x = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
            grad2_y = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)

            # Calculate gradient magnitudes
            mag1 = np.sqrt(grad1_x**2 + grad1_y**2)
            mag2 = np.sqrt(grad2_x**2 + grad2_y**2)

            # Calculate correlation between gradient magnitudes
            correlation = np.corrcoef(mag1.flatten(), mag2.flatten())[0, 1]

            if np.isnan(correlation):
                correlation = 0.5

            return max(0.0, min(1.0, correlation))

        except Exception as e:
            self.analyzer.logger.main_logger.error(f"Error calculating face structure consistency: {str(e)}")
            return 0.5

    def _analyze_eye_blink_patterns(self, face_regions: List[np.ndarray]) -> float:
        """Analyze eye blink patterns for naturalness."""

        try:
            if len(face_regions) < 10:
                return 0.7  # Default when insufficient data

            blink_indicators = []

            for face in face_regions[:30]:  # Analyze up to 30 faces
                try:
                    # Convert to grayscale
                    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                    # Detect eyes using Haar cascade
                    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                    eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5)

                    if len(eyes) >= 2:
                        # Analyze eye openness for first two detected eyes
                        eye_openness = []
                        for (ex, ey, ew, eh) in eyes[:2]:
                            eye_region = gray_face[ey:ey+eh, ex:ex+ew]

                            # Calculate eye aspect ratio (simplified)
                            if eye_region.size > 0:
                                # Use horizontal vs vertical intensity variation
                                h_variation = np.std(np.mean(eye_region, axis=0))
                                v_variation = np.std(np.mean(eye_region, axis=1))

                                openness = h_variation / (v_variation + 1e-10)
                                eye_openness.append(openness)

                        if eye_openness:
                            avg_openness = np.mean(eye_openness)
                            blink_indicators.append(avg_openness)

                except Exception:
                    continue

            if len(blink_indicators) < 5:
                return 0.6

            # Analyze blink pattern naturalness
            # Natural blink patterns have variation
            blink_variance = np.var(blink_indicators)

            # Too uniform = suspicious, too varied = also suspicious
            if 0.1 <= blink_variance <= 2.0:
                naturalness_score = 1.0
            elif 0.05 <= blink_variance <= 5.0:
                naturalness_score = 0.8
            else:
                naturalness_score = 0.4

            return naturalness_score

        except Exception as e:
            self.analyzer.logger.main_logger.error(f"Error analyzing eye blink patterns: {str(e)}")
            return 0.5

    def _analyze_micro_expressions(self, face_regions: List[np.ndarray]) -> float:
        """Analyze micro-expressions for naturalness."""

        try:
            if len(face_regions) < 5:
                return 0.7

            expression_variations = []

            # Sample faces for micro-expression analysis
            sample_indices = np.linspace(0, len(face_regions)-1, min(15, len(face_regions)), dtype=int)

            for i in range(1, len(sample_indices)):
                try:
                    face1 = face_regions[sample_indices[i-1]]
                    face2 = face_regions[sample_indices[i]]

                    # Resize for consistency
                    target_size = (96, 96)
                    face1_resized = cv2.resize(face1, target_size)
                    face2_resized = cv2.resize(face2, target_size)

                    # Focus on mouth and eye regions for micro-expressions
                    expression_diff = self._calculate_expression_difference(face1_resized, face2_resized)
                    expression_variations.append(expression_diff)

                except Exception:
                    continue

            if not expression_variations:
                return 0.6

            # Analyze variation pattern
            avg_variation = np.mean(expression_variations)
            variation_std = np.std(expression_variations)

            # Natural expressions have moderate, varied changes
            if 0.01 <= avg_variation <= 0.1 and variation_std > 0.005:
                naturalness_score = 1.0
            elif 0.005 <= avg_variation <= 0.2 and variation_std > 0.002:
                naturalness_score = 0.8
            else:
                naturalness_score = 0.5

            return naturalness_score

        except Exception as e:
            self.analyzer.logger.main_logger.error(f"Error analyzing micro-expressions: {str(e)}")
            return 0.5

    def _calculate_expression_difference(self, face1: np.ndarray, face2: np.ndarray) -> float:
        """Calculate difference in facial expressions between two faces."""

        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)

            # Focus on key facial regions (lower and upper face)
            h, w = gray1.shape

            # Upper face (eyes region)
            upper1 = gray1[h//4:h//2, :]
            upper2 = gray2[h//4:h//2, :]

            # Lower face (mouth region)
            lower1 = gray1[h*2//3:, :]
            lower2 = gray2[h*2//3:, :]

            # Calculate normalized cross-correlation for each region
            if upper1.size > 0 and upper2.size > 0 and lower1.size > 0 and lower2.size > 0:
                upper_diff = 1.0 - cv2.matchTemplate(upper1, upper2, cv2.TM_CCOEFF_NORMED).max()
                lower_diff = 1.0 - cv2.matchTemplate(lower1, lower2, cv2.TM_CCOEFF_NORMED).max()

                # Combine differences (mouth changes are more significant for expressions)
                expression_diff = (upper_diff * 0.3 + lower_diff * 0.7)
                return max(0.0, min(1.0, expression_diff))

            return 0.05  # Default small difference

        except Exception as e:
            self.analyzer.logger.main_logger.error(f"Error calculating expression difference: {str(e)}")
            return 0.05

    def _perform_statistical_analysis(self, frame_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Perform statistical analysis on frame results."""

        try:
            # Extract metrics from frame results
            overall_scores = []
            quality_scores = []
            artifact_scores = []
            temporal_scores = []

            for result in frame_results:
                if 'overall_score' in result:
                    overall_scores.append(result['overall_score'])

                if 'frame_quality' in result:
                    quality_scores.append(result['frame_quality'].get('overall_quality', 0.5))

                if 'artifacts' in result:
                    artifact_scores.append(result['artifacts'].get('overall_artifacts', 0.5))

                if 'temporal_score' in result:
                    temporal_scores.append(result['temporal_score'])

            # Calculate statistical metrics
            stats = {}

            if overall_scores:
                stats['overall_mean'] = np.mean(overall_scores)
                stats['overall_std'] = np.std(overall_scores)
                stats['overall_min'] = np.min(overall_scores)
                stats['overall_max'] = np.max(overall_scores)

                # Consistency analysis
                stats['score_consistency'] = 1.0 / (1.0 + stats['overall_std'] * 5)

            if quality_scores:
                stats['quality_mean'] = np.mean(quality_scores)
                stats['quality_consistency'] = 1.0 / (1.0 + np.std(quality_scores) * 5)

            if artifact_scores:
                stats['artifact_mean'] = np.mean(artifact_scores)
                stats['artifact_consistency'] = 1.0 / (1.0 + np.std(artifact_scores) * 5)

            if temporal_scores:
                stats['temporal_mean'] = np.mean(temporal_scores)
                stats['temporal_consistency'] = 1.0 / (1.0 + np.std(temporal_scores) * 5)

            return stats

        except Exception as e:
            self.analyzer.logger.main_logger.error(f"Error in statistical analysis: {str(e)}")
            return {}

    def _analyze_artifact_patterns(self, frame_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze patterns in artifacts across frames."""

        try:
            artifact_patterns = {
                'compression_pattern': 0.5,
                'blur_pattern': 0.5,
                'noise_pattern': 0.5,
                'edge_pattern': 0.5
            }

            # Extract artifact data
            compression_scores = []
            blur_scores = []
            noise_scores = []
            edge_scores = []

            for result in frame_results:
                artifacts = result.get('artifacts', {})

                compression_scores.append(artifacts.get('compression_artifacts', 0.5))
                blur_scores.append(artifacts.get('blur_artifacts', 0.5))
                noise_scores.append(artifacts.get('noise_artifacts', 0.5))
                edge_scores.append(artifacts.get('edge_artifacts', 0.5))

            # Analyze patterns
            if compression_scores:
                compression_consistency = 1.0 / (1.0 + np.std(compression_scores) * 3)
                artifact_patterns['compression_pattern'] = compression_consistency

            if blur_scores:
                blur_consistency = 1.0 / (1.0 + np.std(blur_scores) * 3)
                artifact_patterns['blur_pattern'] = blur_consistency

            if noise_scores:
                noise_consistency = 1.0 / (1.0 + np.std(noise_scores) * 3)
                artifact_patterns['noise_pattern'] = noise_consistency

            if edge_scores:
                edge_consistency = 1.0 / (1.0 + np.std(edge_scores) * 3)
                artifact_patterns['edge_pattern'] = edge_consistency

            return artifact_patterns

        except Exception as e:
            self.analyzer.logger.main_logger.error(f"Error analyzing artifact patterns: {str(e)}")
            return {'compression_pattern': 0.5, 'blur_pattern': 0.5, 
                   'noise_pattern': 0.5, 'edge_pattern': 0.5}

    def _create_detection_result(self, video_path: str, frame_results: List[Dict[str, Any]],
                               enhanced_results: Dict[str, Any], start_time: float) -> DetectionResult:
        """Create final detection result."""

        try:
            # Extract frame scores
            frame_scores = [result.get('overall_score', 0.5) for result in frame_results]

            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(frame_results, enhanced_results)

            # Determine if deepfake
            is_deepfake = overall_confidence < 0.5

            # Calculate individual component scores
            anatomical_score = self._extract_anatomical_score(frame_results)
            face_quality_score = self._extract_face_quality_score(frame_results)
            artifact_score = self._extract_artifact_score(frame_results)

            # Extract enhanced analysis scores
            temporal_consistency = enhanced_results.get('temporal_consistency', 0.5)
            face_landmarks_consistency = enhanced_results.get('face_consistency', 0.5)
            eye_blink_naturalness = enhanced_results.get('eye_blink_naturalness', 0.5)
            micro_expression_analysis = enhanced_results.get('micro_expression_score', 0.5)

            # Aggregate frequency-domain score from frames and convert to artifact score (higher worse)
            try:
                freq_scores = [float(r.get('frequency_score', 0.5)) for r in frame_results]
                avg_freq_score = float(np.mean(freq_scores)) if freq_scores else 0.5
            except Exception:
                avg_freq_score = 0.5
            frequency_artifact_score = max(0.0, min(1.0, 1.0 - avg_freq_score))

            # Count statistics
            frames_analyzed = len(frame_results)
            faces_detected = sum(result.get('faces_detected', 0) for result in frame_results)

            # Analysis methods used
            analysis_methods = ['enhanced_anatomical', 'face_quality', 'artifacts', 
                              'temporal_consistency', 'frequency_analysis']

            # Detect errors
            detected_errors = []
            for result in frame_results:
                if 'error' in result:
                    detected_errors.append({'frame_error': result['error']})

            # Confidence breakdown
            confidence_breakdown = {
                'anatomical': anatomical_score,
                'face_quality': face_quality_score,
                'artifacts': 1.0 - artifact_score,  # Invert artifacts score
                'temporal': temporal_consistency,
                'overall': overall_confidence
            }

            # Create result
            result = DetectionResult(
                is_deepfake=is_deepfake,
                confidence=overall_confidence,
                processing_time=time.time() - start_time,
                anatomical_score=anatomical_score,
                face_quality_score=face_quality_score,
                artifact_score=artifact_score,
                temporal_consistency=temporal_consistency,
                frequency_artifact_score=frequency_artifact_score,
                frame_scores=frame_scores,
                face_landmarks_consistency=face_landmarks_consistency,
                eye_blink_naturalness=eye_blink_naturalness,
                micro_expression_analysis=micro_expression_analysis,
                frames_analyzed=frames_analyzed,
                faces_detected=faces_detected,
                analysis_methods=analysis_methods,
                detected_errors=detected_errors,
                confidence_breakdown=confidence_breakdown,
                video_path=video_path,
                analysis_timestamp=datetime.datetime.now().isoformat(),
                model_version="4.0-improved"
            )

            return result

        except Exception as e:
            self.analyzer.logger.main_logger.error(f"Error creating detection result: {str(e)}")
            # Return fallback result
            return DetectionResult(
                is_deepfake=False,
                confidence=0.0,
                processing_time=time.time() - start_time,
                anatomical_score=0.0,
                face_quality_score=0.0,
                artifact_score=0.0,
                temporal_consistency=0.0,
                frequency_artifact_score=0.0,
                frame_scores=[],
                face_landmarks_consistency=0.0,
                eye_blink_naturalness=0.0,
                micro_expression_analysis=0.0,
                frames_analyzed=0,
                faces_detected=0,
                analysis_methods=['error'],
                detected_errors=[{'creation_error': str(e)}],
                confidence_breakdown={'error': 1.0},
                video_path=video_path,
                analysis_timestamp=datetime.datetime.now().isoformat(),
                model_version="4.0-improved"
            )

    def _calculate_overall_confidence(self, frame_results: List[Dict[str, Any]], 
                                    enhanced_results: Dict[str, Any]) -> float:
        """Calculate overall confidence score."""

        try:
            # Frame-level confidence
            frame_scores = [result.get('overall_score', 0.5) for result in frame_results]
            frame_confidence = np.mean(frame_scores) if frame_scores else 0.5

            # Enhanced analysis confidence
            temporal_confidence = enhanced_results.get('temporal_consistency', 0.5)
            face_confidence = enhanced_results.get('face_consistency', 0.5)
            eye_confidence = enhanced_results.get('eye_blink_naturalness', 0.5)
            expression_confidence = enhanced_results.get('micro_expression_score', 0.5)

            # Statistical confidence
            stats = enhanced_results.get('statistical_analysis', {})
            statistical_confidence = stats.get('score_consistency', 0.5)

            # Weighted combination
            weights = {
                'frames': 0.4,
                'temporal': 0.2,
                'faces': 0.15,
                'eyes': 0.1,
                'expressions': 0.1,
                'statistics': 0.05
            }

            overall_confidence = (
                weights['frames'] * frame_confidence +
                weights['temporal'] * temporal_confidence +
                weights['faces'] * face_confidence +
                weights['eyes'] * eye_confidence +
                weights['expressions'] * expression_confidence +
                weights['statistics'] * statistical_confidence
            )

            return max(0.0, min(1.0, overall_confidence))

        except Exception as e:
            self.analyzer.logger.main_logger.error(f"Error calculating overall confidence: {str(e)}")
            return 0.5

    def _extract_anatomical_score(self, frame_results: List[Dict[str, Any]]) -> float:
        """Extract average anatomical score from frame results."""

        anatomical_scores = []

        for result in frame_results:
            face_analysis = result.get('face_analysis', [])
            for face_result in face_analysis:
                anatomical_analysis = face_result.get('anatomical_analysis', {})
                score = anatomical_analysis.get('overall_anatomical_score', 0.5)
                anatomical_scores.append(score)

        return np.mean(anatomical_scores) if anatomical_scores else 0.5

    def _extract_face_quality_score(self, frame_results: List[Dict[str, Any]]) -> float:
        """Extract average face quality score from frame results."""

        quality_scores = []

        for result in frame_results:
            face_analysis = result.get('face_analysis', [])
            for face_result in face_analysis:
                quality_analysis = face_result.get('quality_analysis', {})
                score = quality_analysis.get('overall_quality', 0.5)
                quality_scores.append(score)

        return np.mean(quality_scores) if quality_scores else 0.5

    def _extract_artifact_score(self, frame_results: List[Dict[str, Any]]) -> float:
        """Extract average artifact score from frame results."""

        artifact_scores = []

        for result in frame_results:
            artifacts = result.get('artifacts', {})
            score = artifacts.get('overall_artifacts', 0.5)
            artifact_scores.append(score)

        return np.mean(artifact_scores) if artifact_scores else 0.5


