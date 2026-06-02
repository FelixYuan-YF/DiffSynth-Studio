"""
Epipolar geometry-based reward for GRPO training.

Evaluates 3D consistency between video frames using epipolar geometry
(SIFT/LightGlue feature matching + Sampson distance).

Adapted from epipolar-dpo/metrics/video_evaluation/epipolar.py
"""

from typing import Dict, Any, Tuple, List, Optional
from abc import ABC, abstractmethod

import numpy as np
import cv2
import torch


# ======================================================================
# Keypoint Matchers
# ======================================================================

class KeypointMatcher(ABC):
    """Abstract base class for keypoint detection and matching algorithms."""

    @abstractmethod
    def get_matched_points(
        self, frame1: np.ndarray, frame2: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, Dict[str, Any]]:
        pass


class SIFTMatcher(KeypointMatcher):
    """SIFT-based keypoint detection and matching."""

    def __init__(self, ratio_thresh: float = 0.75, min_matches: int = 20):
        self.ratio_thresh = ratio_thresh
        self.min_matches = min_matches
        self.sift = cv2.SIFT_create()

    def detect_and_compute(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        kp, desc = self.sift.detectAndCompute(gray, None)
        return kp, desc

    def match_features(self, desc1: np.ndarray, desc2: np.ndarray):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc1, desc2, k=2)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_thresh * n.distance:
                    good_matches.append(m)
        return good_matches

    def get_matched_points(
        self, frame1: np.ndarray, frame2: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, Dict[str, Any]]:
        kp1, desc1 = self.detect_and_compute(frame1)
        kp2, desc2 = self.detect_and_compute(frame2)

        metadata = {
            'keypoints1': len(kp1),
            'keypoints2': len(kp2),
            'descriptor_type': 'sift',
        }

        if len(kp1) < 8 or len(kp2) < 8:
            metadata['error'] = 'Not enough keypoints detected'
            return None, None, 0, metadata

        if desc1 is None or desc2 is None:
            metadata['error'] = 'Failed to compute descriptors'
            return None, None, 0, metadata

        matches = self.match_features(desc1, desc2)

        if len(matches) < self.min_matches:
            metadata['error'] = (
                f'Too few matches ({len(matches)}) - minimum {self.min_matches} required'
            )
            return None, None, len(matches), metadata

        pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)
        return pts1, pts2, len(matches), metadata


class LightGlueMatcher(KeypointMatcher):
    """LightGlue-based keypoint detection and matching."""

    def __init__(self, min_matches: int = 20):
        self.min_matches = min_matches
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained("ETH-CVG/lightglue_superpoint")
        self.model = AutoModel.from_pretrained("ETH-CVG/lightglue_superpoint")

    def get_matched_points(
        self, frame1: np.ndarray, frame2: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, Dict[str, Any]]:
        from PIL import Image

        try:
            if len(frame1.shape) == 3:
                image1 = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
                image2 = Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
            else:
                image1 = Image.fromarray(frame1)
                image2 = Image.fromarray(frame2)

            inputs = self.processor([image1, image2], return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)

            image_sizes = [[(image.height, image.width) for image in [image1, image2]]]
            results = self.processor.post_process_keypoint_matching(
                outputs, image_sizes, threshold=0.2
            )

            if not results:
                return None, None, 0, {
                    'error': 'No results from LightGlue',
                    'descriptor_type': 'lightglue',
                }

            result = results[0]
            num_matches = len(result["keypoints0"])

            metadata = {
                'descriptor_type': 'lightglue',
                'threshold': 0.2,
                'total_matches': num_matches,
            }

            if num_matches < self.min_matches:
                metadata['error'] = (
                    f'Too few matches ({num_matches}) - minimum {self.min_matches} required'
                )
                return None, None, num_matches, metadata

            pts1 = result["keypoints0"].cpu().numpy()
            pts2 = result["keypoints1"].cpu().numpy()
            return pts1, pts2, num_matches, metadata

        except Exception as e:
            return None, None, 0, {
                'error': f'LightGlue processing failed: {str(e)}',
                'descriptor_type': 'lightglue',
            }


# ======================================================================
# Epipolar Evaluator (self-contained, no external base class)
# ======================================================================

class EpipolarEvaluator:
    """
    Evaluator that analyzes 3D consistency between frames using epipolar geometry.
    Computes Sampson distance between matched features to quantify inconsistency.
    """

    def __init__(
        self,
        sampling_rate: int = 15,
        descriptor_type: str = "sift",
        ratio_thresh: float = 0.75,
        ransac_thresh: float = 1.0,
        min_matches: int = 20,
    ):
        self.sampling_rate = sampling_rate
        self.descriptor_type = descriptor_type
        self.ransac_thresh = ransac_thresh
        self.frames: List[np.ndarray] = []

        if descriptor_type == "sift":
            self.matcher = SIFTMatcher(ratio_thresh=ratio_thresh, min_matches=min_matches)
        elif descriptor_type == "lightglue":
            self.matcher = LightGlueMatcher(min_matches=min_matches)
        else:
            raise ValueError(
                f"Unsupported descriptor type: {descriptor_type}. Choose 'sift' or 'lightglue'"
            )

    # ------------------------------------------------------------------
    # Core geometry (OpenCV FM_8POINT)
    # ------------------------------------------------------------------
    def compute_fundamental_matrix(self, pts1: np.ndarray, pts2: np.ndarray):
        """
        Compute fundamental matrix using OpenCV FM_8POINT.

        Note: OpenCV's FM_8POINT already:
        1. Applies Hartley normalization internally
        2. Uses 8-point algorithm (SVD-based, rank-2 enforced)
        3. Returns de-normalized F matrix
        
        We only need to normalize F[2,2] = 1 (Kornia's normalize_transformation).
        """
        try:
            F_matrix, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
            if F_matrix is None:
                return None, None, None

            # Normalize so that F[2,2] = 1 (matches Kornia's normalize_transformation)
            if abs(F_matrix[2, 2]) > 1e-12:
                F_matrix = F_matrix / F_matrix[2, 2]

            if np.isnan(F_matrix).any():
                return None, None, None

            return F_matrix, pts1, pts2
        except Exception:
            return None, None, None

    @staticmethod
    def compute_sampson_distances(F_matrix, pts1, pts2):
        """Compute Sampson distances (pure numpy, equivalent to Kornia)."""
        try:
            pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])  # (N, 3)
            pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])  # (N, 3)

            Fx1 = (F_matrix @ pts1_h.T).T      # (N, 3)
            Ftx2 = (F_matrix.T @ pts2_h.T).T    # (N, 3)

            x2tFx1 = np.sum(pts2_h * Fx1, axis=1)  # (N,)

            # Kornia formula: sampson^2 = (x'^T F x)^2 / (||Fx||^2 + ||F^T x'||^2)
            # Note: only first 2 components of Fx and F^T x' are used
            denom = Fx1[:, 0]**2 + Fx1[:, 1]**2 + Ftx2[:, 0]**2 + Ftx2[:, 1]**2

            sampson_dist_sq = x2tFx1**2 / (denom + 1e-8)
            return np.sqrt(sampson_dist_sq)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Per-pair metric
    # ------------------------------------------------------------------
    def compute_metric_for_pair(self, frame1: np.ndarray, frame2: np.ndarray):
        pts1, pts2, num_matches, metadata = self.matcher.get_matched_points(frame1, frame2)

        base_result = {
            'num_matches': num_matches,
            'descriptor_type': self.descriptor_type,
            **metadata,
        }

        if pts1 is None or pts2 is None:
            return {
                **base_result,
                'epipolar_error': None,
                'inlier_rate': None,
                'error': metadata.get('error', 'Failed to get matched points'),
            }

        # Use pixel coordinates directly (consistent with official Kornia implementation)
        F_matrix, points1_out, points2_out = self.compute_fundamental_matrix(pts1, pts2)
        if F_matrix is None:
            return {
                **base_result,
                'epipolar_error': None,
                'inlier_rate': None,
                'error': 'Failed to compute fundamental matrix',
            }

        sampson_distances = self.compute_sampson_distances(F_matrix, points1_out, points2_out)
        if sampson_distances is None:
            return {
                **base_result,
                'epipolar_error': None,
                'inlier_rate': None,
                'error': 'Failed to compute Sampson distances',
            }

        mean_sampson = np.mean(sampson_distances)
        inlier_threshold = 5.0  # pixels, consistent with official implementation
        inliers = sampson_distances <= inlier_threshold
        inlier_rate = np.mean(inliers)

        return {
            **base_result,
            'epipolar_error': mean_sampson,
            'inlier_rate': inlier_rate,
        }

    # ------------------------------------------------------------------
    # Batch metrics
    # ------------------------------------------------------------------
    def compute_metrics(self, frame_pairs: List[Tuple[np.ndarray, np.ndarray]]):
        results = []
        for frame1, frame2 in frame_pairs:
            result = self.compute_metric_for_pair(frame1, frame2)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def aggregate_metrics(self, frame_metrics) -> Tuple[float, Dict[str, Any]]:
        if len(frame_metrics) == 0:
            return -1, {'mean_epipolar_error': -1, 'mean_inlier_rate': -1, 'total_pairs': 0}

        valid_metrics = [
            m for m in frame_metrics
            if m.get('epipolar_error') is not None
            and not np.isinf(m.get('epipolar_error', float('inf')))
        ]

        if not valid_metrics:
            return -1, {
                'mean_epipolar_error': -1,
                'mean_inlier_rate': -1,
                'total_pairs': len(frame_metrics),
                'valid_pairs': 0,
            }

        epipolar_errors = [m['epipolar_error'] for m in valid_metrics]
        match_counts = [m['num_matches'] for m in valid_metrics]
        inlier_rates = [
            m['inlier_rate'] for m in valid_metrics if m.get('inlier_rate') is not None
        ]

        result = {
            'mean_epipolar_error': float(np.mean(epipolar_errors)),
            'mean_matches': float(np.mean(match_counts)),
            'mean_inlier_rate': float(np.mean(inlier_rates)) if inlier_rates else -1,
            'total_pairs': len(frame_metrics),
            'valid_pairs': len(valid_metrics),
        }

        return result['mean_epipolar_error'], result

    # ------------------------------------------------------------------
    # Main entry: evaluate a video file
    # ------------------------------------------------------------------
    def evaluate_video(self, video_path: str) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate a video file for epipolar consistency.

        Args:
            video_path: Path to the video file

        Returns:
            Tuple of (mean_epipolar_error, detailed_metrics_dict)
            Returns (-1, {...}) on failure.
        """
        self.frames = []

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not cap.isOpened():
            return -1, {'error': f'Could not open video: {video_path}'}

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.sampling_rate == 0:
                self.frames.append(frame.copy())
            frame_idx += 1

        cap.release()

        # Create consecutive frame pairs
        frame_pairs = []
        for i in range(len(self.frames) - 1):
            frame_pairs.append((self.frames[i], self.frames[i + 1]))

        if frame_pairs:
            pair_metrics = self.compute_metrics(frame_pairs)
            avg_score, result = self.aggregate_metrics(pair_metrics)

            result.update({
                'video_path': video_path,
                'original_fps': float(fps),
                'total_frames': total_frames,
                'sampling_rate': self.sampling_rate,
                'sampled_frames': len(self.frames),
                'frame_pairs_evaluated': len(frame_pairs),
            })

            return avg_score, result
        else:
            return -1, {'error': 'Not enough frames for evaluation'}
