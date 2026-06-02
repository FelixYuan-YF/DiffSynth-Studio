"""
HPSv3 (Human Preference Score v3) reward for GDPO training.

Evaluates image quality and text-image alignment using the HPSv3 model
(based on Qwen2-VL architecture).

Reference: https://github.com/MizzenAI/HPSv3
"""

from typing import List, Optional, Tuple
import numpy as np
import torch


class HPSv3Evaluator:
    """HPSv3 reward evaluator for video frames.

    Scores video frames using HPSv3 (Human Preference Score v3) model,
    which evaluates image quality and prompt alignment.

    The evaluator samples frames from the video at a given sampling rate
    (shared with EpipolarEvaluator), scores each frame against the prompt,
    and returns the mean score as the reward.

    Args:
        checkpoint_path: Local path or HuggingFace model ID for HPSv3 checkpoint.
                    Defaults to "MizzenAI/HPSv3" (auto-download).
        device: Device to load the model on. Defaults to "cuda".
        sampling_rate: Sample every N-th frame for scoring (shared with epipolar).
    """

    def __init__(
        self,
        checkpoint_path: str = "MizzenAI/HPSv3",
        device: str = "cuda",
        sampling_rate: int = 15,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.sampling_rate = sampling_rate
        self._inferencer = None
        self._current_device = None  # Track current device for model

    def _lazy_init(self):
        """Lazily initialize the HPSv3 model (first call only)."""
        if self._inferencer is not None:
            return
        try:
            from hpsv3 import HPSv3RewardInferencer
            # Initialize on CPU first to save GPU memory
            self._inferencer = HPSv3RewardInferencer(
                checkpoint_path=self.checkpoint_path,
                device="cpu",
            )
            self._current_device = "cpu"
            print(f"[HPSv3Reward] Model loaded from: {self.checkpoint_path} (on CPU)")
        except ImportError:
            raise ImportError(
                "HPSv3 package not found. Install it with:\n"
                "  git clone https://github.com/MizzenAI/HPSv3.git && cd HPSv3 && pip install -e .\n"
                "Or install via: pip install hpsv3"
            )

    def to_gpu(self):
        """Move model to GPU for inference."""
        self._lazy_init()
        if self._inferencer is None:
            return
        
        if self._current_device != "cuda":
            # Move model to GPU
            if hasattr(self._inferencer, 'model') and hasattr(self._inferencer.model, 'to'):
                self._inferencer.model.to('cuda')
                self._current_device = "cuda"
                print(f"[HPSv3Reward] Model moved to GPU")
            elif hasattr(self._inferencer, 'to'):
                self._inferencer.to('cuda')
                self._current_device = "cuda"
                print(f"[HPSv3Reward] Model moved to GPU")
            else:
                print(f"[HPSv3Reward] Warning: Cannot move model to GPU, device management not supported")

    def to_cpu(self):
        """Move model to CPU to save GPU memory."""
        if self._inferencer is None:
            return
        
        if self._current_device == "cuda":
            # Move model to CPU
            if hasattr(self._inferencer, 'model') and hasattr(self._inferencer.model, 'to'):
                self._inferencer.model.to('cpu')
                self._current_device = "cpu"
                print(f"[HPSv3Reward] Model moved to CPU")
            elif hasattr(self._inferencer, 'to'):
                self._inferencer.to('cpu')
                self._current_device = "cpu"
                print(f"[HPSv3Reward] Model moved to CPU")
            
            # Force GPU memory cleanup
            import torch
            torch.cuda.empty_cache()

    def score_frames(
        self,
        frames: List,
        prompt: str,
    ) -> Tuple[float, dict]:
        """Score a list of PIL Image frames against a text prompt.

        Args:
            frames: List of PIL.Image.Image objects (video frames).
            prompt: Text prompt used to generate the video.

        Returns:
            (mean_score, metrics_dict):
              - mean_score: Average HPSv3 score across all frames.
              - metrics_dict: Detailed per-frame scores and statistics.
        """
        self._lazy_init()

        if not frames or len(frames) < 1:
            return 0.0, {"error": "No frames provided", "num_frames": 0}

        # Score each frame
        import tempfile
        import os
        scores = []
        temp_dir = tempfile.mkdtemp(prefix="hpsv3_")

        try:
            image_paths = []
            prompts = []
            for idx, frame in enumerate(frames):
                # Save frame to temp file (HPSv3 expects file paths)
                temp_path = os.path.join(temp_dir, f"frame_{idx:04d}.png")
                frame.save(temp_path)
                image_paths.append(temp_path)
                prompts.append(prompt)

            # Batch inference
            with torch.no_grad():
                rewards = self._inferencer.reward(image_paths, prompts)

            for reward in rewards:
                # reward[0] = mu (preference score), reward[1] = sigma (uncertainty)
                mu = reward[0].item()
                scores.append(mu)

        except Exception as e:
            print(f"[HPSv3Reward] Scoring failed: {e}")
            return 0.0, {"error": str(e)}
        finally:
            # Clean up temp files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

        if not scores:
            return 0.0, {"error": "No valid scores", "num_frames": len(frames)}

        mean_score = float(np.mean(scores))
        metrics = {
            "mean_score": mean_score,
            "std_score": float(np.std(scores)) if len(scores) > 1 else 0.0,
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "num_sampled_frames": len(frames),
            "total_frames": len(frames),
            "per_frame_scores": scores,
        }
        return mean_score, metrics

    def evaluate_video(
        self,
        video_path: str,
        prompt: str,
    ) -> Tuple[float, dict]:
        """Evaluate a video file using HPSv3.

        Args:
            video_path: Path to the video file.
            prompt: Text prompt used to generate the video.

        Returns:
            (mean_score, metrics_dict)
        """
        import cv2
        from PIL import Image

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0, {"error": f"Could not open video: {video_path}"}

        frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.sampling_rate == 0:
                # Convert BGR (OpenCV) to RGB (PIL)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb_frame)
                frames.append(pil_frame)
            frame_idx += 1
        cap.release()

        if not frames:
            return 0.0, {"error": "No frames extracted from video"}

        # Frames are already sampled in this method, score them directly
        score, metrics = self.score_frames(frames, prompt)

        metrics["video_path"] = video_path
        metrics["total_video_frames"] = frame_idx
        return score, metrics
