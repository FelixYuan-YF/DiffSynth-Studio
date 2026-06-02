import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from diffsynth.diffusion.epipolar_reward import EpipolarEvaluator

dir_path = f"./results"
video_files = [f for f in sorted(os.listdir(dir_path)) if f.endswith(".mp4")]
num_workers = 16

# Evaluator 配置（每个 worker 进程独立创建实例，避免 pickle 序列化问题）
EVALUATOR_CONFIG = {
    "sampling_rate": 5,        # 与训练相同
    "descriptor_type": "sift",  # 与训练相同
    "ratio_thresh": 0.75,       # 与训练相同
    "min_matches": 20,          # 与训练相同
    "ransac_thresh": 1.0,       # 使用默认值
}

# 全局变量，存储每个进程的 evaluator 实例
_evaluator = None


def init_evaluator():
    """在每个进程初始化时创建 evaluator 实例。"""
    global _evaluator
    _evaluator = EpipolarEvaluator(**EVALUATOR_CONFIG)


def evaluate_one(video_path: str) -> dict:
    """单个视频评估，在子进程中执行。"""
    global _evaluator
    mean_error, metrics = _evaluator.evaluate_video(video_path)
    print(f"Evaluated {os.path.basename(video_path)}: mean_error={mean_error:.4f}, inlier_rate={metrics.get('mean_inlier_rate', -1):.4f}")
    return {
        "video": os.path.basename(video_path),
        "mean_error": mean_error,
        "inlier_rate": metrics.get("mean_inlier_rate", -1),
        "matches": metrics.get("mean_matches", -1),
        "valid_pairs": metrics.get("valid_pairs", 0),
    }

if __name__ == "__main__":
    video_paths = [os.path.join(dir_path, v) for v in video_files]

    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_evaluator) as executor:
        # executor.map 保持原始顺序，tqdm 显示进度
        results = list(
            tqdm(
                executor.map(evaluate_one, video_paths),
                total=len(video_paths),
                desc="Evaluating epipolar",
            )
        )

    # 一次性构建 DataFrame，比逐行 df.loc[i] 更高效
    df = pd.DataFrame(results)
    df.to_csv("epipolar_metrics.csv", index=False)
    print(f"\nResults saved to epipolar_metrics.csv")
    print(f"Average mean_error: {df['mean_error'].mean():.4f}")