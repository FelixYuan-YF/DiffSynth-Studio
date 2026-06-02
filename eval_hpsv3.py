import os
import pandas as pd
from tqdm import tqdm
from diffsynth.diffusion.hpsv3_reward import HPSv3Evaluator
import argparse
import multiprocessing
import torch

# Evaluator 配置（每个 worker 进程独立创建实例，避免 pickle 序列化问题）
EVALUATOR_CONFIG = {
    "checkpoint_path": "./models/HPSv3/HPSv3.safetensors",  # HuggingFace model ID
    "device": "cuda",                      # 使用 GPU
    "sampling_rate": 5,                   # 与训练相同
}

# 全局变量，存储每个进程的 evaluator 实例
_evaluator = None


def init_evaluator():
    """在每个进程初始化时创建 evaluator 实例，并分配到指定的 GPU。"""
    global _evaluator
    
    # 获取当前进程的 ID
    current_process = multiprocessing.current_process()
    worker_id = current_process._identity[0] - 1 if current_process._identity else 0
    
    # 获取可用的 GPU 数量
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPU available")
    
    # 轮询分配 GPU
    gpu_id = worker_id % num_gpus
    device = f"cuda:{gpu_id}"
    
    # 设置当前进程使用的 GPU
    torch.cuda.set_device(device)
    
    # 更新配置中的 device
    config = EVALUATOR_CONFIG.copy()
    config["device"] = device
    
    print(f"Worker {worker_id} (PID: {os.getpid()}) initialized on {device}")
    _evaluator = HPSv3Evaluator(**config)


def evaluate_one(row: pd.Series) -> dict:
    """单个视频评估，在子进程中执行。"""
    global _evaluator
    
    # 如果 evaluator 还未初始化，则初始化（兼容不同的调用方式）
    if _evaluator is None:
        init_evaluator()
    
    video_path = row["video"]
    prompt = row["prompt"]
    video_file = os.path.basename(video_path)
    
    mean_score, metrics = _evaluator.evaluate_video(video_path, prompt)
    
    return {
        "video": video_file,
        "video_path": video_path,
        "mean_score": mean_score,
        "std_score": metrics.get("std_score", -1),
        "min_score": metrics.get("min_score", -1),
        "max_score": metrics.get("max_score", -1),
        "num_sampled_frames": metrics.get("num_sampled_frames", 0),
        "total_frames": metrics.get("total_frames", 0),
        "prompt": prompt,
    }

if __name__ == "__main__":
    # 设置多进程启动方式为 spawn（解决 CUDA fork 问题）
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Evaluate videos using HPSv3")
    parser.add_argument("--input_csv", type=str, required=True, help="Input CSV file containing video_path and prompt columns")
    parser.add_argument("--output_csv", type=str, default="hpsv3_metrics.csv", help="Output CSV file for results")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes (default: number of available GPUs)")
    args = parser.parse_args()
    
    # 检测可用的 GPU 数量
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPU available. This script requires at least one GPU.")
    
    # 确定实际使用的 worker 数量
    if args.num_workers is None:
        actual_workers = num_gpus
    else:
        actual_workers = min(args.num_workers, num_gpus)
    
    print(f"Available GPUs: {num_gpus}")
    print(f"Using {actual_workers} worker processes")
    
    # 读取输入 CSV
    df_input = pd.read_csv(args.input_csv)
    
    # 检查必需的列
    if "video" not in df_input.columns or "prompt" not in df_input.columns:
        raise ValueError("Input CSV must contain 'video' and 'prompt' columns")
    
    # 转换为字典列表，方便多进程处理
    rows = [row for _, row in df_input.iterrows()]

    # 使用 multiprocessing.Pool 来正确分配 GPU
    with multiprocessing.Pool(processes=actual_workers, initializer=init_evaluator) as pool:
        # pool.imap 保持顺序，tqdm 显示进度
        results = list(
            tqdm(
                pool.imap(evaluate_one, rows),
                total=len(rows),
                desc="Evaluating HPSv3",
            )
        )

    # 一次性构建 DataFrame，比逐行 df.loc[i] 更高效
    df_output = pd.DataFrame(results)
    df_output.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")
    print(f"Average mean_score: {df_output['mean_score'].mean():.4f}")