import torch, os, argparse, accelerate, warnings
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        pos_encoder="plucker",
        norm_poses=False,
        # GRPO/GDPO 相关参数
        num_generations=4,
        rl_sampling_steps=10,
        rl_eta=1.0,
        rl_shift=5.0,
        rl_cfg_scale=5.0,
        rl_clip_range=1e-4,
        rl_adv_clip_max=5.0,
        rl_timestep_fraction=1.0,
        rl_reward_output_dir="./rl_videos",
        # Epipolar 奖励相关参数
        epipolar_sampling_rate=15,
        epipolar_descriptor_type="sift",
        epipolar_ratio_thresh=0.75,
        epipolar_min_matches=20,
        # HPSv3 奖励相关参数
        hpsv3_model_path="MizzenAI/HPSv3",
        hpsv3_device=None,
        # GRPO 奖励类型选择（支持多个奖励，如 ["epipolar", "hpsv3"]）
        reward_type="epipolar",
        # GDPO 专有：各奖励权重（dict，如 {"epipolar": 1.0, "hpsv3": 0.5}）
        reward_weights=None,
        # 视频尺寸
        height=480,
        width=832,
        num_frames=81,
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        
        # Inject pos_encoder into the global model_configs so that WanModel
        # instances built from MODEL_CONFIGS get the correct positional encoding mode.
        from diffsynth.configs.model_configs import MODEL_CONFIGS as global_model_configs
        for cfg in global_model_configs:
            if cfg.get("model_name") == "wan_video_dit":
                if "extra_kwargs" not in cfg:
                    cfg["extra_kwargs"] = {}
                cfg["extra_kwargs"]["pos_encoder"] = pos_encoder

        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = self.parse_path_or_model_id(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "grpo": FlowMatchGRPOLoss(
                num_generations=num_generations,
                sampling_steps=rl_sampling_steps,
                eta=rl_eta,
                shift=rl_shift,
                cfg_scale=rl_cfg_scale,
                clip_range=rl_clip_range,
                adv_clip_max=rl_adv_clip_max,
                timestep_fraction=rl_timestep_fraction,
                reward_output_dir=rl_reward_output_dir,
                reward_type=reward_type,
                epipolar_sampling_rate=epipolar_sampling_rate,
                epipolar_descriptor_type=epipolar_descriptor_type,
                epipolar_ratio_thresh=epipolar_ratio_thresh,
                epipolar_min_matches=epipolar_min_matches,
                hpsv3_model_path=hpsv3_model_path,
                hpsv3_device=hpsv3_device,
            ),
            "gdpo": FlowMatchGDPOLoss(
                num_generations=num_generations,
                sampling_steps=rl_sampling_steps,
                eta=rl_eta,
                shift=rl_shift,
                cfg_scale=rl_cfg_scale,
                clip_range=rl_clip_range,
                adv_clip_max=rl_adv_clip_max,
                timestep_fraction=rl_timestep_fraction,
                reward_output_dir=rl_reward_output_dir,
                epipolar_sampling_rate=epipolar_sampling_rate,
                epipolar_descriptor_type=epipolar_descriptor_type,
                epipolar_ratio_thresh=epipolar_ratio_thresh,
                epipolar_min_matches=epipolar_min_matches,
                hpsv3_model_path=hpsv3_model_path,
                hpsv3_device=hpsv3_device,
                reward_weights=reward_weights if reward_weights is not None else {"epipolar": 1.0},
            ),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.norm_poses = norm_poses
        # GRPO 视频尺寸参数
        self.rl_cfg_scale = rl_cfg_scale
        
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        if inputs_shared.get("framewise_decoding", False):
            # WanToDance global model
            inputs_shared["num_frames"] = 4 * (len(data["video"]) - 1) + 1
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": data.get("negative_prompt", "")} if self.task in ("grpo", "gdpo") else {}
        
        if self.task in ("grpo", "gdpo"):
            # GRPO: 视频由模型自己生成，使用指定的尺寸
            inputs_shared = {
                "height": data["video"][0].size[1],
                "width": data["video"][0].size[0],
                "num_frames": len(data["video"]),
                "cfg_scale": self.rl_cfg_scale,
                "tiled": False,
                "rand_device": self.pipe.device,
                "use_gradient_checkpointing": self.use_gradient_checkpointing,
                "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
                "cfg_merge": False,
                "norm_poses": self.norm_poses,
            }
        else:
            # SFT: 从数据集中获取视频
            inputs_shared = {
                "input_video": data["video"],
                "height": data["video"][0].size[1],
                "width": data["video"][0].size[0],
                "num_frames": len(data["video"]),
                "cfg_scale": 1,
                "tiled": False,
                "rand_device": self.pipe.device,
                "use_gradient_checkpointing": self.use_gradient_checkpointing,
                "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
                "cfg_merge": False,
                "vace_scale": 1,
                "max_timestep_boundary": self.max_timestep_boundary,
                "min_timestep_boundary": self.min_timestep_boundary,
                "norm_poses": self.norm_poses,
            }
        
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        result = self.task_to_loss[self.task](self.pipe, *inputs)
        # GRPO 返回字典 {"loss": tensor, "metrics": {...}}，直接透传给 runner
        return result

def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--framewise_decoding", default=False, action="store_true", help="Enable it if this model is a WanToDance global model.")
    parser.add_argument("--pos_encoder", type=str, default="plucker", choices=["plucker", "prope"], help="Type of camera relative positional encoding (plucker or prope).")
    parser.add_argument("--norm_poses", default=False, action="store_true", help="Normalize camera translation scale so that the max pairwise distance equals 1.")
    parser.add_argument("--frame_rate", type=float, default=24, help="Frame rate for video loading.")
    parser.add_argument("--fix_frame_rate", default=False, action="store_true", help="Fix frame rate for video loading.")
    # GRPO/GDPO 相关参数
    parser.add_argument("--num_generations", type=int, default=4, help="每个 prompt 生成的视频数量（GRPO/GDPO 组大小）。")
    parser.add_argument("--rl_sampling_steps", type=int, default=10, help="RL 阶段去噪步数。")
    parser.add_argument("--rl_eta", type=float, default=1.0, help="SDE 噪声强度（>0 才能计算 log_prob）。")
    parser.add_argument("--rl_shift", type=float, default=5.0, help="sigma 时间表非线性偏移系数。")
    parser.add_argument("--rl_cfg_scale", type=float, default=5.0, help="Classifier-Free Guidance 强度（RL 阶段）。")
    parser.add_argument("--rl_clip_range", type=float, default=1e-4, help="PPO clip 范围 ε。")
    parser.add_argument("--rl_adv_clip_max", type=float, default=5.0, help="优势值截断上界。")
    parser.add_argument("--rl_timestep_fraction", type=float, default=1.0, help="训练时使用的时间步比例（<1 可节省显存）。")
    parser.add_argument("--rl_reward_output_dir", type=str, default="./rl_videos", help="生成视频保存目录（用于奖励计算）。")
    # Epipolar 奖励相关参数
    parser.add_argument("--epipolar_sampling_rate", type=int, default=15, help="Epipolar 评估时每隔 N 帧采样一次。")
    parser.add_argument("--epipolar_descriptor_type", type=str, default="sift", choices=["sift", "lightglue"], help="特征描述子类型（sift 或 lightglue）。")
    parser.add_argument("--epipolar_ratio_thresh", type=float, default=0.75, help="SIFT Lowe's ratio test 阈值。")
    parser.add_argument("--epipolar_min_matches", type=int, default=20, help="最少匹配点数。")
    # HPSv3 奖励相关参数
    parser.add_argument("--hpsv3_model_path", type=str, default="MizzenAI/HPSv3", help="HPSv3 模型路径（本地路径或 HuggingFace model ID）。")
    parser.add_argument("--hpsv3_device", type=str, default=None, help="HPSv3 模型运行设备（默认与 pipe 同设备）。")
    # GRPO 奖励类型选择
    parser.add_argument("--reward_type", type=str, default="epipolar", help="GRPO 使用的奖励类型，支持单个（如 'epipolar'）或多个（如 'epipolar,hpsv3'，用逗号分隔）。")
    # GDPO 专有参数
    parser.add_argument("--reward_weights", type=str, default=None, help="GDPO 各奖励权重，JSON 字符串，如 '{\"epipolar\": 1.0, \"hpsv3\": 0.5}'。")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4 if not args.framewise_decoding else 1,
            time_division_remainder=1 if not args.framewise_decoding else 0,
            frame_rate=args.frame_rate,
            fix_frame_rate=args.fix_frame_rate,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            "wantodance_music_path": ToAbsolutePath(args.dataset_base_path),
        }
    )
    # 解析 GDPO reward_weights（JSON 字符串 → dict）
    reward_weights = None
    if getattr(args, "reward_weights", None) is not None:
        import json
        reward_weights = json.loads(args.reward_weights)
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        pos_encoder=args.pos_encoder,
        norm_poses=args.norm_poses,
        # GRPO/GDPO 参数
        num_generations=args.num_generations,
        rl_sampling_steps=args.rl_sampling_steps,
        rl_eta=args.rl_eta,
        rl_shift=args.rl_shift,
        rl_cfg_scale=args.rl_cfg_scale,
        rl_clip_range=args.rl_clip_range,
        rl_adv_clip_max=args.rl_adv_clip_max,
        rl_timestep_fraction=args.rl_timestep_fraction,
        rl_reward_output_dir=args.rl_reward_output_dir,
        epipolar_sampling_rate=args.epipolar_sampling_rate,
        epipolar_descriptor_type=args.epipolar_descriptor_type,
        epipolar_ratio_thresh=args.epipolar_ratio_thresh,
        epipolar_min_matches=args.epipolar_min_matches,
        reward_type=[r.strip() for r in args.reward_type.split(',')],
        reward_weights=reward_weights,
        hpsv3_model_path=args.hpsv3_model_path,
        hpsv3_device=args.hpsv3_device,
        # 视频尺寸复用已有参数
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        metrics_backend=getattr(args, 'log_backend', 'none'),
        log_dir=getattr(args, 'log_dir', './logs'),
        project_name=getattr(args, 'wandb_project', None),
        run_name=getattr(args, 'wandb_run_name', None),
        config=vars(args),
        log_interval=getattr(args, 'log_interval', 1),
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
        "grpo": launch_rl_training_task,
        "gdpo": launch_rl_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
