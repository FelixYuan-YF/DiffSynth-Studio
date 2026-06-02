from .base_pipeline import BasePipeline
import torch
import math
import os


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    
    if "first_frame_latents" in inputs:
        noise_pred = noise_pred[:, :, 1:]
        training_target = training_target[:, :, 1:]
    
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


def FlowMatchSFTAudioVideoLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    # video
    noise = torch.randn_like(inputs["input_latents"])
    inputs["video_latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    # audio
    if inputs.get("audio_input_latents") is not None:
        audio_noise = torch.randn_like(inputs["audio_input_latents"])
        inputs["audio_latents"] = pipe.scheduler.add_noise(inputs["audio_input_latents"], audio_noise, timestep)
        training_target_audio = pipe.scheduler.training_target(inputs["audio_input_latents"], audio_noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred, noise_pred_audio = pipe.model_fn(**models, **inputs, timestep=timestep)

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    if inputs.get("audio_input_latents") is not None:
        loss_audio = torch.nn.functional.mse_loss(noise_pred_audio.float(), training_target_audio.float())
        loss_audio = loss_audio * pipe.scheduler.training_weight(timestep)
        loss = loss + loss_audio
    return loss


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            denom = sigma_ - sigma
            denom = torch.sign(denom) * torch.clamp(denom.abs(), min=1e-6)
            target = (latents_ - inputs_shared["latents"]) / denom
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss


# ============================================================
# GRPO (Group Relative Policy Optimization) for WanVideo
# ============================================================

def sd3_time_shift(shift, t):
    """对时间步 t 做非线性偏移（Flow Matching 中常用）。
    shift > 1 时将采样集中在低噪声阶段，shift < 1 时集中在高噪声阶段。
    公式：t' = shift*t / (1 + (shift-1)*t)
    """
    return (shift * t) / (1 + (shift - 1) * t)


def wan_grpo_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
    grpo: bool,
    sde_solver: bool = True,
):
    """Flow Matching 框架下的单步去噪（Euler/SDE），支持 GRPO log_prob 计算。

    - grpo=True 且 prev_sample=None：采样阶段，自动从高斯分布采样下一步 latent
    - grpo=True 且 prev_sample 不为 None：训练阶段，计算固定轨迹的 log_prob（带梯度）
    - grpo=False：普通推理，返回 prev_sample_mean
    """
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    # Euler 步：z_{t+1} 的均值预测（Flow Matching: dz = v·dσ）
    prev_sample_mean = latents + dsigma * model_output
    # 预测完全去噪后的干净 latent x_0
    pred_original_sample = latents - sigma * model_output

    delta_t = sigma - sigmas[index + 1]
    std_dev_t = eta * math.sqrt(delta_t.item())

    if sde_solver:
        # Ito SDE 修正：利用 score 函数 ∇log p(z_t) 添加一阶修正项
        score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma ** 2
        log_term = -0.5 * eta ** 2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    if grpo and prev_sample is None:
        # 采样阶段：从高斯分布采样下一步 latent
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

    if grpo:
        # 训练阶段：计算固定轨迹样本 prev_sample 在当前策略下的对数概率
        # log p(z_{t+1} | z_t) = -||z_{t+1} - μ||² / (2σ²) - log σ - log√(2π)
        log_prob = (
            -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2)
            / (2 * (std_dev_t ** 2))
            - math.log(std_dev_t)
            - math.log(math.sqrt(2 * math.pi))
        )
        # 在除 batch 维度外的所有维度上取均值，得到每个样本的标量 log_prob
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob
    else:
        return prev_sample_mean, pred_original_sample


class FlowMatchGRPOLoss:
    """GRPO（Group Relative Policy Optimization）损失，用于 WanVideo 在线 RL 训练。

    整体流程：
      1. [Rollout]   用当前策略对每个 prompt 生成 num_generations 个视频，记录轨迹和旧策略 log_prob
      2. [Reward]    用奖励模型对生成视频打分（支持多个奖励类型，简单相加）
      3. [Advantage] 组内归一化奖励得到优势值
      4. [Update]    PPO-clip 策略梯度更新

    支持的 reward_type：
      - "epipolar"：Epipolar 极线几何一致性（越低越好，取负值作为奖励）
      - "hpsv3"：HPSv3 人类偏好评分（越高越好，直接作为奖励）
      - 支持多个奖励类型：如 ["epipolar", "hpsv3"]，多个奖励简单相加
    """

    def __init__(
        self,
        num_generations: int = 4,
        sampling_steps: int = 10,
        eta: float = 1.0,
        shift: float = 5.0,
        cfg_scale: float = 5.0,
        clip_range: float = 1e-4,
        adv_clip_max: float = 5.0,
        timestep_fraction: float = 1.0,
        reward_output_dir: str = "./rl_videos",
        # 奖励类型选择（支持多个奖励类型）
        reward_type = "epipolar",
        # Epipolar 奖励相关参数
        epipolar_sampling_rate: int = 15,
        epipolar_descriptor_type: str = "sift",
        epipolar_ratio_thresh: float = 0.75,
        epipolar_min_matches: int = 20,
        # HPSv3 奖励相关参数
        hpsv3_model_path: str = "MizzenAI/HPSv3",
        hpsv3_device: str = None,
    ):
        self.num_generations = num_generations
        self.sampling_steps = sampling_steps
        self.eta = eta
        self.shift = shift
        self.cfg_scale = cfg_scale
        self.clip_range = clip_range
        self.adv_clip_max = adv_clip_max
        self.timestep_fraction = timestep_fraction
        self.reward_output_dir = reward_output_dir
        # 奖励类型（支持单个字符串或列表）
        if isinstance(reward_type, str):
            self.reward_types = [reward_type]
        else:
            self.reward_types = list(reward_type)
        # Epipolar 奖励参数
        self.epipolar_sampling_rate = epipolar_sampling_rate
        self.epipolar_descriptor_type = epipolar_descriptor_type
        self.epipolar_ratio_thresh = epipolar_ratio_thresh
        self.epipolar_min_matches = epipolar_min_matches
        self._epipolar_evaluator = None
        # HPSv3 奖励参数
        self.hpsv3_model_path = hpsv3_model_path
        self.hpsv3_device = hpsv3_device
        self._hpsv3_evaluator = None

    # ------------------------------------------------------------------
    # 懒加载 Epipolar 奖励评估器
    # ------------------------------------------------------------------
    def _init_reward_model(self, device):
        """懒加载 EpipolarEvaluator（首次调用时初始化）。"""
        if self._epipolar_evaluator is not None:
            return
        from diffsynth.diffusion.epipolar_reward import EpipolarEvaluator
        self._epipolar_evaluator = EpipolarEvaluator(
            sampling_rate=self.epipolar_sampling_rate,
            descriptor_type=self.epipolar_descriptor_type,
            ratio_thresh=self.epipolar_ratio_thresh,
            min_matches=self.epipolar_min_matches,
        )

    # ------------------------------------------------------------------
    # 懒加载 HPSv3 奖励评估器
    # ------------------------------------------------------------------
    def _init_hpsv3_model(self, device):
        """懒加载 HPSv3Evaluator（首次调用时初始化）。"""
        if self._hpsv3_evaluator is not None:
            return
        from diffsynth.diffusion.hpsv3_reward import HPSv3Evaluator
        hpsv3_dev = self.hpsv3_device if self.hpsv3_device is not None else str(device)
        self._hpsv3_evaluator = HPSv3Evaluator(
            checkpoint_path=self.hpsv3_model_path,
            device=hpsv3_dev,
            sampling_rate=self.epipolar_sampling_rate,  # 共享采样率
        )

    # ------------------------------------------------------------------
    # 构建 sigma 时间表
    # ------------------------------------------------------------------
    def _build_sigma_schedule(self, device):
        """构建 sigma 时间表（线性 + 非线性偏移）。"""
        sigma_schedule = torch.linspace(1, 0, self.sampling_steps + 1)
        sigma_schedule = sd3_time_shift(self.shift, sigma_schedule)
        return sigma_schedule

    # ------------------------------------------------------------------
    # 构建 model_fn 输入参数（公共方法）
    # ------------------------------------------------------------------
    def _build_model_inputs(
        self,
        pipe,
        latents,
        context,
        timestep,
        extra_inputs: dict,
        negative_context=None,
    ):
        """构建 model_fn 的输入参数，支持 CFG 模式。

        Args:
            latents: 当前 latent
            context: 文本编码（正向）
            timestep: 当前时间步
            extra_inputs: 额外的模型输入参数
            negative_context: 负向文本编码（CFG 模式需要）

        Returns:
            (models, model_inputs, is_cfg_mode) - 模型字典，模型输入参数字典，是否为 CFG 模式
        """
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        extra_inputs = extra_inputs or {}

        # 构建基础参数
        model_inputs = {
            "latents": latents,
            "context": context,
            "timestep": timestep,
        }

        # 添加额外输入参数
        for k, v in extra_inputs.items():
            if v is not None and k not in model_inputs:
                model_inputs[k] = v

        is_cfg = self.cfg_scale > 1.0 and negative_context is not None
        return models, model_inputs, is_cfg

    def _run_model_fn(self, pipe, models, inputs, is_cfg, negative_context=None):
        """执行 model_fn 并处理 CFG 输出。

        当 is_cfg=True 时，分两次前向传播（正向 + 负向），避免 batch cat 导致显存翻倍。
        这与 DanceGRPO 的 cat 方式在数学上等价，但峰值显存减半。
        """
        if is_cfg and negative_context is not None:
            # 正向前向传播（带梯度）
            with torch.autocast("cuda", torch.bfloat16):
                model_pred = pipe.model_fn(**models, **inputs)

            # 负向前向传播（不需要梯度，仅用于 CFG 引导）
            uncond_inputs = dict(inputs)
            uncond_inputs["context"] = negative_context
            with torch.no_grad():
                with torch.autocast("cuda", torch.bfloat16):
                    uncond_pred = pipe.model_fn(**models, **uncond_inputs)

            pred = uncond_pred.to(torch.float32) + self.cfg_scale * (
                model_pred.to(torch.float32) - uncond_pred.to(torch.float32)
            )
            del uncond_inputs, uncond_pred
            return pred
        else:
            with torch.autocast("cuda", torch.bfloat16):
                pred = pipe.model_fn(**models, **inputs)
            return pred

    # ------------------------------------------------------------------
    # 单个 prompt 的完整去噪采样（rollout）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _rollout_single(
        self,
        pipe,
        latent_shape,
        encoder_hidden_states,
        negative_prompt_embeds,
        sigma_schedule,
        device,
        extra_inputs: dict = None,
    ):
        """对单个 prompt 执行完整去噪采样，记录轨迹和 log_prob。

        返回：(all_latents, all_log_probs, final_pred_original)
          - all_latents:   shape (1, T+1, C, t, h, w)，每步 latent（含初始噪声）
          - all_log_probs: shape (1, T)，每步旧策略 log_prob
          - final_pred_original: shape (1, C, t, h, w)，最终预测的干净 latent

        Args:
            extra_inputs: 额外的模型输入参数（如 y, control_camera_latents_input 等）
        """
        z = torch.randn(latent_shape, device=device, dtype=pipe.torch_dtype)
        all_latents = [z]
        all_log_probs = []

        for i in range(self.sampling_steps):
            sigma = sigma_schedule[i]
            timestep_value = int(sigma.item() * 1000)
            timestep = torch.full(
                [encoder_hidden_states.shape[0]], timestep_value,
                device=device, dtype=pipe.torch_dtype,
            )

            models, model_inputs, is_cfg = self._build_model_inputs(
                pipe, z, encoder_hidden_states, timestep,
                extra_inputs, negative_prompt_embeds,
            )
            pred = self._run_model_fn(pipe, models, model_inputs, is_cfg, negative_prompt_embeds)

            z, pred_original, log_prob = wan_grpo_step(
                pred, z.to(torch.float32), self.eta,
                sigmas=sigma_schedule, index=i,
                prev_sample=None, grpo=True, sde_solver=True,
            )
            z = z.to(pipe.torch_dtype)
            all_latents.append(z)
            all_log_probs.append(log_prob)

        all_latents = torch.stack(all_latents, dim=1)      # (1, T+1, C, t, h, w)
        all_log_probs = torch.stack(all_log_probs, dim=1)  # (1, T)
        return all_latents, all_log_probs, pred_original

    # ------------------------------------------------------------------
    # VAE 解码 + 奖励打分（根据 reward_type 选择 Epipolar 或 HPSv3）
    # ------------------------------------------------------------------
    def _compute_reward(self, pred_original, pipe, prompt, device, sample_id):
        """VAE 解码 latent → 视频 → 根据 reward_type 打分，返回标量奖励 tensor。

        reward_type == "epipolar":
            Epipolar error 越小表示 3D 一致性越好，奖励 = -epipolar_error。
        reward_type == "hpsv3":
            HPSv3 分数越高表示人类偏好越高，奖励 = hpsv3_score。

        若评估失败，返回 0 作为中性奖励。
        """
        import torch.distributed as dist
        import numpy as np
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        os.makedirs(self.reward_output_dir, exist_ok=True)
        video_path = os.path.join(self.reward_output_dir, f"wan_grpo_{rank}_{sample_id}.mp4")

        # VAE 解码
        video_frames = None
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=pipe.torch_dtype):
                latents = pred_original.clone()
                # 反归一化
                if hasattr(pipe.vae, 'config') and hasattr(pipe.vae.config, 'latents_mean'):
                    latents_mean = (
                        torch.tensor(pipe.vae.config.latents_mean)
                        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                        .to(latents.device, latents.dtype)
                    )
                    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
                        1, pipe.vae.config.z_dim, 1, 1, 1
                    ).to(latents.device, latents.dtype)
                    latents = latents / latents_std + latents_mean
                    del latents_mean, latents_std
                video = pipe.vae.decode(latents, device=device, tiled=False)
                
                # 解码为 PIL 帧列表
                video_frames = pipe.vae_output_to_video(video)  # list of PIL images
                
                # 立即释放 video tensor 的显存
                del video, latents
                torch.cuda.empty_cache()
        
        if video_frames is None or len(video_frames) < 2:
            # 释放 video_frames
            if video_frames is not None:
                del video_frames
            return torch.tensor(0.0, device=device)
        
        if len(video_frames) < 2:
            # 释放 video_frames
            del video_frames
            return torch.tensor(0.0, device=device)

        # 保存视频文件（用于调试 & Epipolar 评估）
        try:
            from diffusers.utils import export_to_video
            export_to_video(video_frames, video_path, fps=24)
        except Exception:
            pass

        # 计算多个奖励并简单相加
        total_reward = 0.0
        
        for reward_type in self.reward_types:
            if reward_type == "hpsv3":
                # ---- HPSv3 人类偏好评分 ----
                self._init_hpsv3_model(device)
                try:
                    # 将模型移到 GPU 进行推理
                    self._hpsv3_evaluator.to_gpu()
                    hpsv3_score, _ = self._hpsv3_evaluator.score_frames(video_frames, prompt)
                    # 推理完成后将模型移回 CPU 以节省显存
                    self._hpsv3_evaluator.to_cpu()
                    
                    if np.isnan(hpsv3_score) or np.isinf(hpsv3_score):
                        hpsv3_reward = 0.0
                    else:
                        hpsv3_reward = hpsv3_score  # 分数越高越好，直接作为奖励
                    total_reward += hpsv3_reward
                except Exception as e:
                    print(f"[HPSv3Reward] 评估失败: {e}")
                    # 确保异常时也将模型移回 CPU
                    if self._hpsv3_evaluator is not None:
                        self._hpsv3_evaluator.to_cpu()
            elif reward_type == "epipolar":
                # ---- Epipolar 极线几何一致性打分 ----
                self._init_reward_model(device)
                try:
                    epipolar_score, detailed_metrics = self._epipolar_evaluator.evaluate_video(video_path)
                    # epipolar_score 是 mean_epipolar_error，越小越好
                    # 返回 -1 表示评估失败
                    if epipolar_score < 0 or np.isnan(epipolar_score) or np.isinf(epipolar_score):
                        epipolar_reward = 0.0
                    else:
                        # 奖励 = -epipolar_error（error 越小，奖励越高）
                        epipolar_reward = -epipolar_score
                    total_reward += epipolar_reward
                except Exception as e:
                    print(f"[EpipolarReward] 评估失败: {e}")
            else:
                print(f"未知的 reward_type: {reward_type}")
        
        reward = total_reward
        
        # 释放 video_frames
        del video_frames
        torch.cuda.empty_cache()

        return torch.tensor(reward, device=device, dtype=torch.float32).unsqueeze(0)

    # ------------------------------------------------------------------
    # 训练阶段单步前向：计算新策略 log_prob（带梯度）
    # ------------------------------------------------------------------
    def _grpo_one_step(
        self,
        pipe,
        latents,
        pre_latents,
        encoder_hidden_states,
        negative_prompt_embeds,
        timestep,
        sigma_idx,
        sigma_schedule,
        extra_inputs: dict = None,
    ):
        """训练阶段单步前向：用新策略计算固定轨迹上的 log_prob（带梯度）。"""
        models, model_inputs, is_cfg = self._build_model_inputs(
            pipe, latents, encoder_hidden_states, timestep,
            extra_inputs, negative_prompt_embeds,
        )
        pred = self._run_model_fn(pipe, models, model_inputs, is_cfg, negative_prompt_embeds)

        _, _, log_prob = wan_grpo_step(
            pred, latents.to(torch.float32), self.eta,
            sigma_schedule, sigma_idx,
            prev_sample=pre_latents.to(torch.float32),
            grpo=True, sde_solver=True,
        )
        return log_prob

    # ------------------------------------------------------------------
    # 完整 GRPO 训练步骤
    # ------------------------------------------------------------------
    def __call__(self, pipe, inputs_shared, inputs_posi, inputs_nega, accelerator=None):
        """完整的 GRPO 训练步骤，返回标量 loss（已完成 backward）。

        被 WanTrainingModule.forward() 调用。
        """
        import torch.distributed as dist

        device = pipe.device
        prompt = inputs_posi.get("prompt", "")
        negative_prompt = inputs_nega.get("negative_prompt", "")

        # 从 inputs_shared 中获取视频尺寸参数
        height = inputs_shared.get("height", 480)
        width = inputs_shared.get("width", 832)
        num_frames = inputs_shared.get("num_frames", 81)

        # 计算 latent 空间维度（Wan VAE 的下采样比例）
        SPATIAL_DOWNSAMPLE = 8
        TEMPORAL_DOWNSAMPLE = 4
        IN_CHANNELS = 16
        latent_t = ((num_frames - 1) // TEMPORAL_DOWNSAMPLE) + 1
        latent_h = height // SPATIAL_DOWNSAMPLE
        latent_w = width // SPATIAL_DOWNSAMPLE
        latent_shape = (1, IN_CHANNELS, latent_t, latent_h, latent_w)

        # 获取文本编码（已由 pipe.units 中的 PromptEmbedder 处理完毕）
        # WanVideo pipeline 中文本编码的 key 是 "context"
        encoder_hidden_states = inputs_posi.get("context")
        negative_prompt_embeds = inputs_nega.get("context")

        # 兼容其他可能的 key 名
        if encoder_hidden_states is None:
            encoder_hidden_states = inputs_posi.get("encoder_hidden_states") or inputs_posi.get("prompt_embeds")
        if negative_prompt_embeds is None:
            negative_prompt_embeds = inputs_nega.get("encoder_hidden_states") or inputs_nega.get("prompt_embeds")

        # 构建 sigma 时间表
        sigma_schedule = self._build_sigma_schedule(device)

        # 提取额外的模型输入参数（如 y, control_camera_latents_input, camera_viewmats 等）
        # 这些参数由 pipeline units 处理后存入 inputs_shared
        EXTRA_INPUT_KEYS = [
            "y", "clip_feature", "reference_latents", "vace_context", "vace_scale",
            "control_camera_latents_input", "camera_viewmats", "camera_Ks",
            "motion_bucket_id", "pose_latents", "face_pixel_values",
            "wantodance_refimage_feature", "wantodance_fps", "music_feature",
            "skip_9th_layer", "audio_embeds", "s2v_pose_latents",
            "vap_hidden_state", "vap_clip_feature", "context_vap",
            "tea_cache", "use_unified_sequence_parallel",
            "longcat_latents", "sliding_window_size", "sliding_window_stride",
            "cfg_merge", "fuse_vae_embedding_in_latents",
        ]
        extra_inputs = {}
        for key in EXTRA_INPUT_KEYS:
            if key in inputs_shared and inputs_shared[key] is not None:
                extra_inputs[key] = inputs_shared[key]

        # ---- 1. Rollout：对每个 prompt 生成 num_generations 个视频 ----
        # 将 prompt 重复 num_generations 次（组内对比）
        if isinstance(prompt, str):
            prompts = [prompt] * self.num_generations
        else:
            prompts = [p for p in prompt for _ in range(self.num_generations)]

        # 重复文本编码
        enc_hs = encoder_hidden_states.repeat_interleave(self.num_generations, dim=0) \
            if encoder_hidden_states is not None else None
        neg_embeds = negative_prompt_embeds.repeat_interleave(self.num_generations, dim=0) \
            if negative_prompt_embeds is not None else None

        all_latents_list = []
        all_log_probs_list = []
        all_rewards_list = []

        # 逐个生成（显存限制，每次处理 1 个）
        for gen_idx in range(len(prompts)):
            enc_hs_i = enc_hs[gen_idx:gen_idx+1] if enc_hs is not None else None
            neg_embeds_i = neg_embeds[gen_idx:gen_idx+1] if neg_embeds is not None else None

            # 采样轨迹
            batch_latents, batch_log_probs, pred_original = self._rollout_single(
                pipe, latent_shape, enc_hs_i, neg_embeds_i, sigma_schedule, device,
                extra_inputs=extra_inputs,
            )
            
            # 立即计算奖励,避免保存 pred_original
            reward = self._compute_reward(pred_original, pipe, prompts[gen_idx], device, gen_idx)
            all_rewards_list.append(reward)
            
            # 将 latents 移到 CPU 以节省 GPU 显存（训练时再移回 GPU）
            all_latents_list.append(batch_latents.cpu())
            all_log_probs_list.append(batch_log_probs.cpu())
            del pred_original, batch_latents, batch_log_probs
            torch.cuda.empty_cache()

        all_latents = torch.cat(all_latents_list, dim=0)                  # (G, T+1, C, t, h, w) 保留在 CPU
        all_log_probs = torch.cat(all_log_probs_list, dim=0)              # (G, T) 保留在 CPU
        all_rewards = torch.cat(all_rewards_list, dim=0).to(torch.float32)  # (G,)
        
        # 清理 CPU 上的临时列表
        del all_latents_list, all_log_probs_list, all_rewards_list
        torch.cuda.empty_cache()

        batch_size = all_latents.shape[0]  # = num_generations（每个 prompt）

        # ---- 2. 跨 GPU 汇总奖励（用于全局统计）----
        all_rewards_gpu = all_rewards.to(device)
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            gathered_rewards = [torch.zeros_like(all_rewards_gpu) for _ in range(world_size)]
            dist.all_gather(gathered_rewards, all_rewards_gpu)
            gathered_rewards = torch.cat(gathered_rewards, dim=0)
        else:
            gathered_rewards = all_rewards_gpu

        # ---- 3. 计算优势（组内归一化）----
        advantages = torch.zeros_like(all_rewards)
        n_groups = batch_size // self.num_generations
        for g in range(n_groups):
            start = g * self.num_generations
            end = (g + 1) * self.num_generations
            group_rewards = all_rewards[start:end]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            advantages[start:end] = (group_rewards - group_mean) / group_std

        # 收集 reward 统计信息（用于日志记录）
        reward_metrics = {
            "reward/mean": all_rewards.mean().item(),
            "reward/std": all_rewards.std().item(),
            "reward/max": all_rewards.max().item(),
            "reward/min": all_rewards.min().item(),
            "reward/mean_advantage": advantages.mean().item(),
        }

        del all_rewards_gpu, gathered_rewards
        torch.cuda.empty_cache()

        # ---- 4. 整理 samples 字典（保留在 CPU 上，训练时逐步移到 GPU）----
        timestep_values = [int(sigma.item() * 1000) for sigma in sigma_schedule[:self.sampling_steps]]
        timesteps_tensor = torch.tensor(
            [timestep_values] * batch_size, dtype=pipe.torch_dtype
        )  # (B, T) - CPU

        samples = {
            "timesteps": timesteps_tensor[:, :-1].detach().clone(),       # (B, T-1)
            "latents": all_latents[:, :-1][:, :-1].detach().clone(),      # (B, T-1, C, t, h, w)
            "next_latents": all_latents[:, 1:][:, :-1].detach().clone(),  # (B, T-1, C, t, h, w)
            "log_probs": all_log_probs[:, :-1].detach().clone(),          # (B, T-1)
            "advantages": advantages,                                       # (B,)
            "context": enc_hs,                                             # (B, seq, dim) - GPU
            "negative_context": neg_embeds,                                # (B, seq, dim) - GPU
        }

        # 立即释放原始大 tensor
        del all_latents, all_log_probs, all_rewards, advantages
        torch.cuda.empty_cache()

        # ---- 5. 随机打乱时间步顺序 ----
        num_train_timesteps = samples["timesteps"].shape[1]
        perms = torch.stack(
            [torch.randperm(num_train_timesteps) for _ in range(batch_size)]
        )  # CPU
        for key in ["timesteps", "latents", "next_latents", "log_probs"]:
            samples[key] = samples[key][
                torch.arange(batch_size)[:, None],
                perms,
            ]

        # ---- 6. PPO-clip 策略梯度更新（对齐 DanceGRPO 实现）----
        # 关键：将 samples 转为 (B, 1, ...) 格式，然后拆成逐样本列表
        # 这与 DanceGRPO 的 samples_batched / samples_batched_list 完全一致
        samples_batched = {k: v.unsqueeze(1) for k, v in samples.items()}
        samples_batched_list = [
            dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
        ]
        del samples, samples_batched
        torch.cuda.empty_cache()

        train_timesteps = int(num_train_timesteps * self.timestep_fraction)
        total_loss = 0.0

        for i, sample in enumerate(samples_batched_list):
            for t_idx in range(train_timesteps):
                # 逐步将当前时间步的数据移到 GPU（而非一次性全部加载）
                latents_t = sample["latents"][:, t_idx].to(device)
                next_latents_t = sample["next_latents"][:, t_idx].to(device)
                timestep_t = sample["timesteps"][:, t_idx].to(device)
                log_probs_t = sample["log_probs"][:, t_idx].to(device)
                adv = torch.clamp(
                    sample["advantages"].to(device),
                    -self.adv_clip_max, self.adv_clip_max,
                )

                # 用新策略计算 log_prob（带梯度）
                new_log_probs = self._grpo_one_step(
                    pipe,
                    latents_t,
                    next_latents_t,
                    sample["context"],
                    sample["negative_context"],
                    timestep_t,
                    perms[i][t_idx],
                    sigma_schedule,
                    extra_inputs=extra_inputs,
                )

                # 重要性采样比率
                ratio = torch.exp(new_log_probs - log_probs_t)

                # PPO-clip 损失
                unclipped_loss = -adv * ratio
                clipped_loss = -adv * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) / (batch_size * train_timesteps)

                if accelerator is not None:
                    accelerator.backward(loss)
                else:
                    loss.backward()

                total_loss += loss.detach().item()

                # 及时释放当前时间步的中间变量
                del latents_t, next_latents_t, timestep_t, log_probs_t
                del new_log_probs, ratio, unclipped_loss, clipped_loss, loss, adv
            
            # 每个样本训练完成后清理显存
            torch.cuda.empty_cache()
        
        # 清理所有剩余 tensor
        del samples_batched_list, enc_hs, neg_embeds, perms
        torch.cuda.empty_cache()

        return {
            "loss": torch.tensor(total_loss, device=device, requires_grad=False),
            "metrics": reward_metrics,
        }


# ============================================================
# GDPO (Group reward-Decoupled Normalization Policy Optimization)
# ============================================================

class FlowMatchGDPOLoss(FlowMatchGRPOLoss):
    """GDPO 损失：在 GRPO 基础上对每个奖励分别做组内归一化，再加权求和得到优势值。

    核心改动（相对于 FlowMatchGRPOLoss）：
      - _compute_rewards()：返回 dict[str, Tensor]，每个 key 对应一个奖励维度
      - __call__()：对每个奖励维度分别归一化后加权求和，其余逻辑完全复用父类

    当前奖励：
      - "epipolar"：Epipolar 极线几何一致性（越低越好，取负值作为奖励）
      - "hpsv3"：HPSv3 人类偏好评分（越高越好，直接作为奖励）

    预留接口：
      - reward_weights: dict[str, float]，控制各奖励的权重，默认 {"epipolar": 1.0}
      - 未来只需在 _compute_rewards() 中添加新奖励 key，并在 reward_weights 中配置权重即可
    """

    def __init__(
        self,
        num_generations: int = 4,
        sampling_steps: int = 10,
        eta: float = 1.0,
        shift: float = 5.0,
        cfg_scale: float = 5.0,
        clip_range: float = 1e-4,
        adv_clip_max: float = 5.0,
        timestep_fraction: float = 1.0,
        reward_output_dir: str = "./rl_videos",
        # Epipolar 奖励相关参数
        epipolar_sampling_rate: int = 15,
        epipolar_descriptor_type: str = "sift",
        epipolar_ratio_thresh: float = 0.75,
        epipolar_min_matches: int = 20,
        # HPSv3 奖励相关参数
        hpsv3_model_path: str = "MizzenAI/HPSv3",
        hpsv3_device: str = None,
        # GDPO 专有：各奖励权重
        reward_weights: dict = None,
    ):
        super().__init__(
            num_generations=num_generations,
            sampling_steps=sampling_steps,
            eta=eta,
            shift=shift,
            cfg_scale=cfg_scale,
            clip_range=clip_range,
            adv_clip_max=adv_clip_max,
            timestep_fraction=timestep_fraction,
            reward_output_dir=reward_output_dir,
            reward_type="epipolar",  # GDPO 不使用父类的单一 reward_type，但需要传一个默认值
            epipolar_sampling_rate=epipolar_sampling_rate,
            epipolar_descriptor_type=epipolar_descriptor_type,
            epipolar_ratio_thresh=epipolar_ratio_thresh,
            epipolar_min_matches=epipolar_min_matches,
            hpsv3_model_path=hpsv3_model_path,
            hpsv3_device=hpsv3_device,
        )
        # 各奖励权重，默认只有 epipolar，权重为 1.0
        self.reward_weights = reward_weights if reward_weights is not None else {"epipolar": 1.0}

    # ------------------------------------------------------------------
    # 共享 VAE 解码：一次解码，多个奖励共用
    # ------------------------------------------------------------------
    def _decode_latents_to_frames(self, pred_original, pipe, device):
        """VAE 解码 latent → 视频帧列表（PIL Image），供多个奖励共用。

        Returns:
            video_frames: list of PIL.Image.Image, or None on failure.
        """
        video_frames = None
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=pipe.torch_dtype):
                latents = pred_original.clone()
                # 反归一化
                if hasattr(pipe.vae, 'config') and hasattr(pipe.vae.config, 'latents_mean'):
                    latents_mean = (
                        torch.tensor(pipe.vae.config.latents_mean)
                        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                        .to(latents.device, latents.dtype)
                    )
                    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
                        1, pipe.vae.config.z_dim, 1, 1, 1
                    ).to(latents.device, latents.dtype)
                    latents = latents / latents_std + latents_mean
                    del latents_mean, latents_std
                video = pipe.vae.decode(latents, device=device, tiled=False)
                video_frames = pipe.vae_output_to_video(video)  # list of PIL images
                del video, latents
                torch.cuda.empty_cache()
        return video_frames

    # ------------------------------------------------------------------
    # 多奖励计算（返回 dict，每个 key 对应一个奖励维度）
    # ------------------------------------------------------------------
    def _compute_rewards(self, pred_original, pipe, prompt, device, sample_id):
        """计算所有奖励维度，返回 dict[str, Tensor(scalar)]。

        当前实现：
          - "epipolar"：Epipolar 极线几何一致性（越低越好，取负值作为奖励）
          - "hpsv3"：HPSv3 人类偏好评分（越高越好，直接作为奖励）

        两个奖励共享 VAE 解码结果，避免重复解码。
        """
        import torch.distributed as dist
        import numpy as np

        rewards = {}
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        need_epipolar = "epipolar" in self.reward_weights
        need_hpsv3 = "hpsv3" in self.reward_weights

        # 如果不需要任何奖励，直接返回
        if not need_epipolar and not need_hpsv3:
            return rewards

        # ---- 共享 VAE 解码 ----
        video_frames = self._decode_latents_to_frames(pred_original, pipe, device)

        if video_frames is None or len(video_frames) < 2:
            if need_epipolar:
                rewards["epipolar"] = torch.tensor(0.0, device=device).unsqueeze(0)
            if need_hpsv3:
                rewards["hpsv3"] = torch.tensor(0.0, device=device).unsqueeze(0)
            if video_frames is not None:
                del video_frames
            return rewards

        # ---- 保存视频文件（Epipolar 需要视频路径，同时用于调试）----
        video_path = None
        if need_epipolar:
            os.makedirs(self.reward_output_dir, exist_ok=True)
            video_path = os.path.join(
                self.reward_output_dir, f"wan_gdpo_{rank}_{sample_id}.mp4"
            )
            try:
                from diffusers.utils import export_to_video
                export_to_video(video_frames, video_path, fps=24)
            except Exception:
                video_path = None

        # ---- Epipolar 奖励 ----
        if need_epipolar:
            self._init_reward_model(device)
            try:
                if video_path is not None:
                    epipolar_score, _ = self._epipolar_evaluator.evaluate_video(video_path)
                else:
                    epipolar_score = -1
                if epipolar_score < 0 or np.isnan(epipolar_score) or np.isinf(epipolar_score):
                    epipolar_reward = 0.0
                else:
                    epipolar_reward = -epipolar_score
            except Exception as e:
                print(f"[EpipolarReward] 评估失败: {e}")
                epipolar_reward = 0.0
            rewards["epipolar"] = torch.tensor(
                epipolar_reward, device=device, dtype=torch.float32
            ).unsqueeze(0)

        # ---- HPSv3 奖励 ----
        if need_hpsv3:
            self._init_hpsv3_model(device)
            try:
                hpsv3_score, hpsv3_metrics = self._hpsv3_evaluator.score_frames(
                    video_frames, prompt
                )
                if np.isnan(hpsv3_score) or np.isinf(hpsv3_score):
                    hpsv3_score = 0.0
            except Exception as e:
                print(f"[HPSv3Reward] 评估失败: {e}")
                hpsv3_score = 0.0
            rewards["hpsv3"] = torch.tensor(
                hpsv3_score, device=device, dtype=torch.float32
            ).unsqueeze(0)

        # 释放视频帧
        del video_frames
        torch.cuda.empty_cache()

        return rewards

    # ------------------------------------------------------------------
    # GDPO 核心：对每个奖励分别归一化，再加权求和
    # ------------------------------------------------------------------
    @staticmethod
    def _decoupled_normalize_advantages(
        rewards_dict: dict,
        reward_weights: dict,
        num_generations: int,
        batch_size: int,
    ):
        """GDPO 核心算法：对每个奖励维度分别做组内归一化，再加权求和。

        Args:
            rewards_dict:    dict[str, Tensor(B,)]，每个奖励维度的原始奖励值
            reward_weights:  dict[str, float]，各奖励权重
            num_generations: 每个 prompt 的生成数量（组大小）
            batch_size:      总样本数 = num_prompts * num_generations

        Returns:
            advantages: Tensor(B,)，加权归一化后的优势值
        """
        n_groups = batch_size // num_generations
        advantages = torch.zeros(batch_size, dtype=torch.float32)

        for reward_name, raw_rewards in rewards_dict.items():
            weight = reward_weights.get(reward_name, 1.0)
            if weight == 0.0:
                continue

            # 对该奖励维度分别做组内归一化
            normalized = torch.zeros_like(raw_rewards)
            for g in range(n_groups):
                start = g * num_generations
                end = (g + 1) * num_generations
                group_r = raw_rewards[start:end]
                group_mean = group_r.mean()
                group_std = group_r.std() + 1e-8
                normalized[start:end] = (group_r - group_mean) / group_std

            # 加权累加
            advantages += weight * normalized

        return advantages

    # ------------------------------------------------------------------
    # 完整 GDPO 训练步骤（重写 __call__，仅优势值计算部分不同）
    # ------------------------------------------------------------------
    def __call__(self, pipe, inputs_shared, inputs_posi, inputs_nega, accelerator=None):
        """完整的 GDPO 训练步骤，返回标量 loss（已完成 backward）。"""
        import torch.distributed as dist

        device = pipe.device
        prompt = inputs_posi.get("prompt", "")

        # 从 inputs_shared 中获取视频尺寸参数
        height = inputs_shared.get("height", 480)
        width = inputs_shared.get("width", 832)
        num_frames = inputs_shared.get("num_frames", 81)

        # 计算 latent 空间维度
        SPATIAL_DOWNSAMPLE = 8
        TEMPORAL_DOWNSAMPLE = 4
        IN_CHANNELS = 16
        latent_t = ((num_frames - 1) // TEMPORAL_DOWNSAMPLE) + 1
        latent_h = height // SPATIAL_DOWNSAMPLE
        latent_w = width // SPATIAL_DOWNSAMPLE
        latent_shape = (1, IN_CHANNELS, latent_t, latent_h, latent_w)

        # 获取文本编码
        encoder_hidden_states = inputs_posi.get("context")
        negative_prompt_embeds = inputs_nega.get("context")
        if encoder_hidden_states is None:
            encoder_hidden_states = inputs_posi.get("encoder_hidden_states") or inputs_posi.get("prompt_embeds")
        if negative_prompt_embeds is None:
            negative_prompt_embeds = inputs_nega.get("encoder_hidden_states") or inputs_nega.get("prompt_embeds")

        # 构建 sigma 时间表
        sigma_schedule = self._build_sigma_schedule(device)

        # 提取额外的模型输入参数
        EXTRA_INPUT_KEYS = [
            "y", "clip_feature", "reference_latents", "vace_context", "vace_scale",
            "control_camera_latents_input", "camera_viewmats", "camera_Ks",
            "motion_bucket_id", "pose_latents", "face_pixel_values",
            "wantodance_refimage_feature", "wantodance_fps", "music_feature",
            "skip_9th_layer", "audio_embeds", "s2v_pose_latents",
            "vap_hidden_state", "vap_clip_feature", "context_vap",
            "tea_cache", "use_unified_sequence_parallel",
            "longcat_latents", "sliding_window_size", "sliding_window_stride",
            "cfg_merge", "fuse_vae_embedding_in_latents",
        ]
        extra_inputs = {}
        for key in EXTRA_INPUT_KEYS:
            if key in inputs_shared and inputs_shared[key] is not None:
                extra_inputs[key] = inputs_shared[key]

        # ---- 1. Rollout ----
        if isinstance(prompt, str):
            prompts = [prompt] * self.num_generations
        else:
            prompts = [p for p in prompt for _ in range(self.num_generations)]

        enc_hs = encoder_hidden_states.repeat_interleave(self.num_generations, dim=0) \
            if encoder_hidden_states is not None else None
        neg_embeds = negative_prompt_embeds.repeat_interleave(self.num_generations, dim=0) \
            if negative_prompt_embeds is not None else None

        all_latents_list = []
        all_log_probs_list = []
        # 每个奖励维度单独收集：dict[str, list[Tensor]]
        all_rewards_dict = {k: [] for k in self.reward_weights}

        for gen_idx in range(len(prompts)):
            enc_hs_i = enc_hs[gen_idx:gen_idx+1] if enc_hs is not None else None
            neg_embeds_i = neg_embeds[gen_idx:gen_idx+1] if neg_embeds is not None else None

            batch_latents, batch_log_probs, pred_original = self._rollout_single(
                pipe, latent_shape, enc_hs_i, neg_embeds_i, sigma_schedule, device,
                extra_inputs=extra_inputs,
            )

            # 计算多维奖励
            rewards_i = self._compute_rewards(pred_original, pipe, prompts[gen_idx], device, gen_idx)
            for k in self.reward_weights:
                all_rewards_dict[k].append(rewards_i.get(k, torch.tensor(0.0, device=device)))

            all_latents_list.append(batch_latents.cpu())
            all_log_probs_list.append(batch_log_probs.cpu())
            del pred_original, batch_latents, batch_log_probs
            torch.cuda.empty_cache()

        all_latents = torch.cat(all_latents_list, dim=0)    # (G, T+1, C, t, h, w) CPU
        all_log_probs = torch.cat(all_log_probs_list, dim=0)  # (G, T) CPU
        # 将每个奖励维度拼接为 (G,) tensor
        all_rewards_dict = {
            k: torch.cat(v, dim=0).to(torch.float32)
            for k, v in all_rewards_dict.items()
        }

        del all_latents_list, all_log_probs_list
        torch.cuda.empty_cache()

        batch_size = all_latents.shape[0]

        # ---- 2. 跨 GPU 汇总奖励（用于日志统计）----
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            gathered_rewards_dict = {}
            for k, v in all_rewards_dict.items():
                v_gpu = v.to(device)
                gathered = [torch.zeros_like(v_gpu) for _ in range(world_size)]
                dist.all_gather(gathered, v_gpu)
                gathered_rewards_dict[k] = torch.cat(gathered, dim=0).cpu()
        else:
            gathered_rewards_dict = {k: v for k, v in all_rewards_dict.items()}

        # ---- 3. GDPO 核心：对每个奖励分别归一化，再加权求和 ----
        advantages = self._decoupled_normalize_advantages(
            rewards_dict=all_rewards_dict,
            reward_weights=self.reward_weights,
            num_generations=self.num_generations,
            batch_size=batch_size,
        )  # (B,) CPU

        # 收集 reward 统计信息
        reward_metrics = {}
        for k, v in all_rewards_dict.items():
            reward_metrics[f"reward/{k}/mean"] = v.mean().item()
            reward_metrics[f"reward/{k}/std"] = v.std().item()
            reward_metrics[f"reward/{k}/max"] = v.max().item()
            reward_metrics[f"reward/{k}/min"] = v.min().item()
        reward_metrics["reward/mean_advantage"] = advantages.mean().item()

        del gathered_rewards_dict
        torch.cuda.empty_cache()

        # ---- 4. 整理 samples 字典 ----
        timestep_values = [int(sigma.item() * 1000) for sigma in sigma_schedule[:self.sampling_steps]]
        timesteps_tensor = torch.tensor(
            [timestep_values] * batch_size, dtype=pipe.torch_dtype
        )  # (B, T) CPU

        samples = {
            "timesteps":      timesteps_tensor[:, :-1].detach().clone(),       # (B, T-1)
            "latents":        all_latents[:, :-1][:, :-1].detach().clone(),    # (B, T-1, C, t, h, w)
            "next_latents":   all_latents[:, 1:][:, :-1].detach().clone(),     # (B, T-1, C, t, h, w)
            "log_probs":      all_log_probs[:, :-1].detach().clone(),          # (B, T-1)
            "advantages":     advantages,                                        # (B,)
            "context":        enc_hs,                                           # GPU
            "negative_context": neg_embeds,                                     # GPU
        }

        del all_latents, all_log_probs, all_rewards_dict, advantages
        torch.cuda.empty_cache()

        # ---- 5. 随机打乱时间步顺序 ----
        num_train_timesteps = samples["timesteps"].shape[1]
        perms = torch.stack(
            [torch.randperm(num_train_timesteps) for _ in range(batch_size)]
        )
        for key in ["timesteps", "latents", "next_latents", "log_probs"]:
            samples[key] = samples[key][
                torch.arange(batch_size)[:, None],
                perms,
            ]

        # ---- 6. PPO-clip 策略梯度更新（与 GRPO 完全相同）----
        samples_batched = {k: v.unsqueeze(1) for k, v in samples.items()}
        samples_batched_list = [
            dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
        ]
        del samples, samples_batched
        torch.cuda.empty_cache()

        train_timesteps = int(num_train_timesteps * self.timestep_fraction)
        total_loss = 0.0

        for i, sample in enumerate(samples_batched_list):
            for t_idx in range(train_timesteps):
                latents_t      = sample["latents"][:, t_idx].to(device)
                next_latents_t = sample["next_latents"][:, t_idx].to(device)
                timestep_t     = sample["timesteps"][:, t_idx].to(device)
                log_probs_t    = sample["log_probs"][:, t_idx].to(device)
                adv = torch.clamp(
                    sample["advantages"].to(device),
                    -self.adv_clip_max, self.adv_clip_max,
                )

                new_log_probs = self._grpo_one_step(
                    pipe,
                    latents_t,
                    next_latents_t,
                    sample["context"],
                    sample["negative_context"],
                    timestep_t,
                    perms[i][t_idx],
                    sigma_schedule,
                    extra_inputs=extra_inputs,
                )

                ratio = torch.exp(new_log_probs - log_probs_t)
                unclipped_loss = -adv * ratio
                clipped_loss   = -adv * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) / (batch_size * train_timesteps)

                if accelerator is not None:
                    accelerator.backward(loss)
                else:
                    loss.backward()

                total_loss += loss.detach().item()

                del latents_t, next_latents_t, timestep_t, log_probs_t
                del new_log_probs, ratio, unclipped_loss, clipped_loss, loss, adv

            torch.cuda.empty_cache()

        del samples_batched_list, enc_hs, neg_embeds, perms
        torch.cuda.empty_cache()

        return {
            "loss": torch.tensor(total_loss, device=device, requires_grad=False),
            "metrics": reward_metrics,
        }
