import os, torch
from accelerate import Accelerator


class MetricsLogger:
    """支持 TensorBoard 和 WandB 的训练指标记录器。

    仅在主进程（rank 0）初始化后端，其他进程的调用为空操作。
    """

    def __init__(self, log_dir="./logs", backend="tensorboard", project_name=None, run_name=None, config=None):
        """
        Args:
            log_dir: 日志输出目录（TensorBoard 使用）。
            backend: "tensorboard", "wandb", 或 "none"。
            project_name: WandB 项目名。
            run_name: WandB 运行名称。
            config: 传给 WandB 的超参数字典。
        """
        self.backend = backend.lower() if backend else "none"
        self.log_dir = log_dir
        self._writer = None
        self._initialized = False

        if self.backend == "none":
            return

        if self.backend == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(log_dir, exist_ok=True)
                self._writer = SummaryWriter(log_dir=log_dir)
                self._initialized = True
            except ImportError:
                print("[MetricsLogger] tensorboard 未安装，回退到 none 模式。请运行: pip install tensorboard")
                self.backend = "none"
        elif self.backend == "wandb":
            try:
                import wandb
                if not wandb.run:
                    wandb.init(
                        project=project_name or "diffsynth-training",
                        name=run_name,
                        config=config or {},
                    )
                self._writer = wandb
                self._initialized = True
            except ImportError:
                print("[MetricsLogger] wandb 未安装，回退到 none 模式。请运行: pip install wandb")
                self.backend = "none"
        else:
            print(f"[MetricsLogger] 未知的 backend: {self.backend}，回退到 none 模式。")
            self.backend = "none"

    def log_scalar(self, tag, value, step):
        """记录单个标量指标。"""
        if not self._initialized:
            return
        if self.backend == "tensorboard":
            self._writer.add_scalar(tag, value, global_step=step)
        elif self.backend == "wandb":
            self._writer.log({tag: value}, step=step)

    def log_scalars(self, tag_value_dict, step):
        """批量记录标量指标。

        Args:
            tag_value_dict: {tag: value} 字典。
            step: 全局步数。
        """
        if not self._initialized:
            return
        if self.backend == "tensorboard":
            for tag, value in tag_value_dict.items():
                self._writer.add_scalar(tag, value, global_step=step)
        elif self.backend == "wandb":
            self._writer.log(tag_value_dict, step=step)

    def close(self):
        """关闭日志后端，刷新缓冲区。"""
        if not self._initialized:
            return
        if self.backend == "tensorboard" and self._writer is not None:
            self._writer.close()
        elif self.backend == "wandb" and self._writer is not None:
            self._writer.finish()
        self._initialized = False


class ModelLogger:
    def __init__(
        self,
        output_path,
        remove_prefix_in_ckpt=None,
        state_dict_converter=lambda x: x,
        metrics_backend="none",
        log_dir="./logs",
        project_name=None,
        run_name=None,
        config=None,
        log_interval=1,
    ):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        self.log_interval = log_interval

        # MetricsLogger 将在 _init_metrics_logger 中延迟初始化
        self._metrics_backend = metrics_backend
        self._log_dir = log_dir
        self._project_name = project_name
        self._run_name = run_name
        self._config = config
        self.metrics_logger = None

    def _init_metrics_logger(self, is_main_process):
        """延迟初始化 MetricsLogger，仅在主进程创建实际后端。"""
        if self.metrics_logger is not None:
            return
        if is_main_process and self._metrics_backend != "none":
            self.metrics_logger = MetricsLogger(
                log_dir=self._log_dir,
                backend=self._metrics_backend,
                project_name=self._project_name,
                run_name=self._run_name,
                config=self._config,
            )
        else:
            # 非主进程或 backend=none 时使用空操作 logger
            self.metrics_logger = MetricsLogger(backend="none")

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, **kwargs):
        self.num_steps += 1

        # 延迟初始化 metrics logger
        self._init_metrics_logger(accelerator.is_main_process)

        # 记录训练指标
        if accelerator.is_main_process and self.num_steps % self.log_interval == 0:
            metrics = {}
            if "loss" in kwargs and kwargs["loss"] is not None:
                loss_val = kwargs["loss"]
                if isinstance(loss_val, torch.Tensor):
                    loss_val = loss_val.detach().item()
                metrics["train/loss"] = loss_val
            if "learning_rate" in kwargs and kwargs["learning_rate"] is not None:
                metrics["train/learning_rate"] = kwargs["learning_rate"]
            if "grad_norm" in kwargs and kwargs["grad_norm"] is not None:
                grad_norm_val = kwargs["grad_norm"]
                if isinstance(grad_norm_val, torch.Tensor):
                    grad_norm_val = grad_norm_val.detach().item()
                metrics["train/grad_norm"] = grad_norm_val
            if "step_time" in kwargs and kwargs["step_time"] is not None:
                metrics["train/step_time"] = kwargs["step_time"]
            if "samples_per_sec" in kwargs and kwargs["samples_per_sec"] is not None:
                metrics["train/samples_per_sec"] = kwargs["samples_per_sec"]
            # RL/GRPO reward 指标
            if "reward_metrics" in kwargs and kwargs["reward_metrics"] is not None:
                for key, value in kwargs["reward_metrics"].items():
                    if isinstance(value, torch.Tensor):
                        value = value.detach().item()
                    metrics[key] = value

            if metrics:
                self.metrics_logger.log_scalars(metrics, step=self.num_steps)

        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
        # 关闭 metrics logger
        if self.metrics_logger is not None:
            self.metrics_logger.close()


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
