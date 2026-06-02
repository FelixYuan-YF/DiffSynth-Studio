import os, torch, time
from contextlib import nullcontext
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def _compute_grad_norm(model):
    """计算模型可训练参数的梯度 L2 范数。"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.detach().data.norm(2).item() ** 2
    return total_norm ** 0.5


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            step_start = time.time()
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                grad_norm = _compute_grad_norm(model)
                optimizer.step()
                step_time = time.time() - step_start
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else learning_rate
                model_logger.on_step_end(
                    accelerator, model, save_steps,
                    loss=loss,
                    learning_rate=current_lr,
                    grad_norm=grad_norm,
                    step_time=step_time,
                    samples_per_sec=1.0 / step_time if step_time > 0 else 0,
                )
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)


def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None, 
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")


def launch_rl_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    """RL（GRPO）训练循环。

    与 launch_training_task 的区别：
    - 不使用 accelerator.accumulate（GRPO 内部自己管理梯度累积）
    - loss 由 FlowMatchGRPOLoss 内部计算并 backward，外部只做 optimizer.step()
    - 数据集只提供 prompt，不需要真实视频
    """
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)

    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            step_start = time.time()
            optimizer.zero_grad()
            # GRPO 的 backward 在 FlowMatchGRPOLoss.__call__ 内部逐步完成。
            # 必须使用 no_sync() 禁用 DDP 在 forward/backward 期间的 bucket 管理，
            # 否则 DDP 会在 forward 内部 backward 后尝试 _rebuild_buckets 导致
            # "initialize_buckets must NOT be called during autograd execution" 错误。
            # 梯度同步由 optimizer.step() 时 accelerator 自动处理（或手动 all-reduce）。
            no_sync_ctx = model.no_sync() if hasattr(model, "no_sync") else nullcontext()
            with no_sync_ctx:
                result = model(data)
            # 解析返回值
            if isinstance(result, dict):
                loss = result["loss"]
                reward_metrics = result.get("metrics")
            else:
                loss = result
                reward_metrics = None
            # loss 已经在内部 backward 完毕，手动同步梯度后 step
            if hasattr(model, "module"):
                # DDP 模式：手动 all-reduce 梯度（因为跳过了 DDP 自动同步）
                for param in model.parameters():
                    if param.grad is not None:
                        torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.AVG)
            grad_norm = _compute_grad_norm(model)
            optimizer.step()
            step_time = time.time() - step_start
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else learning_rate
            model_logger.on_step_end(
                accelerator, model, save_steps,
                loss=loss,
                learning_rate=current_lr,
                grad_norm=grad_norm,
                step_time=step_time,
                samples_per_sec=1.0 / step_time if step_time > 0 else 0,
                reward_metrics=reward_metrics,
            )
            scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)
