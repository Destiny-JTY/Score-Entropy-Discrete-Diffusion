import torch
import torch.optim as optim
import numpy as np
import os
import argparse
from tqdm import tqdm

# --- 导入必要的库 ---
from load_model import load_model  
from data import get_sft_dataloader
from model import utils as mutils
from peft import LoraConfig, get_peft_model

# --- 配置类 ---
class Config:
    def __init__(self, model_path):
        self.ngpus = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 模型参数
        self.model = type('obj', (object,), {'length': 1024})
        
        # 训练参数 (4090 单卡优化版)
        # batch_size=8, accum=4 => 等效 Global Batch 32
        self.training = type('obj', (object,), {'batch_size': 8, 'accum': 4}) 
        self.eval = type('obj', (object,), {'batch_size': 8})
        
        # 优化器参数
        self.optim = type('obj', (object,), {
            'lr': 1e-4, 
            'weight_decay': 0.01,
            'beta1': 0.9, 
            'beta2': 0.999, 
            'eps': 1e-8
        })
        
        self.data = type('obj', (object,), {'train': "simplescaling/s1k-1.1"})

def get_loss_fn(noise, graph, mode='train'):
    """
    工厂函数：根据模式返回对应的 Loss 计算逻辑
    mode='train': 使用二次方采样 + 权重截断 (优化 SFT 效果)
    mode='eval': 使用均匀采样 + 无权重 (计算真实 NLL)
    """
    def loss_fn(model, batch, loss_mask):
        B = batch.shape[0]
        device = batch.device
        
        # 1. 时间步 t 的采样策略
        if mode == 'train':
            # [科研级优化] SFT Bias Sampling
            # 使用 u^2 分布，让 t 更倾向于 0 (低噪声区)
            # 这有助于模型学习精细的 Token 补全，而非仅仅学大轮廓
            u = torch.rand(B, device=device)
            t = (1 - 1e-3) * (u ** 2) + 1e-3
        else:
            # Eval 模式：保持均匀采样，为了公平评估 PPL
            t = (1 - 1e-3) * torch.rand(B, device=device) + 1e-3
        
        # 2. 获取噪声参数
        sigma, dsigma = noise(t)
        
        # [数值稳定性优化]
        # 当 t -> 0 时，dsigma 会变得极大，可能导致 bfloat16 溢出
        # SFT 场景下截断到 50.0 通常是安全的
        if mode == 'train':
            dsigma = dsigma.clamp(max=50.0)
        
        # 3. 加噪
        perturbed_batch = graph.sample_transition(batch, sigma[:, None])
        
        # 4. 模型前向 (Score)
        # eval 模式下不需要梯度
        context = torch.enable_grad() if mode == 'train' else torch.no_grad()
        with context:
            log_score_fn = mutils.get_score_fn(model, train=(mode=='train'), sampling=False)
            log_score = log_score_fn(perturbed_batch, sigma)
        
        # 5. 计算 Entropy Loss (Per-Token)
        loss_per_token = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
        
        # 6. Apply SFT Mask
        masked_loss = loss_per_token * loss_mask
        
        # 7. 最终 Loss 计算
        if mode == 'train':
            # 训练目标：加权 Score Matching
            weighted_loss = dsigma[:, None] * masked_loss
            # 归一化：只除以有效 Token 数
            final_loss = weighted_loss.sum() / (loss_mask.sum() + 1e-6)
        else:
            # 评估目标：NLL (Negative Log-Likelihood) 近似
            # 不乘 dsigma，直接看 Entropy
            final_loss = masked_loss.sum() / (loss_mask.sum() + 1e-6)
            
        return final_loss

    return loss_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="预训练模型路径")
    parser.add_argument("--output_dir", type=str, default="sft_output")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    cfg = Config(args.model_path)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 开启 AMP
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using device: {cfg.device}, AMP dtype: {amp_dtype}")

    # --- 加载模型 ---
    print(f"Loading model from {args.model_path}...")
    model, graph, noise = load_model(args.model_path, cfg.device)

    # --- 注入 LoRA ---
    print("Injecting LoRA...")
    target_modules = ["attn_qkv", "attn_out", "mlp.0", "mlp.2"]
    
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05, 
        bias="none",
        target_modules=target_modules,
        # task_type="CAUSAL_LM" # [已删除] 避免兼容性报错
    )
    
    model = get_peft_model(model, peft_config)
  
# 强制确保所有 LoRA 参数 requires_grad=True
    has_trainable = False
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            has_trainable = True

    if not has_trainable:
    # 打印所有模块名，帮你找出正确的 target_modules
        print("❌ ERROR: No LoRA parameters created. Available modules:")
        for name, _ in model.named_modules():
            print(name)
        raise RuntimeError("LoRA targets not found. Check the printed module names above.")
    model.print_trainable_parameters()
    
    model.to(cfg.device)
    model.train()

    # --- 数据准备 ---
    train_loader, valid_loader = get_sft_dataloader(cfg)
    from transformers import GPT2TokenizerFast
    # 1. 手动实例化一个相同的 tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    # 2. 这里的 pad_token 设置必须与 data.py 里的逻辑一致
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    from data import debug_dataset_sample
    debug_dataset_sample(train_loader.dataset, tokenizer)  
    # --- 优化器 [精准剪裁准备] ---
    # 只传入 LoRA 参数
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.AdamW(
        trainable_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.eps
    )
    
    # [新增] 训练前自检：确保有参数需要训练
    print("-" * 30)
    print("Training Sanity Check:")
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_trainable}")
    
    if num_trainable == 0:
        raise RuntimeError("⚠️ 致命错误：可训练参数量为 0！请检查 target_modules 是否匹配模型结构。")
    
    # [新增] 数据自检：确保 loss_mask 不全为 0
    print("Checking first batch...")
    check_batch = next(iter(train_loader))
    mask_sum = check_batch['loss_mask'].sum()
    print(f"First batch effective tokens (mask sum): {mask_sum.item()}")
    
    if mask_sum == 0:
        raise RuntimeError("⚠️ 致命错误：第一个 Batch 的 Loss Mask 全为 0！模型没有训练目标。请检查 data.py 的字段读取逻辑。")
    
    print("Sanity check passed. Starting training loop...")
    print("-" * 30)
    
    # --- Loss 函数 (区分训练和验证) ---
    train_loss_fn = get_loss_fn(noise, graph, mode='train')
    eval_loss_fn = get_loss_fn(noise, graph, mode='eval')

    # --- 训练循环 ---
    print("Starting SFT...")
    
    global_step = 0
    micro_step = 0
    accum_steps = cfg.training.accum
    
    optimizer.zero_grad()
    
    pbar = tqdm(total=args.steps, desc="Training")
    train_iter = iter(train_loader)
    
    # 用于平滑显示 Loss
    running_loss = 0.0
    
    while global_step < args.steps:
        try:
            batch_data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch_data = next(train_iter)
            
        input_ids = batch_data['input_ids'].to(cfg.device)
        loss_mask = batch_data['loss_mask'].to(cfg.device)
        
        # [Training Forward]
        with torch.cuda.amp.autocast(dtype=amp_dtype):
            loss = train_loss_fn(model, input_ids, loss_mask)
            loss = loss / accum_steps # Normalize for accumulation
        
        if global_step == 0 and micro_step == 0:
            print(f"DEBUG: loss_mask sum = {loss_mask.sum().item()}")
            print(f"DEBUG: input_ids shape = {input_ids.shape}")
        
        # [Backward]
        scaler.scale(loss).backward()
        
        # 记录用于显示的 Loss (还原 accum)
        running_loss += loss.item() * accum_steps
        micro_step += 1
        
        # [Optimizer Step]
        if micro_step % accum_steps == 0:
            # Unscale & Clip
            scaler.unscale_(optimizer)
            
            # [精准剪裁] 只对可训练参数做 Clip
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            global_step += 1
            pbar.update(1)
            
            # 计算平滑 Loss
            avg_loss = running_loss / accum_steps
            running_loss = 0.0
            
            pbar.set_description(f"Step {global_step} | Train Loss: {avg_loss:.4f}")
            
            # [Evaluation Loop] 每 200 步验证一次
            if global_step % 200 == 0:
                model.eval()
                eval_loss = 0
                eval_steps = 0
                # 验证 20 个 batch 即可，不用跑完
                print("\nRunning Evaluation...")
                for i, val_batch in enumerate(valid_loader):
                    if i >= 20: break 
                    val_ids = val_batch['input_ids'].to(cfg.device)
                    val_mask = val_batch['loss_mask'].to(cfg.device)
                    
                    with torch.cuda.amp.autocast(dtype=amp_dtype):
                        # 使用 eval_loss_fn (无权重)
                        e_loss = eval_loss_fn(model, val_ids, val_mask)
                    eval_loss += e_loss.item()
                    eval_steps += 1
                
                avg_eval_loss = eval_loss / eval_steps
                print(f"Validation NLL: {avg_eval_loss:.4f}")
                model.train()
            
            # [Checkpoint]
            if global_step % 500 == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                model.save_pretrained(save_path)
                print(f"Saved checkpoint to {save_path}")

    pbar.close()
    print("Training finished!")
    
    final_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_path)
    print(f"Final model saved to {final_path}")

if __name__ == "__main__":
    main()