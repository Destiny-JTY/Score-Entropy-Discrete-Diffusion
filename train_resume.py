import torch
import torch.optim as optim
import os
import argparse
import glob
import re
from tqdm import tqdm

# --- 核心依赖 (请确保这些文件在当前目录下) ---
from load_model import load_model  
from data import get_sft_dataloader
from model import utils as mutils
from peft import PeftModel 

def get_latest_checkpoint(output_dir, initial_lora):
    """
    智能路径寻找逻辑
    """
    # 1. 寻找 checkpoint-resume-xxxx 文件夹
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-resume-*"))
    if checkpoints:
        def extract_step(path):
            match = re.search(r"checkpoint-resume-(\d+)", path)
            return int(match.group(1)) if match else 0
        latest_path = max(checkpoints, key=extract_step)
        return latest_path, extract_step(latest_path)
    
    # 2. 如果没有 checkpoint 文件夹，看看有没有上次运行完生成的 final 文件夹
    resumed_final = os.path.join(output_dir, "final_model_resumed")
    if os.path.exists(resumed_final):
        print(f">>> Found previously completed final model in {output_dir}")
        return resumed_final, 2000 # 假设你之前跑了2000步
    
    # 3. 兜底：使用最初的 final model
    print(f">>> No progress found in {output_dir}, falling back to initial: {initial_lora}")
    return initial_lora, 0

class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = type('obj', (object,), {'length': 1024})
        self.training = type('obj', (object,), {'batch_size': 8, 'accum': 2}) 
        self.eval = type('obj', (object,), {'batch_size': 8})
        self.optim = type('obj', (object,), {
            'lr': 3e-4, 
            'weight_decay': 0.01,
            'beta1': 0.9, 
            'beta2': 0.999, 
            'eps': 1e-8
        })
        self.data = type('obj', (object,), {'train': "simplescaling/s1k-1.1"})

def get_loss_fn(noise, graph, mode='train'):
    def loss_fn(model, batch, loss_mask):
        B = batch.shape[0]
        device = batch.device
        u = torch.rand(B, device=device)
        t = (1 - 1e-3) * (u ** 2) + 1e-3 
        sigma, dsigma = noise(t)
        dsigma = dsigma.clamp(max=50.0)
        
        perturbed_batch = graph.sample_transition(batch, sigma[:, None])
        log_score_fn = mutils.get_score_fn(model, train=True, sampling=False)
        log_score = log_score_fn(perturbed_batch, sigma)
        
        loss_per_token = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
        masked_loss = loss_per_token * loss_mask
        final_loss = (dsigma[:, None] * masked_loss).sum() / (loss_mask.sum() + 1e-6)
        return final_loss
    return loss_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="louaaron/sedd-small")
    parser.add_argument("--initial_lora", type=str, default="sft_output/final_model")
    parser.add_argument("--output_dir", type=str, default="sft_output_resume")
    parser.add_argument("--total_steps", type=int, default=5000, help="目标累计总步数")
    args = parser.parse_args()

    cfg = Config()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # 1. 加载基座
    model, graph, noise = load_model(args.base_model, cfg.device)

    # 2. 核心：寻找最先进度
    resume_path, start_step = get_latest_checkpoint(args.output_dir, args.initial_lora)
    print(f">>> LOADING WEIGHTS FROM: {resume_path}")
    print(f">>> STARTING FROM GLOBAL STEP: {start_step}")

    # 加载已有的 LoRA 权重
    model = PeftModel.from_pretrained(model, resume_path, is_trainable=True)
    model.to(cfg.device)
    
    # 确保参数可训练
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    # 3. 数据与优化器
    train_loader, _ = get_sft_dataloader(cfg)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.optim.lr)
    train_loss_fn = get_loss_fn(noise, graph)
    
    # 4. 训练循环
    if start_step >= args.total_steps:
        print(f"Goal {args.total_steps} already reached. Exit.")
        return

    global_step = start_step
    micro_step = 0
    accum_steps = cfg.training.accum
    optimizer.zero_grad()
    
    pbar = tqdm(total=args.total_steps, initial=start_step, desc="SFT 接力训练")
    train_iter = iter(train_loader)
    
    model.train()
    while global_step < args.total_steps:
        try:
            batch_data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch_data = next(train_iter)
            
        input_ids = batch_data['input_ids'].to(cfg.device)
        loss_mask = batch_data['loss_mask'].to(cfg.device)
        
        with torch.cuda.amp.autocast(dtype=amp_dtype):
            loss = train_loss_fn(model, input_ids, loss_mask)
            loss_scaled = loss / accum_steps
        
        scaler.scale(loss_scaled).backward()
        micro_step += 1
        
        if micro_step % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            global_step += 1
            pbar.update(1)
            pbar.set_description(f"Step {global_step} | Loss: {loss.item():.4f}")
            
            # 每 500 步保存一次
            if global_step % 500 == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-resume-{global_step}")
                model.save_pretrained(save_path)

    # 最终保存
    final_save_path = os.path.join(args.output_dir, "final_model_resumed")
    model.save_pretrained(final_save_path)
    print(f"训练完成！最终模型保存在: {final_save_path}")

if __name__ == "__main__":
    main()