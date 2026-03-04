import torch
import torch.nn.functional as F
import os
import json
import argparse
import re
from tqdm import tqdm
from peft import PeftModel

# --- 引用你现有的项目模块 ---
from load_model import load_model 
from data import get_sft_dataloader
from model import utils as mutils
from sampling import get_pc_sampler # 用于生成测试

def get_step_from_path(path):
    """从路径中提取 step 数字，例如 'checkpoint-resume-1000' -> 1000"""
    match = re.search(r"checkpoint-(\d+)", path)
    if match:
        return int(match.group(1))
    if "final" in path:
        return "final"
    return "unknown"

def calculate_metrics_at_t(model, input_ids, loss_mask, t_val, graph, noise, device):
    """
    计算特定时间步 t 的 Score Entropy Loss
    """
    B = input_ids.shape[0]
    t = torch.full((B,), t_val, device=device)
    
    sigma, dsigma = noise(t)
    perturbed_batch = graph.sample_transition(input_ids, sigma[:, None])
    
    with torch.no_grad():
        log_score_fn = mutils.get_score_fn(model, train=True, sampling=False)
        log_score = log_score_fn(perturbed_batch, sigma)
    
    loss_per_token = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, input_ids)
    
    # 直接使用传入的 loss_mask
    final_loss = (loss_per_token * loss_mask).sum() / (loss_mask.sum() + 1e-6)
    return final_loss.item(), log_score

def generate_samples(model, graph, noise, batch, tokenizer, num_samples=5):
    """
    生成固定的样本用于人工评估 (Level 2 & 3)
    [修复]：使用 next(model.parameters()).device 替代 model.device
    """
    results = []
    model.eval()
    
    # [关键修复] 获取模型当前所在的设备
    device = next(model.parameters()).device
    
    # 只取前 num_samples 个
    input_ids = batch['input_ids'][:num_samples].to(device)
    loss_mask = batch['loss_mask'][:num_samples].to(device)
    
    B_gen = input_ids.shape[0]
    L = input_ids.shape[1]
    
    original_ids = input_ids.clone()
    
    def batch_projector(x):
        # x: [B, L]
        # loss_mask: [B, L] (0 for prompt, 1 for response)
        # 将 loss_mask 为 0 的位置还原为 original_ids
        mask = (loss_mask == 0).long()
        x = x * (1 - mask) + original_ids * mask
        return x

    # 构造 Sampler
    sampler = get_pc_sampler(
        graph=graph,
        noise=noise,
        batch_dims=(B_gen, L),
        predictor="analytic",
        steps=1024, 
        denoise=True,
        device=device, # [关键修复] 传入正确的 device
        proj_fun=batch_projector
    )
    
    # 设置随机种子以保证确定性
    torch.manual_seed(1234)
    
    print(f"Generating {B_gen} samples for qualitative eval...")
    with torch.no_grad():
        output_tokens = sampler(model)
        
    # 解码
    for i in range(B_gen):
        # 分离 Prompt 和 Response (简单按 mask 分离)
        prompt_len = (loss_mask[i] == 0).sum().item()
        prompt_text = tokenizer.decode(original_ids[i, :prompt_len], skip_special_tokens=True)
        
        # 截断 Response (去除 padding/eos)
        resp_ids = output_tokens[i, prompt_len:]
        
        # 处理 EOS 截断
        if tokenizer.eos_token_id in resp_ids:
            eos_idx = (resp_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0]
            resp_ids = resp_ids[:eos_idx]
            
        response_text = tokenizer.decode(resp_ids, skip_special_tokens=True)
        
        results.append({
            "prompt": prompt_text,
            "response_sft": response_text
        })
        
    return results

def main():
    parser = argparse.ArgumentParser()
    # 默认修改为 sedd-medium
    parser.add_argument("--base_model", type=str, default="louaaron/sedd-medium")
    parser.add_argument("--lora_path", type=str, required=True, help="SFT 训练后的 checkpoint 路径")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--num_batches", type=int, default=10, help="评估多少个 Batch")
    args = parser.parse_args()

    # 1. 初始化
    class Config:
        def __init__(self, base_model_name):
            self.device = torch.device("cuda")
            self.model = type('obj', (object,), {'length': 1024})
            self.training = type('obj', (object,), {'batch_size': 8, 'accum': 1}) # Eval Batch Size
            self.eval = type('obj', (object,), {'batch_size': 8})
            self.data = type('obj', (object,), {'train': "simplescaling/s1k-1.1"})
            self.base_model_name = base_model_name
            
    cfg = Config(args.base_model)
    os.makedirs(args.output_dir, exist_ok=True)
    step_num = get_step_from_path(args.lora_path)
    print(f">>> [Evaluation] Target: {args.base_model} | Step: {step_num}")

    # 2. 加载数据 (只加载验证集)
    print(">>> Loading Validation Data...")
    _, val_loader = get_sft_dataloader(cfg)
    
    # 3. 加载模型
    print(f">>> Loading Base Model: {args.base_model}...")
    model, graph, noise = load_model(args.base_model, cfg.device)
    
    # 加载 LoRA
    print(f">>> Loading LoRA Adapter: {args.lora_path}...")
    model = PeftModel.from_pretrained(model, args.lora_path)
    model.to(cfg.device)
    model.eval()

    # 4. 定义要扫描的时间步 t
    # Medium 模型对 t 的敏感度更高，我们增加细粒度扫描
    t_list = [0.01, 0.02, 0.05, 0.1, 0.3, 0.5, 0.8]
    metrics = {t: {"sft_loss": [], "base_loss": [], "cosine_sim": []} for t in t_list}

    print(">>> Starting Metric Evaluation...")
    
    data_iter = iter(val_loader)
    for _ in tqdm(range(args.num_batches), desc="Evaluating Batches"):
        try:
            batch_data = next(data_iter)
        except StopIteration:
            break
            
        # 显式提取两个 Tensor
        input_ids = batch_data['input_ids'].to(cfg.device)
        loss_mask = batch_data['loss_mask'].to(cfg.device)
        
        for t in t_list:
            # A. 计算 SFT 模型的表现 (传入 loss_mask)
            model.enable_adapter_layers()
            loss_sft, score_sft = calculate_metrics_at_t(
                model, input_ids, loss_mask, t, graph, noise, cfg.device
            )
            
            # B. 计算 Base 模型的表现
            with model.disable_adapter():
                loss_base, score_base = calculate_metrics_at_t(
                    model, input_ids, loss_mask, t, graph, noise, cfg.device
                )
            
            # C. 计算余弦相似度 (衡量 SFT 对原模型的改变程度)
            # Flatten 维度: [B, L, V] -> [B*L, V]
            cos_sim = F.cosine_similarity(score_sft.reshape(-1, score_sft.shape[-1]), 
                                          score_base.reshape(-1, score_base.shape[-1]), dim=-1)
            avg_cos = cos_sim.mean().item()

            # 记录
            metrics[t]["sft_loss"].append(loss_sft)
            metrics[t]["base_loss"].append(loss_base)
            metrics[t]["cosine_sim"].append(avg_cos)

    # 5. 汇总结果
    final_report = {
        "step": step_num,
        "base_model": args.base_model,
        "metrics_by_t": {}
    }

    print("\n=== Evaluation Report (SEDD-Medium) ===")
    for t in t_list:
        avg_sft = sum(metrics[t]["sft_loss"]) / len(metrics[t]["sft_loss"])
        avg_base = sum(metrics[t]["base_loss"]) / len(metrics[t]["base_loss"])
        avg_cos = sum(metrics[t]["cosine_sim"]) / len(metrics[t]["cosine_sim"])
        
        final_report["metrics_by_t"][str(t)] = {
            "loss_sft": avg_sft,
            "loss_base": avg_base,
            "gap": avg_sft - avg_base, # 负数越小越好，说明 SFT 比 Base 更确定
            "cosine_sim": avg_cos
        }
        
        # 打印关键指标 (t <= 0.1 是重点)
        if t <= 0.1:
            print(f"t={t:<4} | Loss: {avg_sft:.4f} (Base: {avg_base:.4f}) | CosSim: {avg_cos:.4f}")

    # 6. 生成一个定性测试案例 (Qualitative Sample)
    # 4. Level 2 & 3: Qualitative Sampling
    # 重新取一个 batch 做生成
    print("\nGenerating Samples for Qualitative Review...")
    _, val_loader = get_sft_dataloader(cfg)
    sample_batch = next(iter(val_loader))

    # 需要 Tokenizer 进行解码
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    samples = generate_samples(model, graph, noise, sample_batch, tokenizer, num_samples=4)
    final_report["qualitative_samples"] = samples
    
    # 5. 保存结果
    import re
    match = re.search(r"checkpoint-(\d+)", args.lora_path)
    step_num = match.group(1) if match else "unknown"
    output_name = f"eval_report_step_{step_num}.json"
    # 保存时使用这个带步数的文件名
    with open(output_name, 'w') as f:
        json.dump(final_report, f, indent=2)
    print(f"Report saved to {output_name}")

if __name__ == "__main__":
    main()