import torch
import torch.nn.functional as F
import argparse
import json
import os
from tqdm import tqdm
from peft import PeftModel

# 项目依赖
from load_model import load_model
from data import get_sft_dataloader
from model import utils as mutils
from sampling import get_pc_sampler # 确保你有 sampling.py

# --- Level 1 & 5: 固定 t 的 Score Loss ---
def calc_fixed_t_loss(model, graph, noise, batch, loss_mask, t_val):
    """
    计算特定 t 下的 Conditional Score Loss (Level 1 & 5)
    只关注 Response 部分 (loss_mask)
    """
    B = batch.shape[0]
    device = batch.device
    
    # 1. 固定 t (广播到 Batch)
    t = torch.full((B,), t_val, device=device)
    
    # 2. 加噪
    sigma, dsigma = noise(t)
    perturbed_batch = graph.sample_transition(batch, sigma[:, None])
    
    # 3. 计算 Score (Eval 模式，不乘 dsigma)
    with torch.no_grad():
        log_score_fn = mutils.get_score_fn(model, train=False, sampling=False)
        log_score = log_score_fn(perturbed_batch, sigma)
    
    # 4. 计算 Entropy (Loss)
    loss_per_token = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
    
    # 5. Masking (只看 Response)
    masked_loss = loss_per_token * loss_mask
    
    # 6. Normalize
    # 避免除以 0
    num_tokens = loss_mask.sum() + 1e-6
    avg_loss = masked_loss.sum() / num_tokens
    
    return avg_loss.item(), num_tokens.item()

# --- Level 4: Score Field Cosine Similarity ---
def calc_score_cosine(model, graph, noise, batch, loss_mask, t_val=0.02):
    """
    计算 SFT 前后 Score Field 的余弦相似度 (Level 4)
    用于检测 SFT 是否真正改变了模型的去噪方向
    """
    B = batch.shape[0]
    device = batch.device
    t = torch.full((B,), t_val, device=device)
    sigma, _ = noise(t)
    perturbed_batch = graph.sample_transition(batch, sigma[:, None]) # 同一扰动
    
    # 1. 获取 SFT 后的 Score
    with torch.no_grad():
        score_fn_sft = mutils.get_score_fn(model, train=False, sampling=False)
        # log_score: [B, L, Vocab]
        log_score_sft = score_fn_sft(perturbed_batch, sigma)
    
    # 2. 获取 Base 模型的 Score (临时禁用 LoRA)
    with torch.no_grad():
        with model.disable_adapter(): # PEFT 提供的黑魔法
            score_fn_base = mutils.get_score_fn(model, train=False, sampling=False)
            log_score_base = score_fn_base(perturbed_batch, sigma)
            
    # 3. 计算 Cosine Similarity
    # 我们只关心 Response 部分的 Vector 方向变化
    # Flatten 到 [Total_Valid_Tokens, Vocab]
    mask_bool = loss_mask.bool()
    
    vec_sft = log_score_sft[mask_bool]   # [N, V]
    vec_base = log_score_base[mask_bool] # [N, V]
    
    # 计算余弦相似度
    cosine = F.cosine_similarity(vec_sft, vec_base, dim=-1)
    
    return cosine.mean().item()

# --- Level 2: Deterministic Sampling ---
# --- 修正后的 generate_samples 函数 ---
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

# --- Main Evaluation Script ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="louaaron/sedd-medium")
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="eval_report.json")
    parser.add_argument("--num_batches", type=int, default=20, help="评估多少个 Batch")
    args = parser.parse_args()
    
    # 1. 配置与加载
    class Config:
        def __init__(self):
            self.device = torch.device("cuda")
            self.model = type('obj', (object,), {'length': 1024})
            self.training = type('obj', (object,), {'batch_size': 8, 'accum': 1}) # Eval Batch Size
            self.eval = type('obj', (object,), {'batch_size': 8})
            self.data = type('obj', (object,), {'train': "simplescaling/s1k-1.1"})
            
    cfg = Config()
    
    print("Loading Model...")
    model, graph, noise = load_model(args.base_model, cfg.device)
    model = PeftModel.from_pretrained(model, args.lora_path)
    
   # print("-" * 30)
# 遍历并检查 LoRA A 和 B 矩阵的数值
    #has_data = False
    #for name, param in model.named_parameters():
        #if "lora_A" in name or "lora_B" in name:
            #weight_sum = param.abs().sum().item()
            #print(f"LoRA Weight [{name}] sum of abs: {weight_sum:.8f}")
            #if weight_sum > 1e-5:
                #has_data = True

    #if not has_data:
        #print("⚠️ 警告：LoRA 权重几乎为 0！说明训练过程中没有发生有效的梯度更新。")
    #else:
        #print("✅ 确认：LoRA 权重包含有效数值。")
    #print("-" * 30)
    model.to(cfg.device)
    model.eval()
    
    print("Loading Validation Data...")
    _, valid_loader = get_sft_dataloader(cfg)
    
    # 2. 定义评估指标容器
    # Level 5: Noise Sensitivity Curve (Loss vs t)
    t_list = [0.01, 0.02, 0.05, 0.1, 0.5, 0.7, 0.9]
    metrics = {f"loss_t_{t}": [] for t in t_list}
    metrics["cosine_sim_t_0.02"] = [] # Level 4
    
    # 3. 循环计算指标
    print("Running Quantitative Evaluation (Level 1, 4, 5)...")
    iterator = iter(valid_loader)
    
    for _ in tqdm(range(args.num_batches)):
        try:
            batch = next(iterator)
        except StopIteration:
            break
            
        input_ids = batch['input_ids'].to(cfg.device)
        loss_mask = batch['loss_mask'].to(cfg.device)
        
        # Level 1 & 5: Calculate Loss at different t
        for t_val in t_list:
            loss, _ = calc_fixed_t_loss(model, graph, noise, input_ids, loss_mask, t_val)
            metrics[f"loss_t_{t_val}"].append(loss)
            
        # Level 4: Calculate Score Cosine at t=0.02 (Low noise region)
        cos_sim = calc_score_cosine(model, graph, noise, input_ids, loss_mask, t_val=0.02)
        metrics["cosine_sim_t_0.02"].append(cos_sim)
        
    # 汇总平均值
    final_report = {}
    for k, v in metrics.items():
        final_report[k] = sum(v) / len(v)
        
    print("\n" + "="*40)
    print("EVALUATION REPORT")
    print("="*40)
    print(f"Level 1: Low-Noise Score Loss (t=0.02): {final_report['loss_t_0.02']:.4f}")
    print(f"Level 4: Score Field Cosine (t=0.02):   {final_report['cosine_sim_t_0.02']:.4f}")
    print("-" * 40)
    print("Level 5: Noise Sensitivity Curve:")
    for t in t_list:
        print(f"  t={t:<4}: {final_report[f'loss_t_{t}']:.4f}")
    print("="*40)
    
    # 4. Level 2 & 3: Qualitative Sampling
    # 重新取一个 batch 做生成
    print("\nGenerating Samples for Qualitative Review...")
    sample_batch = next(iter(valid_loader)) # 取第一个 batch
    
    # 需要 Tokenizer 进行解码
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    samples = generate_samples(model, graph, noise, sample_batch, tokenizer, num_samples=4)
    final_report["qualitative_samples"] = samples
    
    # 5. 保存结果
    import re
    match = re.search(r"checkpoint-(\d+)", args.lora_path)
    step_num = match.group(1) if match else "unknown"
    output_name = f"eval_report{step_num}.json"
    # 保存时使用这个带步数的文件名
    with open(output_name, 'w') as f:
        json.dump(final_report, f, indent=2)
    print(f"Report saved to {output_name}")

if __name__ == "__main__":
    main()