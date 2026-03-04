import torch
import json
import os
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from transformers import GPT2Tokenizer
from peft import PeftModel
from datetime import datetime

# --- 引入项目模块 ---
from load_model import load_model
from model import utils as mutils
from sampling import get_pc_sampler 

# ================= 配置区域 =================
class EvalConfig:
    # 模型路径配置
    BASE_MODEL_NAME = "louaaron/sedd-small"  # 或 medium
    SFT_ADAPTER_PATH = "sft_output_resume/checkpoint-resume-10000" # 你的 Adapter 路径
    
    # API 配置 (DeepSeek)
    API_KEY = "sk-a924a883817d47989743759e61386396" # 替换你的 Key
    BASE_URL = "https://api.deepseek.com" # 不要带 /v1 或 /chat
    
    # 推理参数
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAMPLING_STEPS = 1024 
    MAX_LENGTH = 1024

# 初始化 OpenAI 客户端
client = OpenAI(api_key=EvalConfig.API_KEY, base_url=EvalConfig.BASE_URL)

# ================= 1. 核心生成函数 =================
def generate_sedd(model, graph, noise, tokenizer, prompt, steps=512):
    """SEDD 采样推理核心逻辑"""
    model.eval()
    device = EvalConfig.DEVICE
    
    # 构造 Prompt
    full_prompt = f"User: {prompt}\nAssistant: "
    prefix_ids = tokenizer.encode(full_prompt)
    prefix_tensor = torch.tensor([prefix_ids], device=device)
    prefix_len = prefix_tensor.shape[1]
    
    # 初始化噪声和 Mask
    initial_samples = torch.randint(0, graph.dim, (1, EvalConfig.MAX_LENGTH), device=device)
    initial_samples[:, :prefix_len] = prefix_tensor
    sampling_mask = torch.ones_like(initial_samples)
    sampling_mask[:, :prefix_len] = 0 
    
    # 投影函数
    def batch_projector(x):
        return x * sampling_mask + initial_samples * (1 - sampling_mask)

    # 获取采样器
    sampler = get_pc_sampler(
        graph=graph, noise=noise, batch_dims=(1, EvalConfig.MAX_LENGTH),
        predictor="analytic", steps=steps, denoise=True, device=device,
        proj_fun=batch_projector
    )
    
    with torch.no_grad():
        # [关键修复] 只传 model
        final_samples = sampler(model)
        
    # 解码
    full_text = tokenizer.decode(final_samples[0], skip_special_tokens=True)
    if "Assistant:" in full_text:
        return full_text.split("Assistant:")[-1].strip()
    return full_text

# ================= 2. 趋势分析裁判 =================
def llm_judge_trend(prompt, ans_base, ans_sft):
    """让 AI 判断模型是变好了还是变差了"""
    system_prompt = """你是一个模型训练评估专家。请对比 Base 模型和 SFT 模型对同一问题的回答。
    即使两个模型都答错了，也要判断 SFT 是否在"格式"、"逻辑结构"或"去噪能力"上有提升。
    
    请返回 JSON:
    {
        "trend": "BETTER" (变好) | "WORSE" (变差/崩塌) | "SAME" (无区别),
        "reason": "简短的一句话评价，例如：SFT学会了使用LaTeX公式，尽管答案仍有误。",
        "score_sft": 0-10 (主观打分)
    }
    """
    
    user_prompt = f"问题: {prompt}\n\n[Base回答]:\n{ans_base[:800]}\n\n[SFT回答]:\n{ans_sft[:800]}"
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0, response_format={"type": "json_object"}, timeout=20
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Judge Error: {e}")
        return {"trend": "SAME", "reason": "API调用失败", "score_sft": 0}

# ================= 3. 生成美观 HTML 报告 =================
# ================= 3. 生成美观 HTML 报告 (修复版) =================
def generate_html_report(results, filename="sedd_arena_report.html"):
    # 注意：CSS 部分的大括号全部变成了双层 {{ }}，以避免 Python format 报错
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SEDD Training Arena Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; padding: 20px; }}
            h1 {{ text-align: center; color: #333; }}
            .summary {{ text-align: center; margin-bottom: 20px; font-size: 1.2em; }}
            table {{ width: 100%; border-collapse: collapse; background-color: white; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            th, td {{ border: 1px solid #ddd; padding: 15px; text-align: left; vertical-align: top; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .prompt {{ font-weight: bold; color: #555; width: 15%; }}
            .base-col {{ width: 35%; color: #666; font-family: monospace; white-space: pre-wrap; font-size: 0.9em; }}
            .sft-col {{ width: 35%; font-family: monospace; white-space: pre-wrap; font-size: 0.9em; }}
            .judge-col {{ width: 15%; }}
            
            /* 状态颜色 */
            .status-better {{ background-color: #d4edda; border-left: 5px solid #28a745; }}
            .status-worse {{ background-color: #f8d7da; border-left: 5px solid #dc3545; }}
            .status-same {{ background-color: #fff3cd; border-left: 5px solid #ffc107; }}
            
            .badge {{ display: inline-block; padding: 5px 10px; border-radius: 4px; color: white; font-size: 0.8em; font-weight: bold; }}
            .badge-better {{ background-color: #28a745; }}
            .badge-worse {{ background-color: #dc3545; }}
            .badge-same {{ background-color: #ffc107; color: #333; }}
        </style>
    </head>
    <body>
        <h1>SEDD Model Evaluation Arena</h1>
        <div class="summary">
            Generated on: {timestamp} <br>
            Base Model: {base_name} | Adapter: {adapter_path}
        </div>
        <table>
            <tr>
                <th>Prompt (User)</th>
                <th>Base Model Output</th>
                <th>SFT Model Output</th>
                <th>AI Judge Analysis</th>
            </tr>
            {rows}
        </table>
    </body>
    </html>
    """
    
    rows_html = ""
    for res in results:
        trend = res.get('trend', 'SAME') # 防止 Key 不存在
        # 简单的安全映射，防止 trend 是其他字符串导致 CSS 失效
        safe_trend = trend.lower() if trend in ["BETTER", "WORSE", "SAME"] else "same"
        
        css_class = f"status-{safe_trend}"
        badge_class = f"badge-{safe_trend}"
        
        # 处理可能的 None 值
        base_ans = res.get('base_ans', '') or ""
        sft_ans = res.get('sft_ans', '') or ""
        reason = res.get('reason', '') or ""
        score = res.get('score_sft', 0)
        
        # 将换行符转换为 HTML <br> 以便在网页正确显示
        base_ans = base_ans.replace("\n", "<br>")
        sft_ans = sft_ans.replace("\n", "<br>")
        
        rows_html += f"""
        <tr>
            <td class="prompt">{res['prompt']}</td>
            <td class="base-col">{base_ans}</td>
            <td class="sft-col {css_class}">{sft_ans}</td>
            <td class="judge-col">
                <span class="badge {badge_class}">{trend}</span>
                <br><br>
                <b>Score:</b> {score}/10
                <br><br>
                <i>"{reason}"</i>
            </td>
        </tr>
        """
        
    final_html = html_template.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        base_name=EvalConfig.BASE_MODEL_NAME,
        adapter_path=EvalConfig.SFT_ADAPTER_PATH,
        rows=rows_html
    )
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(final_html)
    print(f"\n✅ HTML Report saved to: {filename}")

# ================= 4. 主执行流程 =================
def run_arena():
    # 1. 准备测试集 (10 个案例)
    test_prompts = [
        "Calculate 15 + 27 and explain the steps.",
        "Let $x + y = 10$ and $x - y = 2$. Find the value of $xy$.",
        "Solve the equation: $2x + 5 = 15$.",
        "What is the derivative of $f(x) = x^2$?",
        "Simplify the fraction $\\frac{12}{16}$.",
        "If a circle has a radius of 3, what is its area?",
        "Translate the following logic to math: 'A number plus five equals ten'.",
        "Write the Pythagorean theorem formula.",
        "Is 17 a prime number? Why?",
        "Calculate $3^3 + 4^2$."
    ]

    # 2. 加载 Tokenizer
    print(">>> Loading Tokenizer...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    except:
        tokenizer = GPT2Tokenizer.from_pretrained(EvalConfig.BASE_MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 3. 加载模型 (Load Once)
    print(">>> Loading Model & Adapter...")
    base_model, graph, noise = load_model(EvalConfig.BASE_MODEL_NAME, EvalConfig.DEVICE)
    model = PeftModel.from_pretrained(base_model, EvalConfig.SFT_ADAPTER_PATH)
    model.to(EvalConfig.DEVICE)

    results = []
    print(f">>> Starting Evaluation on {len(test_prompts)} prompts...")

    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Processing Case {i+1}/{len(test_prompts)} ---")
        
        # A. Base 生成
        print("  Generating Base...")
        with model.disable_adapter():
            ans_base = generate_sedd(model, graph, noise, tokenizer, prompt, steps=EvalConfig.SAMPLING_STEPS)
            
        # B. SFT 生成
        print("  Generating SFT...")
        model.enable_adapter_layers()
        ans_sft = generate_sedd(model, graph, noise, tokenizer, prompt, steps=EvalConfig.SAMPLING_STEPS)
        
        # C. 裁判打分
        print("  Judging...")
        judge_res = llm_judge_trend(prompt, ans_base, ans_sft)
        
        results.append({
            "prompt": prompt,
            "base_ans": ans_base,
            "sft_ans": ans_sft,
            "trend": judge_res.get("trend", "SAME"),
            "reason": judge_res.get("reason", "N/A"),
            "score_sft": judge_res.get("score_sft", 0)
        })

    # 4. 导出报告
    generate_html_report(results, filename="sedd_arena_report.html")

if __name__ == "__main__":
    run_arena()