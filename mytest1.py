import json
import requests
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import re
import time

# ==================== 配置部分 ====================
API_URL = "http://192.168.110.45:1234/v1/completions"
DATA_PATH = "D:/BABABA/ppyyppyy/pythonProject/result5.json"
OUTPUT_FILE = "raw_responses.txt"
PLOT_FILE = "raw_model_performance.png"


# ==================== 数据加载 ====================
def load_data():
    """加载测试数据集"""
    try:
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            cases = json.load(f)
        print(f"✅ 成功加载 {len(cases)} 个病例")
        return cases
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return []


# ==================== 优化提示词构建 ====================
def build_prompt(case):
    """构建明确指令的提示词"""
    return f"""
[系统指令]
你是一个专业的泌尿放射科AI助手，专门进行膀胱肿瘤VI-RADS评分。必须严格按以下要求执行：
1. 只根据提供的影像特征评分
2. 只输出1个1-5的数字
3. 禁止解释或添加其他文字

[评分标准]
1分：低信号线完整 + 无强化
2分：低信号线完整 + 轻度强化  
3分：不确定
4分：低信号线中断 + 明显强化
5分：侵犯肌层外 + 强强化

[病例特征]
位置：{case['位置']}
大小：{case['大小']}
浸润：{case['浸润深度']}
T2WI：{case['T2WI']}
DWI：{case['DWI']}
DCE：{case['DCE']}

[输出]
VI-RADS评分：
"""


# ==================== 模型测试 ====================
def test_optimized_model(cases):
    """使用优化提示词测试模型性能"""
    results = []
    print("🔍 开始优化模型测试...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for i, case in enumerate(cases):
            # 使用优化提示词
            prompt = build_prompt(case)

            # 调用模型
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "prompt": prompt,
                        "max_tokens": 3,
                        "temperature": 0.1,
                        "frequency_penalty": 1.0,
                        "presence_penalty": 1.0,
                        "top_p": 0.9,
                        "stop": ["\n", "。", "！", "分", "评分", "VI-RADS"]
                    },
                    timeout=30
                )

                # 提取评分
                raw_output = response.json()["choices"][0]["text"].strip()
                # 使用正则表达式提取数字
                numbers = re.findall(r'\d+', raw_output)
                pred_score = int(numbers[0]) if numbers else None

            except Exception as e:
                print(f"❌ 病例 {i + 1} 测试失败: {e}")
                pred_score = None
                raw_output = f"错误: {e}"

            # 保存结果
            result = {
                "case_id": i + 1,
                "true_score": case["评分"],
                "pred_score": pred_score,
                "raw_output": raw_output,
                "is_correct": pred_score == case["评分"] if pred_score is not None else False,
                "is_near": abs(pred_score - case["评分"]) <= 1 if pred_score is not None else False
            }
            results.append(result)

            # 写入文件
            f.write(f"病例 {result['case_id']}:\n")
            f.write(f"  真实评分: {result['true_score']}\n")
            f.write(f"  预测评分: {result['pred_score']}\n")
            f.write(f"  模型输出: {repr(result['raw_output'])}\n")
            f.write(f"  是否正确: {'✅' if result['is_correct'] else '❌'}\n")
            f.write("-" * 50 + "\n")

            # 进度显示
            if (i + 1) % 10 == 0:
                print(f"  已完成 {i + 1}/{len(cases)} 个病例")
                time.sleep(0.5)  # 避免请求过快

    return results


# ==================== 计算指标 ====================
def calculate_metrics(results):
    """计算准确率指标"""
    valid_results = [r for r in results if r['pred_score'] is not None]

    if not valid_results:
        return {"error": "无有效结果"}

    n_valid = len(valid_results)

    # 精确准确率
    accuracy = sum(1 for r in valid_results if r['is_correct']) / n_valid

    # 邻近准确率（±1分算正确）
    near_accuracy = sum(1 for r in valid_results if r['is_near']) / n_valid

    # 95%置信区间
    accuracy_ci = stats.norm.interval(
        0.95,
        loc=accuracy,
        scale=np.sqrt(accuracy * (1 - accuracy) / n_valid)
    )

    near_ci = stats.norm.interval(
        0.95,
        loc=near_accuracy,
        scale=np.sqrt(near_accuracy * (1 - near_accuracy) / n_valid)
    )

    return {
        "n_total": len(results),
        "n_valid": n_valid,
        "accuracy": accuracy,
        "near_accuracy": near_accuracy,
        "accuracy_ci": accuracy_ci,
        "near_ci": near_ci,
        "valid_rate": n_valid / len(results)
    }


# ==================== 可视化 ====================
def plot_results(metrics):
    """绘制结果图表"""
    plt.figure(figsize=(12, 8))

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    # 数据准备
    labels = ['精确准确率\n(完全匹配)', '邻近准确率\n(相差≤1分)']
    values = [metrics['accuracy'], metrics['near_accuracy']]
    ci_errors = [
        values[0] - metrics['accuracy_ci'][0],
        values[1] - metrics['near_ci'][0]
    ]

    # 绘制柱状图
    colors = ['#3498db', '#2ecc71']
    bars = plt.bar(labels, values, yerr=ci_errors, capsize=15,
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1)

    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{value:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 图表装饰
    plt.title('MedGemma-4B 优化提示词性能评估\n(明确指令 + 上下文隔离)', fontsize=16, pad=20)
    plt.ylabel('准确率', fontsize=14)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加统计信息
    stats_text = f"""
统计信息：
• 总病例数: {metrics['n_total']}
• 有效测试: {metrics['n_valid']} ({metrics['valid_rate']:.1%})
• 精确准确率: {metrics['accuracy']:.1%} (95%CI: {metrics['accuracy_ci'][0]:.3f}-{metrics['accuracy_ci'][1]:.3f})
• 邻近准确率: {metrics['near_accuracy']:.1%} (95%CI: {metrics['near_ci'][0]:.3f}-{metrics['near_ci'][1]:.3f})

说明：
• 精确准确率：预测评分与真实评分完全一致
• 邻近准确率：预测评分与真实评分相差不超过1分
• 误差线表示95%置信区间范围
"""

    plt.figtext(0.5, -0.25, stats_text, ha='center', va='top', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5))

    plt.tight_layout()
    plt.savefig(PLOT_FILE, bbox_inches='tight', dpi=300)
    plt.show()


# ==================== 主执行流程 ====================
def main():
    """主函数"""
    print("=" * 60)
    print("MedGemma-4B 优化提示词性能测试")
    print("=" * 60)

    # 1. 加载数据
    cases = load_data()
    if not cases:
        return

    # 2. 测试模型（使用优化提示词）
    results = test_optimized_model(cases)

    # 3. 计算指标
    print("\n📊 计算性能指标...")
    metrics = calculate_metrics(results)

    if "error" in metrics:
        print("❌ 无法计算指标: 无有效结果")
        return

    # 4. 显示结果
    print("\n" + "=" * 60)
    print("📈 测试结果汇总")
    print("=" * 60)
    print(f"总病例数: {metrics['n_total']}")
    print(f"有效测试: {metrics['n_valid']} ({metrics['valid_rate']:.1%})")
    print(f"精确准确率: {metrics['accuracy']:.2%}")
    print(f"邻近准确率: {metrics['near_accuracy']:.2%}")
    print(f"95%置信区间 - 精确: [{metrics['accuracy_ci'][0]:.3f}, {metrics['accuracy_ci'][1]:.3f}]")
    print(f"95%置信区间 - 邻近: [{metrics['near_ci'][0]:.3f}, {metrics['near_ci'][1]:.3f}]")

    # 5. 生成可视化
    print("\n📈 生成可视化图表...")
    plot_results(metrics)

    print("\n🎉 测试完成！")
    print(f"  详细结果: {OUTPUT_FILE}")
    print(f"  性能图表: {PLOT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()