import json
import requests
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import re
import time
from typing import List, Dict, Any

# ==================== 配置部分 ====================
API_URL = "http://192.168.110.45:1234/v1/completions"
DATA_PATH = "D:/BABABA/ppyyppyy/pythonProject/result5.json"
OUTPUT_FILE = "strict_prompt_responses.txt"
PLOT_FILE = "strict_prompt_performance.png"
REQUEST_TIMEOUT = 30
DELAY_BETWEEN_REQUESTS = 0.5


# ==================== 严格提示词模板 ====================
def build_strict_prompt(case: Dict) -> str:
    """构建严格完整的提示词（每个病例都包含完整规则）"""
    return f"""
你好，接下来我将会给你一些规则和膀胱癌肿瘤的病例，你来根据这些规则对这些肿瘤进行VI-RADS评分。注意评分之后只需要给我输出一个数字，不用解释你的推理过程，我只需要一个数字可以吗？

【VI-RADS评分规则】

【步骤1：T2WI评估】
特征：{case['T2WI']}
判断：低信号线是否完整？病变大小？肿瘤形态？
SC类别：1-5（1=完整+<1cm，2=完整+>1cm，3=不确定，4=中断，5=膀胱外）

【步骤2：DWI评估】
特征：{case['DWI']}
判断：信号特点？扩散受限范围？
DW类别：1-5（1=中等信号，2=中等信号+大病变，3=不确定，4=高信号+局灶侵犯，5=高信号+全层侵犯）

【步骤3：DCE评估】
特征：{case['DCE']}
判断：强化模式？强化范围？
CE类别：1-5（1=无肌层强化，2=内层强化，3=不确定，4=局灶肌层强化，5=全层强化）

【步骤4：综合评分】
根据SC/DW/CE类别组合确定VI-RADS评分：
1分：SC1+DW1+CE1
2分：SC2-3+(DW2或CE2)
3分：SC3+DW3+CE3
4分：SC3-5+(DW4或CE4)
5分：SC4-5+(DW5或CE5)

以上是规则 以下是病例
患者：男，{case.get('年龄', '未知')}岁
位置：{case['位置']}
大小：{case['大小']}
带蒂：{case['带蒂']}
浸润深度：{case['浸润深度']}
T2WI：{case['T2WI']}
DWI：{case['DWI']}
DCE：{case['DCE']}

请根据规则对这个病例进行打分，然后告诉我你的打分。
输出一个数字即可，不用解释为什么。
除了你的评分（一个数字），其它什么都不要回答。

"""


# ==================== 模型测试 ====================
def test_strict_prompt_model(cases: List[Dict]) -> List[Dict]:
    """使用严格提示词测试模型性能"""
    results = []
    print("🔍 开始严格提示词测试（每个病例完整规则）...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for i, case in enumerate(cases):
            # 为每个病例构建完整提示词
            prompt = build_strict_prompt(case)

            try:
                response = requests.post(
                    API_URL,
                    json={
                        "prompt": prompt,
                        "max_new_tokens": 3,  # 严格控制输出长度
                        "temperature": 0.01,  # 极低随机性
                        "do_sample": False,  # 禁用随机采样
                        "top_p": 0.9,
                        "repetition_penalty": 2.0,  # 强力防止重复
                        "frequency_penalty": 1.5,  # 惩罚频率
                        "presence_penalty": 1.5,  # 惩罚出现
                        "stop": ["\n", "。", "！", "分", "评分", "VI-RADS", "输出", "回答"],
                        "seed": 42  # 固定随机种子
                    },
                    timeout=REQUEST_TIMEOUT
                )

                # 严格提取评分数字
                raw_output = response.json()["choices"][0]["text"].strip()
                numbers = re.findall(r'[1-5]', raw_output)
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

            # 写入详细记录
            f.write(f"病例 {result['case_id']}:\n")
            f.write(f"  真实评分: {result['true_score']}\n")
            f.write(f"  预测评分: {result['pred_score']}\n")
            f.write(f"  模型输出: {repr(result['raw_output'])}\n")
            f.write(f"  是否正确: {'✅' if result['is_correct'] else '❌'}\n")
            f.write("-" * 50 + "\n")

            # 进度显示
            if (i + 1) % 10 == 0:
                print(f"  已完成 {i + 1}/{len(cases)} 个病例")
                time.sleep(DELAY_BETWEEN_REQUESTS)

    return results


# ==================== 统计分析 ====================
def calculate_metrics(results: List[Dict]) -> Dict:
    """计算性能指标"""
    valid_results = [r for r in results if r['pred_score'] is not None]

    if not valid_results:
        return {"error": "无有效结果"}

    n_valid = len(valid_results)

    # 精确准确率
    accuracy = sum(1 for r in valid_results if r['is_correct']) / n_valid

    # 邻近准确率（±1分算正确）
    near_accuracy = sum(1 for r in valid_results if r['is_near']) / n_valid

    # 95%置信区间（Wilson方法）
    def wilson_ci(p, n, z=1.96):
        denominator = 1 + z ** 2 / n
        center = (p + z ** 2 / (2 * n)) / denominator
        width = z * np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / denominator
        return (center - width, center + width)

    accuracy_ci = wilson_ci(accuracy, n_valid)
    near_ci = wilson_ci(near_accuracy, n_valid)

    # 错误分析
    error_analysis = {
        "total_errors": sum(1 for r in valid_results if not r['is_correct']),
        "major_errors": sum(1 for r in valid_results if not r['is_near']),
        "minor_errors": sum(1 for r in valid_results if not r['is_correct'] and r['is_near'])
    }

    return {
        "n_total": len(results),
        "n_valid": n_valid,
        "accuracy": accuracy,
        "near_accuracy": near_accuracy,
        "accuracy_ci": accuracy_ci,
        "near_ci": near_ci,
        "valid_rate": n_valid / len(results),
        "error_analysis": error_analysis
    }


# ==================== 可视化 ====================
def plot_results(metrics: Dict):
    """绘制结果图表"""
    plt.figure(figsize=(15, 10))
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 准确率对比
    plt.subplot(2, 2, 1)
    labels = ['精确准确率\n(完全匹配)', '邻近准确率\n(相差≤1分)']
    values = [metrics['accuracy'], metrics['near_accuracy']]
    ci_errors = [values[0] - metrics['accuracy_ci'][0], values[1] - metrics['near_ci'][0]]

    colors = ['#3498db', '#2ecc71']
    bars = plt.bar(labels, values, yerr=ci_errors, capsize=15, color=colors, alpha=0.8)

    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{value:.1%}',
                 ha='center', va='bottom', fontweight='bold')

    plt.title('严格提示词性能评估', fontsize=14, pad=20)
    plt.ylabel('准确率')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)

    # 2. 错误分析
    plt.subplot(2, 2, 2)
    error_labels = ['严重错误\n(相差>1分)', '轻微错误\n(相差1分)', '正确预测']
    error_values = [
        metrics['error_analysis']['major_errors'],
        metrics['error_analysis']['minor_errors'],
        metrics['n_valid'] - metrics['error_analysis']['total_errors']
    ]
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    plt.pie(error_values, labels=error_labels, colors=colors, autopct='%1.1f%%')
    plt.title('错误类型分布')

    # 3. 置信区间
    plt.subplot(2, 2, 3)
    plt.errorbar([1, 2], [metrics['accuracy'], metrics['near_accuracy']],
                 yerr=[[metrics['accuracy'] - metrics['accuracy_ci'][0],
                        metrics['near_accuracy'] - metrics['near_ci'][0]],
                       [metrics['accuracy_ci'][1] - metrics['accuracy'],
                        metrics['near_ci'][1] - metrics['near_accuracy']]],
                 fmt='o', capsize=5)
    plt.xticks([1, 2], ['精确准确率', '邻近准确率'])
    plt.title('95%置信区间')
    plt.grid(True, alpha=0.3)

    # 4. 样本有效性
    plt.subplot(2, 2, 4)
    valid_labels = ['有效响应', '无效响应']
    valid_values = [metrics['n_valid'], metrics['n_total'] - metrics['n_valid']]
    plt.pie(valid_values, labels=valid_labels, autopct='%1.1f%%',
            colors=['#27ae60', '#e74c3c'])
    plt.title('响应有效性分析')

    # 统计信息
    stats_text = f"""
统计摘要（总病例数: {metrics['n_total']}）
• 有效测试: {metrics['n_valid']} ({metrics['valid_rate']:.1%})
• 精确准确率: {metrics['accuracy']:.1%} (95%CI: {metrics['accuracy_ci'][0]:.3f}-{metrics['accuracy_ci'][1]:.3f})
• 邻近准确率: {metrics['near_accuracy']:.1%} (95%CI: {metrics['near_ci'][0]:.3f}-{metrics['near_ci'][1]:.3f})
• 严重错误: {metrics['error_analysis']['major_errors']}例
• 轻微错误: {metrics['error_analysis']['minor_errors']}例
"""

    plt.figtext(0.5, -0.1, stats_text, ha='center', fontsize=10,
                bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5))

    plt.tight_layout()
    plt.savefig(PLOT_FILE, bbox_inches='tight', dpi=300)
    plt.show()


# ==================== 主函数 ====================
def main():
    """主执行流程"""
    print("=" * 60)
    print("VI-RADS评分系统 - 严格提示词版本")
    print("=" * 60)

    # 1. 加载数据
    try:
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            cases = json.load(f)
        print(f"✅ 成功加载 {len(cases)} 个病例")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 2. 测试模型（严格提示词）
    results = test_strict_prompt_model(cases)

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
    print(f"严重错误（相差>1分）: {metrics['error_analysis']['major_errors']}例")
    print(f"轻微错误（相差1分）: {metrics['error_analysis']['minor_errors']}例")

    # 5. 生成可视化
    print("\n📈 生成可视化图表...")
    plot_results(metrics)

    print("\n🎉 测试完成！")
    print(f"  详细结果: {OUTPUT_FILE}")
    print(f"  性能图表: {PLOT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()