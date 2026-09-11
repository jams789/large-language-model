import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== 配置部分 ====================
INPUT_FILE = "result51.json"  # 原始病例数据
RESPONSE_FILE = "corrected_responses.json"  # 模型响应数据
OUTPUT_STATS = "scoring_performance.json"
CHART_FILE = "performance_analysis.png"


# ==================== 数据分析函数 ====================
def analyze_scoring_performance():
    """分析评分性能并生成可视化"""

    # 1. 加载数据
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        cases = json.load(f)

    with open(RESPONSE_FILE, 'r', encoding='utf-8') as f:
        responses = json.load(f)

    # 2. 数据配对
    case_dict = {case['case_id']: case for case in cases}
    paired_data = []

    for resp in responses:
        case_id = resp['case_id']
        if case_id in case_dict:
            paired_data.append({
                "case_id": case_id,
                "true_score": case_dict[case_id]['true_score'],
                "pred_score": int(resp['response']) if resp['response'].isdigit() else None
            })

    # 3. 计算指标
    valid_pairs = [d for d in paired_data if d['pred_score'] is not None]
    n_valid = len(valid_pairs)

    # 精确准确率（完全匹配）
    accuracy = sum(1 for d in valid_pairs if d['pred_score'] == d['true_score']) / n_valid

    # 邻近准确率（±1分内）
    near_accuracy = sum(1 for d in valid_pairs if abs(d['pred_score'] - d['true_score']) <= 1) / n_valid

    # 95%置信区间计算
    def wilson_ci(p, n, z=1.96):
        center = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
        width = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
        return (center - width, center + width)

    accuracy_ci = wilson_ci(accuracy, n_valid)
    near_ci = wilson_ci(near_accuracy, n_valid)

    # 4. 保存统计结果
    stats = {
        "total_cases": len(paired_data),
        "valid_cases": n_valid,
        "accuracy": accuracy,
        "near_accuracy": near_accuracy,
        "accuracy_ci": accuracy_ci,
        "near_ci": near_ci,
        "score_distribution": {
            "true_scores": [d['true_score'] for d in valid_pairs],
            "pred_scores": [d['pred_score'] for d in valid_pairs]
        }
    }

    with open(OUTPUT_STATS, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # 5. 生成可视化
    generate_performance_chart(stats)

    return stats


def generate_performance_chart(stats: dict):
    """生成专业性能分析图表"""
    plt.figure(figsize=(14, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    # ========== 主图：准确率对比 ==========
    plt.subplot(2, 2, 1)
    labels = ['精确准确率\n(完全匹配)', '邻近准确率\n(相差≤1分)']
    values = [stats['accuracy'], stats['near_accuracy']]
    ci_errors = [
        [values[0] - stats['accuracy_ci'][0]],
        [values[1] - stats['near_ci'][0]]
    ]

    colors = ['#3498db', '#2ecc71']
    bars = plt.bar(labels, values, yerr=ci_errors, capsize=15,
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1)

    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{value:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title('VI-RADS评分性能分析', fontsize=16, pad=20)
    plt.ylabel('准确率', fontsize=14)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # ========== 评分分布热力图 ==========
    plt.subplot(2, 2, 2)

    # 创建混淆矩阵
    max_score = max(max(stats['score_distribution']['true_scores']),
                    max(stats['score_distribution']['pred_scores']))
    confusion = np.zeros((max_score, max_score))

    for true, pred in zip(stats['score_distribution']['true_scores'],
                          stats['score_distribution']['pred_scores']):
        confusion[true - 1][pred - 1] += 1

    # 绘制热力图
    sns.heatmap(confusion, annot=True, fmt='g', cmap='Blues',
                xticklabels=range(1, max_score + 1),
                yticklabels=range(1, max_score + 1))
    plt.title('真实评分 vs 预测评分分布', fontsize=14)
    plt.xlabel('预测评分', fontsize=12)
    plt.ylabel('真实评分', fontsize=12)

    # ========== 误差分布 ==========
    plt.subplot(2, 2, 3)
    errors = [pred - true for true, pred in
              zip(stats['score_distribution']['true_scores'],
                  stats['score_distribution']['pred_scores'])]

    plt.hist(errors, bins=range(-4, 5), edgecolor='black', align='left', rwidth=0.8)
    plt.xticks(range(-4, 5))
    plt.title('评分误差分布', fontsize=14)
    plt.xlabel('预测误差（预测-真实）', fontsize=12)
    plt.ylabel('病例数量', fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    # ========== 统计信息 ==========
    plt.subplot(2, 2, 4)
    plt.axis('off')

    stats_text = f"""
统计摘要（总病例数: {stats['total_cases']}）
• 有效测试: {stats['valid_cases']} ({stats['valid_cases'] / stats['total_cases']:.1%})
• 精确准确率: {stats['accuracy']:.1%} (95%CI: {stats['accuracy_ci'][0]:.3f}-{stats['accuracy_ci'][1]:.3f})
• 邻近准确率: {stats['near_accuracy']:.1%} (95%CI: {stats['near_ci'][0]:.3f}-{stats['near_ci'][1]:.3f})

误差分析:
• 平均绝对误差: {np.mean(np.abs(errors)):.2f}分
• 最大误差: {max(np.abs(errors))}分
• 完全正确: {sum(1 for e in errors if e == 0)}例
• 轻微误差(±1分): {sum(1 for e in errors if abs(e) == 1)}例
• 严重误差(≥2分): {sum(1 for e in errors if abs(e) >= 2)}例
"""

    plt.text(0.1, 0.5, stats_text, ha='left', va='center', fontsize=11,
             bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5))

    plt.tight_layout()
    plt.savefig(CHART_FILE, dpi=300, bbox_inches='tight')
    plt.show()


# ==================== 执行入口 ====================
if __name__ == "__main__":
    print("🔍 开始分析VI-RADS评分性能...")
    stats = analyze_scoring_performance()

    print("\n📊 分析完成！")
    print(f"✅ 精确准确率: {stats['accuracy']:.2%}")
    print(f"✅ 邻近准确率: {stats['near_accuracy']:.2%}")
    print(f"📈 结果已保存到 {OUTPUT_STATS}")
    print(f"📊 图表已生成到 {CHART_FILE}")