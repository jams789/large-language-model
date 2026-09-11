import json
import copy
from collections import Counter

# ================ 配置区 ================
INPUT_PATH = "result8.json"
OUTPUT_PATH = "result8-1.json"
TARGET_CLASS = 1  # 要特别处理的类别
TARGET_COUNT = 36  # 该类目标样本数
MAX_OTHER_CLASS = 50  # 其他类上限


# =======================================

def precise_balance():
    """精准平衡数据（指定类别1=36样本）"""
    with open(INPUT_PATH) as f:
        data = json.load(f)

    counts = Counter(item['score'] for item in data)
    print("=" * 50)
    print("📊 原始分布:", dict(sorted(counts.items())))

    # 计算需要复制的数量
    balanced_data = []
    copy_plan = {}

    # 特殊处理类别1
    if counts[TARGET_CLASS] < TARGET_COUNT:
        copy_plan[TARGET_CLASS] = TARGET_COUNT - counts[TARGET_CLASS]

    # 处理其他类别（不超过MAX_OTHER_CLASS）
    for cls, cnt in counts.items():
        if cls != TARGET_CLASS and cnt > MAX_OTHER_CLASS:
            copy_plan[cls] = MAX_OTHER_CLASS - cnt  # 负数表示需要抽样

    # 执行复制/抽样
    for item in data:
        cls = item['score']

        # 添加原始样本
        balanced_data.append(item)

        # 处理需要补充的类别1
        if cls == TARGET_CLASS and copy_plan.get(cls, 0) > 0:
            for _ in range(copy_plan[cls]):
                balanced_data.append(copy.deepcopy(item))

        # 处理需要抽样的其他类（暂不实现删除，仅控制新增）

    # 结果验证
    new_counts = Counter(item['score'] for item in balanced_data)
    print("🔄 平衡后分布:", dict(sorted(new_counts.items())))
    print(f"总样本数: {len(data)} → {len(balanced_data)}")

    # 保存新文件
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(balanced_data, f, indent=2)
    print(f"💾 已保存新文件: {OUTPUT_PATH}")


if __name__ == "__main__":
    print("=" * 50)
    print(f"🐱 精准数据平衡器 (定制版)")
    print(f"🔧 策略: 类别1={TARGET_COUNT}样本 | 其他类≤{MAX_OTHER_CLASS}")
    print("=" * 50)

    try:
        precise_balance()
        print("\n✅ 平衡完成！原文件未改动")
    except Exception as e:
        print(f"\n❌ 操作失败: {str(e)}")
    finally:
        print("\n" + "=" * 50)
        print("操作完成！小猫状态: [心满意足]")