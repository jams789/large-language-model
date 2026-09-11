import json
from collections import Counter


def analyze_imaging_descriptions(input_path):
    """分析T2WI/DWI/DCE的所有描述"""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. 统计所有描述
    t2wi_descriptions = [item['T2WI'] for item in data]
    dwi_descriptions = [item['DWI'] for item in data]
    dce_descriptions = [item['DCE'] for item in data]

    # 2. 统计频率
    t2wi_counter = Counter(t2wi_descriptions)
    dwi_counter = Counter(dwi_descriptions)
    dce_counter = Counter(dce_descriptions)

    # 3. 保存结果
    result = {
        "T2WI": dict(t2wi_counter.most_common()),
        "DWI": dict(dwi_counter.most_common()),
        "DCE": dict(dce_counter.most_common())
    }

    with open('imaging_description_stats.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("✅ 分析完成！结果已保存到 imaging_description_stats.json")
    return result


# 使用示例
analyze_imaging_descriptions('result5.json')