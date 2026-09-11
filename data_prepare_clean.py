import json


def remove_missing_descriptions(input_path, output_path):
    """删除所有包含'未描述'的记录"""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 过滤条件：三个影像特征均不含"未描述"
    cleaned_data = [
        item for item in data
        if "未描述" not in item['T2WI']
           and "未描述" not in item['DWI']
           and "未描述" not in item['DCE']
    ]

    # 保存清洗后数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    # 打印清洗报告
    original_count = len(data)
    cleaned_count = len(cleaned_data)
    print(f"✅ 数据清洗完成！\n"
          f"- 原始数据量：{original_count}条\n"
          f"- 删除记录：{original_count - cleaned_count}条\n"
          f"- 剩余有效数据：{cleaned_count}条")


# 使用示例
remove_missing_descriptions(
    input_path='result5.json',
    output_path='cleaned_data.json'
)