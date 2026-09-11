import json


def convert_imaging_features_only(input_path, output_path, imaging_codes_path):
    """仅转换影像特征（其他字段原样保留）"""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(imaging_codes_path, 'r', encoding='utf-8') as f:
        imaging_codes = json.load(f)

    converted = []
    error_count = 0

    for item in data:
        try:
            new_item = item.copy()  # 保留所有原始字段

            # 仅修改这三个字段
            new_item['t2wi'] = imaging_codes['T2WI'][item['t2wi']]
            new_item['dwi'] = imaging_codes['DWI'][item['dwi']]
            new_item['dce'] = imaging_codes['DCE'][item['dce']]

            converted.append(new_item)
        except KeyError as e:
            error_count += 1
            print(f"⚠️ 跳过异常数据: {item.get('location_text', '未知位置')} | 错误: {e}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    print(f"✅ 转换完成！成功: {len(converted)}条, 失败: {error_count}条")


# 使用示例
convert_imaging_features_only(
    input_path='result9.json',
    output_path='result8.json',
    imaging_codes_path='final_imaging_codes.json'
)