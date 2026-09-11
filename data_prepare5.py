import json
import re
from pathlib import Path


def preprocess_with_fixed_mapping(input_path, output_path):
    """严格保持位置编码一致性的预处理"""
    # 1. 加载数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 预定义位置编码（完全匹配你的数据）
    LOCATION_MAP = {
        "三角区": 0,
        "膀胱三角区": 1,
        "前壁": 2,
        "后壁": 3,
        "右侧壁": 4,
        "左侧壁": 5,
        "右后壁": 6,
        "左后壁": 7,
        "顶部": 8,
        "颈部": 9,
        "右下壁": 10,
        "前壁及右侧壁": 11,
        "右前壁": 12,
        "下壁": 13,
        "前下壁": 14,
        "右顶侧壁": 15,
        "内后部": 16,
        "顶壁": 17,
        "顶部及双侧壁": 18
    }

    # 3. 预处理
    processed = []
    for item in data:
        try:
            # 提取三个维度（带异常处理）
            sizes = list(map(float, re.findall(r"\d+\.?\d*", item['大小'])))
            sizes = sizes + [sizes[-1]] * (3 - len(sizes))  # 补全缺失维度

            processed.append({
                'gender': 0 if item['性别'] == '男' else 1,
                'age': int(item['年龄']),
                'location': LOCATION_MAP[item['位置']],  # 使用预定义编码
                'location_text': item['位置'],
                'size_x': round(sizes[0], 2),
                'size_y': round(sizes[1], 2),
                'size_z': round(sizes[2], 2),
                'size_max': round(max(sizes), 2),
                'pedunculated': 1 if item['带蒂'] == '是' else 0,
                't2wi': item['T2WI'],
                'dwi': item['DWI'],
                'dce': item['DCE'],
                'score': int(item['评分'])
            })
        except Exception as e:
            print(f"⚠️ 跳过异常数据: {item.get('位置', '未知')} | 错误: {str(e)}")
            continue

    # 4. 保存结果
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    # 保存编码表（与预处理数据一致）
    with open(output_dir / 'fixed_location_map.json', 'w', encoding='utf-8') as f:
        json.dump(LOCATION_MAP, f, indent=2, ensure_ascii=False)

    print(f"✅ 预处理完成！\n- 数据保存到: {output_path}\n- 编码表保存到: {output_dir}/fixed_location_map.json")


# 使用示例
preprocess_with_fixed_mapping(
    input_path=r"cleaned_data.json",
    output_path=r"D:\BABABA\ppyyppyy\pythonProject\result9.json"
)