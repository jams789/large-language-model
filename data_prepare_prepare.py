import json
import os

# 输入和输出路径
input_path = r"D:\BABABA\ppyyppyy\pythonProject\result5.json"
output_dir = r"D:\BABABA\ppyyppyy\pythonProject"
output_filename = "result51.json"
output_path = os.path.join(output_dir, output_filename)

# VI-RADS评分规则模板
virads_rules = """[系统角色]
你是一名专业泌尿放射科AI医生，请严格按照以下流程进行VI-RADS评分：

[回答要求]
只回答一个数字，即评分

[评分流程]
第一步：T2WI结构评估（SC类别）
SC1：低信号线完整 + 病变<1cm
SC2：低信号线完整 + 病变>1cm
SC3：低信号线不确定
SC4：低信号线中断
SC5：侵犯膀胱外脂肪

第二步：DWI扩散评估（DW类别）
DW1：中等信号 + 病变<1cm
DW2：中等信号 + 病变>1cm
DW3：信号不确定
DW4：高信号 + 局灶肌层侵犯
DW5：高信号 + 全层侵犯

第三步：DCE强化评估（CE类别）
CE1：无肌层强化
CE2：内层早期强化
CE3：强化不确定
CE4：局灶肌层强化
CE5：全层及膀胱外强化

第四步：综合评分
1分：SC1 + DW1 + CE1
2分：SC2-3 + (DW2或CE2)
3分：SC3 + DW3 + CE3
4分：SC3-5 + (DW4或CE4)
5分：SC4-5 + (DW5或CE5)"""


def build_full_prompt(case_data):
    """构建完整提示词"""
    return f"""{virads_rules}

[当前病例]
性别: {case_data['性别']}
年龄: {case_data['年龄']}
位置: {case_data['位置']}
大小: {case_data['大小']}
带蒂: {case_data['带蒂']}
浸润深度: {case_data['浸润深度']}
T2WI: {case_data['T2WI']}
DWI: {case_data['DWI']}
DCE: {case_data['DCE']}"""


def process_cases(input_path, output_path):
    """处理所有病例数据"""
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_cases = json.load(f)

    enhanced_cases = []
    for i, case in enumerate(raw_cases, start=1):
        enhanced_case = {
            "case_id": i,
            "true_score": case["评分"],
            "full_prompt": build_full_prompt(case),
            "metadata": case.copy()  # 保留所有原始数据
        }
        enhanced_cases.append(enhanced_case)

    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_cases, f, indent=2, ensure_ascii=False)

    print(f"✅ 转换完成！共处理 {len(enhanced_cases)} 个病例")
    print(f"输出文件: {output_path}")


# 执行转换
if __name__ == "__main__":
    process_cases(input_path, output_path)