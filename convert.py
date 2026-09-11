#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将原始病灶级病例 JSON 转换为脱敏后的 LLM 输入文本（system prompt + user prompt）。
- 输入：result4.json（每条含 评分/浸润深度/T2WI/DWI/DCE 等字段）
- 输出：resultresult.json（每条含 prompt 文本 + 金标准 gold_score，供后续调用/对比）
脱敏规则：
1. 删除 评分、浸润深度 字段（直接泄露标签）
2. T2WI/DWI/DCE 只保留纯信号强度描述，删除"侵犯""突破""固有肌层"等诊断性结论
3. 性别/年龄/位置/大小/带蒂 为客观描述，予以保留
"""

import json
import re

# ============ 配置 ============
INPUT_FILE = "/data/inputs/result4.json"
OUTPUT_FILE = "/data/workspace/resultresult.json"


# ============ 脱敏处理：把信号描述里的诊断性结论剥离，只留信号强度 ============
def sanitize_signal(text):
    """从 T2WI/DWI/DCE 原始描述中提取纯信号强度，去掉侵犯/突破等结论性表述。"""
    if not text or text == "未描述":
        return "未描述"

    out = []

    # ---- T2WI 部分 ----
    # 低信号线状态
    if "低信号线完整" in text or "完整无中断" in text:
        out.append("T2WI低信号线完整")
    elif "低信号线中断" in text:
        out.append("T2WI低信号线中断")
    elif "低信号" in text and ("未明确中断" in text or "不确定" in text):
        out.append("T2WI低信号线不确定")

    # 肿瘤信号（去掉"侵犯固有肌层/外脂肪"这类结论）
    if "中等信号" in text:
        out.append("T2WI呈中等信号")

    # 形态描述（客观，可保留）
    if "无蒂外生性" in text or "广基无蒂" in text:
        out.append("无蒂外生性")

    # ---- DWI 部分 ----
    if "DWI连续中等信号" in text or ("连续中等信号" in text and "DWI" in text):
        out.append("DWI呈连续中等信号")
    elif "DWI高信号" in text or ("高信号" in text and "DWI" in text):
        out.append("DWI呈高信号")
    elif "固有肌层低信号未明确中断" in text or "低信号未明确中断" in text:
        # 这条是T2WI/DWI混合描述，表示信号不确定
        if not any("DWI" in o for o in out):
            out.append("DWI信号不确定")

    # ---- DCE 部分 ----
    if "无早期强化" in text:
        out.append("DCE无早期强化")
    elif "固有肌内层早期强化" in text:
        out.append("DCE内层早期强化")
    elif "早期强化" in text:
        out.append("DCE呈早期强化")

    result = "；".join(out)
    # 兜底：如果什么都没提取到，返回未描述
    return result if result else "未描述"


def parse_size(size_str):
    """从大小字符串解析最大径（cm），用于 <1cm / >1cm 判断。"""
    if not size_str or size_str == "未描述":
        return None
    nums = re.findall(r"\d+\.?\d*", str(size_str))
    try:
        nums = [float(n) for n in nums if n]
        return max(nums) if nums else None
    except Exception:
        return None


# ============ 构建完整 Prompt ============
SYSTEM_PROMPT = """[系统角色]
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


def build_user_prompt(case):
    """用脱敏后的字段拼接 [当前病例] 部分。"""
    t2wi = sanitize_signal(case.get("T2WI", ""))
    dwi = sanitize_signal(case.get("DWI", ""))
    dce = sanitize_signal(case.get("DCE", ""))

    # 兜底：若独立字段提取为空，从完整T2WI再提一次
    if dwi == "未描述":
        dwi = sanitize_signal("DWI：" + case.get("T2WI", ""))
    if dce == "未描述":
        dce = sanitize_signal("DCE：" + case.get("T2WI", ""))

    lines = [
        "[当前病例]",
        f"性别: {case.get('性别', '')}",
        f"年龄: {case.get('年龄', '')}",
        f"位置: {case.get('位置', '')}",
        f"大小: {case.get('大小', '')}",
        f"带蒂: {case.get('带蒂', '')}",
        f"T2WI: {t2wi}",
        f"DWI: {dwi}",
        f"DCE: {dce}",
    ]
    return "\n".join(lines)


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    leak_fields = ["评分", "浸润深度"]  # 明确删除的泄露字段

    for i, case in enumerate(data):
        # 1. 记录金标准（仅供对比评估，不进入 prompt）
        gold = case.get("评分", None)

        # 2. 构建脱敏输入
        user_prompt = build_user_prompt(case)
        full_prompt = SYSTEM_PROMPT + "\n\n" + user_prompt

        # 3. 脱敏后的字段（供审稿人检查）
        sanitized_fields = {
            "性别": case.get("性别"),
            "年龄": case.get("年龄"),
            "位置": case.get("位置"),
            "大小": case.get("大小"),
            "带蒂": case.get("带蒂"),
            "T2WI": sanitize_signal(case.get("T2WI", "")),
            "DWI": sanitize_signal(case.get("DWI", "")),
            "DCE": sanitize_signal(case.get("DCE", "")),
        }

        results.append({
            "index": i,
            "prompt": full_prompt,          # 实际喂给模型的完整文本
            "sanitized_fields": sanitized_fields,  # 脱敏后的字段（审稿人可查）
            "gold_score": gold,             # 金标准（不进入 prompt）
            "removed_fields": {k: case.get(k) for k in leak_fields},  # 被删除的泄露字段（审稿人可查）
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ============ 打印统计 & 示例 ============
    print(f"✅ 共处理 {len(results)} 条病例")
    print(f"✅ 已保存至 {OUTPUT_FILE}\n")

    # 验证脱敏有效性：只检查 [当前病例] 的用户输入部分，排除 system prompt 中的流程定义
    leak_keywords = ["浸润深度", "侵犯固有肌层", "突破膀胱", "侵犯膀胱", "侵出肌层", "粘膜及粘膜下"]
    leak_count = 0
    leak_examples = []
    for r in results:
        # 只取 [当前病例] 之后的内容（用户输入部分）
        user_part = r["prompt"].split("[当前病例]")[-1] if "[当前病例]" in r["prompt"] else r["prompt"]
        hit = [kw for kw in leak_keywords if kw in user_part]
        if hit:
            leak_count += 1
            if len(leak_examples) < 3:
                leak_examples.append((r["index"], hit, user_part.strip()[:120]))

    print(f"🔍 脱敏检查（仅检查 [当前病例] 用户输入部分，排除评分流程定义）：")
    print(f"   含泄露关键词的病例数 = {leak_count}（应为 0）")
    for idx, kws, snippet in leak_examples:
        print(f"   ⚠️ 第{idx}条 命中{kws}：{snippet}")
    print()

    # 打印第 1 条作为示例
    ex = results[1]
    print("=" * 60)
    print("示例（第 2 条，对应 前壁 1.1*0.9*0.9cm 病灶）：")
    print("=" * 60)
    print(ex["prompt"])
    print("-" * 60)
    print(f"金标准 gold_score = {ex['gold_score']}")
    print(f"被删除字段：{ex['removed_fields']}")


if __name__ == "__main__":
    main()
