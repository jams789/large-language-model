import json
import requests
import time
from datetime import datetime

# ==================== 配置 ====================
API_URL = "http://192.168.110.45:1234/v1/chat/completions"  # ✅ 使用聊天端点
INPUT_FILE = "result51.json"
OUTPUT_FILE = "responses1.json"


# ==================== 聊天格式请求 ====================
def chat_style_request():
    """使用聊天格式的正确请求"""
    print("🚀 开始聊天格式请求...")

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        cases = json.load(f)

    results = []

    for i, case in enumerate(cases):
        try:
            # 构建聊天格式消息
            messages = [
                {
                    "role": "system",
                    "content": "你是一名专业泌尿放射科AI医生，请严格按照VI-RADS评分流程进行评分。只回答一个1-5的数字。"
                },
                {
                    "role": "user",
                    "content": case["full_prompt"]
                }
            ]

            # 发送聊天请求
            response = requests.post(
                API_URL,
                json={
                    "messages": messages,
                    "max_tokens": 3,
                    "temperature": 0.1,
                    "stop": ["\n", "。", "！"]
                },
                timeout=30
            )

            # 提取响应
            raw_output = response.json()["choices"][0]["message"]["content"].strip()

            results.append({
                "case_id": i + 1,
                "prompt": case["full_prompt"],
                "response": raw_output,
                "timestamp": datetime.now().isoformat()
            })

            print(f"✅ 病例 {i + 1}: {raw_output}")

        except Exception as e:
            print(f"❌ 病例 {i + 1} 失败: {e}")
            results.append({
                "case_id": i + 1,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

        time.sleep(0.5)

    # 保存结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"🎉 完成！保存到 {OUTPUT_FILE}")


if __name__ == "__main__":
    chat_style_request()