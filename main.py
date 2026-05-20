import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

def main():
    # 1. 动态加载环境配置
    load_dotenv()
    
    api_key = os.getenv("YUNWU_API_KEY")
    base_url = os.getenv("YUNWU_BASE_URL", "https://yunwu.ai/v1")
    model_name = os.getenv("LLM_MODEL_NAME", "gpt-5.4")

    if not api_key:
        print("启动失败: 未检测到 YUNWU_API_KEY 环境变量", file=sys.stderr)
        sys.exit(1)

    print("✅ 配置加载成功")
    print(f"🔌 正在连接接口: {base_url}")
    print(f"🤖 目标模型: {model_name}")

    # 2. 实例化客户端
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    # 3. 发起调用测试
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个AI助手。"},
                {"role": "user", "content": "你是gpt5.4吗？请简单介绍一下你自己。"}
            ],
            timeout=15.0 # 设置合理的超时时间
        )
        print("\n🎉 测试成功！模型返回信息:")
        print(response.choices[0].message.content)
        
    except Exception as e:
        print(f"\n❌ 测试失败！请检查网络连通性或密钥是否正确。\n错误详情: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()