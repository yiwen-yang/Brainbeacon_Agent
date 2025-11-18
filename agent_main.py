#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage

# === 自定义工具 ===
from tools.csv_analyzer import analyze_csv
from tools.tf_coregulation_tool import check_tf_coregulation
from tools.opentargets_tool import query_opentargets
from tools.memory_setup import setup_memory
from tools.brainbeacon_ko_tool import brainbeacon_ko_summary
# =============================
# 1️⃣ 环境变量
# =============================
load_dotenv()
api_key = os.getenv("DS_API_KEY") or os.getenv("OPENAI_API_KEY")
base_url = "https://api.deepseek.com/v1" if os.getenv("DS_API_KEY") else None

# =============================
# 2️⃣ 初始化 LLM 模型
# =============================
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    openai_api_key=api_key,
    openai_api_base=base_url,
)

# =============================
# 3️⃣ 定义 system 提示
# =============================
system_prompt = SystemMessage(
    content=(
        "你是一个科研智能助理。你可以使用以下工具："
        "analyze_csv、brainbeacon_ko_summary、check_tf_coregulation、query_opentargets。"

        "【最重要规则（必须遵守）】\n"
        "当用户提到以下关键词："
        "“BrainBeacon 敲除”、“KO 后哪些 gene 最显著”、“敲除实验结果”、"
        "“KO 最强基因”、“最显著的基因变化”、“OE 实验结果”等问题时，"
        "你必须自动调用工具 brainbeacon_ko_summary 来回答，不要要求用户提供 CSV 路径。"

        "brainbeacon_ko_summary 默认读取 data/gene_scores.csv, 无需用户输入路径。\n"

        "如果用户没有提到 KO/OE/敲除/过表达，則按普通对话逻辑处理。\n"

        "所有回答必须使用中文。"
        "查询 TF 共调控网络和 Open Targets 基因功能。"
        "当用户提到 BrainBeacon、敲除/KO/过表达/OE、哪些基因最显著/"
        "变化最大等问题时，应该优先调用工具 brainbeacon_ko_summary，"
        "它会从 data/gene_scores.csv 中读取结果。"
        "请使用中文回答，并在多轮对话中保持上下文一致。"
        "当使用 TRRUST 工具 check_tf_coregulation 时，"
        "如果用户没有指定物种，可以先用 species='auto' 调用；"
        "若 human 和 mouse 都有结果，请在回答中主动提醒用户选择物种，"
        "并在用户明确物种后再次调用工具以给出更精确结果。"
    )
)
# =============================
# 4️⃣ 工具注册与记忆设置
# =============================
tools = [
    analyze_csv,
    brainbeacon_ko_summary,   # 👈 新增：BrainBeacon 专用 KO/OE 结果工具
    check_tf_coregulation,
    query_opentargets,
]
checkpointer, store = setup_memory()

# =============================
# 5️⃣ 构建 Agent
# =============================
agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer,
    store=store,
)

# =============================
# 6️⃣ 主循环
# =============================
if __name__ == "__main__":
    print("🤖 LangAgent 已启动（含短期记忆）——输入问题（exit 退出）")

    # 在每轮对话开头注入 system message
    messages = [system_prompt]

    while True:
        query = input("你：")
        if query.lower() in ["exit", "quit"]:
            print("Agent 已结束。")
            break

        # 将用户输入加入上下文
        messages.append(HumanMessage(content=query))

        # 让 Agent 执行
        result = agent.invoke(
            {"messages": messages},
            config={"configurable": {"thread_id": "session1"}},
        )

        # 输出结果并更新会话
        reply = result["messages"][-1].content
        print("Agent：", reply)
        messages.append(result["messages"][-1])