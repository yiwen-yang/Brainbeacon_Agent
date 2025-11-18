#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import uuid

# === 自定义工具 ===
from tools.csv_analyzer import analyze_csv
from tools.tf_coregulation_tool import check_tf_coregulation
from tools.opentargets_tool import query_opentargets
from tools.brainbeacon_ko_tool import brainbeacon_ko_summary
from tools.memory_setup import setup_memory

# =============================
# 初始化 Flask 应用
# =============================
app = Flask(__name__)
CORS(app)

# =============================
# 环境变量
# =============================
load_dotenv()
api_key = os.getenv("DS_API_KEY") or os.getenv("OPENAI_API_KEY")
base_url = "https://api.deepseek.com/v1" if os.getenv("DS_API_KEY") else None

# =============================
# 初始化 LLM 模型
# =============================
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    openai_api_key=api_key,
    openai_api_base=base_url,
)

# =============================
# 强化版 system_prompt
# =============================
system_prompt = SystemMessage(
    content=(
        "你是一名科研智能助理，能够使用以下工具：\n"
        "- analyze_csv：分析用户指定的 CSV 文件；\n"
        "- brainbeacon_ko_summary：分析 BrainBeacon 敲除/过表达（KO/OE）实验，"
        "自动读取 data/gene_scores.csv；\n"
        "- check_tf_coregulation：查询 TRRUST 转录因子调控关系；\n"
        "- query_opentargets：查询基因功能与疾病关联。\n\n"

        "==============================\n"
        "【必须遵守的核心规则】\n"
        "==============================\n"
        "① 当用户提到以下关键词时，你必须自动调用 brainbeacon_ko_summary：\n"
        "   “BrainBeacon 敲除”、 “KO 哪些 gene 最强”、 “KO 后最显著变化”、\n"
        "   “敲除实验结果”、 “OE 实验”、 “虚拟扰动结果”、\n"
        "   “哪些基因最显著/最强/影响最大”。\n"
        "   - 不要要求用户提供 CSV 路径；\n"
        "   - 不要反问；\n"
        "   - 自动调用 brainbeacon_ko_summary。\n\n"

        "② 对于 TRRUST（check_tf_coregulation）：\n"
        "   - 若用户未指定物种，先以 species='auto' 调用；\n"
        "   - 如果 human 与 mouse 都存在结果，你需要提示用户选择物种；\n"
        "   - 用户明确 species 后，再次调用并给出精确回答。\n\n"

        "③ 对于 OpenTargets（query_opentargets）：\n"
        "   - 输入基因名称即可获取功能、疾病关联和 gene type。\n\n"

        "④ 所有回答必须使用中文，并保持多轮对话上下文一致。\n"
        "   回答需学术、准确、逻辑清晰。\n"
    )
)

# =============================
# 工具注册与记忆设置
# =============================
tools = [
    analyze_csv,
    check_tf_coregulation,
    query_opentargets,
    brainbeacon_ko_summary
]

checkpointer, store = setup_memory()

# =============================
# 构建 Agent（含短期记忆）
# =============================
agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer,
    store=store,
)

# 存储每个会话的消息历史
sessions = {}

IDENTITY_RESPONSE = (
    "我是大脑启智（BrainBeacon）的智能助理，专为多模态与跨物种空间转录组研究设计。\n\n"
    "作为您的研究伙伴，我可以帮助您处理多个层面的生物信息学任务：\n\n"
    "⸻\n\n"
    "🧠 ✨ **核心能力：BrainBeacon 虚拟空间扰动分析**\n"
    "我与 BrainBeacon（大脑启智）模型深度集成，能够：\n"
    "- 自动分析基因敲除（KO）与过表达（OE）的空间扰动结果\n"
    "- 提取最显著变化基因与微环境变化\n"
    "- 解读目标基因对邻域细胞的影响\n\n"
    "这是我的核心专长 —— 智能解读 BrainBeacon 产生的虚拟扰动数据。\n\n"
    "⸻\n\n"
    "🔬 **数据分析能力**\n"
    "- 自动提取最显著变化的基因\n"
    "- 比较扰动前后细胞 embedding 的变化\n"
    "- 汇总 KO/OE 结果，生成清晰、生物学导向的解释\n\n"
    "⸻\n\n"
    "🧬 **基因调控网络查询**\n"
    "- 基于 TRRUST 查询转录因子调控关系\n"
    "- 进行共调控分析（自动识别 human/mouse，物种不明确时会询问您）\n"
    "- 提供转录因子 → 靶基因的调控方式（激活/抑制）\n\n"
    "⸻\n\n"
    "🎯 **基因功能与疾病关联查询**\n"
    "- 接入 OpenTargets API 获取基因功能\n"
    "- 提供相关疾病与关联评分\n"
    "- 汇总关键通路或生物学过程\n\n"
    "⸻\n\n"
    "💡 **我的特点**\n"
    "- 理解上下文并保持多轮对话一致\n"
    "- 自动选择合适的工具进行分析\n"
    "- 支持跨物种、跨平台的空间转录组任务\n\n"
    "⸻\n\n"
    "🗣️ 您现在想查询基因功能、调控网络，还是让大脑启智帮您分析一次虚拟扰动？"
)


# =============================
# 路由
# =============================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """处理用户消息"""
    try:
        data = request.json
        user_message = data.get('message', '')
        session_id = data.get('session_id', 'default')

        if not user_message:
            return jsonify({'error': '消息不能为空'}), 400

        # 初始化会话
        if session_id not in sessions:
            sessions[session_id] = {
                'messages': [system_prompt],
                'thread_id': f"thread_{session_id}"
            }

        session = sessions[session_id]

        # 添加用户消息
        session['messages'].append(HumanMessage(content=user_message))

        normalized = user_message.strip().lower()
        identity_keywords = [
            "你是谁", "你是誰", "who are you", "你叫什么", "你叫什麼",
            "你是干什么的", "你是干甚麼的", "你可以做什么", "你能做什么",
            "who r u", "what can you do"
        ]

        if any(keyword in user_message or keyword in normalized for keyword in identity_keywords):
            reply_content = IDENTITY_RESPONSE
            session['messages'].append(AIMessage(content=reply_content))
            return jsonify({
                "response": reply_content,
                "session_id": session_id
            })

        # 调用 agent
        result = agent.invoke(
            {"messages": session['messages']},
            config={"configurable": {"thread_id": session['thread_id']}}
        )

        # 获取 agent 回复
        reply_msg = result["messages"][-1]
        reply_content = reply_msg.content

        # 保存到会话
        session['messages'].append(reply_msg)

        return jsonify({
            "response": reply_content,
            "session_id": session_id
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/new_session', methods=['POST'])
def new_session():
    """创建新会话"""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        'messages': [system_prompt],
        'thread_id': f"thread_{session_id}"
    }
    return jsonify({"session_id": session_id})


@app.route('/api/clear_session', methods=['POST'])
def clear_session():
    """清除会话"""
    data = request.json
    session_id = data.get('session_id', 'default')

    if session_id in sessions:
        sessions[session_id] = {
            'messages': [system_prompt],
            'thread_id': f"thread_{session_id}"
        }

    return jsonify({'status': 'cleared'})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)