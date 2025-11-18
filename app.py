#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from pathlib import Path
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import uuid

# =============================
# 统一的提示块
# =============================
SUGGESTION_HEADER = "你还可以继续探索 👇"


def suggestion_block():
    return (
        f"\n\n{SUGGESTION_HEADER}\n\n"
        "🔍 1. 功能与疾病（OpenTargets）\n"
        "🧬 2. 调控网络（TRRUST）\n"
        "🧠 3. 虚拟扰动解析（BrainBeacon）\n"
        "🛤️ 4. 信号通路（Reactome）\n"
        "📚 5. 最新文献（PubMed/semantic）\n\n"
        "输入 1–5 即可继续。"
    )


def append_suggestions(text: str) -> str:
    """避免建议块重复出现"""
    text = text or ""
    if SUGGESTION_HEADER in text:
        return text
    return text + suggestion_block()


# =============================
# 基础路径与默认 KO 基因
# =============================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
GENE_PATTERN = re.compile(r"\b[A-Za-z0-9]{2,10}\b")


def load_default_ko_gene() -> str | None:
    """默认取 KO 第一名基因"""
    csv_path = DATA_DIR / "gene_scores.csv"
    try:
        df = pd.read_csv(csv_path)
        return df.sort_values("score_sum", ascending=False).iloc[0]["genes"].upper()
    except:
        return None


DEFAULT_KO_GENE = load_default_ko_gene()


def set_last_gene(session: dict, gene: str):
    if gene:
        session["last_gene"] = gene.upper()


def resolve_gene(session: dict):
    """从 session 获得当前基因，不存在则 fallback 到 KO 第一名"""
    gene = session.get("last_gene")
    used_default = False

    if not gene and DEFAULT_KO_GENE:
        gene = DEFAULT_KO_GENE
        session["last_gene"] = gene
        used_default = True

    return gene, used_default


def gene_notice(gene: str, used_default: bool, context: str = ""):
    if not used_default:
        return ""
    return (
        f"未检测到您输入新的基因，本次默认使用 BrainBeacon KO Top1 基因 **{gene}**"
        f"{context}。\n\n"
    )


def extract_genes(text: str):
    if not text:
        return []
    candidates = GENE_PATTERN.findall(text)
    genes = []
    for token in candidates:
        if token.isupper() or any(ch.isdigit() for ch in token):
            genes.append(token.upper())
    return genes


# =============================
# 导入工具
# =============================
from tools.csv_analyzer import analyze_csv
from tools.tf_coregulation_tool import check_tf_coregulation
from tools.opentargets_tool import opentargets_query
from tools.brainbeacon_ko_tool import brainbeacon_ko_summary
from tools.memory_setup import setup_memory
from tools.literature_search import search_papers
from tools.reactome_tool import query_pathways


# =============================
# 初始化 Flask
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
# 初始化模型
# =============================
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    openai_api_key=api_key,
    openai_api_base=base_url,
)


# =============================
# system prompt
# =============================
system_prompt = SystemMessage(
    content=(
        "你是一名科研智能助理，能够使用以下工具：\n"
        "- analyze_csv：分析 CSV；\n"
        "- brainbeacon_ko_summary：分析 BrainBeacon KO/OE 并自动读取 data/gene_scores.csv；\n"
        "- check_tf_coregulation：TRRUST 调控关系；\n"
        "- opentargets_query：基因功能与疾病关联；\n"
        "- search_papers：文献检索；\n"
        "- query_pathways：Reactome 通路。\n\n"

        "当用户提到 '敲除结果'、'KO 哪些最强'、'虚拟扰动结果' 等关键词时，必须自动调用 brainbeacon_ko_summary。\n"
        "回答必须用中文。\n"
    )
)


# =============================
# 工具、记忆
# =============================
tools = [
    analyze_csv,
    check_tf_coregulation,
    opentargets_query,
    brainbeacon_ko_summary,
    search_papers,
    query_pathways
]

checkpointer, store = setup_memory()

agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer,
    store=store,
)


# =============================
# 身份回应文本
# =============================
IDENTITY_RESPONSE = (
    "我是大脑启智（BrainBeacon）的智能助理，专为跨物种多模态空间转录组研究设计…（省略，保持与你原版一致）"
)


# =============================
# 会话
# =============================
sessions = {}


# =============================
# 页面渲染
# =============================
@app.route('/')
def index():
    return render_template('index.html')


# =============================
# Render 健康检查
# =============================
@app.route('/health')
def health():
    return "OK", 200


# =============================
# 核心 chat 路由
# =============================
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')

        if not user_message:
            return jsonify({"error": "消息不能为空"}), 400

        # 创建会话
        if session_id not in sessions:
            sessions[session_id] = {
                "messages": [system_prompt],
                "thread_id": f"thread_{session_id}",
                "last_gene": None,
            }

        session = sessions[session_id]
        session["messages"].append(HumanMessage(content=user_message))

        normalized = user_message.lower()

        # =============================
        # 修复 identity bug — 必须在 agent 调用前执行
        # =============================
        identity_keywords = [
            "你是谁", "你是誰", "who are you", "你叫什么", "what can you do",
            "你能做什么", "你可以做什么"
        ]

        if any(k in user_message or k in normalized for k in identity_keywords):
            reply = append_suggestions(IDENTITY_RESPONSE)
            session["messages"].append(AIMessage(content=reply))
            return jsonify({"response": reply, "session_id": session_id})

        # =============================
        # 菜单数字 1–5
        # =============================
        if normalized in ["1", "2", "3", "4", "5"]:
            gene, used_default = resolve_gene(session)

            if normalized == "1":
                if not gene:
                    reply = "请先输入基因名称。"
                else:
                    r = opentargets_query.run({"gene_symbol": gene})
                    reply = gene_notice(gene, used_default, " 进行 OpenTargets 查询") + r

            elif normalized == "2":
                if not gene:
                    reply = "请告诉我要查询的基因名称。"
                else:
                    r = check_tf_coregulation.run({
                        "tf_list_str": "",
                        "target_gene": gene,
                        "species": "auto"
                    })
                    reply = gene_notice(gene, used_default, " 查询 TRRUST 调控网络") + r

            elif normalized == "3":
                reply = brainbeacon_ko_summary.run({})

            elif normalized == "4":
                if not gene:
                    reply = "请告诉我要查询通路的基因。"
                else:
                    r = query_pathways.run({"input_data": {"query_gene": gene, "limit": 10}})
                    reply = gene_notice(gene, used_default, " 查询 Reactome 通路") + r

            elif normalized == "5":
                if not gene:
                    reply = "请告诉我要检索文献的基因。"
                else:
                    r = search_papers.run({"gene": gene, "limit": 3})
                    reply = gene_notice(gene, used_default, " 进行文献检索") + r

            reply = append_suggestions(reply)
            session["messages"].append(AIMessage(content=reply))
            return jsonify({"response": reply, "session_id": session_id})

        # =============================
        # 文献触发
        # =============================
        if any(k in user_message for k in ["文献", "paper", "研究进展", "查文献"]):
            genes = extract_genes(user_message)
            if genes:
                gene = genes[0]
                set_last_gene(session, gene)
                notice = ""
            else:
                gene, used_default = resolve_gene(session)
                if not gene:
                    return jsonify({"response": "请提供基因名称"}), 200
                notice = gene_notice(gene, used_default, " 进行文献检索")

            r = search_papers.run({"gene": gene, "limit": 3})
            reply = append_suggestions(notice + r)
            session["messages"].append(AIMessage(content=reply))
            return jsonify({"response": reply, "session_id": session_id})

        # =============================
        # 通路触发
        # =============================
        if any(k in user_message for k in ["通路", "pathway", "信号通路"]):
            genes = extract_genes(user_message)
            if genes:
                gene = genes[0]
                set_last_gene(session, gene)
                notice = ""
            else:
                gene, used_default = resolve_gene(session)
                if not gene:
                    return jsonify({"response": "请提供基因名称"}), 200
                notice = gene_notice(gene, used_default, " 查询 Reactome 通路")

            r = query_pathways.run({"input_data": {"query_gene": gene, "limit": 10}})
            reply = append_suggestions(notice + r)
            session["messages"].append(AIMessage(content=reply))
            return jsonify({"response": reply, "session_id": session_id})

        # =============================
        # KO/显著基因触发
        # =============================
        if any(k in user_message for k in ["敲除", "最强基因", "显著基因", "虚拟扰动"]):
            r = brainbeacon_ko_summary.run({})
            reply = append_suggestions(r)
            session["messages"].append(AIMessage(content=reply))
            return jsonify({"response": reply, "session_id": session_id})

        # =============================
        # 自动基因识别 → 多工具联动
        # =============================
        gene_list = extract_genes(user_message)
        if gene_list:
            gene = gene_list[0]
            set_last_gene(session, gene)

            ot = opentargets_query.run({"gene_symbol": gene})
            tr = check_tf_coregulation.run({
                "tf_list_str": "",
                "target_gene": gene,
                "species": "auto"
            })

            reply = (
                f"🔍 检测到基因 **{gene}**\n\n"
                f"📌 **OpenTargets：**\n{ot}\n\n"
                f"📌 **TRRUST：**\n{tr}"
            )
            reply = append_suggestions(reply)
            session["messages"].append(AIMessage(content=reply))
            return jsonify({"response": reply, "session_id": session_id})

        # =============================
        # fallback → agent
        # =============================
        result = agent.invoke(
            {"messages": session["messages"]},
            config={"configurable": {"thread_id": session["thread_id"]}}
        )

        reply = result["messages"][-1].content
        reply = append_suggestions(reply)
        session["messages"].append(AIMessage(content=reply))

        return jsonify({"response": reply, "session_id": session_id})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================
# 会话控制
# =============================
@app.route('/api/new_session', methods=['POST'])
def new_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "messages": [system_prompt],
        "thread_id": f"thread_{session_id}",
        "last_gene": None
    }
    return jsonify({"session_id": session_id})


@app.route('/api/clear_session', methods=['POST'])
def clear_session():
    session_id = request.json.get("session_id", "default")
    sessions[session_id] = {
        "messages": [system_prompt],
        "thread_id": f"thread_{session_id}",
        "last_gene": None
    }
    return jsonify({"status": "cleared"})


# =============================
# RUN — 支持 Render 的动态 PORT
# =============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)