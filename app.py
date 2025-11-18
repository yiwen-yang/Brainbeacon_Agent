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
    """Ensure suggestion block appears at most once."""
    text = text or ""
    if SUGGESTION_HEADER in text:
        return text
    return text + suggestion_block()


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
GENE_PATTERN = re.compile(r"\b[A-Za-z0-9]{2,10}\b")


def load_default_ko_gene() -> str | None:
    """Load KO top1 gene as default fallback."""
    csv_path = DATA_DIR / "gene_scores.csv"
    try:
        df = pd.read_csv(csv_path)
        top_gene = df.sort_values("score_sum", ascending=False).iloc[0]["genes"]
        return str(top_gene).upper()
    except Exception:
        return None


DEFAULT_KO_GENE = load_default_ko_gene()


def set_last_gene(session: dict, gene: str) -> None:
    if not gene:
        return
    session["last_gene"] = gene.upper()


def resolve_gene(session: dict) -> tuple[str | None, bool]:
    """Return active gene for session, optionally falling back to KO top1."""
    gene = session.get("last_gene")
    used_default = False
    if not gene and DEFAULT_KO_GENE:
        gene = DEFAULT_KO_GENE
        session["last_gene"] = gene
        used_default = True
    return gene, used_default


def gene_notice(gene: str, used_default: bool, context: str = "") -> str:
    if not used_default or not gene:
        return ""
    suffix = context if context else ""
    return (
        f"未检测到您输入新的基因，本次默认使用 BrainBeacon KO Top1 基因 **{gene}**"
        f"{suffix}。\n\n"
    )


def extract_genes(text: str) -> list[str]:
    """Extract likely gene symbols from free text."""
    if not text:
        return []
    candidates = GENE_PATTERN.findall(text)
    genes = []
    for token in candidates:
        if any(ch.isdigit() for ch in token) or token.isupper():
            genes.append(token.upper())
    return genes

# === 自定义工具 ===
from tools.csv_analyzer import analyze_csv
from tools.tf_coregulation_tool import check_tf_coregulation
from tools.opentargets_tool import opentargets_query
from tools.brainbeacon_ko_tool import brainbeacon_ko_summary
from tools.memory_setup import setup_memory
from tools.literature_search import search_papers
from tools.reactome_tool import query_pathways

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
    opentargets_query,
    brainbeacon_ko_summary,
    search_papers,
    query_pathways
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
                'thread_id': f"thread_{session_id}",
                'last_gene': None
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

        # =============================
        # 菜单数字识别（1–5 自动映射工具）
        # =============================
        if normalized in ["1", "2", "3", "4", "5"]:
            if normalized == "1":
                gene, used_default = resolve_gene(session)
                if not gene:
                    reply = "暂未检测到可用的基因，请先输入基因名称。"
                else:
                    result = opentargets_query.run({"gene_symbol": gene})
                    reply = gene_notice(gene, used_default, " 进行 OpenTargets 查询") + result
            elif normalized == "2":
                gene, used_default = resolve_gene(session)
                if not gene:
                    reply = "请告诉我要查询的转录因子或基因名称。"
                else:
                    result = check_tf_coregulation.run({
                        "tf_list_str": "",
                        "target_gene": gene,
                        "species": "auto"
                    })
                    reply = gene_notice(gene, used_default, " 查询 TRRUST 调控网络") + result
            elif normalized == "3":
                reply = brainbeacon_ko_summary.run({})
            elif normalized == "4":
                gene, used_default = resolve_gene(session)
                if not gene:
                    reply = "请提供要查询的基因名称，我才能检索 Reactome 通路。"
                else:
                    result = query_pathways.run({
                        "input_data": {
                            "query_gene": gene,
                            "limit": 10
                        }
                    })
                    reply = gene_notice(gene, used_default, " 查询 Reactome 通路") + result
            elif normalized == "5":
                gene, used_default = resolve_gene(session)
                if not gene:
                    reply = "请告诉我需要检索文献的基因。"
                else:
                    result = search_papers.run({"gene": gene, "limit": 3})
                    reply = gene_notice(gene, used_default, " 进行文献检索") + result

            reply = append_suggestions(reply)
            session['messages'].append(AIMessage(content=reply))
            return jsonify({"response": reply, "session_id": session_id})

        # 文献查询触发词
        literature_keywords = ["文献", "paper", "最新研究", "研究进展", "related papers", "查文献"]
        if any(keyword in user_message for keyword in literature_keywords):
            genes = extract_genes(user_message)
            notice = ""
            if genes:
                gene = genes[0]
                set_last_gene(session, gene)
            else:
                gene, used_default = resolve_gene(session)
                if not gene:
                    reply = "暂未检测到要检索的基因，请先提供基因名称（如 TP53、MEG3）。"
                    session['messages'].append(AIMessage(content=reply))
                    return jsonify({"response": reply, "session_id": session_id})
                notice = gene_notice(gene, used_default, " 进行文献检索")

            tool_result = search_papers.run({"gene": gene, "limit": 3})
            reply = append_suggestions(notice + tool_result)
            session['messages'].append(AIMessage(content=reply))
            return jsonify({"response": reply, "session_id": session_id})

        # Pathway 查询触发词
        pathway_keywords = ["通路", "pathway", "信号通路", "代谢通路", "reactome"]
        if any(keyword in user_message for keyword in pathway_keywords):
            genes = extract_genes(user_message)
            notice = ""
            if genes:
                gene = genes[0]
                set_last_gene(session, gene)
            else:
                gene, used_default = resolve_gene(session)
                if not gene:
                    reply = "您想查询哪个基因的通路信息？例如：TP53、STAT1、MEG3。"
                    session['messages'].append(AIMessage(content=reply))
                    return jsonify({"response": reply, "session_id": session_id})
                notice = gene_notice(gene, used_default, " 查询 Reactome 通路")

            tool_result = query_pathways.run({
                "input_data": {
                    "query_gene": gene,
                    "limit": 10
                }
            })
            reply = append_suggestions(notice + tool_result)
            session['messages'].append(AIMessage(content=reply))
            return jsonify({"response": reply, "session_id": session_id})

        # CSV 自动分析触发词
        csv_keywords = ["最高分", "top 基因", "显著基因", "最强基因"]
        if any(keyword in user_message for keyword in csv_keywords):
            csv_path = "data/gene_scores.csv"
            tool_result = analyze_csv.run({"file_path": csv_path, "top_n": 5})
            reply = append_suggestions(tool_result)
            session['messages'].append(AIMessage(content=reply))
            return jsonify({"response": reply, "session_id": session_id})

        # =============================
        # 基因名称自动识别 + 多工具联动
        # =============================
        gene_candidates = extract_genes(user_message)

        if gene_candidates:
            gene = gene_candidates[0]
            set_last_gene(session, gene)

            # 联动：OpenTargets + TRRUST
            opentargets_result = opentargets_query.run({"gene_symbol": gene})
            trrust_result = check_tf_coregulation.run({
                "tf_list_str": "",
                "target_gene": gene,
                "species": "auto"
            })

            combo_reply = (
                f"🔍 **检测到基因：{gene}**\n\n"
                f"📌 **OpenTargets 结果：**\n{opentargets_result}\n\n"
                f"📌 **TRRUST 调控关系：**\n{trrust_result}\n\n"
                "如需继续查询其他基因，请告诉我基因名称。"
            )

            reply = append_suggestions(combo_reply)
            session['messages'].append(AIMessage(content=reply))
            return jsonify({"response": reply, "session_id": session_id})

        # 调用 agent
        result = agent.invoke(
            {"messages": session['messages']},
            config={"configurable": {"thread_id": session['thread_id']}}
        )

        # 获取 agent 回复
        reply_msg = result["messages"][-1]
        reply_content = append_suggestions(reply_msg.content)

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
        'thread_id': f"thread_{session_id}",
        'last_gene': None
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
            'thread_id': f"thread_{session_id}",
            'last_gene': None
        }

    return jsonify({'status': 'cleared'})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)