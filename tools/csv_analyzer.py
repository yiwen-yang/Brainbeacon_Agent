# tools/csv_analyzer.py
import pandas as pd
from langchain.tools import tool

@tool("analyze_csv", return_direct=True)
def analyze_csv(file_path: str, top_n: int = 5) -> str:
    """读取 CSV 文件并返回得分最高的前 N 个基因（美化 Markdown 输出）"""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return f"❌ 无法读取 CSV 文件：{e}"

    # 必须包含的字段
    required_cols = ["genes", "n_cells_perturbed", "delta_cos_target", "delta_cos_niche", "score_sum"]
    for col in required_cols:
        if col not in df.columns:
            return f"CSV 格式错误，缺少必要列：{col}"

    # 取 Top N
    top_genes = df.nlargest(top_n, "score_sum")

    # ==== Markdown 美化输出 ====
    md = f"## 🧬 BrainBeacon KO 敲除结果（Top {top_n} 基因）\n\n"
    md += "| 排名 | 基因 | 目标变化 Δtarget | 微环境变化 Δniche | 综合得分 score_sum |\n"
    md += "|------|------|------------------|-------------------|--------------------|\n"

    for i, row in enumerate(top_genes.itertuples(), 1):
        md += (
            f"| {i} | **{row.genes}** "
            f"| {row.delta_cos_target:.4f} "
            f"| {row.delta_cos_niche:.4f} "
            f"| **{row.score_sum:.4f}** |\n"
        )

    # 追加说明
    md += (
        "\n📌 **说明**：\n"
        "- `delta_cos_target`：目标细胞 embedding 变化\n"
        "- `delta_cos_niche`：邻域细胞 microenvironment 变化\n"
        "- `score_sum`：综合影响分数，越高表示 KO 后影响越显著\n"
    )

    return md