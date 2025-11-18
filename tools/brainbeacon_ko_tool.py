# tools/brainbeacon_ko_tool.py
from pathlib import Path
import pandas as pd
from langchain.tools import tool

DEFAULT_TOP_N = 10
REQUIRED_COLS = [
    "genes",
    "delta_cos_target",
    "delta_cos_niche",
    "score_sum",
]


def _format_table(df: pd.DataFrame, top_n: int) -> str:
    """Format KO/OE scores into a markdown table."""
    header = (
        "| 排名 | 基因 | target变化 Δtarget | niche变化 Δniche | 综合 score_sum |\n"
        "|-----|------|--------------------|------------------|----------------|\n"
    )

    rows = []
    for rank, (_, row) in enumerate(df.head(top_n).iterrows(), start=1):
        rows.append(
            "| {rank} | **{gene}** | {target:.6f} | {niche:.6f} | **{score:.6f}** |".format(
                rank=rank,
                gene=row["genes"],
                target=row["delta_cos_target"],
                niche=row["delta_cos_niche"],
                score=row["score_sum"],
            )
        )

    body = "\n".join(rows)
    return (
        f"基于 BrainBeacon 敲除结果，按 score_sum 排序的前 {top_n} 个基因如下：\n\n"
        f"{header}{body}"
    )


@tool("brainbeacon_ko_summary", return_direct=True)
def brainbeacon_ko_summary(top_n: int = DEFAULT_TOP_N) -> str:
    """
    分析 BrainBeacon KO/OE 结果，返回得分最高的前 N 个基因（默认 10 个）。
    自动读取 data/gene_scores.csv。
    """

    try:
        top_n = int(top_n)
    except (ValueError, TypeError):
        top_n = DEFAULT_TOP_N

    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "data" / "gene_scores.csv"

    if not csv_path.exists():
        return f"❌ 未找到 {csv_path}，请确认文件是否存在。"

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return f"❌ 无法读取 CSV 文件：{exc}"

    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        return f"❌ CSV 文件缺少必要列：{', '.join(missing)}"

    df_sorted = df.sort_values("score_sum", ascending=False)
    return _format_table(df_sorted, top_n)