# tools/brainbeacon_ko_tool.py
from pathlib import Path

import pandas as pd
from langchain.tools import tool

@tool("brainbeacon_ko_summary", return_direct=True)
def brainbeacon_ko_summary(top_n: int = 5, sort_by: str = "score_sum") -> str:
    """
    基于 BrainBeacon 敲除（KO/OE）实验结果，返回最显著的基因列表。
    
    使用约定：
    - 默认从 data/gene_scores.csv 读取结果；
    - 按 sort_by 列降序排序（默认使用 score_sum）；
    - 当用户问「BrainBeacon 敲除后哪些基因最显著」「KO 哪些 gene 变化最大」等问题时，你应该调用本工具。
    
    参数
    ----
    top_n : int
        返回的基因数量，默认 5。
    sort_by : str
        按哪一列排序，默认 "score_sum"。
    
    返回
    ----
    str
        格式化的文本结果，包含 top_n 个基因及其指标。
    """
    try:
        # 1) 读取固定文件（约定为 BrainBeacon 敲除结果）
        base_dir = Path(__file__).resolve().parent.parent
        path = base_dir / "data" / "gene_scores.csv"

        if not path.exists():
            return f"未找到 {path}, 请确认文件是否存在或路径是否正确。"

        df = pd.read_csv(path)

        if sort_by not in df.columns:
            return f"列 '{sort_by}' 在 gene_scores.csv 中不存在，可选列包括：{list(df.columns)}"

        # 2) 按指定指标降序排序
        df_sorted = df.sort_values(sort_by, ascending=False).head(top_n)

        # 3) 生成一个易读的文本说明
        summary_lines = [
            f"基于 BrainBeacon 敲除结果，按 {sort_by} 排序的前 {top_n} 个基因如下：",
            df_sorted.to_string(index=False),
        ]

        # 如果有你之前那几个列，就顺便解释一下含义
        col_set = set(df.columns)
        if {"delta_cos_target", "delta_cos_niche", "score_sum"} <= col_set:
            summary_lines.append(
                "\n说明：delta_cos_target 表示敲除后目标细胞 embedding 变化，"
                "delta_cos_niche 表示邻域微环境变化，score_sum 为两者之和。"
            )

        return "\n".join(summary_lines)

    except FileNotFoundError:
        return "未找到 data/gene_scores.csv，请确认文件路径。"
    except Exception as e:
        return f"在解析 BrainBeacon 敲除结果时出错: {e}"