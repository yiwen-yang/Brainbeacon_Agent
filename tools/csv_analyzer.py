# tools/csv_analyzer.py
import pandas as pd
from langchain.tools import tool

@tool("analyze_csv", return_direct=True)
def analyze_csv(file_path: str, top_n: int = 5) -> str:
    """读取 CSV 文件并返回得分最高的前 N 个基因"""
    df = pd.read_csv(file_path)
    if "genes" not in df.columns or "score_sum" not in df.columns:
        return "CSV 格式错误，必须包含 'genes' 和 'score_sum' 列。"
    top_genes = df.nlargest(top_n, "score_sum")
    return top_genes.to_string(index=False)