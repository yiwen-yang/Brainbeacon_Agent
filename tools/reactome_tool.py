# tools/reactome_tool.py
import requests
from langchain.tools import tool

@tool("query_pathways", return_direct=True)
def query_pathways(input_data: dict) -> str:
    """
    使用 Reactome API 查询基因相关的通路。
    仅返回查询结果，不包含任何数字导航菜单（导航由 app.py 统一处理）。
    输入：{"query_gene": "TP53", "limit": 10}
    """

    gene = input_data.get("query_gene")
    limit = input_data.get("limit", 10)

    if not gene:
        return "必须提供 query_gene，例如 {'query_gene': 'TP53'}"

    url = f"https://reactome.org/ContentService/data/pathways/low/entity/{gene}"
    r = requests.get(url)

    # Reactome 返回 404 = 无该基因记录（常见于 lncRNA / 非蛋白编码基因）
    if r.status_code == 404:
        return (
            f"🛤️ **Reactome 未收录 {gene} 的相关通路。**\n\n"
            f"常见原因：\n"
            f"- {gene} 是 lncRNA 或非蛋白编码基因（如 MEG3）\n"
            f"- Reactome 主要包含蛋白质通路，未涵盖该基因\n"
        )

    # 其他错误（500, 503 等）
    if r.status_code != 200:
        return f"Reactome 查询失败（HTTP {r.status_code}）"

    pathways = r.json()
    if not pathways:
        return (
            f"🛤️ **Reactome 未找到与 {gene} 相关的通路。**\n"
            f"（若该基因为 lncRNA，这是正常现象。）\n"
        )

    # 正常返回结果
    pathways = pathways[:limit]
    result = [f"🛤️ **{gene} 的 Reactome 通路：**\n"]
    for p in pathways:
        result.append(f"- {p.get('displayName')}  (ID: {p.get('stId')})")

    return "\n".join(result)