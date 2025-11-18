# tools/opentargets_tool.py
import requests
from langchain.tools import tool

OPENTARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"

SEARCH_QUERY = """
query searchTarget($queryString: String!) {
  search(
    queryString: $queryString
    entityNames: ["target"]
    page: { index: 0, size: 1 }
  ) {
    hits {
      id
      entity
      object {
        ... on Target {
          id
          approvedSymbol
          approvedName
          biotype
          associatedDiseases {
            rows {
              disease {
                id
                name
              }
              score
            }
          }
        }
      }
    }
  }
}
"""


@tool("opentargets_query", return_direct=True)
def opentargets_query(gene_symbol: str) -> str:
    """
    查询 OpenTargets 平台中某个基因的功能与疾病关联。
    """

    symbol = (gene_symbol or "").strip()
    if not symbol:
        return "必须提供基因 symbol，例如 'TP53'。"

    try:
        response = requests.post(
            OPENTARGETS_GRAPHQL_URL,
            json={"query": SEARCH_QUERY, "variables": {"queryString": symbol}},
            timeout=20,
        )
    except Exception as exc:
        return f"OpenTargets 请求失败：{exc}"

    if response.status_code != 200:
        print("OpenTargets raw response:", response.status_code, response.text[:500])
        return f"OpenTargets 查询失败: 状态码 {response.status_code}"

    data = response.json()
    hits = data.get("data", {}).get("search", {}).get("hits", [])

    if not hits:
        return f"未在 OpenTargets 中找到基因 {symbol} 的记录。"

    target_obj = (hits[0].get("object") or {})
    approved_symbol = target_obj.get("approvedSymbol") or symbol.upper()
    approved_name = target_obj.get("approvedName") or "暂无基因名称描述"
    biotype = target_obj.get("biotype") or "未知生物类型"

    disease_rows = (target_obj.get("associatedDiseases", {}).get("rows", []))[:5]
    if disease_rows:
        disease_lines = [
            f"- {row.get('disease', {}).get('name', '未知疾病')}（score={row.get('score', 0):.3f}）"
            for row in disease_rows
        ]
        disease_section = "\n".join(disease_lines)
    else:
        disease_section = "暂无疾病关联数据。"

    return (
        f"🧬 **{approved_symbol} — {approved_name}**（{biotype}）\n"
        f"🔹 主要疾病关联（Top 5）：\n{disease_section}"
    )