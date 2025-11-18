# tools/opentargets_tool.py
import requests
from langchain.tools import tool

@tool("query_opentargets", return_direct=True)
def query_opentargets(gene_symbol: str) -> str:
    """
    查询 Open Targets 平台中某个基因的基本功能信息和主要疾病关联。
    
    参数
    ----
    gene_symbol : str
        基因符号（如 "TP53", "CXCL9"）。
    
    返回
    ----
    str
        包含基因名称/简要功能说明，以及若干高分疾病关联的文本描述。
    """
    try:
        base_url = "https://api.platform.opentargets.org/api/v4/graphql"

        # 使用 search，根据 gene_symbol 找到对应的 Target 对象，再直接拿它的 associatedDiseases
        query_string = """
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

        variables = {"queryString": gene_symbol}

        resp = requests.post(
            base_url,
            json={"query": query_string, "variables": variables},
            timeout=20,
        )

        # 调试用：如果出问题，可以在终端看到返回内容
        if resp.status_code != 200:
            # 打印到终端，方便你 debug
            print("OpenTargets raw response:", resp.status_code, resp.text[:500])
            return f"OpenTargets 查询失败: 状态码 {resp.status_code}"

        data = resp.json()
        search_data = data.get("data", {}).get("search", {})
        hits = (search_data or {}).get("hits", [])

        if not hits:
            return f"未在 Open Targets 中找到基因 {gene_symbol} 的记录。"

        target_obj = hits[0].get("object") or {}
        symbol = target_obj.get("approvedSymbol") or gene_symbol
        name = target_obj.get("approvedName") or "暂无基因名称描述"
        biotype = target_obj.get("biotype") or "未知生物类型"

        # 取前 5 个疾病关联
        disease_rows = (
            target_obj
            .get("associatedDiseases", {})
            .get("rows", [])
        )[:5]

        if not disease_rows:
            disease_info = "暂无疾病关联数据。"
        else:
            disease_lines = []
            for row in disease_rows:
                dis = row.get("disease", {}) or {}
                dname = dis.get("name", "未知疾病")
                score = row.get("score", 0.0)
                disease_lines.append(f"- {dname}（关联得分: {score:.2f}）")
            disease_info = "\n".join(disease_lines)

        return (
            f"（{biotype}）\n"
            f"基因名称：{name}\n"
            f"主要疾病关联：\n{disease_info}"
        )

    except Exception as e:
        # 防御性兜底
        return f"查询出错: {e}"