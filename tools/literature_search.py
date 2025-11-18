import requests
from langchain.tools import tool

@tool("search_papers", return_direct=True)
def search_papers(gene: str, limit: int = 5) -> str:
    """
    查询某个基因的最新文献（使用 Semantic Scholar）。
    做了优雅兜底：429、网络错误、空结果均会返回友好的提示。
    """

    url = (
        "https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={gene}&limit={limit}&fields=title,year,externalIds,url"
    )

    # --- 网络异常兜底 ---
    try:
        resp = requests.get(url, timeout=10)
    except Exception:
        return (
            "📚 文献检索暂时无法连接到 Semantic Scholar。\n"
            "可能是网络不稳定或服务暂时不可用，请稍后再试。"
        )

    # --- 状态码兜底（特别处理 429） ---
    if resp.status_code == 429:
        return (
            "📚 文献服务当前请求过于频繁（HTTP 429）。\n\n"
            "建议：\n"
            "- 稍等几分钟再试；\n"
            "- 或继续查看 OpenTargets、TRRUST、BrainBeacon 的其他信息。\n"
        )

    if resp.status_code != 200:
        return f"📚 文献查询失败（HTTP {resp.status_code}）。请稍后重试。"

    # --- 正常解析数据 ---
    data = resp.json().get("data", [])
    if not data:
        return f"📚 未找到与 **{gene}** 相关的文献。"

    results = []
    for p in data:
        title = p.get("title", "无标题")
        year = p.get("year", "未知年份")
        external = p.get("externalIds", {})
        doi = external.get("DOI", None)

        url = p.get("url", "无链接")

        line = f"📄 **{title}** ({year})\n🔗 链接：{url}"
        if doi:
            line += f"\n🆔 DOI：{doi}"
        results.append(line + "\n")

    return "\n".join(results)