# tools/tf_coregulation_tool.py
import os
import pandas as pd
from langchain.tools import tool


@tool("check_tf_coregulation", return_direct=True)
def check_tf_coregulation(
    tf_list_str: str,
    target_gene: str,
    species: str = "auto",
) -> str:
    """
    查询 TRRUST 转录因子调控网络中，某个基因是否被给定 TF 共调控。

    用法说明（写给大模型看的）：
    - 这个工具会读取本地 TRRUST 文件：
        data/trrust_rawdata.human.tsv
        data/trrust_rawdata.mouse.tsv
    - 参数 `tf_list_str` 为逗号分隔的 TF 列表，例如：
        "PU.1, IRF8, BATF3"
      如果传入空字符串，则表示“列出所有调控该基因的 TF”。
    - 参数 `target_gene` 为目标基因符号，例如 "CXCL9"。
    - 参数 `species`：
        * "human"  → 只查人类 TRRUST 文件；
        * "mouse"  → 只查小鼠 TRRUST 文件；
        * "auto"   →（默认）同时在 human 和 mouse 中搜索：
            - 如果只有一个物种有结果，直接返回该物种结果；
            - 如果两个物种都有结果，会分别列出，并提醒用户可以指定物种。
    
    作为 Agent 使用建议：
    - 当用户没有明确提“人/小鼠/物种”时，可以先用 species="auto" 调用本工具；
    - 如果返回结果中提示“人和小鼠都有结果”，你应该追问用户希望关注哪个物种，
      然后再用指定的 species 重新调用一次，以给出更精确的回答。
    """
    try:
        target_gene = target_gene.strip().upper()
        tf_list = [
            t.strip().upper()
            for t in tf_list_str.split(",")
            if t.strip()
        ]
        use_tf_filter = len(tf_list) > 0

        def _load_trrust(sp: str):
            path = f"data/trrust_rawdata.{sp}.tsv"
            if not os.path.exists(path):
                return None, f"未找到 {sp} 物种的 TRRUST 文件：{path}"
            df = pd.read_csv(path, sep="\t", header=None)
            df.columns = ["TF", "Target", "Direction", "PMID"]
            return df, None

        def _query_one_species(sp: str):
            df, err = _load_trrust(sp)
            if err is not None:
                return None, err
            sub = df[df["Target"].str.upper() == target_gene]
            if use_tf_filter:
                sub = sub[sub["TF"].str.upper().isin(tf_list)]
            if sub.empty:
                return [], None
            records = []
            for _, row in sub.iterrows():
                records.append(
                    f"TF {row['TF']} → {row['Target']} "
                    f"({row['Direction']}, PMID: {row['PMID']})"
                )
            return records, None

        # -------------------------------
        # 1) 指定物种：human 或 mouse
        # -------------------------------
        if species.lower() in {"human", "mouse"}:
            sp = species.lower()
            records, err = _query_one_species(sp)
            if err is not None:
                return err
            if not records:
                tf_part = f"{', '.join(tf_list)} " if use_tf_filter else ""
                return f"在 {sp} 的 TRRUST 数据中，未找到 {tf_part}{target_gene} 的调控记录。"
            header = f"{target_gene} 在 {sp} 中的调控证据："
            return header + "\n" + "\n".join(records)

        # -------------------------------
        # 2) species = auto：同时查 human + mouse
        # -------------------------------
        species_list = ["human", "mouse"]
        results = {}
        msgs = []

        for sp in species_list:
            records, err = _query_one_species(sp)
            if err is not None:
                msgs.append(err)   # 文件缺失也记录下来
                continue
            results[sp] = records

        # 如果两个物种都没有任何记录
        if all((not records for records in results.values())):
            tf_part = f"{', '.join(tf_list)} " if use_tf_filter else ""
            base_msg = (
                f"在 human 和 mouse 的 TRRUST 数据中，都未找到 "
                f"{tf_part}{target_gene} 的调控记录。"
            )
            if msgs:
                base_msg += "\n（附加信息：{}）".format("；".join(msgs))
            return base_msg

        # 只有一个物种有结果：直接返回该物种，并说明是自动推断
        non_empty_species = [sp for sp, rec in results.items() if rec]
        if len(non_empty_species) == 1:
            sp = non_empty_species[0]
            header = (
                f"自动在 TRRUST 中搜索 {target_gene}，"
                f"仅在 {sp} 中发现调控记录："
            )
            return header + "\n" + "\n".join(results[sp])

        # 两个物种都有结果：分别列出，并提醒用户指定
        lines = [
            f"在 TRRUST 中，human 和 mouse 均存在 {target_gene} 的调控记录。",
            "建议你在后续问题中明确指定物种（human 或 mouse），以便得到更精确的回答。\n"
        ]
        for sp in species_list:
            if results.get(sp):
                lines.append(f"--- {sp} ---")
                lines.extend(results[sp])
                lines.append("")
        return "\n".join(lines).strip()

    except Exception as e:
        return f"查询 TRRUST 时出错: {e}"