# tools/memory_setup.py
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

def setup_memory():
    """
    初始化短期记忆模块。
    MemorySaver 用于保存对话历史；
    InMemoryStore 存储当前会话状态（仅保存在内存中）。
    """
    checkpointer = MemorySaver()     # 保存对话记录（RAM）
    store = InMemoryStore()          # 存储 agent 状态（RAM）
    return checkpointer, store