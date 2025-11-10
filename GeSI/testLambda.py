import networkx as nx

def find_lambda(obj, path="root"):
    """递归查找对象中的 lambda 函数"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            find_lambda(v, f"{path}[{repr(k)}]")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            find_lambda(v, f"{path}[{i}]")
    elif isinstance(obj, tuple):
        for i, v in enumerate(obj):
            find_lambda(v, f"{path}({i})")
    elif callable(obj) and obj.__name__ == "<lambda>":
        print(f"Found lambda at: {path}")

# 1. 遍历 `tree.graph`
