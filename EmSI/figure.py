import networkx as nx
import plotly.graph_objects as go

# 创建一个有向图
G = nx.DiGraph()

# 添加节点和边
G.add_node(1, label="Node 1 Info")
G.add_node(2, label="Node 2 Info")
G.add_node(3, label="Node 3 Info")
G.add_edge(1, 2)
G.add_edge(2, 3)
G.add_edge(3, 1)

# 生成图布局
pos = nx.spring_layout(G)

# 创建节点和边的数据
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

node_x = []
node_y = []
node_text = []

for node in G.nodes(data=True):
    x, y = pos[node[0]]
    node_x.append(x)
    node_y.append(y)
    node_text.append(node[1].get("label", f"Node {node[0]}"))

# 创建边的绘图
edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color="#888"),
    hoverinfo="none",
    mode="lines")

# 创建节点的绘图
node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode="markers+text",
    text=node_text,
    hoverinfo="text",
    marker=dict(
        showscale=True,
        colorscale="YlGnBu",
        size=20,
        colorbar=dict(
            thickness=15,
            title="Node Connections",
            xanchor="left",
            titleside="right"
        )
    )
)

# 创建图表对象
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title="Interactive Graph",
                    titlefont_size=16,
                    showlegend=False,
                    hovermode="closest",
                    margin=dict(b=0, l=0, r=0, t=40),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False)
                ))

# 输出到 HTML 文件
fig.write_html("network_graph.html")
