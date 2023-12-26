import networkx as nx
import plotly.graph_objs as go

def draw_interactive_graph(graph,file_path=None):
    """
    绘制交互式图表的函数。

    参数:
    graph: 一个 networkx.DiGraph 对象，其节点拥有多个属性。

    返回:
    无，但会显示一个交互式图表。
    """

    # 使用 NetworkX 的布局
    graph_layout = nx.drawing.nx_agraph.graphviz_layout(graph, prog="dot", args="-Grankdir=TB")

    # 创建边的散点图对象
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in graph.edges():
        x0, y0 = graph_layout[edge[0]]
        x1, y1 = graph_layout[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    # 创建节点的散点图对象
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            line_width=2))

    for node in graph_layout:
        x, y = graph_layout[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        # 添加节点的属性
        node_info = f"Node: {node}"
        for key in ['type', 'label', 'Purity']:# 'tables'
            if key in graph.nodes[node]:
                node_info += f"<br>{key}: {graph.nodes[node][key]}"
        node_trace['text'] += tuple([node_info])

    # 创建图形
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                       # title='<br>Network graph made with Python',
                       # titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    # 显示图形
    fig.show()
    fig.write_html(file_path)


# 使用示例
"""tree = nx.DiGraph()
tree.add_node('A', label='animal', type='Mammal', Purity='High', data='SampleData')
tree.add_node('B', label='bird', type='Avian', Purity='Medium', data='SampleData')
tree.add_node('C', label='dog', type='Canine', Purity='Low', data='SampleData')
tree.add_edge('A', 'B')
tree.add_edge('B', 'C')
tree.add_edge('C', 'A')

draw_interactive_graph(tree)
"""