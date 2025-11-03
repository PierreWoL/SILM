import networkx as nx
import plotly.graph_objs as go
import os
import pickle
os.environ['PATH'] += r';C:\Program Files\Graphviz\bin'


def draw_interactive_graph(graph, file_path=None):
    graph_layout = nx.drawing.nx_agraph.graphviz_layout(graph, prog="dot", args="-Grankdir=TB")

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
        node_info = f"Node: {node}"
        for key in ['type', 'label', 'Purity','name']:  # 'tables'
            if key in graph.nodes[node]:
                node_info += f"<br>{key}: {graph.nodes[node][key]}"
        node_trace['text'] += tuple([node_info])

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        # title='<br>Network graph made with Python',
                        # titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    fig.show()
    if file_path is not None:
        fig.write_html(file_path)

test_path = f"datasets/GDS/"
with open(os.path.join(test_path, "graphGroundTruth.pkl"),"rb") as f:
    data = pickle.load(f)
draw_interactive_graph(data, os.path.join(test_path, "tree.html"))
"""numbers = [0,1,2]
dataset="WDC"
for i in numbers:
    test_path = f"E:/SILLM/Result/{dataset}/Detail/COL_Zero/{str(i)}/"
    with open(os.path.join(test_path, "tree.pkl"),"rb") as f:
        data = pickle.load(f)
    draw_interactive_graph(data, os.path.join(test_path, "tree.html"))"""