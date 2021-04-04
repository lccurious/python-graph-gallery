import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from matplotlib import cm


def plot_domain_adaptation(predictions, labels, title_text):
    """
    Plot the domain adaptation classification results

    :param predictions: the predicted labels for instances
    :type predictions: numpy.ndarray
    :param labels: the real label correspondence to predictions
    :type labels: numpy.ndarray
    :param title_text: title of figure
    :type title_text: str
    """
    # Unique labels for node creating
    label_list = list(np.unique(labels))
    G = nx.MultiGraph()
    G.add_weighted_edges_from(zip(predictions, labels, [1]*len(predictions)))

    # create adjacent matrix for valued transform
    adjacent_matrix = nx.adjacency_matrix(G)
    source_list = []
    target_list = []
    value_list = []
    adj = adjacent_matrix.tocoo()
    for ei, ej, v in zip(adj.col, adj.row, adj.data):
        source_list.append(ei)
        target_list.append(ej + len(label_list))
        value_list.append(v)

    # create colormap
    cmap = cm.get_cmap('viridis', len(label_list))
    color_list = cmap(np.linspace(0, 1, len(label_list)))
    node_color_list = ['rgba' + str(tuple(c)) for c in color_list]
    link_color_list = ['rgba' + str(tuple(np.append(c[:-1], 0.4))) for c in color_list]
    
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            label = label_list + label_list,
            color = node_color_list + node_color_list
        ),
        link = dict(
            source = source_list, # indices correspond to labels, eg A1, A2, A1, B1, ...
            target = target_list,
            value = value_list,
            color = [link_color_list[i] for i in source_list]
    ))])

    fig.update_layout(title_text=title_text, font_size=10,
                      autosize=False, width=800, height=800)
    fig.show()
