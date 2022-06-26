import csv
import os
import pickle

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import torch
from matplotlib import pyplot as plt
from torch_geometric.utils import degree, get_laplacian, to_dense_adj
from tqdm import tqdm

from environment_setup import PROJECT_ROOT_DIR, get_configurations_dtype_string

viz_save_rel_path = get_configurations_dtype_string(section='VISUALIZATIONS', key='SAVE_HTML_REL_PATH')
plot_title = get_configurations_dtype_string(section='VISUALIZATIONS', key='PLOT_TITLE')
viz_save_path = os.path.join(PROJECT_ROOT_DIR, viz_save_rel_path)
# Make the directory
os.makedirs(viz_save_path, exist_ok=True)


def plot_heterogeneous_3d_graph(hetero_dataset, scan_to_patients, out_file):
    # We can iterate over the different edge types in our dataset and then generate plots for them
    nodes = hetero_dataset.x_dict['lesion']
    lesion_location = [patient_info[5] for scan_name, patient_info in scan_to_patients.items()]
    edge_colors = ['black', 'blue']
    edge_widhts = [0.05, 2.5]
    G = nx.Graph()
    # This line needs to be executed unfortunately. This is an issue with networkx.
    # If we do not pass different node names and rather use nodes.numpy().tolist() -> All values are 1
    # networkx can't distinguish unless we pass in the edges. Both ways are clumsy.
    # Perhaps this is a bit less clumsy.
    G.add_nodes_from(range(nodes.size(0)))
    pos, x_nodes, y_nodes, z_nodes = generate_fixed_spring_layout_for_nodes(m_graph=G)
    trace_nodes = create_node_trace(node_display_info='markers', node_labels=lesion_location, size=10, x_nodes=x_nodes, y_nodes=y_nodes, z_nodes=z_nodes)
    # Let us generate separate edge traces for each of the edge types
    all_edge_traces = create_edge_trace_hetero(pos=pos, hetero_dataset=hetero_dataset, edge_colors=edge_colors, edge_widths=edge_widhts)
    # Include the traces we want to plot and create a figure
    layout = create_layout()
    data = [*all_edge_traces, trace_nodes]
    # To make our central nodes more visible, we can create another trace for the same.
    fig = go.Figure(data=data, layout=layout)
    pio.write_html(fig, os.path.join(viz_save_path, out_file))




def plot_3d_graph(edge_list, m_graph, scan_to_patients, out_file='sample.html', node_display_info='markers', size=10):
    # Keep track of `chexpert label` nodes since we would like to have them distinct
    # We use lesion locations as index.
    # This is useful when we are working with the smaller dataframe of a single patient
    patient_names = [patient_info[5] for scan_name, patient_info in scan_to_patients.items()]

    x_edges, x_nodes, y_edges, y_nodes, z_edges, z_nodes = spring_layout(edge_list, m_graph)
    trace_edges = create_edge_trace(x_edges, y_edges, z_edges)
    # create a trace for the nodes
    trace_nodes = create_node_trace(node_display_info, patient_names, size, x_nodes, y_nodes, z_nodes)
    layout = create_layout()
    # Include the traces we want to plot and create a figure
    data = [trace_edges, trace_nodes]
    # To make our central nodes more visible, we can create another trace for the same.
    fig = go.Figure(data=data, layout=layout)
    pio.write_html(fig, os.path.join(viz_save_path, out_file))


def create_layout():
    # we need to set the axis for the plot
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')
    # also need to create the layout for our plot
    layout = go.Layout(title=plot_title,
                       showlegend=False,
                       scene=dict(xaxis=dict(axis),
                                  yaxis=dict(axis),
                                  zaxis=dict(axis),
                                  ),
                       margin=dict(t=100),
                       hovermode='closest')
    print("Layout created")
    return layout


def create_node_trace(node_display_info, node_labels, size, x_nodes, y_nodes, z_nodes):
    trace_nodes = go.Scatter3d(x=x_nodes,
                               y=y_nodes,
                               z=z_nodes,
                               mode=node_display_info,
                               textfont={
                                   "size": size,
                                   "family": "Sans-serif"
                               },
                               marker=dict(color=[20*x for x in node_labels]),
                               text=node_labels,
                               hoverinfo='text')
    return trace_nodes


def create_edge_trace_hetero(pos, hetero_dataset, edge_colors, edge_widths):
    # create a trace for the edges
    all_edge_traces = []
    for idx, (edge_type, edge_indices) in enumerate(hetero_dataset.edge_index_dict.items()):
        x_edges, y_edges, z_edges = generate_edge_layout_based_on_node_layout(edge_indices.T.numpy().tolist(), pos)
        trace_edges = go.Scatter3d(x=x_edges,
                                   y=y_edges,
                                   z=z_edges,
                                   mode='lines',
                                   line=dict(color=edge_colors[idx], width=edge_widths[idx]),
                                   hoverinfo='none')
        all_edge_traces.append(trace_edges)
    return all_edge_traces

def create_edge_trace(x_edges, y_edges, z_edges, edge_color='black', edge_width=2):
    # create a trace for the edges
    trace_edges = go.Scatter3d(x=x_edges,
                               y=y_edges,
                               z=z_edges,
                               mode='lines',
                               line=dict(color=edge_color, width=edge_width),
                               hoverinfo='none')
    return trace_edges


def spring_layout(edge_list, m_graph):
    pos, x_nodes, y_nodes, z_nodes = generate_fixed_spring_layout_for_nodes(m_graph)
    # we  need to create lists that contain the starting and ending coordinates of each edge.
    x_edges, y_edges, z_edges = generate_edge_layout_based_on_node_layout(edge_list, pos)
    return x_edges, x_nodes, y_edges, y_nodes, z_edges, z_nodes


def generate_edge_layout_based_on_node_layout(edge_list, pos):
    x_edges = []
    y_edges = []
    z_edges = []
    # need to fill these with all of the coordiates
    for edge in tqdm(edge_list):
        # format: [beginning,ending,None]
        x_coords = [pos[edge[0]][0], pos[edge[1]][0], None]
        x_edges += x_coords

        y_coords = [pos[edge[0]][1], pos[edge[1]][1], None]
        y_edges += y_coords

        z_coords = [pos[edge[0]][2], pos[edge[1]][2], None]
        z_edges += z_coords
    return x_edges, y_edges, z_edges


def generate_fixed_spring_layout_for_nodes(m_graph):
    pio.renderers.default = "browser"
    pos = nx.spring_layout(m_graph, dim=3, seed=42)
    x_nodes = [pos[i][0] for i in pos.keys()]  # x-coordinates of nodes
    y_nodes = [pos[i][1] for i in pos.keys()]  # y-coordinates
    z_nodes = [pos[i][2] for i in pos.keys()]  # z-coordinates
    return pos, x_nodes, y_nodes, z_nodes


def to_gephi_data():
    pyG_hetero_dataobject = torch.load(os.path.join(PROJECT_ROOT_DIR, 'bootstrap', f'pop_graph_hetero.pth'))
    edge_type_key = ('Patient', 'Radiomic', 'Patient')
    all_edges = pyG_hetero_dataobject.edge_index_dict[edge_type_key]
    # Iterate over the edges, get the corresponding MRI names and then plot them together in the csv
    SPLIT_FILES_PATH = os.path.join(PROJECT_ROOT_DIR, 'dataset', 'split_files')
    scan_order = pickle.load(open(os.path.join(SPLIT_FILES_PATH, "scan_order.pkl"), "rb"))
    int_2_mri_name_mapping = {mri_name: int_value for int_value, mri_name in scan_order.items()}
    node_list = []
    for edge in all_edges.t():
        src, tgt = edge[0].item(), edge[1].item()
        node_list.append((int_2_mri_name_mapping[src], int_2_mri_name_mapping[tgt]))
    data = pd.DataFrame(node_list)
    data.to_csv('sample.csv', header=False, index=False)


def to_csv(edge_type_key=None, out_file='info.csv'):
    pyG_hetero_dataobject = torch.load(os.path.join(PROJECT_ROOT_DIR, 'bootstrap', f'pop_graph_hetero.pth'))
    if edge_type_key is None:
        # edge_type_key = ('Patient', 'Centre_Dist', 'Patient')
        edge_type_key = ('Patient', 'Radiomic', 'Patient')
    edges = pyG_hetero_dataobject.edge_index_dict[edge_type_key]
    laplacian = get_laplacian(edges)
    adj = to_dense_adj(laplacian[0], edge_attr=laplacian[1])
    eig_vals = torch.linalg.eigvals(adj[0])

    print("Eigen values are:")
    print(eig_vals)
    edges = edges.T.numpy().tolist()
    with open(out_file, 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerows(edges)


def to_networkx_fail_safe(data, node_attrs=None, edge_attrs=None, to_undirected=False,
                remove_self_loops=False):
    """
    Same as to_networkx but would also work with empty graphs. In case there are no edges, it will
    just return the nodes with node features.
    :param data: Homogeneous PyG dataset
    :param node_attrs:
    :param edge_attrs:
    :param to_undirected: default: False
    :param remove_self_loops: default: False
    :return: networkx object
    """
    import networkx as nx

    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    G.add_nodes_from(range(data.num_nodes))

    node_attrs, edge_attrs = node_attrs or [], edge_attrs or []

    values = {}
    for key, item in data(*(node_attrs + edge_attrs)):
        if torch.is_tensor(item):
            values[key] = item.squeeze().tolist()
        else:
            values[key] = item
        if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
            values[key] = item[0]

    # The transpose operation would fail in case the edges are empty
    if data.edge_index is not None:
        for i, (u, v) in enumerate(data.edge_index.t().tolist()):

            if to_undirected and v > u:
                continue

            if remove_self_loops and u == v:
                continue

            G.add_edge(u, v)

            for key in edge_attrs:
                G[u][v][key] = values[key][i]

    for key in node_attrs:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})

    return G


def plot_bar_plot(dictionary_to_plot, y_label, title, filename, output_dir, fontsize=15, color='g'):
    dictionary_to_plot_sorted = {k: v for k, v in
                                 sorted(dictionary_to_plot.items(), key=lambda item: item[0].key_iden)}
    plt.title(title, fontsize=fontsize)
    plt.bar([x.key_name for x in dictionary_to_plot_sorted.keys()], dictionary_to_plot_sorted.values(), color=color)
    # plt.xticks(range(1, len(labels) + 1), labels, rotation=90, fontsize=10)
    plt.ylabel(y_label, fontsize=fontsize)
    # plt.xlabel('Methods', fontsize=40)
    plt.xlabel('Graph size', fontsize=fontsize)
    plt.tight_layout()
    # Let us also save the results
    if output_dir is not None:
        save_results(output_dir=output_dir, filename=filename, plt=plt, is_img=True, dpi=200)
    plt.show()


def save_results(output_dir, filename, is_img, plt=None, df=None, **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, filename)
    if is_img:
        plt.savefig(filename, **kwargs)
    else:
        df.to_csv(filename, **kwargs)


if __name__ == '__main__':
    to_gephi_data()