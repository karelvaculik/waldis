# -*- coding: utf-8 -*-

import pkg_resources
import json
import pandas as pd
import numpy as np
from waldis.common_utils import AttributeType
from waldis.dynamic_graph import Vertex, Edge, DynamicGraph

# load json data_original

# input_filename = pkg_resources.resource_filename(__name__, "../data_original/dblp/tmp_dblp.json")
input_filename = pkg_resources.resource_filename(__name__, "data_original/dblp/tmp_dblp.json")
with open(input_filename) as file:
    data = json.load(file)

# restrict data_original to only these conferences
selected_confs = ['kdd', 'sigmod', 'www', 'vldb', 'sigir', 'icde', 'cikm', 'icml', 'nips', 'cvpr', 'iccv', 'pkdd', 'ecml', 'ida', 'pakdd', 'sdm']
restricted_data = [x for x in data if x[0].startswith('conf') and x[0].split('/')[1] in selected_confs]

# replace the specific conference name with the general name
for i in range(len(restricted_data)):
    restricted_data[i][0] = restricted_data[i][0].split('/')[1]


def create_pair_combinations(values):
    if len(values) < 2:
        return None
    result_1 = []
    result_2 = []
    for i in range(len(values)):
        for j in range(i+1, len(values)):
            result_1.append(values[i])
            result_2.append(values[j])
    return result_1, result_2


def replace_vertices_with_integers(edges):
    original_columns = list(edges.columns)
    all_values = np.unique(np.concatenate([edges['src'].values, edges['dst'].values]))
    values_mapping = pd.DataFrame({'value': all_values, 'vertex_id': range(len(all_values))})
    edges = pd.merge(edges, values_mapping, how='left', left_on='src', right_on='value').drop('src', axis=1).rename(columns={'vertex_id': 'src'})[original_columns]
    edges = pd.merge(edges, values_mapping, how='left', left_on='dst', right_on='value').drop('dst', axis=1).rename(columns={'vertex_id': 'dst'})[original_columns]
    return edges


def create_vertices_from_edges(edges_df):
    all_vertex_ids = np.unique(np.concatenate([edges_df['src'].values, edges_df['dst'].values]))
    vertex_list = [Vertex(vertex_id, {'label': 0}) for vertex_id in all_vertex_ids]
    return vertex_list


def create_edges(edges_df):
    edge_list = [Edge(row['id'], row['src'], row['dst'], row['time'], {"label": row['label']})
                 for index, row in edges_df.iterrows()]
    return edge_list


# -------------------------------
# Data for experiment
# -------------------------------


def create_data_experiment_1(dblp_records):
    src_nodes = []
    dst_nodes = []
    attr_conf = []
    attr_time = []

    for i in range(len(dblp_records)):
        authors = dblp_records[i][2]
        if len(authors) > 1:
            authors_src, authors_dst = create_pair_combinations(authors)
            src_nodes += authors_src
            dst_nodes += authors_dst
            attr_conf += [dblp_records[i][0]] * len(authors_src)
            attr_time += [dblp_records[i][3]] * len(authors_src)
    result = pd.DataFrame({'id': range(len(attr_time)), 'time': attr_time, 'src': src_nodes, 'dst': dst_nodes,
                           'label': attr_conf})
    return result

exp_1_edges = create_data_experiment_1(restricted_data)
exp_1_edges = replace_vertices_with_integers(exp_1_edges)

vertices = create_vertices_from_edges(exp_1_edges)
edges = create_edges(exp_1_edges)

dynamic_graph = DynamicGraph(vertices=vertices, edges=edges, vertex_schema=[],
                             edge_schema=[("label", AttributeType.NOMINAL)], undirected=True)

dynamic_graph.to_json_file("data_graphs/dblp_graph_exp1.json.gz", gz=True)


