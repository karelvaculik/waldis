# -*- coding: utf-8 -*-

import pkg_resources
import pandas as pd
import numpy as np
import re
from waldis.common_utils import AttributeType
from waldis.dynamic_graph import Vertex, Edge, DynamicGraph

rank_maping = dict()
rank_maping['Employee'] = 'Emp'
rank_maping['In House Lawyer'] = 'Law'
rank_maping['Vice President'] = 'VP'
rank_maping['Manager'] = 'Man'
rank_maping['Trader'] = 'Trad'
rank_maping['Managing Director  Legal Department'] = 'MDLP'
rank_maping['Managing Director'] = 'MD'
rank_maping['President'] = 'Pres'
rank_maping['Director'] = 'Dir'
rank_maping['CEO'] = 'CEO'


def read_employee_list():
    """
    Creates a list of employees, ignores those that have no rank (3rd column) or it is N/A.
    :return: 
    """
    employees_filename = pkg_resources.resource_filename(__name__, "data_original/enron/employees.txt")
    with open(employees_filename) as fp:
        employee_ids = []
        employee_ranks = []
        index = 0
        for line in fp:
            employee_record = re.compile("\\t| {3,}").split(line.replace('\n', ''))
            emp_rank = employee_record[2] if len(employee_record) >= 3 else ''
            if emp_rank in rank_maping:
                employee_ids.append(index)
                employee_ranks.append(rank_maping[emp_rank])
            index += 1
    # create new ids with numbers 0 to N
    df = pd.DataFrame({'original_id': employee_ids, 'rank': employee_ranks, 'id': np.arange(len(employee_ids))})
    return df


def read_email_categories():
    return pd.read_csv(pkg_resources.resource_filename(__name__, 'data_original/enron/LDCtopics.csv'))


def read_email_messages(employees, remove_loops=True):
    messages_df = pd.read_csv(pkg_resources.resource_filename(__name__,
                                                              'data_original/enron/execs.email.linesnum.ldctopic'),
                              sep=' ')
    # skip the first rows with strange time value:
    messages_df = messages_df.loc[messages_df['time'] >= 910948020].reset_index(drop=True)

    # restrict the messages to only those with sender and receiver in employees
    messages_df = pd.merge(messages_df, employees[['original_id', 'id']], how='inner',
                           left_on='sender', right_on='original_id')
    messages_df.rename(columns={'id': 'sender_id'}, inplace=True)
    messages_df = pd.merge(messages_df, employees[['original_id', 'id']], how='inner',
                           left_on='receiver', right_on='original_id')
    messages_df.rename(columns={'id': 'receiver_id'}, inplace=True)
    messages_df = messages_df[['time', 'LDCtopic', 'sender_id', 'receiver_id']]

    if remove_loops:
        messages_df = messages_df.loc[messages_df['sender_id'] != messages_df['receiver_id']]
    messages_df['id'] = np.arange(messages_df.shape[0])
    messages_df.reset_index(drop=True, inplace=True)
    return messages_df


def create_vertices(data):
    vertex_list = [Vertex(row['id'], {'rank': row['rank']})
                   for index, row in data.iterrows()]
    return vertex_list


def create_edges(data):
    edge_list = [Edge(row['id'], row['sender_id'], row['receiver_id'], row['time'], {"label": row['LDCtopic']})
                 for index, row in data.iterrows()]
    return edge_list


df_vertices = read_employee_list()
df_categories = read_email_categories()
df_edges = read_email_messages(df_vertices)

vertices = create_vertices(df_vertices)
edges = create_edges(df_edges)

dynamic_graph = DynamicGraph(vertices=vertices, edges=edges, vertex_schema=[],
                             edge_schema=[("label", AttributeType.NOMINAL)], undirected=False)

dynamic_graph.to_json_file("data_graphs/enron_graph.json.gz", gz=True)
