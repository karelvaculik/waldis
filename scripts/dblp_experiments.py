# -*- coding: utf-8 -*-

from waldis.dynamic_graph import DynamicGraph
from waldis.waldis import mine_patterns, evaluate_pattern
import numpy as np

# graph = DynamicGraph.from_json_file("../data_graphs/dblp_graph_exp1.json.gz", gz=True)
graph = DynamicGraph.from_json_file("data_graphs/dblp_graph_exp1.json.gz", gz=True)

positive_edge_label = 'icml'

np.random.seed(123)
n = 30

positive_edge_ids, positive_vertex_ids, positive_vertex_timestamps = graph.identify_edge_events("label", positive_edge_label, sample_size=n * 2)
positive_edge_ids_test = positive_edge_ids[:, n:(n+n)]
positive_vertex_ids_test = positive_vertex_ids[:, n:(n+n)]
positive_vertex_timestamps_test = positive_vertex_timestamps[:, n:(n+n)]

positive_edge_ids = positive_edge_ids[:, 0:n]
positive_vertex_ids = positive_vertex_ids[:, 0:n]
positive_vertex_timestamps = positive_vertex_timestamps[:, 0:n]

negative_edge_label = 'kdd'
negative_edge_ids, negative_vertex_ids, negative_vertex_timestamps = graph.identify_edge_events("label", negative_edge_label, sample_size=n * 2)
negative_edge_ids_test = negative_edge_ids[:, n:(n+n)]
negative_vertex_ids_test = negative_vertex_ids[:, n:(n+n)]
negative_vertex_timestamps_test = negative_vertex_timestamps[:, n:(n+n)]

negative_edge_ids = negative_edge_ids[:, 0:n]
negative_vertex_ids = negative_vertex_ids[:, 0:n]
negative_vertex_timestamps = negative_vertex_timestamps[:, 0:n]

# TRAIN DATA
positive_starting_times = positive_vertex_timestamps.min(axis=0)
positive_starting_times = positive_starting_times - 1

negative_starting_times = negative_vertex_timestamps.min(axis=0)
negative_starting_times = negative_starting_times - 1

# TEST DATA
positive_starting_times_test = positive_vertex_timestamps_test.min(axis=0) - 1
negative_starting_times_test = negative_vertex_timestamps_test.min(axis=0) - 1


time_unit_primary = 1.0
time_unit_secondary = 0.5

pattern = mine_patterns(graph=graph, positive_event_vertices=positive_vertex_ids, positive_event_times=positive_starting_times,
              positive_event_edges=positive_edge_ids, negative_event_vertices=negative_vertex_ids,
              negative_event_times=negative_starting_times, negative_event_edges=negative_edge_ids,
              use_vertex_attributes=False, time_unit_primary=time_unit_primary, time_unit_secondary=time_unit_secondary,
              random_walks=1000, prob_restart=0.1, max_pattern_edges=20, verbose=True)

print("EDGES")
print(pattern.pattern_edges)

# print(pattern.pattern_scores)
# print(pattern.pattern_edges)
# print(pattern.pattern_timestamps)


pattern.plot_statistics(add_one_to_timestamp=True)


positive_scores, negative_scores = evaluate_pattern(graph, pattern, positive_vertex_ids_test, positive_starting_times_test,
                     negative_vertex_ids_test, negative_starting_times_test, time_unit_secondary,
                     random_walks=10)

import matplotlib.pyplot as plt

plt.boxplot([positive_scores, negative_scores])
plt.title("Pattern matching score on test data")
plt.xticks([1, 2], ["positive instances", "negative instances"])
