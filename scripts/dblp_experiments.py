# -*- coding: utf-8 -*-

from waldis.dynamic_graph import DynamicGraph
from waldis.waldis import mine_patterns, evaluate_pattern
import numpy as np
import time
import os

# for reproducibility
np.random.seed(123)

graph = DynamicGraph.from_json_file("data_graphs/dblp_graph_exp1.json.gz", gz=True)

positive_edge_label = 'icml'
negative_edge_label = 'kdd'
n = 30


data_name = "dblp"
param_time_unit_primary = [2.0, 1.0, 1.0, 1.0, 1.0]
param_time_unit_secondary = [1.0, 1.0, 0.5, 0.5, 0.5]
param_prob_restart = [0.1, 0.1, 0.1, 0.1, 0.3]
param_max_param_edges = [10, 10, 10, 10, 10]
param_random_walks = [1000, 1000, 1000, 5000, 5000]
evaluation_random_walks = 10
num_of_repetitions = 10

for i in range(len(param_time_unit_primary)):

    print("PARAM SETTINGS: " + str(i+1))

    for repetition in range(num_of_repetitions):

        start_time_total = time.time()
        print("RUN " + str(repetition+1))

        positive_edge_ids, positive_vertex_ids, positive_vertex_timestamps = graph.identify_edge_events("label",
                                                                                                        positive_edge_label,
                                                                                                        sample_size=n * 2)
        positive_edge_ids_test = positive_edge_ids[:, n:(n + n)]
        positive_vertex_ids_test = positive_vertex_ids[:, n:(n + n)]
        positive_vertex_timestamps_test = positive_vertex_timestamps[:, n:(n + n)]

        positive_edge_ids = positive_edge_ids[:, 0:n]
        positive_vertex_ids = positive_vertex_ids[:, 0:n]
        positive_vertex_timestamps = positive_vertex_timestamps[:, 0:n]

        negative_edge_ids, negative_vertex_ids, negative_vertex_timestamps = graph.identify_edge_events("label",
                                                                                                        negative_edge_label,
                                                                                                        sample_size=n * 2)
        negative_edge_ids_test = negative_edge_ids[:, n:(n + n)]
        negative_vertex_ids_test = negative_vertex_ids[:, n:(n + n)]
        negative_vertex_timestamps_test = negative_vertex_timestamps[:, n:(n + n)]

        negative_edge_ids = negative_edge_ids[:, 0:n]
        negative_vertex_ids = negative_vertex_ids[:, 0:n]
        negative_vertex_timestamps = negative_vertex_timestamps[:, 0:n]

        # subtract 1 from the starting timestamps so that we find only edge patterns older that events
        # TRAIN DATA
        positive_starting_times = positive_vertex_timestamps.min(axis=0)
        positive_starting_times = positive_starting_times - 1

        negative_starting_times = negative_vertex_timestamps.min(axis=0)
        negative_starting_times = negative_starting_times - 1

        # TEST DATA
        positive_starting_times_test = positive_vertex_timestamps_test.min(axis=0) - 1
        negative_starting_times_test = negative_vertex_timestamps_test.min(axis=0) - 1

        experiment_directory = "experiments/"+data_name+"/"
        if not os.path.exists(experiment_directory):
            os.makedirs(experiment_directory)

        pattern = mine_patterns(graph=graph, positive_event_vertices=positive_vertex_ids,
                                positive_event_times=positive_starting_times,
                                positive_event_edges=positive_edge_ids, negative_event_vertices=negative_vertex_ids,
                                negative_event_times=negative_starting_times, negative_event_edges=negative_edge_ids,
                                use_vertex_attributes=False, time_unit_primary=param_time_unit_primary[i],
                                time_unit_secondary=param_time_unit_secondary[i],
                                random_walks=param_random_walks[i], prob_restart=param_prob_restart[i],
                                max_pattern_edges=param_max_param_edges[i], verbose=False)

        positive_scores, negative_scores = evaluate_pattern(graph, pattern, positive_vertex_ids_test,
                                                            positive_starting_times_test,
                                                            negative_vertex_ids_test, negative_starting_times_test,
                                                            param_time_unit_secondary[i],
                                                            random_walks=evaluation_random_walks)

        end_time_total = time.time()
        print("TIME: " + str(int(end_time_total - start_time_total)))

        with open(experiment_directory + 'experiments_info_and_results-' + str(i).rjust(2, '0') + '-run' +
                  str(repetition).rjust(2, '0') + '.txt', 'w') as the_file:
            the_file.write("DATA:" + str(data_name) + "\n")
            the_file.write("TOTAL_TIME:" + str(int(end_time_total - start_time_total)) + "\n")
            the_file.write("TIME_UNIT_PRIMARY:" + str(param_time_unit_primary[i]) + "\n")
            the_file.write("TIME_UNIT_SECONDARY:" + str(param_time_unit_secondary[i]) + "\n")
            the_file.write("RANDOM_WALKS:" + str(param_random_walks[i]) + "\n")
            the_file.write("PROB_RESTART:" + str(param_prob_restart[i]) + "\n")
            the_file.write("MAX_PATTERN_EDGES:" + str(param_max_param_edges[i]) + "\n")
            the_file.write("EVALUATION_RANDOM_WALKS:" + str(evaluation_random_walks) + "\n")
            the_file.write("TEST_EVALUATION_POSITIVE:" + str(positive_scores) + "\n")
            the_file.write("TEST_EVALUATION_NEGATIVE:" + str(negative_scores) + "\n")
            the_file.write("PATTERN_EDGES:" + str(pattern.pattern_edges) + "\n")
            the_file.write("PATTERN_SCORES:" + str(pattern.pattern_scores) + "\n")
            the_file.write("PATTERN_TIMESTAMPS:" + str(pattern.pattern_timestamps) + "\n")
            the_file.write("PATTERN_ATTRIBUTES:" + str(pattern.pattern_attributes) + "\n")
            the_file.write("PATTERN_DIRECTIONS:" + str(pattern.pattern_directions) + "\n")
            the_file.write("PATTERN_UNDIRECTED:" + str(graph.undirected) + "\n")
