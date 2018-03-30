# -*- coding: utf-8 -*-

from waldis.random_walker import perform_many_walks
from waldis.pattern_extractor import extract_pattern, update_positive_scores, compute_margin_scores, \
    weighted_set_cover_problem, reduce_pattern, interpret_pattern
from waldis.pattern import Pattern
from waldis.pattern_checker import check_pattern_in_instance
import numpy as np


def mine_patterns(graph, positive_event_vertices, positive_event_times, positive_event_edges=None,
                  negative_event_vertices=None, negative_event_times=None, negative_event_edges=None,
                  use_vertex_attributes=True, time_unit_primary=1.0, time_unit_secondary=1.0, random_walks=1000,
                  prob_restart=0.1, max_pattern_edges=10, verbose=False):
    # compute statistics by using random walk algorithm
    edge_pairs_dictionary_positive, edge_pairs_dictionary_negative, edge_pairs_counter, simple_counter = \
        perform_many_walks(graph=graph, positive_event_vertices=positive_event_vertices,
                           positive_event_times=positive_event_times, positive_event_edges=positive_event_edges,
                           negative_event_vertices=negative_event_vertices, negative_event_times=negative_event_times,
                           negative_event_edges=negative_event_edges, use_vertex_attributes=use_vertex_attributes,
                           time_unit_primary=time_unit_primary, time_unit_secondary=time_unit_secondary,
                           random_walks=random_walks, prob_restart=prob_restart, verbose=verbose)

    # extract patterns by using the computed statistics
    pattern, pattern_scores, negative_pattern_scores, vertex_mapping = \
        extract_pattern(graph=graph, positive_event_vertices=positive_event_vertices,
                        edge_pairs_dictionary_positive=edge_pairs_dictionary_positive,
                        edge_pairs_dictionary_negative=edge_pairs_dictionary_negative,
                        edge_pairs_counter=edge_pairs_counter.most_common(),
                        n_positive=len(positive_event_times),
                        n_negative=(len(negative_event_times) if negative_event_times is not None else 0),
                        max_pattern_edges=max_pattern_edges)

    pattern_scores_updated = update_positive_scores(pattern_scores, negative_pattern_scores)
    instance_scores, parallel_edge_scores = compute_margin_scores(pattern_scores_updated)
    dropped_inst, dropped_para = weighted_set_cover_problem(pattern_scores_updated, instance_scores,
                                                            parallel_edge_scores)
    reduced_pattern, edge_scores = reduce_pattern(pattern, dropped_inst, dropped_para, parallel_edge_scores)

    pattern_edges, pattern_attributes, pattern_timestamps, pattern_directions = \
        interpret_pattern(graph=graph, pattern=reduced_pattern, vertex_mapping=vertex_mapping,
                          starting_times=positive_event_times)
    return Pattern(pattern_edges=pattern_edges, pattern_attributes=pattern_attributes,
                   pattern_timestamps=pattern_timestamps, pattern_directions=pattern_directions,
                   pattern_scores=edge_scores, edge_schema=graph.edge_schema)


def evaluate_pattern(graph, pattern, positive_vertex_ids_test, positive_starting_times_test,
                     negative_vertex_ids_test, negative_starting_times_test, time_unit_secondary,
                     random_walks=10):
    positive_scores = []
    negative_scores = []

    for i in range(len(positive_vertex_ids_test[0])):
        score = check_pattern_in_instance(graph, pattern.pattern_edges, pattern.pattern_attributes,
                                          pattern.pattern_timestamps, pattern.pattern_directions,
                                          # starting_vertices=positive_vertex_ids_test[0][i:(i + 1)],
                                          # checked_timestamp=positive_starting_times_test[i:(i + 1)],
                                          graph_starting_vertices=positive_vertex_ids_test[:, i],
                                          graph_starting_timestamp=positive_starting_times_test[i],
                                          secondary_time_unit=time_unit_secondary,
                                          edge_weights=np.array(pattern.pattern_scores) / sum(pattern.pattern_scores),
                                          num_of_random_walks=random_walks)
        positive_scores.append(score)

    for i in range(len(negative_vertex_ids_test[0])):
        score = check_pattern_in_instance(graph, pattern.pattern_edges, pattern.pattern_attributes,
                                          pattern.pattern_timestamps, pattern.pattern_directions,
                                          # starting_vertices=negative_vertex_ids_test[0][i:(i + 1)],
                                          # checked_timestamp=negative_starting_times_test[i:(i + 1)],
                                          graph_starting_vertices=negative_vertex_ids_test[:, i],
                                          graph_starting_timestamp=negative_starting_times_test[i],
                                          secondary_time_unit=time_unit_secondary,
                                          edge_weights=np.array(pattern.pattern_scores) / sum(pattern.pattern_scores),
                                          num_of_random_walks=random_walks)
        negative_scores.append(score)
    return positive_scores, negative_scores

