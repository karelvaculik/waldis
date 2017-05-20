# -*- coding:utf-8 -*-

import numpy as np
import math
from collections import Counter
from .common_utils import expon_pdf, AttributeType
import copy


def prob_succeed(prob):
    # this function returns True with probability = prob
    value = np.random.uniform(low=0.0, high=1.0)
    return value < prob


def edge_similarity(graph, primary_edge, secondary_edge, secondary_event_start_time, secondary_edge_expected_timestamp,
                    time_unit, edge_schema, use_vertex_attributes):
    # for edges with opposite direction or with timestamp later than the event start time we return 0 similarity
    if primary_edge.direction != secondary_edge.direction or secondary_edge.timestamp > secondary_event_start_time:
        return 0

    # compute mixed-euclidean distance
    total_distance = 0
    for att_name, att_type in edge_schema:
        if att_type == AttributeType.NOMINAL:
            # this will be 0.0 for the same values and 1.0 for different values:
            total_distance += float(primary_edge.attributes[att_name] != secondary_edge.attributes[att_name])
        elif att_type == AttributeType.NUMERIC:
            # for numeric, it is distance squared
            total_distance += (primary_edge.attributes[att_name] - secondary_edge.attributes[att_name]) ** 2
    if use_vertex_attributes:
        for att_name, att_type in graph.vertex_schema:
            if att_type == AttributeType.NOMINAL:
                # this will be 0.0 for the same values and 1.0 for different values:
                total_distance += float(graph.vertices[primary_edge.from_vertex_id].attributes[att_name] !=
                                        graph.vertices[secondary_edge.from_vertex_id].attributes[att_name])
                total_distance += float(graph.vertices[primary_edge.to_vertex_id].attributes[att_name] !=
                                        graph.vertices[secondary_edge.to_vertex_id].attributes[att_name])
            elif att_type == AttributeType.NUMERIC:
                # for numeric, it is distance squared
                total_distance += (graph.vertices[primary_edge.from_vertex_id].attributes[att_name] -
                                   graph.vertices[secondary_edge.from_vertex_id].attributes[att_name]) ** 2
                total_distance += (graph.vertices[primary_edge.to_vertex_id].attributes[att_name] -
                                   graph.vertices[secondary_edge.to_vertex_id].attributes[att_name]) ** 2
    # now take square root of that sum
    total_distance = math.sqrt(total_distance)
    if use_vertex_attributes:
        max_distance = math.sqrt(len(edge_schema) + 2.0 * len(graph.vertex_schema))
    else:
        max_distance = math.sqrt(len(edge_schema))

    # print(primary_edge)
    # print(secondary_edge)
    # print("TOTAL DISTANCE: " + str(total_distance))
    # print("MAX DISTANCE: " +str(max_distance))

    attribute_similarity = max(0.1, 1.0 - total_distance / max_distance)

    time_similarity = expon_pdf(abs(secondary_edge_expected_timestamp - secondary_edge.timestamp), lamb=1,
                                time_unit=time_unit)

    total_similarity = 2 * attribute_similarity * time_similarity / (attribute_similarity + time_similarity)
    # total_similarity = (attribute_similarity + time_similarity) / 2.0
    # TMP
    # total_similarity = 1.0

    return total_similarity


def vertex_similarity():
    pass


def select_primary_edge(graph, current_node, start_time, time_unit_primary, already_visited_edges):
    edges = graph.adjacency_list[current_node]
    allowed_edges = [e for e in edges if e.original_edge_id not in already_visited_edges]

    if len(allowed_edges) == 0:
        return None

    probs = np.array([expon_pdf(start_time - e.timestamp, lamb=1, time_unit=time_unit_primary) for e in allowed_edges])

    # add 0.1 for "nonselection":
    probs = np.append(probs, [0.05])
    new_probs = probs / probs.sum()

    #
    selected_edge_index = np.random.choice(np.arange(len(new_probs)), 1, p=new_probs)[0]
    if selected_edge_index == len(new_probs) - 1:
        return None
    else:
        return allowed_edges[selected_edge_index]


def select_secondary_edge(graph, current_node, secondary_event_start_time,
                          secondary_edge_expected_timestamp, time_unit, primary_edge,
                          already_visited_edges, use_vertex_attributes):
    # print("PRIMARY EDGE:")
    # print(primary_edge)
    # print("EDGES")
    edges = graph.adjacency_list[current_node]
    # print(edges)
    allowed_edges = [e for e in edges if e.original_edge_id not in already_visited_edges]
    # print(allowed_edges)
    # print("DONE")

    probs = np.array(
        [edge_similarity(graph, primary_edge, secondary_edge, secondary_event_start_time,
                         secondary_edge_expected_timestamp,
                         time_unit, graph.edge_schema, use_vertex_attributes=use_vertex_attributes)
         for secondary_edge in allowed_edges])

    # add 0.1 for "nonselection":
    # print("PROBS")
    probs = np.append(probs, [0.05])
    # print(probs)
    new_probs = probs / probs.sum()
    selected_edge_index = np.random.choice(np.arange(len(new_probs)), 1, p=new_probs)[0]
    # if we selected the last one, it is "nonselection"
    if selected_edge_index == len(new_probs) - 1:
        return None, 0.0
    else:
        return allowed_edges[selected_edge_index], probs[selected_edge_index]


def perform_one_walk(graph, start_nodes, start_nodes_times, n_positive_instances, use_vertex_attributes, time_unit_primary,
                     time_unit_secondary, edge_pairs_dictionary_positive, edge_pairs_dictionary_negative,
                     edge_pair_counter, prob_restart, already_used_edge_sets, simple_counts):
    n_instances = len(start_nodes)
    used_edge_sets = copy.deepcopy(already_used_edge_sets)
    # which occurrence graphs can be still traversed (initially all of them)
    instances_alive = np.arange(n_instances)
    # which instances from the positives are alive (will be the same if there are no negative events)
    positive_instances_alive = np.arange(n_positive_instances)

    # for each occurence, we keep here ID of the current node (in traversal); copy the starting nodes:
    current_nodes = copy.deepcopy(start_nodes)
    # go until we restart
    while not prob_succeed(prob_restart):
        # if there less than 2 instances alive
        if len(positive_instances_alive) <= 1:
            break
        currently_traversed_edges = [-1] * n_instances
        # select occurrence which is used for start
        primary_occurrence_index = np.random.choice(positive_instances_alive)
        # select next edge of such occurrence
        selected_primary_edge = select_primary_edge(graph, current_nodes[primary_occurrence_index],
                                                    start_nodes_times[primary_occurrence_index], time_unit_primary,
                                                    used_edge_sets[primary_occurrence_index])
        # for debugging
        simple_counts[
            (primary_occurrence_index, None if selected_primary_edge is None else selected_primary_edge.edge_id)] += 1

        if selected_primary_edge is None:
            break

        # this is non-negative
        primary_edge_timestamp_difference = start_nodes_times[
                                                primary_occurrence_index] - selected_primary_edge.timestamp

        used_edge_sets[primary_occurrence_index].add(selected_primary_edge.original_edge_id)
        currently_traversed_edges[primary_occurrence_index] = selected_primary_edge.edge_id
        current_nodes[primary_occurrence_index] = selected_primary_edge.to_vertex_id
        new_instances_alive = [primary_occurrence_index]
        new_positive_instances_alive = [primary_occurrence_index]
        # select secondary edges in other occurrences
        for instance_index in instances_alive:
            # skip the already traversed graph:
            if instance_index != primary_occurrence_index:
                secondary_edge_expected_timestamp = start_nodes_times[
                                                        instance_index] - primary_edge_timestamp_difference
                selected_secondary_edge, similarity = select_secondary_edge(graph,
                                                                            current_nodes[instance_index],
                                                                            start_nodes_times[instance_index],
                                                                            secondary_edge_expected_timestamp,
                                                                            time_unit_secondary,
                                                                            selected_primary_edge,
                                                                            used_edge_sets[instance_index],
                                                                            use_vertex_attributes)
                if selected_secondary_edge is not None:
                    if instance_index < n_positive_instances:
                        # save counts for positive instance
                        if (
                        primary_occurrence_index, selected_primary_edge.edge_id) not in edge_pairs_dictionary_positive:
                            edge_pairs_dictionary_positive[
                                (primary_occurrence_index, selected_primary_edge.edge_id)] = Counter()
                        if (instance_index, selected_secondary_edge.edge_id) not in edge_pairs_dictionary_positive:
                            edge_pairs_dictionary_positive[
                                (instance_index, selected_secondary_edge.edge_id)] = Counter()
                        edge_pairs_dictionary_positive[(primary_occurrence_index, selected_primary_edge.edge_id)][
                            (instance_index, selected_secondary_edge.edge_id)] += similarity
                        edge_pairs_dictionary_positive[(instance_index, selected_secondary_edge.edge_id)][
                            (primary_occurrence_index, selected_primary_edge.edge_id)] += similarity
                        if primary_occurrence_index < instance_index:
                            edge_pair_counter[((primary_occurrence_index, selected_primary_edge.edge_id),
                                               (instance_index, selected_secondary_edge.edge_id))] += similarity
                        else:
                            edge_pair_counter[((instance_index, selected_secondary_edge.edge_id),
                                               (primary_occurrence_index, selected_primary_edge.edge_id))] += similarity
                    else:
                        # save counts for negative instance (but only the first direction)
                        if (
                        primary_occurrence_index, selected_primary_edge.edge_id) not in edge_pairs_dictionary_negative:
                            edge_pairs_dictionary_negative[
                                (primary_occurrence_index, selected_primary_edge.edge_id)] = Counter()
                        edge_pairs_dictionary_negative[(primary_occurrence_index, selected_primary_edge.edge_id)][
                            (instance_index, selected_secondary_edge.edge_id)] += similarity

                    currently_traversed_edges[instance_index] = selected_secondary_edge.edge_id
                    current_nodes[instance_index] = selected_secondary_edge.to_vertex_id
                    new_instances_alive.append(instance_index)
                    if instance_index < n_positive_instances:
                        new_positive_instances_alive.append(instance_index)
                    used_edge_sets[instance_index].add(selected_secondary_edge.original_edge_id)
                else:
                    currently_traversed_edges[instance_index] = -1
        instances_alive = new_instances_alive
        positive_instances_alive = new_positive_instances_alive


def perform_many_walks(graph, positive_event_vertices, positive_event_times, positive_event_edges=None,
                       negative_event_vertices=None, negative_event_times=None, negative_event_edges=None,
                       use_vertex_attributes=True, time_unit_primary=1.0, time_unit_secondary=1.0, random_walks=1000,
                       prob_restart=0.2, verbose=False):
    """
    
    :param graph: Graph to be traversed
    :param positive_event_vertices:
    :param positive_event_times:
    :param time_unit_primary: time (epoch) that represents common time unit for selecting the primary edge
    :param time_unit_secondary: time (epoch) that represents common time unit for selecting the primary edge
    :param random_walks: number of random walks
    :param prob_restart: probability of random walk restart
    :param positive_event_edges: (N, M) np.array of edge ids that are part of the events (and thus are already occupied)
    :return: Counter with edge tuples walked together 
    """
    if positive_event_vertices.ndim != 2:
        raise ValueError("event_vertices must have 2 dimensions")
    if positive_event_times.ndim != 1:
        raise ValueError("event_times must have 1 dimension")
    if positive_event_vertices.shape[1] != positive_event_times.shape[0]:
        raise ValueError("event_vertices and event_times must have values for the same number of instances")
    if positive_event_edges is not None and positive_event_edges.ndim != 2:
        raise ValueError("event_edges must have 2 dimensions")
    if positive_event_edges is not None and positive_event_vertices.shape[1] != positive_event_edges.shape[1]:
        raise ValueError("event_vertices and event_edges must have values for the same number of instances")

    # for each instance, we keep a set of occupied edges
    already_used_edge_sets = []
    if positive_event_edges is not None:
        for i in range(positive_event_edges.shape[1]):
            already_used_edge_sets.append(set(positive_event_edges[:, i]))
    else:
        for i in range(positive_event_vertices.shape[1]):
            already_used_edge_sets.append(set())
    # if negative events are used too, mark the their event edges too
    if negative_event_edges is not None:
        for i in range(negative_event_edges.shape[1]):
            already_used_edge_sets.append(set(negative_event_edges[:, i]))
    elif negative_event_vertices is not None:
        for i in range(negative_event_vertices.shape[1]):
            already_used_edge_sets.append(set())

    positive_instances_count = positive_event_vertices.shape[1]
    if negative_event_vertices is not None:
        positive_event_vertices = np.concatenate([positive_event_vertices, negative_event_vertices], axis=1)
        positive_event_times = np.concatenate([positive_event_times, negative_event_times])

    edge_pairs_dictionary_positive = dict()
    # here is the subset (from positive to negative only):
    edge_pairs_dictionary_negative = dict()
    edge_pairs_counter = Counter()

    # this is temporary (for debugging)
    simple_counts = Counter()
    ten_percents = int(random_walks / 10)
    for i in range(random_walks):
        if verbose and i > 0 and i % ten_percents == 0:
            print(str(int(i / ten_percents) * 10) + "%")
        # which nodes should be the starting ones
        which_nodes = np.random.choice(positive_event_vertices.shape[0], size=1)[0]
        perform_one_walk(graph, positive_event_vertices[which_nodes], positive_event_times, positive_instances_count,
                         use_vertex_attributes, time_unit_primary, time_unit_secondary,
                         edge_pairs_dictionary_positive, edge_pairs_dictionary_negative, edge_pairs_counter,
                         prob_restart, already_used_edge_sets, simple_counts)
    return edge_pairs_dictionary_positive, edge_pairs_dictionary_negative, edge_pairs_counter, simple_counts
