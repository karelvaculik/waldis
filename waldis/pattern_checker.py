# -*- coding: utf-8 -*-


import numpy as np
from waldis.dynamic_graph import Edge
from waldis.random_walker import edge_similarity


def check_pattern_in_instance(graph, pattern_edges, pattern_attributes, pattern_timestamps, pattern_directions,
                              starting_vertices, checked_timestamp, secondary_time_unit, edge_weights,
                              use_vertex_attributes=True, num_of_random_walks=100):
    """
    Looks into the graph, whether there is the given pattern on  with respect to the given
    vertices and timestamp.

    :param graph:
    :param pattern_edges:
    :param pattern_attributes:
    :param pattern_timestamps:
    :param starting_vertices:
    :param checked_timestamp: assumed event time in the graph
    :param secondary_time_unit: time (epoch) that represents common time unit for selecting the primary edge
    :return:
    """
    if len(pattern_edges) == 0:
        # return full score because empty pattern can be everywhere
        return 1.0
    else:
        number_of_instances_in_pattern = len(pattern_attributes[0])
    # prepare adjacency list from pattern edges
    adjacency_list = dict()
    # we also remember which edge is linked to each such adjacency information - it is necessary for score computation
    adjacency_links_to_edges = dict()
    # whether this is the original direction or the added (opposite) one:
    adjacency_list_is_opposite = dict()
    for i in range(len(pattern_edges)):
        src = pattern_edges[i][0]
        dst = pattern_edges[i][1]
        if src not in adjacency_list:
            adjacency_list[src] = [dst]
            adjacency_links_to_edges[src] = [i]
            adjacency_list_is_opposite[src] = [False]
        else:
            adjacency_list[src].append(dst)
            adjacency_links_to_edges[src].append(i)
            adjacency_list_is_opposite[src].append(False)
        if dst not in adjacency_list:
            adjacency_list[dst] = [src]
            adjacency_links_to_edges[dst] = [i]
            adjacency_list_is_opposite[dst] = [True]
        else:
            adjacency_list[dst].append(src)
            adjacency_links_to_edges[dst].append(i)
            adjacency_list_is_opposite[dst].append(True)

    all_total_scores = []

    for which_instance in range(number_of_instances_in_pattern):

        edges_scores = list()
        edges_counts = list()
        for i in range(len(pattern_edges)):
            edges_scores.append(0.0)
            edges_counts.append(0.0)

        for i in range(num_of_random_walks):

            # we don't allow to use one pattern edge many times (both in pattern and graph) in one random walk
            already_visited_pattern_edges = set()
            # here we keep the original ids used in the graph in this single random walk
            already_visited_graph_edges = set()
            # first, choose the starting vertices (for the pattern and the corresponding one from graph)
            current_pattern_vertex = np.random.choice(len(starting_vertices), size=1)[0]
            current_graph_vertex = starting_vertices[current_pattern_vertex]
            # keep the mapping from pattern vertices to graph vertices and use it to check the vertex consistency:
            vertex_mapping = {current_pattern_vertex: current_graph_vertex}
            # if walking in the graph goes wrong, we continue only in the pattern
            walking_pattern_only = False
            # now perform the random walk controled by the pattern
            # the random walk is not stopped until there are no more edges in pattern to use for moving forward

            while True:
                # which pattern edges can be used to move forward
                which_adjacent_edges_could_be_used = [i for i, x in enumerate(adjacency_links_to_edges[current_pattern_vertex]) if x not in already_visited_pattern_edges]

                if len(which_adjacent_edges_could_be_used) == 0:
                    # we cannot go further in the pattern so finish the current random walk
                    break
                # if there are some usable edges, select one at random
                index_in_adjacency_list = np.random.choice(which_adjacent_edges_could_be_used, 1)[0]
                pattern_edge_index = adjacency_links_to_edges[current_pattern_vertex][index_in_adjacency_list]
                # increase the counts
                edges_counts[pattern_edge_index] += 1
                already_visited_pattern_edges.add(pattern_edge_index)

                next_current_pattern_vertex = adjacency_list[current_pattern_vertex][index_in_adjacency_list]

                if not walking_pattern_only:
                    # now check that we can move forward in the graph
                    graph_edges = graph.adjacency_list[current_graph_vertex]
                    graph_allowed_edges = [e for e in graph_edges if e.original_edge_id not in already_visited_graph_edges]

                    # remove edges that have inconsistent vertices (we check dst vertex)
                    if next_current_pattern_vertex in vertex_mapping:
                        graph_allowed_edges = [e for e in graph_allowed_edges if e.to_vertex_id == vertex_mapping[next_current_pattern_vertex]]

                    if len(graph_allowed_edges) == 0:
                        # in this case we are not able to move in the original graph, but we are able to move in the
                        # pattern, continue only in pattern
                        walking_pattern_only = True
                    else:
                        # select the edge in the graph and move forward both in the graph and pattern
                        # only attributes and direction is necessary for edge similarity function
                        primary_edge = Edge(0, 0, 0, timestamp=0, attributes=pattern_attributes[pattern_edge_index][which_instance],
                                            direction=pattern_directions[pattern_edge_index], original_edge_id=None)
                        # now check whether we should take the opposite:
                        if adjacency_list_is_opposite[current_pattern_vertex][index_in_adjacency_list]:
                            # take the opposite:
                            primary_edge = primary_edge.create_opposite_edge(graph.undirected, 0)

                        probs = np.array(
                            [edge_similarity(graph, primary_edge, secondary_edge, secondary_event_start_time=checked_timestamp,
                                             secondary_edge_expected_timestamp=(checked_timestamp - pattern_timestamps[pattern_edge_index][which_instance]),
                                             time_unit=secondary_time_unit, edge_schema=graph.edge_schema,
                                             use_vertex_attributes=use_vertex_attributes)
                             for secondary_edge in graph_allowed_edges])

                        probs_sum = probs.sum()
                        if probs_sum == 0:
                            # we cannot select anything, so continue again just with pattern
                            walking_pattern_only = True
                        else:
                            # we are able to select an edge in the graph, so select one and compute the similarity
                            new_probs = probs / probs.sum()
                            selected_graph_edge_index = np.random.choice(np.arange(len(new_probs)), 1, p=new_probs)[0]
                            selected_graph_edge = graph_allowed_edges[selected_graph_edge_index]
                            computed_similarity = probs[selected_graph_edge_index]
                            already_visited_graph_edges.add(selected_graph_edge.original_edge_id)
                            edges_scores[pattern_edge_index] += computed_similarity
                            current_graph_vertex = selected_graph_edge.to_vertex_id
                            # add the dst vertices to he mapping (from pattern dst vertex to graph dst vertex)
                            vertex_mapping[next_current_pattern_vertex] = selected_graph_edge.to_vertex_id

                # update the current pattern vertex
                current_pattern_vertex = next_current_pattern_vertex

        weighted_edges_scores = np.array(edges_scores) * edge_weights

        total_score = 0.0
        for i in range(len(weighted_edges_scores)):
            if edges_counts[i] > 0:
                total_score += weighted_edges_scores[i] / edges_counts[i]

        all_total_scores.append(total_score)

    return np.array(all_total_scores).mean()
    # return np.array(all_total_scores).max()
