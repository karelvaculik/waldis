# -*- coding: utf-8 -*-


import numpy as np
from waldis.dynamic_graph import Edge
from waldis.random_walker import edge_similarity


def check_pattern_in_instance(graph, pattern_edges, pattern_attributes, pattern_timestamps, pattern_directions,
                              graph_starting_vertices, graph_starting_timestamp, secondary_time_unit, edge_weights,
                              use_vertex_attributes=True, num_of_random_walks=100):
    """
    Looks into the graph, whether there is the given pattern on  with respect to the given
    vertices and timestamp.

    :param graph:
    :param pattern_edges: list of pairs (src, dst)
    :param pattern_attributes:
    :param pattern_timestamps:
    :param graph_starting_vertices:
    :param graph_starting_timestamp: assumed event time in the graph
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
    # not all starting vertices occur in the pattern, so use 0 selection probability for those missing
    # starting_vertices_probs = np.repeat(0.0, len(starting_vertices))
    starting_vertices_usable = np.repeat(False, len(graph_starting_vertices))
    for pattern_edge in pattern_edges:
        if pattern_edge[0] < len(graph_starting_vertices):
            # starting_vertices_probs[pattern_edge[0]] = 1.0
            starting_vertices_usable[pattern_edge[0]] = True
        if pattern_edge[1] < len(graph_starting_vertices):
            # starting_vertices_probs[pattern_edge[1]] = 1.0
            starting_vertices_usable[pattern_edge[1]] = True
    # starting_vertices_probabilities = starting_vertices_probs / starting_vertices_probs.sum()

    all_total_scores = []

    # go through all pattern instances
    for pattern_instance_index in range(number_of_instances_in_pattern):

        walks_scores = list()
        for i in range(num_of_random_walks):
            # try one mapping of the pattern

            # what is the score of this walk and how many steps we did
            walk_score = 0.0
            walk_length = 0.0
            # we don't allow to use one pattern edge many times (both in pattern and graph) in one random walk
            already_visited_pattern_edges = set()
            # here we keep the original ids used in the graph in this single random walk
            already_visited_graph_edges = set()
            # first, choose the starting vertices (for the pattern and the corresponding one from graph)

            # here we keep the vertices from which we can go one edge
            available_vertices = set()
            # here we keep the vertices that have been already added to available_vertices
            # so that we do not add them there again (once they are removed)

            already_tried_vertices = set()
            # keep the mapping from pattern vertices to graph vertices and use it to check the vertex consistency:
            vertex_mapping = dict()
            for pattern_vertex in range(len(graph_starting_vertices)):
                # but add only those that are in the pattern
                if starting_vertices_usable[pattern_vertex]:
                    available_vertices.add(pattern_vertex)
                    already_tried_vertices.add(pattern_vertex)
                    vertex_mapping[pattern_vertex] = graph_starting_vertices[pattern_vertex]

            # if walking in the graph goes wrong, we continue only in the pattern
            # walking_pattern_only = False

            # now perform the occupation of the graph by the pattern
            while len(available_vertices) > 0:
                # while we have some available vertices (i.e. available edges)
                # pick one edge randomly
                current_pattern_vertex = np.random.choice(list(available_vertices))
                current_graph_vertex = vertex_mapping[current_pattern_vertex]

                # which pattern edges can be used to move forward
                which_adjacent_edges_could_be_used = [i for i, x in
                                                      enumerate(adjacency_links_to_edges[current_pattern_vertex]) if
                                                      x not in already_visited_pattern_edges]
                if len(which_adjacent_edges_could_be_used) == 0:
                    # if there are no available pattern edges going from this vertex,
                    # remove it from our set and try another round
                    available_vertices.remove(current_pattern_vertex)
                    continue

                # if there are some usable edges, select one at random
                index_in_adjacency_list = np.random.choice(which_adjacent_edges_could_be_used, 1)[0]
                pattern_edge_index = adjacency_links_to_edges[current_pattern_vertex][index_in_adjacency_list]

                # increase the counts
                # edges_counts[pattern_edge_index] += 1 # OLD
                if pattern_attributes[pattern_edge_index][pattern_instance_index] is not None:
                    # NEW - don't count walk length if there None here
                    walk_length += 1  # NEW
                already_visited_pattern_edges.add(pattern_edge_index)

                pattern_to_vertex = adjacency_list[current_pattern_vertex][index_in_adjacency_list]

                if pattern_to_vertex not in already_tried_vertices:
                    # if the vertex is not in the available_vertices, add it there
                    # available_vertices.add(pattern_to_vertex)
                    already_tried_vertices.add(pattern_to_vertex)

                graph_edges = graph.adjacency_list[current_graph_vertex]
                graph_allowed_edges = [e for e in graph_edges if e.original_edge_id not in already_visited_graph_edges]

                # remove edges that have inconsistent vertices (we check dst vertex)
                if pattern_to_vertex in vertex_mapping:
                    graph_allowed_edges = [e for e in graph_allowed_edges if
                                           e.to_vertex_id == vertex_mapping[pattern_to_vertex]]
                if len(graph_allowed_edges) == 0:
                    # update available vertices and continue by selecting another edge
                    available_vertices.remove(current_pattern_vertex)
                    continue
                else:
                    # select the edge in the graph and move forward both in the graph and pattern
                    # only attributes and direction are necessary for edge similarity function
                    if pattern_attributes[pattern_edge_index][pattern_instance_index] is not None:
                        # don't count walk length if there None here
                        primary_edge = Edge(0, 0, 0, timestamp=0,
                                            attributes=pattern_attributes[pattern_edge_index][pattern_instance_index],
                                            direction=pattern_directions[pattern_edge_index], original_edge_id=None)
                        # now check whether we should take the opposite:
                        if adjacency_list_is_opposite[current_pattern_vertex][index_in_adjacency_list]:
                            # take the opposite:
                            primary_edge = primary_edge.create_opposite_edge(graph.undirected, 0)

                        probs = np.array(
                            [edge_similarity(graph, primary_edge, secondary_edge,
                                                 secondary_event_start_time=graph_starting_timestamp,
                                                 secondary_edge_expected_timestamp=(
                                                         graph_starting_timestamp -
                                                         pattern_timestamps[pattern_edge_index][
                                                             pattern_instance_index]),
                                                 time_unit=secondary_time_unit, edge_schema=graph.edge_schema,
                                                 use_vertex_attributes=use_vertex_attributes)
                             for secondary_edge in graph_allowed_edges])
                    else:
                        # if the pattern edge is None, you can use any edge from the graph
                        probs = np.repeat(1.0, len(graph_allowed_edges))

                    probs_sum = probs.sum()
                    if probs_sum == 0:
                        # we cannot select anything, so continue by selecting another edge
                        # also update the available vertices
                        available_vertices.remove(current_pattern_vertex)
                        continue
                    else:
                        # we are able to select an edge in the graph, so select one and compute the similarity
                        new_probs = probs / probs.sum()
                        selected_graph_edge_index = np.random.choice(np.arange(len(new_probs)), 1, p=new_probs)[0]
                        selected_graph_edge = graph_allowed_edges[selected_graph_edge_index]
                        computed_similarity = probs[selected_graph_edge_index]
                        already_visited_graph_edges.add(selected_graph_edge.original_edge_id)
                        # edges_scores[pattern_edge_index] += computed_similarity # OLD
                        if pattern_attributes[pattern_edge_index][pattern_instance_index] is not None:
                            # NEW - update score only if the edge wasn't None
                            # walk_score += computed_similarity * edge_weights[pattern_edge_index]  # NEW
                            walk_score += computed_similarity  # NEW

                        # current_graph_vertex = selected_graph_edge.to_vertex_id
                        # available_vertices.add(pattern_to_vertex)

                        if pattern_to_vertex not in already_tried_vertices:
                            # if the vertex is not in vertex_mapping, it means that it is not in the available_vertices,
                            # so add it there
                            available_vertices.add(pattern_to_vertex)
                            # already_tried_vertices.add(pattern_to_vertex)
                            # add the dst vertices to he mapping (from pattern dst vertex to graph dst vertex)
                            vertex_mapping[pattern_to_vertex] = selected_graph_edge.to_vertex_id

            # after walk is done, save the walk score divided by the walk length
            if walk_length > 0:
                walks_scores.append(walk_score / walk_length)
            else:
                walks_scores.append(0.0)

        # now take the maximum walk score and save it as a score of this pattern instance:
        all_total_scores.append(max(walks_scores))

    # NOW RETURN mean across all pattern instances
    return np.array(all_total_scores).mean()
