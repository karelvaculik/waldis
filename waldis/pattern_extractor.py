# -*- coding: utf-8 -*-

import sys
from collections import Counter


def _get_vertex_set_of_a_vertex(vertex_mapping, instance_id, vertex_id):
    if instance_id not in vertex_mapping:
        return None
    else:
        if vertex_id not in vertex_mapping[instance_id]:
            return None
        else:
            return vertex_mapping[instance_id][vertex_id]


def extract_pattern(graph, positive_event_vertices, edge_pairs_dictionary_positive, edge_pairs_dictionary_negative,
                    edge_pairs_counter, n_positive, n_negative, max_pattern_edges):

    pattern_edges_done = 0

    pair_index = 0
    # here we keep all (INSTANCE, ORIGINAL EDGE ID) pairs of already selected edges, so we can check that
    # we do not select an edge twice for an instance
    all_occupied_original_edges_ids_with_instances = set()

    # here we keep the selected edges - there is a tuple of edges for each pattern edge
    # (more precisely, there is no tuple but a dictionary INSTANCE -> EDGE ID)
    edge_tuples = []
    # here we keep the score of each edge
    # edges from starting pair has the same score - from the starting pair stats
    # other edges have the score from edge_pairs_dictionary, where they were selected
    edge_tuples_scores = []
    edge_tuples_negative_scores = []

    # here we have mapping of vertices to vertex-sets for each instance
    # vertex-set is a set of vertices that should be mapped to each other when doing isomorphism
    # this dictionary has key for each instance and value is another map from vertex id to vertex set id
    vertex_mapping = dict()
    vertex_sets_current_id = 0

    # here is a little different mapping; for each instance, there is dictionary vertex-set -> vertex-id
    # this serves for preventing edge contractions, because there must be always at most one vertex-id
    vertex_mapping_reversed = dict()

    # add the initial pattern nodes:
    for vertex_list in positive_event_vertices:
        for instance_id, vertex_id in enumerate(vertex_list):
            if instance_id not in vertex_mapping:
                vertex_mapping[instance_id] = dict()
                vertex_mapping_reversed[instance_id] = dict()
            vertex_mapping[instance_id][vertex_id] = vertex_sets_current_id
            vertex_mapping_reversed[instance_id][vertex_sets_current_id] = vertex_id
        vertex_sets_current_id += 1

    while pattern_edges_done < max_pattern_edges and pair_index < len(edge_pairs_counter):
        # in each loop, we try to select one pattern edge from the parallel edges obtained from random walk
        # starting pair is in the following form: (((inst1, edge1), (inst2, edge2)), score)
        starting_pair = edge_pairs_counter[pair_index]

        occupied_instances = set([x[0] for x in starting_pair[0]])
        # here we have ids together with instances
        occupied_edges_ids_with_instances = set(starting_pair[0])
        # for each instance, we keep the score here:
        occupied_edges_scores = {k: starting_pair[1] for k in occupied_instances}

        # and what are the (INSTANCE, ORIGINAL EDGE ID) pairs of these two:
        occupied_original_edges_ids_with_instances = set(
            [(x[0], graph.edges[x[1]].original_edge_id) for x in starting_pair[0]])

        # what are the vertex sets of the from and to vertices in those two instances:
        first_from_vertex = graph.edges[starting_pair[0][0][1]].from_vertex_id
        first_to_vertex = graph.edges[starting_pair[0][0][1]].to_vertex_id
        second_from_vertex = graph.edges[starting_pair[0][1][1]].from_vertex_id
        second_to_vertex = graph.edges[starting_pair[0][1][1]].to_vertex_id

        vert_set_first_from = _get_vertex_set_of_a_vertex(vertex_mapping, starting_pair[0][0][0], first_from_vertex)
        vert_set_first_to = _get_vertex_set_of_a_vertex(vertex_mapping, starting_pair[0][0][0], first_to_vertex)
        vert_set_second_from = _get_vertex_set_of_a_vertex(vertex_mapping, starting_pair[0][1][0], second_from_vertex)
        vert_set_second_to = _get_vertex_set_of_a_vertex(vertex_mapping, starting_pair[0][1][0], second_to_vertex)

        # if vertices from both instances have assigned vertex sets, then these vertex sets must be the same
        are_from_vertices_consistent = vert_set_first_from is None or vert_set_second_from is None or vert_set_first_from == vert_set_second_from
        are_to_vertices_consistent = vert_set_first_to is None or vert_set_second_to is None or vert_set_first_to == vert_set_second_to

        # if some of those first 4 vertices are not in any vertex set yet, put them there
        from_vertex_set = (vert_set_first_from if vert_set_second_from is None else vert_set_second_from)
        to_vertex_set = (vert_set_first_to if vert_set_second_to is None else vert_set_second_to)

        first_instance = starting_pair[0][0][0]
        second_instance = starting_pair[0][1][0]

        # check that we won't contract an edge:
        would_contract_edge = False
        if from_vertex_set is not None and from_vertex_set in vertex_mapping_reversed[first_instance] \
            and vertex_mapping_reversed[first_instance][from_vertex_set] != first_from_vertex:
            would_contract_edge = True
        if from_vertex_set is not None and from_vertex_set in vertex_mapping_reversed[second_instance] \
            and vertex_mapping_reversed[second_instance][from_vertex_set] != second_from_vertex:
            would_contract_edge = True
        if to_vertex_set is not None and to_vertex_set in vertex_mapping_reversed[first_instance] \
            and vertex_mapping_reversed[first_instance][to_vertex_set] != first_to_vertex:
            would_contract_edge = True
        if to_vertex_set is not None and to_vertex_set in vertex_mapping_reversed[second_instance] \
            and vertex_mapping_reversed[second_instance][to_vertex_set] != second_to_vertex:
            would_contract_edge = True

        # check that none of the two starting edges has been selected and
        # the vertices of the selected edges are consistent
        if len(all_occupied_original_edges_ids_with_instances.intersection(
                occupied_original_edges_ids_with_instances)) == 0 \
                and are_from_vertices_consistent and are_to_vertices_consistent and not would_contract_edge:
            # for each selected edge in this while loop, we keep the indices of next candidates
            what_to_check = {k: 0 for k in starting_pair[0]}

            if from_vertex_set is None:
                from_vertex_set = vertex_sets_current_id
                vertex_sets_current_id += 1
            if to_vertex_set is None:
                to_vertex_set = vertex_sets_current_id
                vertex_sets_current_id += 1

            vertex_mapping[starting_pair[0][0][0]][first_from_vertex] = from_vertex_set
            vertex_mapping[starting_pair[0][0][0]][first_to_vertex] = to_vertex_set
            vertex_mapping[starting_pair[0][1][0]][second_from_vertex] = from_vertex_set
            vertex_mapping[starting_pair[0][1][0]][second_to_vertex] = to_vertex_set

            vertex_mapping_reversed[starting_pair[0][0][0]][from_vertex_set] = first_from_vertex
            vertex_mapping_reversed[starting_pair[0][0][0]][to_vertex_set] = first_to_vertex
            vertex_mapping_reversed[starting_pair[0][1][0]][from_vertex_set] = second_from_vertex
            vertex_mapping_reversed[starting_pair[0][1][0]][to_vertex_set] = second_to_vertex

            # try to find more parallel edges until we use all instances
            while len(occupied_instances) < n_positive:
                # take one candidate edge for each already selected edge
                candidate_edges = [(k, edge_pairs_dictionary_positive[k].most_common()[v]) for k, v in what_to_check.items()
                                   if v < len(edge_pairs_dictionary_positive[k].most_common())]
                # if there are no more candidates, we must end this pattern edge
                if len(candidate_edges) == 0:
                    break
                # select the index of the candidate edge with maximal score
                max_idx = max(enumerate(candidate_edges), key=lambda x: x[1][1][1])[0]
                # check that we did not occupy the instance yet and that the edge was not selected (in any round)
                # and that the vertices are consistent
                candidate_instance = candidate_edges[max_idx][1][0][0]
                candidate_edge_id = candidate_edges[max_idx][1][0][1]
                candidate_edge_original_id = graph.edges[candidate_edge_id].original_edge_id

                candidate_edge_from_vertex_set = _get_vertex_set_of_a_vertex(vertex_mapping, candidate_instance,
                                                                             graph.edges[
                                                                                 candidate_edge_id].from_vertex_id)
                candidate_edge_to_vertex_set = _get_vertex_set_of_a_vertex(vertex_mapping, candidate_instance,
                                                                           graph.edges[candidate_edge_id].to_vertex_id)

                are_from_vertices_consistent = candidate_edge_from_vertex_set is None or candidate_edge_from_vertex_set == from_vertex_set
                are_to_vertices_consistent = candidate_edge_to_vertex_set is None or candidate_edge_to_vertex_set == to_vertex_set

                would_contract_edge = False
                if from_vertex_set in vertex_mapping_reversed[candidate_instance] \
                    and vertex_mapping_reversed[candidate_instance][from_vertex_set] != graph.edges[candidate_edge_id].from_vertex_id:
                    would_contract_edge = True
                if to_vertex_set in vertex_mapping_reversed[candidate_instance] \
                    and vertex_mapping_reversed[candidate_instance][to_vertex_set] != graph.edges[candidate_edge_id].to_vertex_id:
                   would_contract_edge = True

                if candidate_instance not in occupied_instances \
                        and (candidate_instance,
                             candidate_edge_original_id) not in all_occupied_original_edges_ids_with_instances \
                        and are_from_vertices_consistent and are_to_vertices_consistent and not would_contract_edge:

                    occupied_instances.add(candidate_instance)
                    occupied_edges_ids_with_instances.add((candidate_instance, candidate_edge_id))
                    occupied_original_edges_ids_with_instances.add((candidate_instance, candidate_edge_original_id))
                    if candidate_edge_from_vertex_set is None:
                        vertex_mapping[candidate_instance][graph.edges[candidate_edge_id].from_vertex_id] = from_vertex_set
                    if candidate_edge_to_vertex_set is None:
                        vertex_mapping[candidate_instance][graph.edges[candidate_edge_id].to_vertex_id] = to_vertex_set
                    # add the new candidate to the what to check list (starting from 0)
                    what_to_check[candidate_edges[max_idx][1][0]] = 0

                    occupied_edges_scores[candidate_instance] = candidate_edges[max_idx][1][1]
                what_to_check[candidate_edges[max_idx][0]] += 1

            # now try to find all negative edges:
            negative_occupied_instances = set()
            negative_what_to_check = {k: 0 for k in occupied_edges_ids_with_instances if k in edge_pairs_dictionary_negative}
            negative_occupied_edges_scores = dict()
            while len(negative_occupied_instances) < n_negative:
                negative_candidate_edges = [(k, edge_pairs_dictionary_negative[k].most_common()[v]) for k, v in negative_what_to_check.items()
                                   if v < len(edge_pairs_dictionary_negative[k].most_common())]
                # if there are no more candidates, we must end this pattern edge
                if len(negative_candidate_edges) == 0:
                    break
                # select the index of the candidate edge with maximal score
                max_idx = max(enumerate(negative_candidate_edges), key=lambda x: x[1][1][1])[0]
                # check that we did not occupy the instance yet
                negative_candidate_instance = negative_candidate_edges[max_idx][1][0][0]
                if negative_candidate_instance not in negative_occupied_instances:
                    negative_occupied_instances.add(negative_candidate_instance)
                    # IMPORTANT: we must multiply the negative score twice so it has the same power
                    # as the positive scores, because positive scores are computed from both sides,
                    # but negative ones only from one side (from positive to negative)
                    # negative_occupied_edges_scores[negative_candidate_instance] = 2 * negative_candidate_edges[max_idx][1][1]
                    negative_occupied_edges_scores[negative_candidate_instance] = negative_candidate_edges[max_idx][1][1]
                negative_what_to_check[negative_candidate_edges[max_idx][0]] += 1

            # this is the end of the round save the current pattern edge
            all_occupied_original_edges_ids_with_instances.update(occupied_original_edges_ids_with_instances)
            edge_tuples.append(occupied_edges_ids_with_instances)
            edge_tuples_scores.append(occupied_edges_scores)
            edge_tuples_negative_scores.append(negative_occupied_edges_scores)
            pattern_edges_done += 1
        pair_index += 1
    return edge_tuples, edge_tuples_scores, edge_tuples_negative_scores, vertex_mapping


def update_positive_scores(pattern_scores, negative_pattern_scores):
    negative_edges_average_scores = []
    # for each pattern edge, compute the average score across negative instances
    for neg_edge_scores in negative_pattern_scores:
        cnt = 0
        total = 0
        for k, v in neg_edge_scores.items():
            total += v
            cnt += 1.0
        if cnt > 0:
            # negative_edges_average_scores.append(total / cnt / 2)
            negative_edges_average_scores.append(total / cnt)
        else:
            negative_edges_average_scores.append(0.0)
    # now subtract the average negative scores from the patterns scores
    updated_pattern_scores = []
    for i, pos_edge_scores in enumerate(pattern_scores):
        edge_scores = dict()
        for k, v in pos_edge_scores.items():
            edge_scores[k] = v - negative_edges_average_scores[i]
        updated_pattern_scores.append(edge_scores)
    return updated_pattern_scores


def compute_margin_scores(pattern_scores):
    # for each instance, keep the total score in such instance
    instance_scores = dict()
    # for each parallel edge, keep the total score
    parallel_edge_scores = []
    for parallel_edge in pattern_scores:
        parallel_edge_score = 0
        for instance, score in parallel_edge.items():
            if instance not in instance_scores:
                instance_scores[instance] = score
            else:
                instance_scores[instance] += score
            parallel_edge_score += score
        parallel_edge_scores.append(parallel_edge_score)
    return instance_scores, parallel_edge_scores


def weighted_set_cover_problem(pattern_scores, instance_scores, parallel_edge_scores):
    all_instances = sorted([k for k, v in instance_scores.items()])
    total_number_of_missing_elements = 0
    cover_instance_elements = dict()
    cover_instance_scores = dict()
    cover_parallel_edge_elements = dict()
    cover_parallel_edge_scores = dict()
    for i in range(len(parallel_edge_scores)):
        for j in all_instances:
            # if there is missing value, add record:
            if j not in pattern_scores[i]:
                # add to instances
                if j not in cover_instance_scores:
                    cover_instance_scores[j] = instance_scores[j]
                    cover_instance_elements[j] = set()
                cover_instance_elements[j].add(total_number_of_missing_elements)

                if i not in cover_parallel_edge_scores:
                    cover_parallel_edge_scores[i] = parallel_edge_scores[i]
                    cover_parallel_edge_elements[i] = set()
                cover_parallel_edge_elements[i].add(total_number_of_missing_elements)

                total_number_of_missing_elements += 1

    # solve greedy weighted cover set problem
    already_covered_elements = set()
    selected_inst = set()
    selected_para = set()
    while len(already_covered_elements) < total_number_of_missing_elements:

        # compute the scores for each considered instance and parallel edge (score is weight / num of new items covered)
        # we want to find the instance or score with minimum score
        # and because Couter saves values in descending order, we save negative score
        pairs_cover_instance_scores_relative = Counter()
        for index, score in cover_instance_scores.items():
            if index not in selected_inst:
                num_of_new_elements = len(
                    already_covered_elements.union(cover_instance_elements[index]).difference(already_covered_elements))
                if num_of_new_elements == 0:
                    relative_score = sys.float_info.max
                else:
                    relative_score = score / num_of_new_elements
                # save negative score (because the Counter sorts in descending order)
                pairs_cover_instance_scores_relative[index] = -relative_score

        pairs_cover_parallel_edge_scores_relative = Counter()
        for index, score in cover_parallel_edge_scores.items():
            if index not in selected_para:
                num_of_new_elements = len(
                    already_covered_elements.union(cover_parallel_edge_elements[index]).difference(
                        already_covered_elements))
                if num_of_new_elements == 0:
                    relative_score = sys.float_info.max
                else:
                    relative_score = score / num_of_new_elements
                # save negative score (because the Counter sorts in descending order)
                pairs_cover_parallel_edge_scores_relative[index] = -relative_score

        # the counter returns pair (index, -score)
        inst_index_score = (pairs_cover_instance_scores_relative.most_common(1)[0] if len(
            pairs_cover_instance_scores_relative) > 0 else None)
        para_index_score = (pairs_cover_parallel_edge_scores_relative.most_common(1)[0] if len(
            pairs_cover_parallel_edge_scores_relative) > 0 else None)

        # at least one of them is not None
        if inst_index_score is None:
            # take para
            already_covered_elements.update(cover_parallel_edge_elements[para_index_score[0]])
            selected_para.add(para_index_score[0])
        elif para_index_score is None or inst_index_score[1] > para_index_score[1]:
            # because the scores are reverted to be negative, we inst_index_score[1] > para_index_score[1] means
            # that the the actual score of inst is lower
            # take inst
            already_covered_elements.update(cover_instance_elements[inst_index_score[0]])
            selected_inst.add(inst_index_score[0])
        else:
            # take para:
            already_covered_elements.update(cover_parallel_edge_elements[para_index_score[0]])
            selected_para.add(para_index_score[0])

    return selected_inst, selected_para


def reduce_pattern(pattern, dropped_inst, dropped_para, parallel_edge_scores):
    """
    Removes the given instances from the pattern as well as given pattern edges.
    Furthermore, it removes pattern edges that have negative score.

    :param pattern:
    :param dropped_inst:
    :param dropped_para:
    :param parallel_edge_scores:
    :return:
    """
    # merge the indices of the edges to be dropped with those that have negative score
    dropped_pattern_edges = dropped_para.union([i for i, x in enumerate(parallel_edge_scores) if x < 0])
    # remove parallel edges
    reduced_pattern = [element for i, element in enumerate(pattern) if i not in dropped_pattern_edges]
    reduced_scores = [element for i, element in enumerate(parallel_edge_scores) if i not in dropped_pattern_edges]
    # remove instances
    result_patterns = [set([y for y in x if y[0] not in dropped_inst]) for x in reduced_pattern]
    return result_patterns, reduced_scores


def interpret_pattern(graph, pattern, vertex_mapping, starting_times):
    edges = []
    edges_attributes = []
    edges_timestamps = []
    edges_directions = []
    for pattern_edge in pattern:
        attributes_list = [graph.edges[x[1]].attributes for x in pattern_edge]
        timestamps_list = [starting_times[x[0]] - graph.edges[x[1]].timestamp for x in pattern_edge]
        first_instance, first_edge = next(iter(pattern_edge))
        new_from_vertex = vertex_mapping[first_instance][graph.edges[first_edge].from_vertex_id]
        new_to_vertex = vertex_mapping[first_instance][graph.edges[first_edge].to_vertex_id]
        edges.append((new_from_vertex, new_to_vertex))
        edges_attributes.append(attributes_list)
        edges_timestamps.append(timestamps_list)
        edges_directions.append(graph.edges[first_edge].direction)
    return edges, edges_attributes, edges_timestamps, edges_directions
