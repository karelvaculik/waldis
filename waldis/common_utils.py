# -*- coding: utf-8 -*-

import datetime
import math
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt


class AttributeType(Enum):
    NUMERIC = 0
    NOMINAL = 1


def index_binary_search_left(array, x):
    """
    Searches for the position of a value in an array.
    If such element is not in the array, it returns the index of the largest value that is smaller than the given value.
    If the element is smaller than all values in the array, -1 is returned.
    It is assumed that there are no duplicities in the array.
    
    :param array: array to be searched
    :param x: value to be found
    :return: index
    """
    if len(array) == 0:
        return -1
    first = 0
    last = len(array) - 1

    while first <= last:
        index = (first + last) // 2
        if array[index] == x:
            return index
        else:
            if x < array[index]:
                last = index - 1
            else:
                first = index + 1
    if array[index] > x:
        return index - 1
    else:
        return index


def expon_pdf(x, lamb, time_unit):
    if x < 0:
        return 0.0
    else:
        return lamb * math.exp(-lamb * x / time_unit)


def convert_to_float(x):
    """
    Converts the value to a float. If it is a datetime object then it returns seconds since epoch.

    :param x: value to be converted.

    :return: float
    """
    if isinstance(x, datetime.datetime):
        return x.timestamp()
    else:
        return float(x)


def plot_results(graph, edges_attributes, edges_timestamps, edge_scores):
    num_of_plots = len(graph.edge_schema) + 2
    plt.figure(1)
    index = 1
    for attr_name, attr_type in graph.edge_schema:
        if attr_type == AttributeType.NUMERIC:
            values = [[y[attr_name] for y in x] for x in edges_attributes]

            plt.subplot(num_of_plots, 1, index)
            plt.title(attr_name)
            plt.boxplot(values)
        elif attr_type == AttributeType.NOMINAL:
            par_edges_cnt = len(edges_attributes)
            possible_values = list(set.union(*[set([y[attr_name] for y in x]) for x in edges_attributes]))
            value_counts = {p: [0]*par_edges_cnt for p in possible_values}

            for i, par_edge in enumerate(edges_attributes):
                for e in par_edge:
                    value_counts[e[attr_name]][i] += 1

            n_rows = len(value_counts)
            colors = plt.cm.jet(np.linspace(0, 1.0, n_rows))
            i_value = 0
            y_offset = np.array([0.0] * par_edges_cnt)
            bar_width = 0.4
            plt.subplot(num_of_plots, 1, index)
            plt.title(attr_name)
            for k, v in value_counts.items():
                plt.bar(np.arange(par_edges_cnt), v, bar_width, bottom=y_offset, color=colors[i_value])
                y_offset = y_offset + np.array(v)
                i_value += 1
            plt.show()
        index += 1

    plt.subplot(num_of_plots, 1, index)
    plt.title("time")
    plt.boxplot(edges_timestamps)
    index += 1
    plt.subplot(num_of_plots, 1, index)
    plt.title("edge scores")
    plt.bar(np.arange(len(edge_scores)), edge_scores, align='center', alpha=0.5)

