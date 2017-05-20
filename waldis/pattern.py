# -*- coding: utf-8 -*-


from waldis.common_utils import AttributeType
import matplotlib.pyplot as plt
import numpy as np
import math


class Pattern:
    def __init__(self, pattern_edges, pattern_attributes, pattern_timestamps, pattern_directions, pattern_scores, edge_schema):
        self.pattern_edges = pattern_edges
        self.pattern_attributes = pattern_attributes
        self.pattern_timestamps = pattern_timestamps
        self.pattern_directions = pattern_directions
        self.pattern_scores = pattern_scores
        self.edge_schema = edge_schema

    def plot_statistics(self, add_one_to_timestamp=False):
        num_of_plots = len(self.edge_schema) + 2
        plt.figure(1)
        index = 1

        xticks = np.arange(len(self.pattern_edges))

        for attr_name, attr_type in self.edge_schema:
            if attr_type == AttributeType.NUMERIC:
                values = [[y[attr_name] for y in x] for x in self.pattern_attributes]

                plt.subplot(num_of_plots, 1, index)
                plt.subplots_adjust(hspace=0.5)
                plt.title("Attribute: " + attr_name)
                plt.xlabel('pattern edge id')
                plt.xticks(xticks, xticks + 1)
                plt.boxplot(values)
            elif attr_type == AttributeType.NOMINAL:
                par_edges_cnt = len(self.pattern_attributes)
                possible_values = list(set.union(*[set([y[attr_name] for y in x]) for x in self.pattern_attributes]))
                value_counts = {p: [0]*par_edges_cnt for p in possible_values}

                for i, par_edge in enumerate(self.pattern_attributes):
                    for e in par_edge:
                        value_counts[e[attr_name]][i] += 1

                n_rows = len(value_counts)
                colors = plt.cm.jet(np.linspace(0, 1.0, n_rows))
                i_value = 0
                y_offset = np.array([0.0] * par_edges_cnt)
                bar_width = 0.4
                plt.subplot(num_of_plots, 1, index)
                plt.subplots_adjust(hspace=0.5)
                nrows = math.ceil(len(value_counts) / 4)
                title_y_shift = (1.15 if nrows == 1 else (1.25 if nrows == 2 else (1.4 if nrows >= 3 else 1.6)))
                plt.title("Attribute: " + attr_name, y=title_y_shift)
                plt.xlabel('pattern edge id')

                for k, v in value_counts.items():
                    plt.bar(np.arange(par_edges_cnt), v, bar_width, bottom=y_offset, color=colors[i_value],
                            label=k)
                    y_offset = y_offset + np.array(v)
                    i_value += 1
                plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)
                plt.xticks(xticks, xticks + 1)
                plt.show()
            index += 1

        plt.subplot(num_of_plots, 1, index)
        plt.subplots_adjust(hspace=0.5)
        plt.title("relative time before events")
        plt.xlabel('pattern edge id')
        plt.xticks(xticks, xticks + 1)
        if add_one_to_timestamp:
            new_timestamps = [[y + 1 for y in x] for x in self.pattern_timestamps]
            plt.boxplot(new_timestamps)
        else:
            plt.boxplot(self.pattern_timestamps)
        index += 1
        plt.subplot(num_of_plots, 1, index)
        plt.subplots_adjust(hspace=0.5)
        plt.title("edge scores")
        plt.xlabel('pattern edge id')
        plt.xticks(xticks, xticks + 1)
        plt.bar(np.arange(len(self.pattern_scores)), self.pattern_scores, align='center', alpha=0.5)

    def __str__(self):
        return str(self.pattern_edges) + "\n" + str(self.pattern_attributes) + "\n" + str(self.pattern_timestamps) \
            + "\n" + str(self.pattern_directions) + "\n" + str(self.pattern_scores)

    def __repr__(self):
        return self.__str__()

