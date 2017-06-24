# WalDis

An algorithm for mining discriminative patterns in dynamic graphs.

## Installation (from the project directory)

```sh
pip3 install .
```

(tested with Python 3)

## Example

```python
from waldis.common_utils import AttributeType
from waldis.dynamic_graph import Vertex, Edge, DynamicGraph
from waldis.waldis import mine_patterns
import numpy as np

vertices = [Vertex(i, {"label": "0"}) for i in range(12)]

edges = [
    Edge(1, 0, 1, 0, {"label": "a"}),
    Edge(2, 0, 2, 2, {"label": "d"}),
    Edge(3, 3, 4, 0, {"label": "a"}),
    Edge(4, 3, 5, 2, {"label": "d"}),
    Edge(5, 6, 7, 0, {"label": "a"}),
    Edge(6, 6, 8, 3, {"label": "x"}),
    Edge(7, 9, 10, 0, {"label": "a"}),
    Edge(8, 9, 11, 3, {"label": "x"}),
]

graph = DynamicGraph(vertices, edges, vertex_schema=[("label", AttributeType.NOMINAL)],
                     edge_schema=[("label", AttributeType.NOMINAL)], undirected=False)

positive_event_vertices = np.array([[0, 3]])
positive_event_times = np.array([5, 5])
negative_event_vertices = np.array([[6, 9]])
negative_event_times = np.array([5, 5])

time_unit_primary = 1
time_unit_secondary = 1

pattern = mine_patterns(graph=graph, positive_event_vertices=positive_event_vertices,
                        positive_event_times=positive_event_times, positive_event_edges=None,
                        negative_event_vertices=negative_event_vertices,
                        negative_event_times=negative_event_times, negative_event_edges=None,
                        use_vertex_attributes=False, time_unit_primary=time_unit_primary,
                        time_unit_secondary=time_unit_secondary,
                        random_walks=1000, prob_restart=0.1, max_pattern_edges=10,
                        verbose=True)

pattern.pattern_edges
pattern.pattern_attributes
pattern.pattern_timestamps
pattern.pattern_scores
```

## Experiments for IDA 2017 paper

- DBLP dataset:

```sh
python3 scripts/dblp_experiments.py
```

- Enron dataset:

```sh
python3 scripts/enron_experiments.py
```

- Telco dataset - unfortunately, this dataset cannot be published

## Results of IDA 2017 experiments

- graphs can be found in `experiments` directory
- NOTE: your graphs may look different when running the experiment scripts as there is no fixed order of barchart groups (the colors of the groups are not fixed either)

## Data preprocessing for IDA 2017 experiments

- scripts used for preparation of networks can be found in `tools` directory
- resulting networks are located in the `data_graphs` directory
- the original data are not part of this project and have to be downloaded from the relevant web pages
- Enron original dataset was downloaded from http://www.cis.jhu.edu/~parky/Enron/
- DBLP original dataset was downloaded from http://dblp.uni-trier.de/xml/ and DBLP2json.py from http://projects.csail.mit.edu/dnd/DBLP/ was used for initial data extraction
