# -*- coding: utf-8 -*-

import gzip
import json
import numpy as np
from waldis.common_utils import index_binary_search_left, convert_to_float, AttributeType


__all__ = ["Vertex", "Edge", "DynamicGraph"]


class JsonEncoder(json.JSONEncoder):
    """
    Json encoder used for DynamicGraph serialization.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)


class Vertex:
    def __init__(self, vertex_id, attributes):
        """
        Creates a vertex with given id from an array of dicts representing attributes
        and an array of corresponding timestamps.
        It is assumed that timestamps are already ordered and that i-th timestamp represents the beginning
        of the interval with i-th row of attribute values on the vertex.
        Value - np.inf can be used for the first timestamp to denote no beginning.
        timestamps can contain either int, float or objects of datetime.datetime class.

        :param vertex_id:
        :param timestamps:
        :param attributes:
        """
        # if len(timestamps) != attributes.shape[0]:
        #     raise ValueError("length of timestamps must be equal to the number rows in attributes dataframe")
        self.vertex_id = vertex_id
        # self.timestamps = np.array([convert_to_float(x) for x in timestamps])
        self.attributes = attributes

    # def get_attributes_at_timestamp(self, timestamp):
    #     """
    #     It returns a pandas Series with attributes for a given timestamp
    #     :param timestamp:
    #     :return:
    #     """
    #
    #     index = index_binary_search_left(self.timestamps, timestamp)
    #     if index == -1:
    #         return None
    #     else:
    #         return self.attributes[index]

    def to_dict(self):
        """
        Creates dict representation of the vertex which can be used for json serialization.
        
        :return: dict
        """
        # return {"id": self.vertex_id, "data_original": [{"timestamp": self.timestamps[i], "attributes": self.attributes[i]}
        #                                        for i in range(len(self.timestamps))]}
        return {"id": self.vertex_id, "attributes": self.attributes}

    @staticmethod
    def from_dict(vertex_dict):
        """
        Creates a new Vertex object from a dictionary.
        
        :param vertex_dict: dict
        :return: Vertex object
        """
        vertex_id = vertex_dict["id"]
        attributes = vertex_dict["attributes"]
        # timestamps = np.array([x["timestamp"] for x in vertex_dict["data_original"]])
        # attributes = np.array([x["attributes"] for x in vertex_dict["data_original"]])
        return Vertex(vertex_id=vertex_id, attributes=attributes)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.vertex_id == other.vertex_id and (self.timestamps == other.timestamps).all() and \
                   (self.attributes == other.attributes).all()
        return False

    def __str__(self):
        # return "V: " + str(self.vertex_id) + " | " + str(self.timestamps) + " | " + str(self.attributes)
        return "V: " + str(self.vertex_id) + " | " + str(self.attributes)

    def __repr__(self):
        return self.__str__()


class Edge:
    def __init__(self, edge_id, from_vertex_id, to_vertex_id, timestamp, attributes,
                 direction=True, original_edge_id=None):
        # each edge has unique original_edge_id, however, for each original_edge_id
        # there can be two edges with different edge_ids denoting opposite directions of the same edge
        # edge_id is needed to reconstruct the patterns from random walks
        # original_edge_id is needed for checking whether the edge is already occupied (walked)
        self.edge_id = edge_id
        # this original_edge_id is either same as edge_id, or it is different when this is the opposite edge
        self.original_edge_id = (original_edge_id if original_edge_id is not None else edge_id)
        self.from_vertex_id = from_vertex_id
        self.to_vertex_id = to_vertex_id
        self.timestamp = convert_to_float(timestamp)
        self.attributes = attributes
        self.direction = direction

    def create_opposite_edge(self, undirected, new_edge_id):
        """
        Creates a new edge that is opposite to the edge itself.
        If the edge is undirected, the direction is always True,
        if it is directed then the direction is opposite.
        It will have the same original edge id but the edge_id will be different.
        
        :param undirected: whether the original edge and the new one are undirected
        :param new_edge_id: new edge id to be used by the new edge
        :return: Edge
        """
        return Edge(new_edge_id, self.to_vertex_id, self.from_vertex_id,
                    self.timestamp, self.attributes, undirected or not self.direction,
                    self.original_edge_id)

    def to_dict(self):
        """
        Creates dict representation of the edge which can be used for json serialization.
        
        :return: dict
        """
        return {"id": self.edge_id, "from_vertex_id": self.from_vertex_id, "to_vertex_id": self.to_vertex_id,
                "timestamp": self.timestamp, "attributes": self.attributes, "direction": self.direction}

    @staticmethod
    def from_dict(edge_dict):
        """
        Creates a new Edge object from a dictionary.
        
        :param edge_dict: dict
        :return: Edge object
        """
        return Edge(edge_id=edge_dict["id"], from_vertex_id=edge_dict["from_vertex_id"],
                    to_vertex_id=edge_dict["to_vertex_id"], timestamp=edge_dict["timestamp"],
                    attributes=edge_dict["attributes"], direction=edge_dict["direction"])

    def __str__(self):
        return "E: " + str(self.edge_id) + " | " + str(self.from_vertex_id) \
            + (" -> " if self.direction else " <- ") \
            + str(self.to_vertex_id) + " | " + str(self.attributes) + " | " + str(self.timestamp)

    def __repr__(self):
        return self.__str__()


class DynamicGraph:
    def __init__(self, vertices, edges, vertex_schema=[], edge_schema=[], undirected=True):
        self.vertices = dict()
        for v in vertices:
            self.vertices[v.vertex_id] = v
        self.edges = dict()
        # we store maximum edge id + 1 in m variable
        m = -1
        for e in edges:
            self.edges[e.edge_id] = e
            m = max(m, e.edge_id)
        m += 1
        # now we store opposite edges with
        for e in edges:
            self.edges[e.edge_id + m] = e.create_opposite_edge(undirected, e.edge_id + m)
        self.vertex_schema = vertex_schema
        self.edge_schema = edge_schema
        self.undirected = undirected
        self._create_adjacency_list(vertices, edges, m)

    def _create_adjacency_list(self, vertices, edges, m):
        """
        Creates adjacency list from list of vertices and edges.
        More specifically, it creates a dict where keys are vertex ids and values are list of edges.
        It also adds opposite edges.
        
        :param vertices: list of vertices
        :param edges: list of edges
        :param m: virtual edge id shift for opposite edges
        :return: dict
        """
        self.adjacency_list = dict()
        # first, initialise empty lists for all vertices
        for v in vertices:
            self.adjacency_list[v.vertex_id] = []
        # now add all edges
        for e in edges:
            self.adjacency_list[e.from_vertex_id].append(e)
        # and also the opposite way
        for e in edges:
            self.adjacency_list[e.to_vertex_id].append(e.create_opposite_edge(self.undirected, e.edge_id + m))

    def get_from_vertex_ids_of_edges(self, edge_ids):
        return [self.edges[k].from_vertex_id for k in edge_ids]

    def get_to_vertex_ids_of_edges(self, edge_ids):
        return [self.edges[k].to_vertex_id for k in edge_ids]

    def to_dict(self):
        """
        Creates dict representation of the dynamic graph which can be used for json serialization.
        It saves only the original ids, not the artificial opposite ones.
        
        :return: dict
        """
        def _encode_attribute_type(t):
            if t == AttributeType.NUMERIC:
                return "NUMERIC"
            elif t == AttributeType.NOMINAL:
                return "NOMINAL"
            else:
                return None
        encoded_vertex_schema = [(x[0], _encode_attribute_type(x[1])) for x in self.vertex_schema]
        encoded_edge_schema = [(x[0], _encode_attribute_type(x[1])) for x in self.edge_schema]
        return {"undirected": self.undirected,
                "vertex_schema": encoded_vertex_schema,
                "edge_schema": encoded_edge_schema,
                "vertices": [v.to_dict() for _, v in self.vertices.items()],
                "edges": [e.to_dict() for _, e in self.edges.items() if e.edge_id == e.original_edge_id]}

    def to_json_file(self, filename, gz=False):
        """
        Serializes the dynamic graph into a text file.
        
        :param filename: name of the file
        :param gz: whether to gzip the file
        """
        if gz:
            with gzip.GzipFile(filename, "w") as data_file:
                data_file.write(json.dumps(self.to_dict(), cls=JsonEncoder).encode('utf-8'))
        else:
            with open(filename, "w") as data_file:
                json.dump(self.to_dict(), data_file, cls=JsonEncoder)

    @staticmethod
    def from_json_file(filename, gz=False):
        """
        Deserializes the dynamic graph from a text file.
        The file can be compressed as tar.gz.
        If that is the case, there must be only one file.
        
        :param filename: name of the file.
        :param gz: whether the file is gzipped
        :return: DynamicGraph object
        """
        def _decode_attribute_type(t):
            if t == "NUMERIC":
                return AttributeType.NUMERIC
            elif t == "NOMINAL":
                return AttributeType.NOMINAL
            else:
                return None
        if gz:
            with gzip.GzipFile(filename, "r") as data_file:
                json_bytes = data_file.read()
                json_str = json_bytes.decode('utf-8')
                data = json.loads(json_str)
        else:
            with open(filename) as data_file:
                data = json.load(data_file)
        vertices = [Vertex.from_dict(vertex_dict) for vertex_dict in data["vertices"]]
        edges = [Edge.from_dict(edge_dict) for edge_dict in data["edges"]]
        decoded_vertex_schema = [(x[0], _decode_attribute_type(x[1])) for x in data["vertex_schema"]]
        decoded_edge_schema = [(x[0], _decode_attribute_type(x[1])) for x in data["edge_schema"]]
        return DynamicGraph(vertices=vertices, edges=edges, vertex_schema=decoded_vertex_schema,
                            edge_schema=decoded_edge_schema, undirected=data["undirected"])

    def identify_vertex_events(self, label):
        raise NotImplementedError

    def identify_edge_events(self, attr_name, label, sample_size = 10):
        result_edge_ids = []
        from_vertex_ids = []
        from_vertex_timestamps = []
        to_vertex_ids = []
        to_vertex_timestamps = []

        vertex_ids = None
        vertex_timestamps = None

        for k, e in self.edges.items():
            # take only edges satisfying the label condition and only the original edges
            if e.attributes[attr_name] == label and e.edge_id == e.original_edge_id:
                result_edge_ids.append(k)
                from_vertex_ids.append(e.from_vertex_id)
                from_vertex_timestamps.append(e.timestamp)
                to_vertex_ids.append(e.to_vertex_id)
                to_vertex_timestamps.append(e.timestamp)

        selected_ids = np.random.choice(range(len(result_edge_ids)), size=sample_size, replace=False)
        result_edge_ids = [result_edge_ids[i] for i in selected_ids]
        from_vertex_ids = [from_vertex_ids[i] for i in selected_ids]
        from_vertex_timestamps = [from_vertex_timestamps[i] for i in selected_ids]
        to_vertex_ids = [to_vertex_ids[i] for i in selected_ids]
        to_vertex_timestamps = [to_vertex_timestamps[i] for i in selected_ids]

        if vertex_ids is None:
            vertex_ids = np.array(from_vertex_ids)
            vertex_timestamps = np.array(from_vertex_timestamps)

        vertex_ids = np.vstack([vertex_ids, to_vertex_ids])
        vertex_timestamps = np.vstack([vertex_timestamps, to_vertex_timestamps])

        return np.array([result_edge_ids]), vertex_ids, vertex_timestamps

    def __str__(self):
        my_str = "|V| = " + str(len(self.vertices)) + ", |E| = " + str(len(self.edges))
        for k, v in self.vertices.items():
            my_str += "\n" + v.__str__()
        for k, e in self.edges.items():
            my_str += "\n" + e.__str__()
        return my_str

    def __repr__(self):
        return self.__str__()

