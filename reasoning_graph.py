import random as rd
import string
import math
import itertools


import numpy as np
from matplotlib import pyplot as plt
import networkx as nx


ENTAILMENT_RELATIONS = {"ent", "neu", "con", "¬ent", "¬neu", "¬con"}
GENERIC_RELATIONS = {"&", "source", "target", "left", "right", "pre", "fol"}
RELATIONS = ENTAILMENT_RELATIONS.union(GENERIC_RELATIONS)
NODE_TYPES = ["Statement", "Domain", "EdgeNode", "AndNode", "CompareNode"]
EDGE_TYPES = ["Entailment", "Compare", "Edge", "Union"]
DOMAIN_TYPES = ["CommonSense", "Legal", "Ethics", "NatureScience"]
COMPARE_OPERATORS = {"<", "<=", "="}
CYPHER_OPERATORS = ["CREATE", "MATCH"]
CHAIN_NODE_TYPES = ["Statement", "AndNode"]


class ReasoningGraph():
    def __init__(self, layout = "planar", font_size_nodes = 12, font_size_edges = 10, node_size = 1200, rad = 0.12, scale = 1.0):
        self.reset_graph()

        #cosmetic settings
        self.layout = layout
        self.font_size_nodes = font_size_nodes
        self.font_size_edges = font_size_edges
        self.node_size = node_size
        self.rad = rad
        self.scale = scale

    def reset_graph(self):
        self.changed_nodes = set()
        self.changed_edges = set()
        self.first_display = True
        self.G = nx.DiGraph()
    
    def check_node_in_graph(self, node):
        for found_node, attributes in self.G.nodes(data = True):
            if node == found_node:
                return attributes
        raise Exception(f"Node {node} have to be present in the graph.")
    
    def check_edge_in_graph(self, source, target):
        if not self.G[source][target]:
            raise Exception(f"Edge ({source}, {target}) has to be present in the graph.")
        return self.G[source][target]
    
    def check_entailment_relation(self, relation):
        if relation not in ENTAILMENT_RELATIONS:
            raise Exception(f"Expected relation of {ENTAILMENT_RELATIONS}, but got {relation} instead.")
        
    def check_relation(self, relation):
        if relation not in RELATIONS:
            raise Exception(f"Expected relation of {RELATIONS}, but got {relation} instead.")
        
    def check_operator(self, operator):
        if operator not in COMPARE_OPERATORS:
            raise Exception(f"Expected operator of {COMPARE_OPERATORS}, but got {operator} instead.")
    

    def _generate_random_name(self):
        nodes = self.G.nodes
        available_names = set()
        counter = 0
        while len(available_names) == 0:
            available_names = set(string.ascii_letters)
            for _ in range(counter):
                available_names = set(itertools.product(available_names, string.ascii_letters))
                available_names_joined = set()
                for name in available_names:
                    available_names_joined.add("".join(name))
                available_names = available_names_joined
            for node in nodes:
                if node in available_names:
                    available_names.remove(node)
            counter += 1
        return rd.choice(list(available_names))
    
    def _check_next_chain_node(self, new_node, next_chain_node, new_node_type, chain_direction = "previous"):
        if new_node_type not in CHAIN_NODE_TYPES:
            raise Exception(f"Expected new node in to be of type {CHAIN_NODE_TYPES} but got {new_node_type} instead.")
        chain_nodes = self.resolve_node(next_chain_node, [chain_direction])
        if new_node in chain_nodes:
            raise Exception(f"New node {new_node} is already connected recursivly with {next_chain_node}, see {chain_nodes}.")
        for chain_node, chain_node_attributes in chain_nodes.items():
            if chain_node_attributes["type"] not in CHAIN_NODE_TYPES:
                raise Exception(f"Expected all nodes in chain to be of types {CHAIN_NODE_TYPES} but got {chain_node}:{chain_node_attributes['type']} instead.")

    def check_node(self, text, name = None, type = "Statement", previous = None, following = None, generated = -1, add = True):
        if not type in NODE_TYPES:
            raise Exception(f"Expected node type of {NODE_TYPES} but got {type} instead.")
        if type == "Domain":
            if text not in DOMAIN_TYPES:
                raise Exception(f"Expected domain of {DOMAIN_TYPES} but got {text} instead.")
        if not name:
            for node, attribute in self.G.nodes(data = True):
                if attribute["text"] == text and attribute["type"] == type and previous == attribute["previous"] and attribute["generated"] == generated:
                    return node
            if add:
                name = self._generate_random_name()
            else:
                return None
        if previous:
            self._check_next_chain_node(name, previous, type, "previous")
        if following:
            self._check_next_chain_node(name, following, type, "following")

        self.G.add_node(name, text = text, type = type, previous = previous, following = following, generated = generated)
        self.changed_nodes.add(name)
        if previous:
            self.add_edge(previous, name, relation = "pre", edge_type = "Union", generated=0)
            self.changed_edges.add((previous, name))
        if following:
            self.add_edge(name, following, relation = "fol", edge_type = "Union", generated=0)
            self.changed_edges.add((name, following))
        return name

    def add_edge(self, source, target, relation = "ent", edge_type = "Entailment" , domain = "CommonSense", generated = -1):
        self.check_node_in_graph(source)
        self.check_node_in_graph(target)
        self.check_relation(relation)            
        if domain not in DOMAIN_TYPES:
            raise Exception(f"Expected domain of {DOMAIN_TYPES} but got {domain} instead.")
        if edge_type not in EDGE_TYPES:
            raise Exception(f"Expected edge_type of {EDGE_TYPES} but got {edge_type} instead.")
        self.changed_edges.add((source, target))
        self.G.add_edge(source, target, relation = relation, edge_type = edge_type, generated = generated, domain = domain)

    def check_edge_node(self, source, target, name = None, generate = True):
        self.check_node_in_graph(source)
        self.check_node_in_graph(target)
        for node, type in self.G.nodes(data="type"):
            potential_source = None
            potential_target = None
            if type == "EdgeNode":
                for some_source, _, attributes in self.G.in_edges(node, data = True):
                    if attributes["relation"] == "source" and source == some_source:
                        potential_source = some_source
                    if attributes["relation"] == "target" and target == some_source:
                        potential_target = some_source
                if potential_source and potential_target:
                    if name and name != node:
                        nx.relabel_nodes(self.G, {node:name})
                        return name
                    else:
                        return node
        if generate:
            if not name:
                name = self._generate_random_name()
            relation = self.G.get_edge_data(source, target)["relation"]
            self.check_node(relation, name = name, type = "EdgeNode", generated = 0)
            self.add_edge(source, name, edge_type = "Edge", relation = "source", generated = 0)
            self.add_edge(target, name, edge_type = "Edge", relation = "target", generated = 0)
            return name
        return None
    
    def get_edges(self, nodes = None):
        if not nodes:
            return self.G.edges()
        else:
            edges = []
            for source, target in self.G.edges():
                if source in nodes and target in nodes:
                    edges.append((source, target))
            return edges

    def check_and_node(self, nodes, name = None, previous = None, following = None, add = True):
        if len(nodes) < 2:
            raise Exception("The and-node can only unite two or more nodes.")
        for node in nodes:
            self.check_node_in_graph(node)
        for node, type in self.G.nodes(data = "type"):
            if type == "AndNode":
                source_nodes = set(nodes)
                for source, _, attributes in self.G.in_edges(node, data=True):
                    if attributes["edge_type"] == "Union" and attributes["relation"] == "&" and source in source_nodes:
                        source_nodes.discard(source)
                if len(source_nodes) == 0:
                    if name == node:
                        return name
                    elif name:
                        nx.relabel_nodes(self.G, {node:name})
                        return name
                    else:
                        return node
        if add:
            if not name:
                name = self._generate_random_name()
            self.check_node("&", name = name, type = "AndNode", generated = 0, previous=previous, following=following)
            for node in nodes:
                self.add_edge(node, name, edge_type = "Union", relation = "&", generated = 0)
        return name
    
    def check_compare_node(self, left, right, operator = "<", name = None, generate = True):
        self.check_node_in_graph(left)
        self.check_node_in_graph(right)
        self.check_operator(operator)
        found_node = None
        for node, type in self.G.nodes(data="type"):
            left_target = None
            right_target = None
            if type == "CompareNode":
                for some_source, some_target, relation in self.G.in_edges(node, data = "relation"):
                    if relation == "left" and left == some_source and some_target == node:
                        left_target = some_target
                    elif relation == "right" and right == some_source and some_target == node:
                        right_target = some_target
                if left_target and right_target:
                    if name and name != node:
                        nx.relabel_nodes(self.G, {node:name})
                        found_node = name
                    else:
                        found_node = node
        if generate:
            if not name:
                name = self._generate_random_name()
            self.check_node(operator, name = name, type = "CompareNode", generated = 0)
            self.add_edge(left, name, edge_type = "Compare", relation = "left", generated = 0)
            self.add_edge(right, name, edge_type = "Compare", relation = "right", generated = 0)
            found_node = name
        return found_node

    def resolve_node(self, node, types = ["generic", "previous", "following"]):
        visited = {}

        def resolve_node_recursive(node, direction = None):
            if node not in visited:
                attributes = self.check_node_in_graph(node)
                visited[node] = attributes
                if "generic" in types and attributes["generated"] == 0:
                    in_edges = self.G.in_edges(node)
                    for (source, _) in in_edges:
                        resolve_node_recursive(source, direction)
                elif (not direction or direction == "previous") and attributes["previous"]:
                    for previous_node, _, previous_attributes in self.G.in_edges(node, data = True):
                        if previous_attributes["edge_type"] == "Union" and previous_attributes["relation"] == "pre":
                            resolve_node_recursive(previous_node, "previous")
                elif (not direction or direction == "following") and attributes["following"]:
                    for _, following_node, following_attributes in self.G.out_edges(node, data = True):
                        if following_attributes["edge_type"] == "Union" and following_attributes["relation"] == "fol":
                            resolve_node_recursive(following_node, "following")
        if node:
            resolve_node_recursive(node)
        return visited

        

    def graph_to_cypher(self, nodes = None, operator = "MATCH"):
        """Generates a Cypher query from a Networkx Graph"""
        if operator not in CYPHER_OPERATORS:
            raise Exception(f"Expected operator of {CYPHER_OPERATORS} but got \"{operator}\" instead.")
        if not nodes:
            nodes = self.G.nodes
        query = ""
        defined_nodes = set()
        node_definitions = {}
        #Partially generate the node representations
        for node, attributes in self.G.nodes(data = True):
            if node in nodes:
                node_definition = f"({node}:{attributes['type']}"
                text = attributes['text']
                if attributes['type'] == "Domain":
                    node_definition += f":{text}"
                node_definition += f"{{text:'{self.sanitize_for_cypher(text)}'}})"
                node_definitions[node] = node_definition                

        #Generate the relationship representations
        for source, target, attributes in self.G.edges(data = True):
            if source in nodes and target in nodes:
                if source not in defined_nodes:
                    defined_nodes.add(source)
                    source = node_definitions[source]
                else:
                    source = f"({source})"
                
                if target not in defined_nodes:
                    defined_nodes.add(target)
                    target = node_definitions[target]
                else:
                    target = f"({target})"
                query += f"{source}-[:{attributes['edge_type']} {{relation: '{attributes['relation']}', generated: '{attributes['generated']}'"
                if attributes['relation'] in ENTAILMENT_RELATIONS:
                    query += f", domain: '{attributes['domain']}'"
                query += f"}}]->{target}, "
        for node in nodes:
            if node not in defined_nodes:
                query += f"{node_definitions[node]}, "
                defined_nodes.add(node)
        if len(query) == 0:
            raise Exception(f"Given (sub-)graph is empty.")
        return f"{operator} {query[:-2]}"
    
    
    def sanitize_for_cypher(self, text):
        return text.replace("'","\\'")
    
    def desanitize_from_cypher(self, text):
        return text.replace("\\'", "'")

    def sigmoid(self, x, maximum = 1.0, gain=1.0):
        return maximum / (1 + math.exp(-gain * x))

    def display_Graph(self, k = 0.1, iterations=50, with_edge_nodes = False):
        '''
        Displays the graph with the following color scheme:
        Nodes and edges that have been generated by the model are colored red.
        Nodes and edges that have been inserted by the user are colored green.
        Generic nodes and edges that are used for unification of nodes and edges are colored transparent.
        Highlights nodes and edges yellow, if they were added since the last display.

        Parameters
        __________
        changed_elems:  tuple[list[str], list[tuple[str,str]]]
                        a tuple with changed nodes (list[str]) with node names and
                        the changed edges with list of node pairs (source, target).
                        Can be None.
        '''
        if self.first_display:
            self.changed_nodes = set()
            self.changed_edges = set()
        all_nodes = dict()
        all_nodes_not_edge_nodes = set()
        all_node_names = set()
        all_edges = []
        for node, attributes in self.G.nodes(data=True):
            if attributes["type"] != "CompareNode":
                all_nodes[node] = attributes
                all_node_names.add(node)
            else:
                for _, _, relation in self.G.in_edges(node, data = "relation"):
                    if relation in ENTAILMENT_RELATIONS:
                        all_nodes[node] = attributes
                        all_node_names.add(node)
            if attributes["type"] != "EdgeNode":
                all_nodes_not_edge_nodes.add(node)
        for source, target, attributes in self.G.edges(data = True):
            if source in all_node_names and target in all_node_names:
                all_edges.append((source, target, attributes))
        #fig_size = self.sigmoid(len(all_nodes)/1.5, len(all_nodes), gain=0.1)
        fig_size = min(len(all_nodes_not_edge_nodes) - 1 if len(all_nodes_not_edge_nodes) > 1 else 1, 50)
        fig_size = (fig_size, fig_size)           # The width size depends on the number of nodes
        plt.figure(3,figsize=fig_size)
        if self.layout == "spring":                            # Switch case depending on the layout
            pos = nx.spring_layout(self.G, k=k, iterations=iterations)
        elif self.layout == "shell":
            pos = nx.shell_layout(self.G, scale = self.scale)
        elif self.layout == "spiral":
            pos = nx.spiral_layout(self.G, scale = self.scale)
        elif self.layout == "spectral":
            pos = nx.spectral_layout(self.G, scale = self.scale)
        elif self.layout == "planar":
            pos = nx.planar_layout(self.G, scale = self.scale)
        elif self.layout == "circular":
            pos = nx.circular_layout(self.G, scale = self.scale)
        nodes_not_generated = []
        nodes_generated = []
        generic_nodes = []
        edge_nodes = {}
        changed_nodes_not_generated = []
        changed_nodes_generated = []
        changed_generic_nodes = []
        
        for node,attributes in all_nodes.items():
            if attributes['generated'] == -1:# Filter all node that have been inserted by the user.
                if not (node in self.changed_nodes):
                    nodes_not_generated.append(node)
                else:
                    changed_nodes_not_generated.append(node)
            elif attributes['generated'] == 1:# Filter all nodes that have been generated by the model.
                if not (node in self.changed_nodes):
                    nodes_generated.append(node)
                else:
                    changed_nodes_generated.append(node)
            else:# Filter all generic nodes.
                if with_edge_nodes or attributes['type'] != "EdgeNode":
                    if not (node in self.changed_nodes):
                        generic_nodes.append(node)
                    else:
                        changed_generic_nodes.append(node)     
                else:
                    source = None
                    target = None
                    for s, _, relation in self.G.in_edges(node, data = "relation"):
                        if relation == "source":
                            source = s
                        elif relation == "target":
                            target = s
                    edge_nodes[(source, target)] = node
        labels_generated = {}
        labels_not_generated = {}
        labels_generic = {}
        for x,y in all_nodes.items():
            if y['generated'] == 1:
                labels_generated[x] = x
            elif y['generated'] == -1:
                labels_not_generated[x] = x
            elif y['generated'] == 0:
                if with_edge_nodes or y['type'] != "EdgeNode":
                    labels_generic[x] = f'{y["text"]}\n{x}'                       # Changes the labels of the generic nodes to their generic counterpart.
        nx.draw_networkx_nodes(                                                                        # Colors user generated nodes green.
            self.G,
            pos,
            nodelist = nodes_not_generated,
            node_color = "tab:green",
            node_size  = self.node_size,        
        )
        nx.draw_networkx_nodes(                                                                        # Colors user generated nodes green.
            self.G,
            pos,
            nodelist = nodes_generated,
            node_color = "tab:red",
            node_size  = self.node_size 
            
        )
        nx.draw_networkx_nodes(                                                                        # Colors user generated nodes green.
            self.G,
            pos,
            nodelist = generic_nodes,
            node_color = "none",
            node_size  = self.node_size           
        )
        nx.draw_networkx_nodes(                                                                        # Colors user generated nodes green.
            self.G,
            pos,
            nodelist = changed_nodes_not_generated,
            node_color = "tab:green",
            node_size  = self.node_size,
            edgecolors="yellow",
            linewidths = 2            
        )
        nx.draw_networkx_nodes(                                                                        # Colors user generated nodes green.
            self.G,
            pos,
            nodelist = changed_nodes_generated,
            node_color = "tab:red",
            node_size  = self.node_size,
            edgecolors="yellow",
            linewidths = 2          
            
        )
        nx.draw_networkx_nodes(                                                                        # Colors user generated nodes green.
            self.G,
            pos,
            nodelist = changed_generic_nodes,
            node_color = "none",
            node_size  = self.node_size,
            edgecolors="yellow",
            linewidths = 2               
        )
        nx.draw_networkx_labels(                                                                             # Colors user generated nodes green.
            self.G,
            pos,
            labels = labels_not_generated,
            font_size=self.font_size_nodes,
            )
        nx.draw_networkx_labels(                                                                            # Colors model generated nodes red.
            self.G,
            pos,
            labels = labels_generated,
            font_size=self.font_size_nodes,
            )
        nx.draw_networkx_labels(                                                                            # Colors generic nodes transparent.
            self.G,
            pos,
            labels = labels_generic,
            font_size=self.font_size_nodes,
            )
        
        curved_edges = []
        straight_edges = []
        curved_edges_genenerated = {}
        curved_edges_not_genenerated = {}
        curved_edges_generic = {}
        straight_edges_genenerated = {}
        straight_edges_not_genenerated = {}
        straight_edges_generic = {}
        changed_curved_edges_genenerated = {}
        changed_curved_edges_not_genenerated = {}
        changed_curved_edges_generic = {}
        changed_straight_edges_genenerated = {}
        changed_straight_edges_not_genenerated = {}
        changed_straight_edges_generic = {}

        def found_straight_edge(edge, attributes, edge_label = ""):        

            straight_edges.append(edge)
            if attributes['generated'] == -1:
                if attributes['domain'] != "CommonSense":
                    if not (edge in self.changed_edges):
                        edge_label += f"{attributes['domain']}\n{attributes['relation']}"
                        straight_edges_not_genenerated[edge] = edge_label
                    else:
                        changed_straight_edges_not_genenerated[edge] = f"{attributes['domain']}\n{attributes['relation']}"
                else:
                    if not (edge in self.changed_edges):
                        edge_label += attributes['relation']
                        straight_edges_not_genenerated[edge] = edge_label
                    else:
                        changed_straight_edges_not_genenerated[edge] = attributes['relation']
            elif attributes["generated"] == 1:
                if attributes['domain'] != "CommonSense":
                    if not (edge in self.changed_edges):
                        edge_label += f"{attributes['domain']}\n{attributes['relation']}"
                        straight_edges_genenerated[edge] = edge_label
                    else:
                        changed_straight_edges_genenerated[edge] = f"{attributes['domain']}\n{attributes['relation']}"
                else:
                    if not (edge in self.changed_edges):
                        edge_label += attributes['relation']
                        straight_edges_genenerated[edge] = edge_label
                    else:
                        edge_label += attributes['relation']
                        changed_straight_edges_genenerated[edge] = edge_label
            else:
                if not (edge in self.changed_edges):
                    edge_label += attributes['relation']
                    straight_edges_generic[edge] = edge_label
                else:
                    edge_label += attributes['relation']
                    changed_straight_edges_generic[edge] = edge_label

        def found_curved_edge(edge, attributes, edge_label = ""):
            curved_edges.append(edge)
            if attributes['generated'] == -1:
                if attributes['domain'] != "CommonSense":
                    if not (edge in self.changed_edges):
                        edge_label += f"{attributes['domain']}\n{attributes['relation']}"
                        curved_edges_not_genenerated[edge] = edge_label
                    else:
                        edge_label += f"{attributes['domain']}\n{attributes['relation']}"
                        changed_curved_edges_not_genenerated[edge] = edge_label
                else:
                    if not (edge in self.changed_edges):
                        edge_label += attributes['relation']
                        curved_edges_not_genenerated[edge] = edge_label
                    else:
                        edge_label += attributes['relation']
                        changed_curved_edges_not_genenerated[edge] = edge_label
            elif attributes["generated"] == 1:
                if attributes['domain'] != "CommonSense":
                    if not (edge in self.changed_edges):
                        edge_label += f"{attributes['domain']}\n{attributes['relation']}"
                        curved_edges_genenerated[edge] = edge_label
                    else:
                        edge_label += f"{attributes['domain']}\n{attributes['relation']}"
                        changed_curved_edges_genenerated[edge] = edge_label
                else:
                    if not (edge in self.changed_edges):
                        edge_label += attributes['relation']
                        curved_edges_genenerated[edge] = edge_label
                    else:
                        edge_label += attributes['relation']
                        changed_curved_edges_genenerated[edge] = edge_label
            else:
                if not (edge in self.changed_edges):
                    edge_label += attributes['relation']
                    curved_edges_generic[edge] = edge_label
                else:
                    edge_label += attributes['relation']
                    changed_curved_edges_generic[edge] = edge_label

        for source, target, attributes in all_edges:
            edge = (source, target)
            edge_label = ""
            if attributes["relation"] in ENTAILMENT_RELATIONS:
                edge_node = self.check_edge_node(source, target, generate = False)
                if edge_node:
                    edge_label = f"{edge_node}\n"
            if with_edge_nodes or (attributes["relation"] != "source" and attributes["relation"] != "target"):
                if reversed(edge) in self.G.edges():
                    for other_source, other_target, other_attributes in all_edges:
                        if (target, source) == (other_source, other_target):
                            if attributes == other_attributes:
                                found_straight_edge(edge, attributes, edge_label)
                            else:
                                found_curved_edge(edge, attributes, edge_label)
                else:
                    found_straight_edge(edge, attributes, edge_label)
                    
        self._my_draw_networkx_edge_node_edges(                                                       # Colors user generated edge labels green.
            pos,
            edge_nodes = edge_nodes,
            edges = straight_edges,
            )
        self._my_draw_networkx_edge_node_edges(                                                       # Colors user generated (curved) edge labels green.
            pos,
            edge_nodes = edge_nodes,
            edges = curved_edges,
            rad = self.rad,
            )
        
        if len(self.changed_edges)>0:                                                                                               # Highligts changed nodes if given.
            straight_changed_edges = []
            curved_changed_edges = []          
            for source, target, attributes in all_edges:
                if with_edge_nodes or (attributes["relation"] != "source" and attributes["relation"] != "target"):
                    if (source, target) in self.changed_edges:
                        if reversed((source, target)) not in self.G.edges():
                            straight_changed_edges.append((source, target))
                        else:
                            for other_source, other_target, other_attributes in all_edges:
                                if (target, source) == (other_source, other_target):
                                    if attributes == other_attributes:
                                        straight_changed_edges.append((source, target))
                                    else:
                                        curved_changed_edges.append((source, target))
            nx.draw_networkx_edges(self.G, pos, edgelist =straight_changed_edges, edge_color= "yellow", width = 4.0, node_size  = self.node_size)
            nx.draw_networkx_edges(self.G, pos, edgelist =curved_changed_edges, edge_color= "yellow", width = 4.0, connectionstyle=f'arc3, rad = {self.rad}', node_size  = self.node_size)
        
        nx.draw_networkx_edges(self.G, pos, edgelist =straight_edges, node_size = self.node_size)
        nx.draw_networkx_edges(self.G, pos, edgelist =curved_edges, connectionstyle=f'arc3, rad = {self.rad}', node_size = self.node_size)
        
        bbox={"boxstyle":"round", "facecolor":"yellow", "edgecolor":(1,1,1), "pad":0}
        self._my_draw_networkx_edge_labels(                                                       # Colors user generated edge labels green.
            all_nodes,
            pos,
            edge_labels = straight_edges_not_genenerated,
            font_color= "tab:green",
            font_size=self.font_size_edges,
            )
        self._my_draw_networkx_edge_labels(                                                       # Colors user generated (curved) edge labels green.
            all_nodes,
            pos,
            edge_labels = curved_edges_not_genenerated,
            font_color= "tab:green",
            font_size=self.font_size_edges,
            rotate=False,
            rad = self.rad,
            )
        self._my_draw_networkx_edge_labels(                                                       # Colors model generated edge labeld red.
            all_nodes,
            pos,
            edge_labels = straight_edges_genenerated,
            font_color= "tab:red",
            font_size=self.font_size_edges
            )
        self._my_draw_networkx_edge_labels(                                                       # Colors model generated edge labeld red.
            all_nodes,
            pos,
            edge_labels = straight_edges_generic,
            font_size=self.font_size_edges
            )
        self._my_draw_networkx_edge_labels(                                                       # Colors model generated edge labeld red.
            all_nodes,
            pos,
            edge_labels = curved_edges_genenerated,
            font_color= "tab:red",
            font_size=self.font_size_edges,
            rotate=False,
            rad = self.rad,
            )
        self._my_draw_networkx_edge_labels(                                                       # Colors model generated edge labeld red.
            all_nodes,
            pos,
            edge_labels = curved_edges_generic,
            font_size=self.font_size_edges,
            rotate=False,
            rad = self.rad,
            )
        
        self._my_draw_networkx_edge_labels(                                                       # Colors user generated edge labels green.
            all_nodes,
            pos,
            bbox=bbox,
            edge_labels = changed_straight_edges_not_genenerated,
            font_color= "tab:green",
            font_size=self.font_size_edges,
            )
        self._my_draw_networkx_edge_labels(                                                       # Colors user generated (curved) edge labels green.
            all_nodes,
            pos,
            bbox=bbox,
            edge_labels = changed_curved_edges_not_genenerated,
            font_color= "tab:green",
            font_size=self.font_size_edges,
            rotate=False,
            rad = self.rad,
            )
        self._my_draw_networkx_edge_labels(                                                       # Colors model generated edge labeld red.
            all_nodes,
            pos,
            bbox=bbox,
            edge_labels = changed_straight_edges_genenerated,
            font_color= "tab:red",
            font_size=self.font_size_edges
            )
        self._my_draw_networkx_edge_labels(                                                       # Colors model generated edge labeld red.
            all_nodes,
            pos,
            bbox=bbox,
            edge_labels = changed_straight_edges_generic,
            font_size=self.font_size_edges
            )
        self._my_draw_networkx_edge_labels(                                                       # Colors model generated edge labeld red.
            all_nodes,
            pos,
            bbox=bbox,
            edge_labels = changed_curved_edges_genenerated,
            font_color= "tab:red",
            font_size=self.font_size_edges,
            rotate=False,
            rad = self.rad,
            )
        self._my_draw_networkx_edge_labels(                                                       # Colors model generated edge labeld red.
            all_nodes,
            pos,
            bbox=bbox,
            edge_labels = changed_curved_edges_generic,
            font_size=self.font_size_edges,
            rotate=False,
            rad = self.rad,
            )
        
        plt.show()
        self.first_display = False
        self.changed_nodes = set()
        self.changed_edges = set()

        #Produces and prints a list of all nodes and edges in a human readable form (truncated if more than 200 characters long).
        prty_print = ""
        for node_name, attributes in all_nodes.items():
            text = attributes["text"]
            if len(text)> 197:
                text = f"{text[:197]}..."
            prty_print += f"{node_name}: {text}\n"
        print(prty_print)

    def _my_draw_networkx_edge_labels(
        self,
        nodes,
        pos,
        edge_labels=None,
        label_pos=0.5,
        font_size=10,
        font_color="k",
        font_family="sans-serif",
        font_weight="normal",
        alpha=None,
        bbox=None,
        horizontalalignment="center",
        verticalalignment="center",
        ax=None,
        rotate=True,
        clip_on=True,
        rad=0
    ):
        """Draw edge labels.
        From https://stackoverflow.com/questions/22785849/drawing-multiple-edges-between-two-nodes-with-networkx user kcoskun tag:6600974

        Parameters
        ----------
        G : graph
            A networkx graph

        pos : dictionary
            A dictionary with nodes as keys and positions as values.
            Positions should be sequences of length 2.

        edge_labels : dictionary (default={})
            Edge labels in a dictionary of labels keyed by edge two-tuple.
            Only labels for the keys in the dictionary are drawn.

        label_pos : float (default=0.5)
            Position of edge label along edge (0=head, 0.5=center, 1=tail)

        font_size : int (default=10)
            Font size for text labels

        font_color : string (default='k' black)
            Font color string

        font_weight : string (default='normal')
            Font weight

        font_family : string (default='sans-serif')
            Font family

        alpha : float or None (default=None)
            The text transparency

        bbox : Matplotlib bbox, optional
            Specify text box properties (e.g. shape, color etc.) for edge labels.
            Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0), pad = 0.0}.

        horizontalalignment : string (default='center')
            Horizontal alignment {'center', 'right', 'left'}

        verticalalignment : string (default='center')
            Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

        ax : Matplotlib Axes object, optional
            Draw the graph in the specified Matplotlib axes.

        rotate : bool (deafult=True)
            Rotate edge labels to lie parallel to edges

        clip_on : bool (default=True)
            Turn on clipping of edge labels at axis boundaries

        Returns
        -------
        dict
            `dict` of labels keyed by edge

        Examples
        --------
        >>> G = nx.dodecahedral_graph()
        >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

        Also see the NetworkX drawing examples at
        https://networkx.org/documentation/latest/auto_examples/index.html

        See Also
        --------
        draw
        draw_networkx
        draw_networkx_nodes
        draw_networkx_edges
        draw_networkx_labels
        """

        if ax is None:
            ax = plt.gca()
        if edge_labels is None:
            labels = {(u, v): d for u, v, d in nodes.items()}
        else:
            labels = edge_labels
        text_items = {}
        for (n1, n2), label in labels.items():
            (x1, y1) = pos[n1]
            (x2, y2) = pos[n2]
            (x, y) = (
                x1 * label_pos + x2 * (1.0 - label_pos),
                y1 * label_pos + y2 * (1.0 - label_pos),
            )
            pos_1 = ax.transData.transform(np.array(pos[n1]))
            pos_2 = ax.transData.transform(np.array(pos[n2]))
            linear_mid = 0.5*pos_1 + 0.5*pos_2
            d_pos = pos_2 - pos_1
            rotation_matrix = np.array([(0,1), (-1,0)])
            ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
            ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
            ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
            bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
            (x, y) = ax.transData.inverted().transform(bezier_mid)

            if rotate:
                # in degrees
                angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
                # make label orientation "right-side-up"
                if angle > 90:
                    angle -= 180
                if angle < -90:
                    angle += 180
                # transform data coordinate angle to screen coordinate angle
                xy = np.array((x, y))
                trans_angle = ax.transData.transform_angles(
                    np.array((angle,)), xy.reshape((1, 2))
                )[0]
            else:
                trans_angle = 0.0
            # use default box of white with white border
            if bbox is None:
                bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0), pad = 0.0)
            if not isinstance(label, str):
                label = str(label)  # this makes "1" and 1 labeled the same

            t = ax.text(
                x,
                y,
                label,
                size=font_size,
                color=font_color,
                family=font_family,
                weight=font_weight,
                alpha=alpha,
                horizontalalignment=horizontalalignment,
                verticalalignment=verticalalignment,
                rotation=trans_angle,
                transform=ax.transData,
                bbox=bbox,
                zorder=1,
                clip_on=clip_on,
            )
            text_items[(n1, n2)] = t

        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )

        return text_items
    
    def _my_draw_networkx_edge_node_edges(
        self,
        pos,
        edge_nodes = None,
        edges=None,
        label_pos=0.5,
        ax=None,
        rad=0
    ):
        """Draw edge labels.
        From https://stackoverflow.com/questions/22785849/drawing-multiple-edges-between-two-nodes-with-networkx user kcoskun tag:6600974

        Parameters
        ----------
        G : graph
            A networkx graph

        pos : dictionary
            A dictionary with nodes as keys and positions as values.
            Positions should be sequences of length 2.

        edge_labels : dictionary (default={})
            Edge labels in a dictionary of labels keyed by edge two-tuple.
            Only labels for the keys in the dictionary are drawn.

        label_pos : float (default=0.5)
            Position of edge label along edge (0=head, 0.5=center, 1=tail)

        font_size : int (default=10)
            Font size for text labels

        font_color : string (default='k' black)
            Font color string

        font_weight : string (default='normal')
            Font weight

        font_family : string (default='sans-serif')
            Font family

        alpha : float or None (default=None)
            The text transparency

        bbox : Matplotlib bbox, optional
            Specify text box properties (e.g. shape, color etc.) for edge labels.
            Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0), pad = 0.0}.

        horizontalalignment : string (default='center')
            Horizontal alignment {'center', 'right', 'left'}

        verticalalignment : string (default='center')
            Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

        ax : Matplotlib Axes object, optional
            Draw the graph in the specified Matplotlib axes.

        rotate : bool (deafult=True)
            Rotate edge labels to lie parallel to edges

        clip_on : bool (default=True)
            Turn on clipping of edge labels at axis boundaries

        Returns
        -------
        dict
            `dict` of labels keyed by edge

        Examples
        --------
        >>> G = nx.dodecahedral_graph()
        >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

        Also see the NetworkX drawing examples at
        https://networkx.org/documentation/latest/auto_examples/index.html

        See Also
        --------
        draw
        draw_networkx
        draw_networkx_nodes
        draw_networkx_edges
        draw_networkx_labels
        """

        if ax is None:
            ax = plt.gca()
        for (n1, n2) in edges:
            (x1, y1) = pos[n1]
            (x2, y2) = pos[n2]
            (x, y) = (
                x1 * label_pos + x2 * (1.0 - label_pos),
                y1 * label_pos + y2 * (1.0 - label_pos),
            )
            pos_1 = ax.transData.transform(np.array(pos[n1]))
            pos_2 = ax.transData.transform(np.array(pos[n2]))
            linear_mid = 0.5*pos_1 + 0.5*pos_2
            d_pos = pos_2 - pos_1
            rotation_matrix = np.array([(0,1), (-1,0)])
            ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
            ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
            ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
            bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
            (x, y) = ax.transData.inverted().transform(bezier_mid)
            if (n1,n2) in edge_nodes:
                pos[edge_nodes[(n1,n2)]] = [x,y]
                nx.draw_networkx_nodes(                                                                        
                    self.G,
                    pos,
                    nodelist = [edge_nodes[(n1,n2)]],
                    node_color = "None",
                    node_size  = self.node_size         
                )
    
    def snapshot_graph_changes(self):
        self.changed_nodes = set()
        self.changed_edges = set()
        self.first_display = False