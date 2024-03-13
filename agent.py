import json
import itertools
from reasoning_graph import ReasoningGraph, ENTAILMENT_RELATIONS, DOMAIN_TYPES

class Agent():

    def __init__(self, layout = "planar", font_size_nodes = 12, font_size_edges = 12, node_size = 500, rad = 0.12, scale = 1.0):
        self.RG = ReasoningGraph(layout = layout, font_size_nodes = font_size_nodes, font_size_edges = font_size_edges, node_size = node_size, rad = rad, scale = scale)

    def check_node(self, text, name = None, type = "Statement", previous = None, following = None, generated = -1, add = True):
        return self.RG.check_node(text, name = name, type = type, previous = previous, following = following, generated = generated, add = add)

    def add_domain_node(self, domain = "CommonSense", name = None, generated = -1):
        if domain not in DOMAIN_TYPES:
            raise Exception(f"Expected domain of {DOMAIN_TYPES} but got {domain} instead.")
        return self.RG.check_node(domain, name = name, type = "Domain", generated = generated)

    def add_edge(self, source, target, edge_type = "Entailment",relation = "ent", domain = "CommonSense", generated = -1):
        self.RG.add_edge(source, target, edge_type = edge_type, relation = relation, domain = domain, generated = generated)

    def check_and_node(self, nodes, previous = None, following = None, add = True):
        return self.RG.check_and_node(nodes, previous = previous, following = following, add = add)

    def check_compare_node(self, left, right, operator = "<"):
        return self.RG.check_compare_node(left, right, operator = operator)
    
    def get_compare_node(self, left_source, left_target, right_source, right_target, context = None):
        return self.RG.get_compare_node(left_source, left_target, right_source, right_target, context)
    
    def get_compare_nodes(self):
        edges = self.RG.get_edges()
        compare_nodes = []
        for (left_source, left_target), (right_source, right_target) in itertools.product(edges, edges):
            left_edge = self.RG.check_edge_node(left_source, left_target, generate=False)
            right_edge = self.RG.check_edge_node(right_source, right_target, generate=False)
            if left_edge and right_edge:
                compare_node = self.RG.check_compare_node(left_edge, right_edge, generate=False)
                if compare_node:
                    compare_nodes.append(compare_node)
        return compare_nodes
        
    def get_edges(self, nodes = None):
        return self.RG.get_edges(nodes)

    def reset_graph(self):
        self.RG.reset_graph()

    def snapshot_graph_changes(self):
        self.RG.snapshot_graph_changes()

    def display_graph(self, k = 0.1, iterations = 50, with_edge_nodes = False):
        self.RG.display_Graph(k = k, iterations = iterations, with_edge_nodes=with_edge_nodes)

    def graph_to_cypher(self, nodes = None, operator = "MATCH"):
        return self.RG.graph_to_cypher(nodes, operator)

    def relation_between(self, source, target, domain = "CommonSense", label = None):
        self.RG.check_node_in_graph(target)
        sources = set(self.RG.resolve_node(source).keys())
        targets = set(self.RG.resolve_node(target).keys())
        nodes = sources.union(targets)
        query = self.RG.graph_to_cypher(nodes)
        match_phrase = f" ({source})-[edge :Entailment]->({target}) RETURN edge.relation AS relation"
        query += f", {match_phrase}"
        relation = self._produce_response(query, dummy_response = label)
        if relation:
            self.add_edge(source, target, relation = relation, domain = domain, generated = 1)
            return query, relation
        return None, None
    
    def generate_entailment(self, premise, name = None, forward_relation = "ent", backward_relation = None, domain = "CommonSense", previous_answers = [], starts_with = None, ends_with = None, label = None):
        self._check_relationship_type(forward_relation)
        if backward_relation:
            self._check_relationship_type(backward_relation)
        nodes = self.RG.resolve_node(premise)
        for previous_answer in previous_answers:
            nodes.update(self.RG.resolve_node(previous_answer))
        query = self.RG.graph_to_cypher(nodes)
        match_phrase = f"({premise})-[:Entailment {{relation: '{forward_relation}'}}]->(hypothesis :Statement)"
        if backward_relation:
            match_phrase += f", ({premise})<-[:Entailment {{relation: '{backward_relation}'}}]-(hypothesis)"
        for previous_answer in previous_answers:
            match_phrase += f", (hypothesis)-[:Entailment {{relation: '¬ent', domain: '{domain}'}}]->({previous_answer})-[:Entailment {{relation: '¬ent', domain: '{domain}'}}]->(hypothesis)"
        if starts_with or ends_with:
            match_phrase +=" WHERE"
        if starts_with:
            match_phrase += f" hypothesis.text STARTS WITH '{starts_with}'"
            if ends_with:
                match_phrase += f" OR"
        if ends_with:
            match_phrase += f" hypothesis.text ENDS WITH '{ends_with}'"
        match_phrase += f" RETURN hypothesis.text AS text"
        query += f", {match_phrase}"
        text = self._produce_response(query, label)
        if text:
            self._check_starts_ends_with(text, starts_with, ends_with)
            name = self.check_node(text, name, generated = 1)
            self.add_edge(premise, name, relation = forward_relation, domain = domain, generated = -1)
            if backward_relation:
                self.add_edge(name, premise, relation = backward_relation, domain = domain, generated = -1)
            for previous_answer in previous_answers:
                self.add_edge(previous_answer, name, relation = "¬ent", domain = domain, generated = -1)
                self.add_edge(name, previous_answer, relation = "¬ent", domain = domain, generated = -1)
            return query, text
        return None, None
    
    def proof_relation(self, source, target, name = None, proof_context = None, context_domain = None, previous_answers = [], relation = "ent", domain = "CommonSense", starts_with = None, ends_with = None, label = None):
        self.RG.check_edge_in_graph(source, target)
        if not domain in DOMAIN_TYPES:
            raise Exception(f"Expected domain of {DOMAIN_TYPES} but got {domain} instead.")
        source_nodes = set(self.RG.resolve_node(source).keys())
        target_nodes = set(self.RG.resolve_node(target).keys())
        nodes = source_nodes.union(target_nodes)
        for previous_answer in previous_answers:
            nodes.update(self.RG.resolve_node(previous_answer))
        edge_node = self.RG.check_edge_node(source, target)
        nodes.add(edge_node)
        if proof_context:
            proof_context_attributes = self.RG.check_node_in_graph(proof_context)
            if not context_domain and proof_context_attributes["type"] == "Domain":
                context_domain = proof_context_attributes["text"]
            proof_context_nodes = self.RG.resolve_node(proof_context)
            nodes.update(proof_context_nodes)
        if not context_domain:
            context_domain = domain

        query = self.RG.graph_to_cypher(nodes)
        match_phrase = f", (edge_node)<-[:Entailment {{relation: '{relation}', domain: '{domain}'}}]-(proof:Statement)"
        if proof_context:
            match_phrase += f"<-[:Entailment {{relation: 'ent', domain: '{context_domain}'}}]-({proof_context})"
        for previous_answer in previous_answers:
            match_phrase += f", (proof)-[:Entailment {{relation: '¬ent', domain: '{domain}'}}]->({previous_answer})-[:Entailment {{relation: '¬ent', domain: '{domain}'}}]->(proof)"
        if starts_with or ends_with:
            match_phrase +=" WHERE"
        if starts_with:
            match_phrase += f" proof.text STARTS WITH '{starts_with}'"
            if ends_with:
                match_phrase += f" OR"
        if ends_with:
            match_phrase += f" proof.text ENDS WITH '{ends_with}'"
        match_phrase +=  " RETURN proof.text AS text"
        query += f", {match_phrase}"
        text = self._produce_response(query, label)
        if text:
            self._check_starts_ends_with(text, starts_with, ends_with)
            self.check_node(text, name, generated = 1)
            self.add_edge(name, edge_node, domain = domain, generated=-1)
            if proof_context:
                self.add_edge(proof_context, name, domain = context_domain, generated=-1)
            for previous_answer in previous_answers:
                self.add_edge(previous_answer, name, relation = "¬ent", domain = domain, generated = -1)
                self.add_edge(name, previous_answer, relation = "¬ent", domain = domain, generated = -1)
            return query, text
        else:
            return None, None

    def relation_adjustment(self, source, target, name = None, previous_answers = [], adjustment_statement_context = None, compare_context = None, adjustment_statement_domain = None, compare_context_domain = None, domain = "CommonSense", direction = 1, starts_with = None, ends_with = None, label = None):
        attributes = self.RG.check_edge_in_graph(source, target)
        if not domain in DOMAIN_TYPES:
            raise Exception(f"Expected domain of {DOMAIN_TYPES} but got {domain} instead.")
        source_nodes = set(self.RG.resolve_node(source).keys())
        target_nodes = set(self.RG.resolve_node(target).keys())
        adjustment_context_nodes = set(self.RG.resolve_node(adjustment_statement_context).keys())
        compare_context_nodes = set(self.RG.resolve_node(compare_context).keys())
        nodes = source_nodes.union(target_nodes)
        nodes.update(adjustment_context_nodes)
        nodes.update(compare_context_nodes)
        for previous_answer in previous_answers:
            nodes.update(self.RG.resolve_node(previous_answer))
        if adjustment_statement_context:
            adjustment_statement_context_attributes = self.RG.check_node_in_graph(adjustment_statement_context)
            if adjustment_statement_context_attributes["type"] == "Domain" and not adjustment_statement_domain:
                adjustment_statement_domain = adjustment_statement_context_attributes["text"]
        if compare_context:
            compare_context_attributes = self.RG.check_node_in_graph(compare_context)
            if compare_context_attributes["type"] == "Domain" and not compare_context_domain:
                compare_context_domain = compare_context_attributes["text"]
        query = self.RG.graph_to_cypher(nodes)
        if direction >= 0:
            left = "edge_node"
            right = "edge_adjustment_node"
        else:
            left = "edge_adjustment_node"
            right = "edge_node"
        query += f", ({target})-[:Edge {{relation: 'target'}}]->(edge_node: EdgeNode)<-[:Edge {{relation: 'source'}}]-({source})-[:Union {{relation: '&'}}]->(and_node :AndNode)<-[:Union {{relation: '&'}}]-(adjustment_node :Statement), (and_node)-[:Edge {{relation: 'source'}}]->(edge_adjustment_node:EdgeNode)<-[:Edge {{relation: 'target'}}]-({target}), ({left})-[:Compare {{relation: 'left'}}]->(compare_node :CompareNode {{text: '<'}})<-[:Compare {{relation: 'right'}}]-({right})"
        if adjustment_statement_context:
            query += f", ({adjustment_statement_context})-[:Entailment {{relation: 'ent', domain: '{adjustment_statement_domain}'}}]->(adjustment_node)"
        if compare_context:
            query += f", ({compare_context})-[:Entailment {{relation: 'ent', domain: '{compare_context_domain}'}}]->(compare_node)"
        # for idx, previous_answer in enumerate(previous_answers):
        #     forward_relation = previous_answer_relations[idx][0] if previous_answer_relations else '¬ent'
        #     backward_relation = previous_answer_relations[idx][1] if previous_answer_relations else '¬ent'
        #     query += f", (adjustment_node)-[:Entailment {{relation: {forward_relation}, domain: '{domain}'}}]->({previous_answers})-[:Entailment {{relation: {backward_relation}, domain: '{domain}'}}]->(adjustment_node)"
        if starts_with or ends_with:
            query +=" WHERE"
        if starts_with:
            query += f" adjustment_node.text STARTS WITH '{starts_with}'"
            if ends_with:
                query += f" OR"
        if ends_with:
            query += f" adjustment_node.text ENDS WITH '{ends_with}'"
        query +=  " RETURN adjustment_node.text AS text"
        text = self._produce_response(query, label)
        if text:
            name =self.check_node(text, name, generated = 1)
            and_node = self.check_and_node([name, source])
            edge_node = self.RG.check_edge_node(source, target)
            self.add_edge(and_node, target, relation = attributes["relation"], generated=attributes["generated"], domain = attributes["domain"])
            adjustment_edge_node = self.RG.check_edge_node(and_node, target)
            if direction >= 0:
                compare_node = self.check_compare_node(edge_node, adjustment_edge_node)
            else:
                compare_node = self.check_compare_node(adjustment_edge_node, edge_node)
            if adjustment_statement_context:
                self.add_edge(adjustment_statement_context, name, domain = adjustment_statement_domain)
            if compare_context:
                self.add_edge(compare_context, compare_node, domain = compare_context_domain)
            for previous_answer in previous_answers:
                self.add_edge(previous_answer, name, relation = "¬ent", domain = domain, generated = -1)
                self.add_edge(name, previous_answer, relation = "¬ent", domain = domain, generated = -1)
            return query, text
        else:
            return None, None
    
    
    def multiple_choice(self, source, targets, domain = "CommonSense", labels = None):
        queries = []
        relations = []
        for target, label in zip(targets, labels):
            query, relation = self.relation_between(source, target, domain=domain, label = label)
            queries.append(query)
            relations.append(relation)
        return queries, relations
    
    def multiple_choice_ordered(self, source, targets, compare_context, domain = "CommonSense", compare_domain = None, labels = None):
        all_queries, all_relations = self.multiple_choice(source, targets, domain, labels["MultipleChoice"])
        edges = []
        for target in targets:
            edges.append((source, target))
        queries, relations = self.compare_relations(edges, compare_context, compare_domain, labels=labels["Compare"])
        all_queries.extend(queries)
        all_relations.extend(relations)
        return all_queries, all_relations
        
    def compare_relations(self, edges, compare_context, operator = "<", compare_domain = None, labels = None):
        context_attributes = self.RG.check_node_in_graph(compare_context)
        edge_nodes = {}
        all_queries = []
        all_relations = []
        if not compare_domain:
            if context_attributes["type"] == "Domain":
                compare_domain = context_attributes["text"]
            else:
                compare_domain = "CommonSense"
        for source, target in edges:
            edge_node = self.RG.check_edge_node(source, target)
            edge_nodes[edge_node] = (source, target)
        compare_nodes = []
        for left, right in itertools.product(edge_nodes.keys(), edge_nodes.keys()):
            if left != right:
                left_edge = edge_nodes[left]
                right_edge = edge_nodes[right]
                if labels and (left_edge, right_edge) in labels:
                    label = labels[(left_edge, right_edge)]
                else:
                    label = None
                compare_node = self.check_compare_node(left, right, operator=operator)
                compare_nodes.append((compare_node, label))
        for compare_node, label in compare_nodes:
            query, relation = self.relation_between(compare_context, compare_node, compare_domain, label)
            if query:
                all_queries.append(query)
                all_relations.append(relation)
        return all_queries, all_relations

    def _produce_response(self, query, dummy_response = None):
        if dummy_response:
            return dummy_response
        else:
            return None
    
    def _generate_cypher_after(self, nodes = None):
        query_after = self.RG.graph_to_cypher(nodes, operator = "CREATE")
        return query_after
        
    def _check_relationship_type(self, relation):
        if relation not in ENTAILMENT_RELATIONS:
            raise Exception(f"Expected textual entailment relationship to be any of {ENTAILMENT_RELATIONS} but got {relation} instead.")
        
    def _check_starts_ends_with(self, text, starts_with = None, ends_with = None):
        if starts_with and not text.startswith(starts_with):
            raise Exception(f"Expected text to start with {starts_with} but it is {text} instead.")
        if ends_with and not text.endswith(ends_with):
            raise Exception(f"Expected text to end with {ends_with} but it is {text} instead.")