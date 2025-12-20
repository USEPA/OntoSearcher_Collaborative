"""
Improved RDF to HeteroData Converter
===================================

This converter properly handles the RDF structure with:
1. Entity nodes (material, assay, result, etc.)
2. Concept nodes (ontology terms from NCIT, NPO, OBO, etc.)
3. Blank nodes (node bags containing structured properties)

The RDF structure is:
- Entity --[relation]--> Concept (e.g., material --[has_chemical_name]--> NCIT_concept)
- Entity --[relation]--> BlankNode --[property]--> Concept (nested structures)
"""

import torch
import networkx as nx
from torch_geometric.data import HeteroData
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
import json
from typing import Dict, List, Tuple, Set

class ImprovedRDFToHeteroData:
    """Convert RDF/NetworkX to HeteroData with proper handling of all node types"""
    
    def __init__(self, nx_graph: nx.MultiDiGraph):
        self.nx_graph = nx_graph
        self.hetero_data = None
        self.node_mappings = {}
        self.reverse_mappings = {}
        
        # Define node type categories
        self.entity_types = {
            'material', 'assay', 'result', 'parameters', 'additive', 
            'medium', 'publication', 'contam', 'materialfg', 'molecularresult'
        }
        
        # Load ontology mapping
        try:
            with open('comprehensive_ontology_mapping.json', 'r') as f:
                self.ontology_mapping = json.load(f)
        except:
            self.ontology_mapping = {'ontology_namespaces': {}, 'relation_meanings': {}}
    
    def classify_all_nodes(self):
        """Classify all nodes into entity, concept, or blank node types"""
        print("Classifying all nodes...")
        
        node_classification = {
            'entities': defaultdict(list),  # entity_type -> [nodes]
            'concepts': defaultdict(list),  # concept_type -> [nodes] 
            'blank_nodes': [],
            'unknown': []
        }
        
        for node in self.nx_graph.nodes():
            node_str = str(node)
            
            # 1. Entity nodes (from example.org)
            if node_str.startswith('http://example.org/'):
                entity_type = self.nx_graph.nodes[node].get('entity_type', 'unknown')
                if entity_type in self.entity_types:
                    node_classification['entities'][entity_type].append(node)
                else:
                    node_classification['unknown'].append(node)
            
            # 2. Blank nodes (node bags)
            elif (isinstance(node, str) and node.startswith('n') and 
                  len(node) > 10 and not node.startswith('http')):
                node_classification['blank_nodes'].append(node)
            
            # 3. Concept nodes (ontology terms)
            elif '://' in node_str:
                concept_type = self._classify_concept_node(node_str)
                node_classification['concepts'][concept_type].append(node)
            
            # 4. Everything else
            else:
                node_classification['unknown'].append(node)
        
        # Print classification summary
        print(f"Node Classification Summary:")
        print(f"  Entities: {sum(len(nodes) for nodes in node_classification['entities'].values()):,}")
        for entity_type, nodes in node_classification['entities'].items():
            print(f"    - {entity_type}: {len(nodes):,}")
        
        print(f"  Concepts: {sum(len(nodes) for nodes in node_classification['concepts'].values()):,}")
        for concept_type, nodes in node_classification['concepts'].items():
            print(f"    - {concept_type}: {len(nodes):,}")
        
        print(f"  Blank nodes: {len(node_classification['blank_nodes']):,}")
        print(f"  Unknown: {len(node_classification['unknown']):,}")
        
        return node_classification
    
    def _classify_concept_node(self, node_uri: str) -> str:
        """Classify a concept node by its namespace"""
        if 'ncicb.nci.nih.gov' in node_uri:
            return 'ncit_concept'
        elif 'purl.obolibrary.org/obo' in node_uri:
            return 'obo_concept'
        elif 'purl.bioontology.org/ontology/npo' in node_uri:
            return 'npo_concept'
        elif 'purl.enanomapper.org' in node_uri:
            return 'enm_concept'
        elif 'semanticscience.org' in node_uri:
            return 'sio_concept'
        elif 'w3.org' in node_uri:
            return 'w3_concept'
        elif 'edamontology.org' in node_uri:
            return 'edam_concept'
        else:
            return 'other_concept'
    
    def create_hetero_data_with_all_types(self, min_nodes_per_type: int = 10):
        """Create HeteroData including entities, concepts, and blank nodes"""
        print("Creating comprehensive HeteroData...")
        
        # Classify all nodes
        node_classification = self.classify_all_nodes()
        
        self.hetero_data = HeteroData()
        
        # 1. Add entity node types
        print("\\nAdding entity node types...")
        for entity_type, nodes in node_classification['entities'].items():
            if len(nodes) >= min_nodes_per_type:
                self._add_node_type(entity_type, nodes, 'entity')
        
        # 2. Add concept node types  
        print("\\nAdding concept node types...")
        for concept_type, nodes in node_classification['concepts'].items():
            if len(nodes) >= min_nodes_per_type:
                self._add_node_type(concept_type, nodes, 'concept')
        
        # 3. Add blank nodes as a single type
        print("\\nAdding blank node type...")
        if len(node_classification['blank_nodes']) >= min_nodes_per_type:
            self._add_node_type('blank_node', node_classification['blank_nodes'], 'blank')
        
        # 4. Create edges between all node types
        print("\\nCreating edges between all node types...")
        self._create_comprehensive_edges()
        
        return self.hetero_data
    
    def _add_node_type(self, node_type: str, nodes: List, category: str):
        """Add a node type to HeteroData with appropriate features"""
        # Create node mapping
        self.node_mappings[node_type] = {node: idx for idx, node in enumerate(nodes)}
        self.reverse_mappings[node_type] = {idx: node for node, idx in self.node_mappings[node_type].items()}
        
        # Generate features based on category
        num_nodes = len(nodes)
        
        if category == 'entity':
            # For entities, use structural features
            features = self._generate_entity_features(nodes)
        elif category == 'concept':
            # For concepts, use semantic features
            features = self._generate_concept_features(nodes, node_type)
        elif category == 'blank':
            # For blank nodes, use connectivity features
            features = self._generate_blank_node_features(nodes)
        else:
            # Default random features
            features = torch.randn(num_nodes, 64)
        
        # Add to HeteroData
        self.hetero_data[node_type].x = features
        self.hetero_data[node_type].num_nodes = num_nodes
        
        print(f"  ✓ Added {num_nodes:,} nodes for '{node_type}' ({category}) with {features.size(1)} features")
    
    def _generate_entity_features(self, nodes: List) -> torch.Tensor:
        """Generate features for entity nodes based on their connectivity"""
        features = []
        
        for node in nodes:
            # Structural features
            in_degree = self.nx_graph.in_degree(node)
            out_degree = self.nx_graph.out_degree(node)
            
            # Relation type diversity
            out_relations = set()
            in_relations = set()
            
            for _, _, data in self.nx_graph.edges(node, data=True):
                out_relations.add(data.get('relation_short', 'unknown'))
            
            for _, _, data in self.nx_graph.in_edges(node, data=True):
                in_relations.add(data.get('relation_short', 'unknown'))
            
            node_features = [
                in_degree, out_degree, 
                len(out_relations), len(in_relations),
                in_degree + out_degree  # total degree
            ]
            
            # Pad to 64 dimensions
            while len(node_features) < 64:
                node_features.append(0.0)
            
            features.append(node_features[:64])
        
        return torch.tensor(features, dtype=torch.float)
    
    def _generate_concept_features(self, nodes: List, concept_type: str) -> torch.Tensor:
        """Generate features for concept nodes"""
        features = []
        
        # Create a simple embedding based on concept type and usage
        type_embedding = hash(concept_type) % 1000 / 1000.0
        
        for node in nodes:
            # Usage-based features
            in_degree = self.nx_graph.in_degree(node)  # How many entities use this concept
            out_degree = self.nx_graph.out_degree(node)  # How many things this concept relates to
            
            # URI-based features
            node_str = str(node)
            uri_hash = hash(node_str) % 1000 / 1000.0
            
            node_features = [
                type_embedding, uri_hash, 
                in_degree, out_degree,
                len(node_str) / 100.0  # URI length as a feature
            ]
            
            # Pad to 64 dimensions
            while len(node_features) < 64:
                node_features.append(0.0)
            
            features.append(node_features[:64])
        
        return torch.tensor(features, dtype=torch.float)
    
    def _generate_blank_node_features(self, nodes: List) -> torch.Tensor:
        """Generate features for blank nodes based on their role as property containers"""
        features = []
        
        for node in nodes:
            # Connectivity features
            in_degree = self.nx_graph.in_degree(node)   # Entities pointing to this blank node
            out_degree = self.nx_graph.out_degree(node) # Properties this blank node contains
            
            # What types of entities point to this blank node
            source_entity_types = set()
            for u, _ in self.nx_graph.in_edges(node):
                entity_type = self.nx_graph.nodes[u].get('entity_type', 'unknown')
                if entity_type in self.entity_types:
                    source_entity_types.add(entity_type)
            
            # What types of concepts this blank node points to
            target_concept_types = set()
            for _, v in self.nx_graph.edges(node):
                if '://' in str(v):
                    concept_type = self._classify_concept_node(str(v))
                    target_concept_types.add(concept_type)
            
            node_features = [
                in_degree, out_degree,
                len(source_entity_types), len(target_concept_types),
                hash(str(node)) % 1000 / 1000.0  # Unique identifier
            ]
            
            # Pad to 64 dimensions
            while len(node_features) < 64:
                node_features.append(0.0)
            
            features.append(node_features[:64])
        
        return torch.tensor(features, dtype=torch.float)
    
    def _create_comprehensive_edges(self):
        """Create edges between all node types"""
        edge_groups = defaultdict(list)
        edge_stats = defaultdict(int)
        
        valid_node_types = set(self.node_mappings.keys())
        
        for u, v, data in self.nx_graph.edges(data=True):
            # Determine source and target types
            source_type = self._get_node_type_in_hetero(u)
            target_type = self._get_node_type_in_hetero(v)
            
            # Skip if either type is not in our valid types
            if source_type not in valid_node_types or target_type not in valid_node_types:
                edge_stats['skipped_invalid_type'] += 1
                continue
            
            # Get relation
            relation = data.get('relation_short', 'related_to')
            
            # Create edge type
            edge_type_key = (source_type, relation, target_type)
            
            try:
                source_idx = self.node_mappings[source_type][u]
                target_idx = self.node_mappings[target_type][v]
                edge_groups[edge_type_key].append((source_idx, target_idx))
                edge_stats['added'] += 1
            except KeyError:
                edge_stats['skipped_mapping_error'] += 1
                continue
        
        # Add edges to HeteroData
        print(f"Edge creation statistics:")
        print(f"  - Added: {edge_stats['added']:,}")
        print(f"  - Skipped (invalid type): {edge_stats['skipped_invalid_type']:,}")
        print(f"  - Skipped (mapping error): {edge_stats['skipped_mapping_error']:,}")
        
        for edge_type_key, edges in edge_groups.items():
            if len(edges) == 0:
                continue
            
            source_type, relation, target_type = edge_type_key
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            
            self.hetero_data[source_type, relation, target_type].edge_index = edge_index
            print(f"  ✓ Added {len(edges):,} edges for ({source_type}, {relation}, {target_type})")
    
    def _get_node_type_in_hetero(self, node):
        """Determine which node type a node belongs to in our HeteroData"""
        node_str = str(node)
        
        # Check entity types
        if node_str.startswith('http://example.org/'):
            entity_type = self.nx_graph.nodes[node].get('entity_type', 'unknown')
            if entity_type in self.node_mappings:
                return entity_type
        
        # Check blank nodes
        elif (isinstance(node, str) and node.startswith('n') and 
              len(node) > 10 and not node.startswith('http')):
            if 'blank_node' in self.node_mappings:
                return 'blank_node'
        
        # Check concept types
        elif '://' in node_str:
            concept_type = self._classify_concept_node(node_str)
            if concept_type in self.node_mappings:
                return concept_type
        
        return None
    
    def validate_comprehensive_heterodata(self):
        """Validate the comprehensive HeteroData"""
        if not self.hetero_data:
            print("❌ No HeteroData to validate")
            return None
        
        print("\\nValidating comprehensive HeteroData...")
        
        node_types = list(self.hetero_data.node_types)
        edge_types = list(self.hetero_data.edge_types)
        
        # Categorize node types
        entity_node_types = [nt for nt in node_types if nt in self.entity_types]
        concept_node_types = [nt for nt in node_types if 'concept' in nt]
        other_node_types = [nt for nt in node_types if nt not in entity_node_types and nt not in concept_node_types]
        
        # Basic statistics
        total_nodes = sum(self.hetero_data[nt].num_nodes for nt in node_types)
        total_edges = sum(self.hetero_data[et].edge_index.size(1) for et in edge_types)
        
        validation_report = {
            'basic_stats': {
                'total_node_types': len(node_types),
                'total_edge_types': len(edge_types),
                'total_nodes': total_nodes,
                'total_edges': total_edges
            },
            'node_type_categories': {
                'entity_types': len(entity_node_types),
                'concept_types': len(concept_node_types),
                'other_types': len(other_node_types)
            },
            'connectivity': {
                'avg_edges_per_node': total_edges / total_nodes if total_nodes > 0 else 0,
                'edge_type_diversity': len(edge_types)
            }
        }
        
        print("✓ Comprehensive HeteroData Validation:")
        print(f"  - Node types: {len(node_types)} (entities: {len(entity_node_types)}, concepts: {len(concept_node_types)}, other: {len(other_node_types)})")
        print(f"  - Edge types: {len(edge_types)}")
        print(f"  - Total nodes: {total_nodes:,}")
        print(f"  - Total edges: {total_edges:,}")
        print(f"  - Avg edges per node: {validation_report['connectivity']['avg_edges_per_node']:.2f}")
        
        # Check for isolated node types
        isolated_types = []
        for node_type in node_types:
            has_edges = any(node_type in str(et) for et in edge_types)
            if not has_edges:
                isolated_types.append(node_type)
        
        if isolated_types:
            print(f"⚠️  Isolated node types: {isolated_types}")
        else:
            print("✅ All node types have edges!")
        
        return validation_report

def main():
    """Test the improved converter"""
    print("=== Testing Improved RDF to HeteroData Converter ===")
    
    # Load NetworkX graph
    with open('networkx_graph.pkl', 'rb') as f:
        nx_graph = pickle.load(f)
    
    # Create improved converter
    converter = ImprovedRDFToHeteroData(nx_graph)
    
    # Convert to HeteroData
    hetero_data = converter.create_hetero_data_with_all_types(min_nodes_per_type=50)
    
    # Validate
    validation_report = converter.validate_comprehensive_heterodata()
    
    # Save results
    torch.save(hetero_data, 'improved_hetero_data.pt')
    print("\\n✓ Improved HeteroData saved to improved_hetero_data.pt")
    
    with open('improved_validation_report.json', 'w') as f:
        json.dump(validation_report, f, indent=2, default=str)
    print("✓ Validation report saved to improved_validation_report.json")

if __name__ == "__main__":
    main()
