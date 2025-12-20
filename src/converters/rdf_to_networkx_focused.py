"""
Focused RDF to NetworkX to HeteroData Pipeline
=============================================

This script focuses on the core conversion pipeline:
1. Load RDF directly (no sampling)
2. Convert to NetworkX with validation
3. Convert to PyTorch Geometric HeteroData with validation
4. Analyze the resulting structures

Based on the EPA NaKnowBase with entity types:
- parameters: 83,233 entities
- result: 24,693 entities  
- assay: 22,329 entities
- material: 374 entities
- additive: 302 entities
- medium: 255 entities
- publication: 129 entities
- contam: 47 entities
- materialfg: 16 entities
- molecularresult: 10 entities
"""

import pandas as pd
import numpy as np
import networkx as nx
from rdflib import Graph, Namespace, URIRef, Literal
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Set, Optional
import pickle
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# PyTorch Geometric imports
try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import HeteroData
    from sklearn.preprocessing import LabelEncoder
    TORCH_AVAILABLE = True
    print("✓ PyTorch Geometric available")
except ImportError:
    print("⚠️  PyTorch Geometric not available. Install with: pip install torch torch-geometric")
    TORCH_AVAILABLE = False

class RDFToNetworkXConverter:
    """Direct RDF to NetworkX conversion with validation"""
    
    def __init__(self, rdf_file_path: str):
        self.rdf_file_path = rdf_file_path
        self.rdf_graph = Graph()
        self.nx_graph = None
        
        # Entity types from the paper
        self.entity_types = {
            'parameters': 'http://example.org/parameters/',
            'result': 'http://example.org/result/',
            'assay': 'http://example.org/assay/',
            'material': 'http://example.org/material/',
            'additive': 'http://example.org/additive/',
            'medium': 'http://example.org/medium/',
            'publication': 'http://example.org/publication/',
            'contam': 'http://example.org/contam/',
            'materialfg': 'http://example.org/materialfg/',
            'molecularresult': 'http://example.org/molecularresult/'
        }
        
        self.entity_counts = Counter()
        self.relation_counts = Counter()
        
    def load_rdf(self):
        """Load the full RDF graph"""
        print(f"Loading RDF from {self.rdf_file_path}...")
        print("This may take a few minutes for large files...")
        
        try:
            self.rdf_graph.parse(self.rdf_file_path, format='turtle')
            print(f"✓ Successfully loaded {len(self.rdf_graph)} triples")
            return True
        except Exception as e:
            print(f"❌ Failed to load RDF: {e}")
            return False
    
    def analyze_rdf_structure(self):
        """Quick analysis of RDF structure"""
        print("Analyzing RDF structure...")
        
        subjects = set()
        predicates = set()
        objects = set()
        
        for s, p, o in tqdm(self.rdf_graph, desc="Analyzing triples"):
            subjects.add(s)
            predicates.add(p)
            if isinstance(o, URIRef):
                objects.add(o)
            
            # Count entity types
            if isinstance(s, URIRef):
                s_str = str(s)
                for entity_type, prefix in self.entity_types.items():
                    if s_str.startswith(prefix):
                        self.entity_counts[entity_type] += 1
                        break
            
            # Count relations
            self.relation_counts[str(p)] += 1
        
        analysis = {
            'total_triples': len(self.rdf_graph),
            'unique_subjects': len(subjects),
            'unique_predicates': len(predicates),
            'unique_objects': len(objects),
            'entity_counts': dict(self.entity_counts.most_common()),
            'top_relations': dict(self.relation_counts.most_common(20))
        }
        
        print(f"✓ RDF Analysis Complete:")
        print(f"  - Total triples: {analysis['total_triples']:,}")
        print(f"  - Unique subjects: {analysis['unique_subjects']:,}")
        print(f"  - Unique predicates: {analysis['unique_predicates']:,}")
        print(f"  - Entity types found: {len(analysis['entity_counts'])}")
        
        return analysis
    
    def convert_to_networkx(self, exclude_literals: bool = True):
        """Convert RDF to NetworkX MultiDiGraph"""
        print("Converting RDF to NetworkX...")
        print(f"Exclude literals: {exclude_literals}")
        
        self.nx_graph = nx.MultiDiGraph()
        
        # Track conversion statistics
        edges_added = 0
        nodes_added = 0
        literals_skipped = 0
        
        for s, p, o in tqdm(self.rdf_graph, desc="Converting to NetworkX"):
            subject_str = str(s)
            predicate_str = str(p)
            object_str = str(o)
            
            # Skip literals if requested
            if exclude_literals and isinstance(o, Literal):
                literals_skipped += 1
                continue
            
            # Add subject node with entity type
            if subject_str not in self.nx_graph:
                subject_type = self._get_entity_type(subject_str)
                self.nx_graph.add_node(
                    subject_str, 
                    entity_type=subject_type,
                    node_type='subject'
                )
                nodes_added += 1
            
            # Add object node if it's a URI
            if isinstance(o, URIRef):
                if object_str not in self.nx_graph:
                    object_type = self._get_entity_type(object_str)
                    self.nx_graph.add_node(
                        object_str,
                        entity_type=object_type,
                        node_type='object'
                    )
                    nodes_added += 1
            else:
                # For literals, add as node attribute instead of separate node
                if not exclude_literals:
                    if object_str not in self.nx_graph:
                        self.nx_graph.add_node(
                            object_str,
                            entity_type='literal',
                            node_type='literal',
                            value=str(o)
                        )
                        nodes_added += 1
            
            # Add edge
            relation_short = self._get_short_relation_name(predicate_str)
            self.nx_graph.add_edge(
                subject_str,
                object_str,
                relation=predicate_str,
                relation_short=relation_short,
                edge_type=f"{self._get_entity_type(subject_str)}_to_{self._get_entity_type(object_str)}"
            )
            edges_added += 1
        
        print(f"✓ NetworkX conversion complete:")
        print(f"  - Nodes added: {nodes_added:,}")
        print(f"  - Edges added: {edges_added:,}")
        print(f"  - Literals skipped: {literals_skipped:,}")
        print(f"  - Final graph: {self.nx_graph.number_of_nodes():,} nodes, {self.nx_graph.number_of_edges():,} edges")
        
        return self.nx_graph
    
    def _get_entity_type(self, uri: str) -> str:
        """Determine entity type from URI"""
        for entity_type, prefix in self.entity_types.items():
            if uri.startswith(prefix):
                return entity_type
        
        # Check for common ontology prefixes
        if any(prefix in uri for prefix in ['ncit:', 'obo:', 'npo:', 'dcterms:', 'rdf:', 'rdfs:']):
            return 'ontology_term'
        
        return 'unknown'
    
    def _get_short_relation_name(self, relation_uri: str) -> str:
        """Get short name for relation"""
        if '#' in relation_uri:
            return relation_uri.split('#')[-1]
        elif '/' in relation_uri:
            return relation_uri.split('/')[-1]
        return relation_uri
    
    def validate_networkx_graph(self):
        """Validate the NetworkX graph structure"""
        if not self.nx_graph:
            print("❌ No NetworkX graph to validate")
            return None
        
        print("Validating NetworkX graph...")
        
        # Basic statistics
        num_nodes = self.nx_graph.number_of_nodes()
        num_edges = self.nx_graph.number_of_edges()
        
        # Node type analysis
        node_types = Counter()
        entity_types = Counter()
        
        for node, data in self.nx_graph.nodes(data=True):
            node_types[data.get('node_type', 'unknown')] += 1
            entity_types[data.get('entity_type', 'unknown')] += 1
        
        # Edge type analysis
        edge_types = Counter()
        relation_types = Counter()
        
        for u, v, data in self.nx_graph.edges(data=True):
            edge_types[data.get('edge_type', 'unknown')] += 1
            relation_types[data.get('relation_short', 'unknown')] += 1
        
        # Connectivity analysis
        is_connected = nx.is_weakly_connected(self.nx_graph)
        num_components = nx.number_weakly_connected_components(self.nx_graph)
        
        if num_components > 0:
            largest_component = max(nx.weakly_connected_components(self.nx_graph), key=len)
            largest_component_size = len(largest_component)
        else:
            largest_component_size = 0
        
        # Degree analysis
        in_degrees = [d for n, d in self.nx_graph.in_degree()]
        out_degrees = [d for n, d in self.nx_graph.out_degree()]
        
        validation_report = {
            'basic_stats': {
                'nodes': num_nodes,
                'edges': num_edges,
                'is_directed': self.nx_graph.is_directed(),
                'is_multigraph': self.nx_graph.is_multigraph()
            },
            'node_analysis': {
                'node_types': dict(node_types.most_common()),
                'entity_types': dict(entity_types.most_common())
            },
            'edge_analysis': {
                'edge_types': dict(edge_types.most_common(10)),
                'relation_types': dict(relation_types.most_common(10))
            },
            'connectivity': {
                'is_weakly_connected': is_connected,
                'num_components': num_components,
                'largest_component_size': largest_component_size,
                'connectivity_ratio': largest_component_size / num_nodes if num_nodes > 0 else 0
            },
            'degree_stats': {
                'avg_in_degree': np.mean(in_degrees) if in_degrees else 0,
                'avg_out_degree': np.mean(out_degrees) if out_degrees else 0,
                'max_in_degree': max(in_degrees) if in_degrees else 0,
                'max_out_degree': max(out_degrees) if out_degrees else 0
            }
        }
        
        print("✓ NetworkX Validation Results:")
        print(f"  - Graph type: {'Directed' if validation_report['basic_stats']['is_directed'] else 'Undirected'} {'MultiGraph' if validation_report['basic_stats']['is_multigraph'] else 'Graph'}")
        print(f"  - Connectivity: {'Connected' if is_connected else f'{num_components} components'}")
        print(f"  - Largest component: {largest_component_size:,} nodes ({validation_report['connectivity']['connectivity_ratio']:.1%})")
        print(f"  - Entity types: {list(validation_report['node_analysis']['entity_types'].keys())}")
        print(f"  - Average degree: in={validation_report['degree_stats']['avg_in_degree']:.1f}, out={validation_report['degree_stats']['avg_out_degree']:.1f}")
        
        return validation_report

class NetworkXToHeteroData:
    """Convert NetworkX to PyTorch Geometric HeteroData with validation"""
    
    def __init__(self, nx_graph: nx.MultiDiGraph):
        self.nx_graph = nx_graph
        self.hetero_data = None
        self.node_mappings = {}
        self.reverse_mappings = {}
        
    def convert_to_heterodata(self, min_nodes_per_type: int = 1):
        """Convert NetworkX graph to HeteroData"""
        if not TORCH_AVAILABLE:
            print("❌ PyTorch Geometric not available")
            return None
        
        print("Converting NetworkX to HeteroData...")
        
        self.hetero_data = HeteroData()
        
        # Group nodes by entity type
        nodes_by_type = defaultdict(list)
        for node, data in self.nx_graph.nodes(data=True):
            entity_type = data.get('entity_type', 'unknown')
            if entity_type != 'unknown':  # Skip unknown types
                nodes_by_type[entity_type].append(node)
        
        print(f"Found {len(nodes_by_type)} entity types:")
        for entity_type, nodes in nodes_by_type.items():
            print(f"  - {entity_type}: {len(nodes):,} nodes")
        
        # Filter entity types with sufficient nodes
        valid_entity_types = {
            entity_type: nodes 
            for entity_type, nodes in nodes_by_type.items() 
            if len(nodes) >= min_nodes_per_type
        }
        
        print(f"Using {len(valid_entity_types)} entity types with >= {min_nodes_per_type} nodes")
        
        # Create node mappings and add to HeteroData
        for entity_type, nodes in valid_entity_types.items():
            # Create mapping from original node IDs to indices
            node_to_idx = {node: idx for idx, node in enumerate(nodes)}
            self.node_mappings[entity_type] = node_to_idx
            self.reverse_mappings[entity_type] = {idx: node for node, idx in node_to_idx.items()}
            
            # Create simple node features (we'll improve this later)
            num_nodes = len(nodes)
            # For now, use random features - we'll replace with proper features later
            node_features = torch.randn(num_nodes, 64)  # 64-dimensional features
            
            # Add to HeteroData
            self.hetero_data[entity_type].x = node_features
            self.hetero_data[entity_type].num_nodes = num_nodes
            
            print(f"  ✓ Added {num_nodes:,} nodes for '{entity_type}' with {node_features.size(1)} features")
        
        # Create edges
        self._create_hetero_edges(valid_entity_types)
        
        return self.hetero_data
    
    def _create_hetero_edges(self, valid_entity_types: Dict[str, List[str]]):
        """Create edges for HeteroData"""
        print("Creating heterogeneous edges...")
        
        # Group edges by (source_type, relation, target_type)
        edge_groups = defaultdict(list)
        skipped_edges = defaultdict(int)
        
        for source, target, edge_data in self.nx_graph.edges(data=True):
            # Get entity types
            source_data = self.nx_graph.nodes[source]
            target_data = self.nx_graph.nodes[target]
            
            source_type = source_data.get('entity_type', 'unknown')
            target_type = target_data.get('entity_type', 'unknown')
            
            # Debug: Track why edges are being skipped
            if source_type not in valid_entity_types:
                skipped_edges[f"source_{source_type}_not_valid"] += 1
                continue
            
            if target_type not in valid_entity_types:
                skipped_edges[f"target_{target_type}_not_valid"] += 1
                
                # Instead of skipping, let's create self-loops or intra-type edges
                # For edges going to 'unknown', create a self-loop or connect to related entities
                relation = edge_data.get('relation_short', 'related_to')
                
                # Strategy 1: Create self-loops for property edges
                if relation in ['type', 'identifier', 'value', 'C42614', 'C68553', 'OBI_0000070']:
                    # These are typically property edges, create as node attributes instead
                    # But for now, let's create self-loops to maintain connectivity
                    edge_type_key = (source_type, relation, source_type)
                    source_idx = self.node_mappings[source_type][source]
                    edge_groups[edge_type_key].append((source_idx, source_idx))
                continue
            
            # Get relation
            relation = edge_data.get('relation_short', 'related_to')
            
            # Create edge type key
            edge_type_key = (source_type, relation, target_type)
            
            # Get node indices
            try:
                source_idx = self.node_mappings[source_type][source]
                target_idx = self.node_mappings[target_type][target]
                edge_groups[edge_type_key].append((source_idx, target_idx))
            except KeyError as e:
                skipped_edges[f"mapping_error_{e}"] += 1
                continue
        
        # Print skipped edge statistics
        print("Edge creation statistics:")
        total_skipped = sum(skipped_edges.values())
        print(f"  - Total edges processed: {self.nx_graph.number_of_edges():,}")
        print(f"  - Edges skipped: {total_skipped:,}")
        
        if skipped_edges:
            print("  - Reasons for skipping (top 10):")
            for reason, count in sorted(skipped_edges.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"    - {reason}: {count:,}")
        
        # Add edges to HeteroData
        edge_type_stats = {}
        for edge_type_key, edges in edge_groups.items():
            if len(edges) == 0:
                continue
            
            source_type, relation, target_type = edge_type_key
            
            # Convert to tensor
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            
            # Add to HeteroData
            self.hetero_data[source_type, relation, target_type].edge_index = edge_index
            
            edge_type_stats[edge_type_key] = len(edges)
            print(f"  ✓ Added {len(edges):,} edges for ({source_type}, {relation}, {target_type})")
        
        print(f"✓ Created {len(edge_type_stats)} edge types")
        return edge_type_stats
    
    def validate_heterodata(self):
        """Validate the HeteroData object"""
        if not self.hetero_data:
            print("❌ No HeteroData to validate")
            return None
        
        print("Validating HeteroData...")
        
        # Basic statistics
        node_types = list(self.hetero_data.node_types)
        edge_types = list(self.hetero_data.edge_types)
        
        # Node analysis
        node_stats = {}
        total_nodes = 0
        
        for node_type in node_types:
            num_nodes = self.hetero_data[node_type].num_nodes
            feature_dim = self.hetero_data[node_type].x.size(1) if hasattr(self.hetero_data[node_type], 'x') else 0
            
            node_stats[node_type] = {
                'num_nodes': num_nodes,
                'feature_dim': feature_dim
            }
            total_nodes += num_nodes
        
        # Edge analysis
        edge_stats = {}
        total_edges = 0
        
        for edge_type in edge_types:
            num_edges = self.hetero_data[edge_type].edge_index.size(1)
            edge_stats[edge_type] = num_edges
            total_edges += num_edges
        
        # Validation checks
        validation_issues = []
        
        # Check for isolated node types (no edges)
        for node_type in node_types:
            has_edges = any(
                node_type in edge_type 
                for edge_type in edge_types
            )
            if not has_edges:
                validation_issues.append(f"Node type '{node_type}' has no edges")
        
        # Check for empty edge types
        for edge_type in edge_types:
            if edge_stats[edge_type] == 0:
                validation_issues.append(f"Edge type '{edge_type}' is empty")
        
        validation_report = {
            'basic_stats': {
                'num_node_types': len(node_types),
                'num_edge_types': len(edge_types),
                'total_nodes': total_nodes,
                'total_edges': total_edges
            },
            'node_types': node_types,
            'edge_types': [str(et) for et in edge_types],
            'node_stats': node_stats,
            'edge_stats': {str(k): v for k, v in edge_stats.items()},
            'validation_issues': validation_issues,
            'is_valid': len(validation_issues) == 0
        }
        
        print("✓ HeteroData Validation Results:")
        print(f"  - Node types: {len(node_types)}")
        print(f"  - Edge types: {len(edge_types)}")
        print(f"  - Total nodes: {total_nodes:,}")
        print(f"  - Total edges: {total_edges:,}")
        
        if validation_issues:
            print("⚠️  Validation Issues:")
            for issue in validation_issues:
                print(f"    - {issue}")
        else:
            print("✅ HeteroData is valid!")
        
        return validation_report

def main():
    """Main pipeline function"""
    print("=" * 80)
    print("RDF → NetworkX → HeteroData Pipeline")
    print("=" * 80)
    
    rdf_file = "/Users/pranavsingh/NNIOntoSearcherEPA/mappings/NKB_RDF_V3.ttl"
    
    # Step 1: Load and convert RDF to NetworkX
    print("\\nStep 1: RDF to NetworkX Conversion")
    print("-" * 40)
    
    converter = RDFToNetworkXConverter(rdf_file)
    
    # Load RDF
    if not converter.load_rdf():
        print("❌ Failed to load RDF file")
        return
    
    # Analyze RDF structure
    rdf_analysis = converter.analyze_rdf_structure()
    
    # Convert to NetworkX
    nx_graph = converter.convert_to_networkx(exclude_literals=True)
    
    # Validate NetworkX graph
    nx_validation = converter.validate_networkx_graph()
    
    # Step 2: Convert NetworkX to HeteroData
    print("\\nStep 2: NetworkX to HeteroData Conversion")
    print("-" * 40)
    
    hetero_converter = NetworkXToHeteroData(nx_graph)
    hetero_data = hetero_converter.convert_to_heterodata(min_nodes_per_type=10)
    
    if hetero_data:
        # Validate HeteroData
        hetero_validation = hetero_converter.validate_heterodata()
        
        # Save results
        print("\\nStep 3: Saving Results")
        print("-" * 40)
        
        # Save NetworkX graph
        with open('networkx_graph.pkl', 'wb') as f:
            pickle.dump(nx_graph, f)
        print("✓ NetworkX graph saved to networkx_graph.pkl")
        
        # Save HeteroData
        torch.save(hetero_data, 'hetero_data.pt')
        print("✓ HeteroData saved to hetero_data.pt")
        
        # Save validation reports
        results = {
            'rdf_analysis': rdf_analysis,
            'networkx_validation': nx_validation,
            'heterodata_validation': hetero_validation
        }
        
        with open('validation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print("✓ Validation results saved to validation_results.json")
        
        print("\\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("Files created:")
        print("- networkx_graph.pkl: NetworkX graph")
        print("- hetero_data.pt: PyTorch Geometric HeteroData")
        print("- validation_results.json: Validation reports")
        
    else:
        print("❌ Failed to create HeteroData")

if __name__ == "__main__":
    main()
