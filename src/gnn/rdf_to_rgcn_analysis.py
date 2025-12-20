"""
Comprehensive RDF to R-GCN Pipeline for NaKnowBase
==================================================

This script provides a complete pipeline from RDF knowledge graph to 
Relational Graph Convolutional Network (R-GCN) for link prediction.

Based on the EPA NaKnowBase semantic mapping paper:
"Translating nanoEHS data using EPA NaKnowBase and the resource description framework"

Entity Types in NKB_RDF_V3.ttl:
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

Total: ~131,388 entities with 10,261,162 triples
"""

import pandas as pd
import numpy as np
import networkx as nx
from rdflib import Graph, Namespace, URIRef, Literal
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Set
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
    from torch_geometric.nn import RGCNConv, Linear
    from torch_geometric.transforms import RandomNodeSplit
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import roc_auc_score, average_precision_score
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch Geometric not available. Install with: pip install torch torch-geometric")
    TORCH_AVAILABLE = False

class RDFAnalyzer:
    """Comprehensive RDF analysis and EDA"""
    
    def __init__(self, rdf_file_path: str):
        self.rdf_file_path = rdf_file_path
        self.graph = Graph()
        self.entity_stats = {}
        self.relation_stats = {}
        self.namespace_stats = {}
        
        # Define entity types based on the paper
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
        
    def load_rdf(self, sample_size: int = None):
        """Load RDF graph with optional sampling for large files"""
        print(f"Loading RDF from {self.rdf_file_path}...")
        
        if sample_size:
            print(f"Sampling first {sample_size} lines for analysis...")
            with open(self.rdf_file_path, 'r', encoding='utf-8') as f:
                lines = []
                line_count = 0
                
                # First, get all prefix lines
                f.seek(0)
                for line in f:
                    if line.startswith('@prefix') or line.startswith('@base'):
                        lines.append(line)
                    elif line.strip() and not line.startswith('@'):
                        lines.append(line)
                        line_count += 1
                        if line_count >= sample_size:
                            break
            
            sampled_content = ''.join(lines)
            
            # Try to parse, if it fails, load without sampling
            try:
                self.graph.parse(data=sampled_content, format='turtle')
            except Exception as e:
                print(f"Sampling failed ({e}), loading full file...")
                self.graph.parse(self.rdf_file_path, format='turtle')
        else:
            self.graph.parse(self.rdf_file_path, format='turtle')
            
        print(f"Loaded {len(self.graph)} triples")
        
    def analyze_entity_types(self):
        """Analyze entity types and their distributions"""
        print("Analyzing entity types...")
        
        entity_counts = defaultdict(int)
        entity_examples = defaultdict(list)
        
        for subject, predicate, obj in tqdm(self.graph):
            if isinstance(subject, URIRef):
                subject_str = str(subject)
                for entity_type, prefix in self.entity_types.items():
                    if subject_str.startswith(prefix):
                        entity_counts[entity_type] += 1
                        if len(entity_examples[entity_type]) < 5:
                            entity_examples[entity_type].append(subject_str)
                        break
        
        self.entity_stats = {
            'counts': dict(entity_counts),
            'examples': dict(entity_examples)
        }
        
        return self.entity_stats
    
    def analyze_relations(self):
        """Analyze predicate/relation types"""
        print("Analyzing relations...")
        
        relation_counts = Counter()
        relation_examples = defaultdict(list)
        
        for subject, predicate, obj in tqdm(self.graph):
            predicate_str = str(predicate)
            relation_counts[predicate_str] += 1
            
            if len(relation_examples[predicate_str]) < 3:
                relation_examples[predicate_str].append({
                    'subject': str(subject),
                    'object': str(obj)
                })
        
        self.relation_stats = {
            'counts': dict(relation_counts.most_common(50)),
            'examples': dict(relation_examples)
        }
        
        return self.relation_stats
    
    def analyze_namespaces(self):
        """Analyze namespace usage"""
        print("Analyzing namespaces...")
        
        namespace_counts = Counter()
        
        for subject, predicate, obj in self.graph:
            # Analyze predicates (most informative for namespaces)
            if isinstance(predicate, URIRef):
                pred_str = str(predicate)
                if '#' in pred_str:
                    namespace = pred_str.split('#')[0] + '#'
                elif '/' in pred_str:
                    parts = pred_str.split('/')
                    namespace = '/'.join(parts[:-1]) + '/'
                else:
                    namespace = pred_str
                    
                namespace_counts[namespace] += 1
        
        self.namespace_stats = dict(namespace_counts.most_common(20))
        return self.namespace_stats
    
    def analyze_completeness(self):
        """Analyze data completeness across entity types"""
        print("Analyzing data completeness...")
        
        completeness_stats = {}
        
        for entity_type, prefix in self.entity_types.items():
            entities = [s for s, p, o in self.graph if str(s).startswith(prefix)]
            unique_entities = set(entities)
            
            if not unique_entities:
                continue
                
            # Count properties per entity
            property_counts = defaultdict(int)
            for entity in unique_entities:
                properties = set()
                for s, p, o in self.graph:
                    if s == entity:
                        properties.add(str(p))
                property_counts[len(properties)] += 1
            
            completeness_stats[entity_type] = {
                'total_entities': len(unique_entities),
                'property_distribution': dict(property_counts),
                'avg_properties': np.mean(list(property_counts.keys())) if property_counts else 0
            }
        
        return completeness_stats
    
    def generate_eda_report(self, output_file: str = 'rdf_eda_report.json'):
        """Generate comprehensive EDA report"""
        print("Generating EDA report...")
        
        report = {
            'file_info': {
                'file_path': self.rdf_file_path,
                'total_triples': len(self.graph),
                'total_subjects': len(set(s for s, p, o in self.graph)),
                'total_predicates': len(set(p for s, p, o in self.graph)),
                'total_objects': len(set(o for s, p, o in self.graph))
            },
            'entity_analysis': self.analyze_entity_types(),
            'relation_analysis': self.analyze_relations(),
            'namespace_analysis': self.analyze_namespaces(),
            'completeness_analysis': self.analyze_completeness()
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"EDA report saved to {output_file}")
        return report
    
    def visualize_entity_distribution(self):
        """Create visualizations of entity distributions"""
        if not self.entity_stats:
            self.analyze_entity_types()
        
        plt.figure(figsize=(15, 10))
        
        # Entity type distribution
        plt.subplot(2, 2, 1)
        entity_counts = self.entity_stats['counts']
        plt.bar(entity_counts.keys(), entity_counts.values())
        plt.title('Entity Type Distribution')
        plt.xticks(rotation=45)
        plt.ylabel('Count')
        
        # Log scale for better visualization
        plt.subplot(2, 2, 2)
        plt.bar(entity_counts.keys(), entity_counts.values())
        plt.title('Entity Type Distribution (Log Scale)')
        plt.xticks(rotation=45)
        plt.ylabel('Count (Log Scale)')
        plt.yscale('log')
        
        # Top relations
        if self.relation_stats:
            plt.subplot(2, 2, 3)
            top_relations = list(self.relation_stats['counts'].items())[:10]
            relations, counts = zip(*top_relations)
            relation_names = [r.split('#')[-1] if '#' in r else r.split('/')[-1] for r in relations]
            plt.barh(relation_names, counts)
            plt.title('Top 10 Relations')
            plt.xlabel('Count')
        
        # Namespace usage
        if self.namespace_stats:
            plt.subplot(2, 2, 4)
            top_namespaces = list(self.namespace_stats.items())[:10]
            namespaces, counts = zip(*top_namespaces)
            namespace_names = [ns.split('/')[-2] if '/' in ns else ns for ns in namespaces]
            plt.barh(namespace_names, counts)
            plt.title('Top 10 Namespaces')
            plt.xlabel('Count')
        
        plt.tight_layout()
        plt.savefig('rdf_entity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

class RDFToNetworkX:
    """Convert RDF to NetworkX graph"""
    
    def __init__(self, rdf_analyzer: RDFAnalyzer):
        self.rdf_analyzer = rdf_analyzer
        self.nx_graph = None
        self.entity_type_mapping = {}
        
    def convert_to_networkx(self, include_literals: bool = False):
        """Convert RDF graph to NetworkX MultiDiGraph"""
        print("Converting RDF to NetworkX...")
        
        self.nx_graph = nx.MultiDiGraph()
        
        for subject, predicate, obj in tqdm(self.rdf_analyzer.graph):
            subject_str = str(subject)
            predicate_str = str(predicate)
            obj_str = str(obj)
            
            # Skip literals unless requested
            if not include_literals and isinstance(obj, Literal):
                continue
            
            # Add nodes with entity type information
            subject_type = self._get_entity_type(subject_str)
            if subject_type:
                self.nx_graph.add_node(subject_str, entity_type=subject_type)
            
            if isinstance(obj, URIRef):
                obj_type = self._get_entity_type(obj_str)
                if obj_type:
                    self.nx_graph.add_node(obj_str, entity_type=obj_type)
            
            # Add edge with relation type
            self.nx_graph.add_edge(
                subject_str, 
                obj_str, 
                relation=predicate_str,
                relation_short=self._get_short_relation_name(predicate_str)
            )
        
        print(f"NetworkX graph created with {self.nx_graph.number_of_nodes()} nodes and {self.nx_graph.number_of_edges()} edges")
        return self.nx_graph
    
    def _get_entity_type(self, uri: str) -> str:
        """Determine entity type from URI"""
        for entity_type, prefix in self.rdf_analyzer.entity_types.items():
            if uri.startswith(prefix):
                return entity_type
        return 'unknown'
    
    def _get_short_relation_name(self, relation_uri: str) -> str:
        """Get short name for relation"""
        if '#' in relation_uri:
            return relation_uri.split('#')[-1]
        elif '/' in relation_uri:
            return relation_uri.split('/')[-1]
        return relation_uri
    
    def analyze_graph_structure(self):
        """Analyze NetworkX graph structure"""
        if not self.nx_graph:
            raise ValueError("NetworkX graph not created yet. Call convert_to_networkx() first.")
        
        analysis = {
            'basic_stats': {
                'nodes': self.nx_graph.number_of_nodes(),
                'edges': self.nx_graph.number_of_edges(),
                'is_directed': self.nx_graph.is_directed(),
                'is_multigraph': self.nx_graph.is_multigraph()
            },
            'connectivity': {
                'is_connected': nx.is_weakly_connected(self.nx_graph),
                'number_of_components': nx.number_weakly_connected_components(self.nx_graph),
                'largest_component_size': len(max(nx.weakly_connected_components(self.nx_graph), key=len))
            },
            'node_types': Counter([data.get('entity_type', 'unknown') 
                                 for _, data in self.nx_graph.nodes(data=True)]),
            'relation_types': Counter([data['relation_short'] 
                                     for _, _, data in self.nx_graph.edges(data=True)])
        }
        
        return analysis

if __name__ == "__main__":
    # Initialize analyzer
    print("=== RDF to R-GCN Analysis Pipeline ===")
    print("Step 1: Comprehensive EDA of RDF Knowledge Graph")
    
    rdf_file = "/Users/pranavsingh/NNIOntoSearcherEPA/mappings/NKB_RDF_V3.ttl"
    analyzer = RDFAnalyzer(rdf_file)
    
    # Load with sampling for initial analysis (full graph is very large)
    analyzer.load_rdf(sample_size=50000)  # Sample for initial analysis
    
    # Generate comprehensive EDA
    eda_report = analyzer.generate_eda_report()
    
    # Create visualizations
    analyzer.visualize_entity_distribution()
    
    print("\n=== EDA Summary ===")
    print(f"Total triples analyzed: {eda_report['file_info']['total_triples']}")
    print(f"Entity types found: {len(eda_report['entity_analysis']['counts'])}")
    print(f"Top entity types: {list(eda_report['entity_analysis']['counts'].keys())[:5]}")
    
    print("\nStep 2: Converting to NetworkX...")
    nx_converter = RDFToNetworkX(analyzer)
    nx_graph = nx_converter.convert_to_networkx(include_literals=False)
    
    # Analyze graph structure
    graph_analysis = nx_converter.analyze_graph_structure()
    print(f"NetworkX graph: {graph_analysis['basic_stats']['nodes']} nodes, {graph_analysis['basic_stats']['edges']} edges")
    
    print("\nNext steps:")
    print("1. Run full graph analysis (without sampling)")
    print("2. Create PyTorch Geometric HeteroData object")
    print("3. Generate node features for each entity type")
    print("4. Implement R-GCN for link prediction")
