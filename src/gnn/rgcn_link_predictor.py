#!/usr/bin/env python3
"""
R-GCN Link Prediction Interface
Author: Pranav Singh

High-level interface for link prediction using trained R-GCN models.
Provides methods for predicting new links and retrieving top-k candidates.

Usage:
    from rgcn_link_predictor import RGCNLinkPredictor
    predictor = RGCNLinkPredictor()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv
import numpy as np
import pandas as pd
import json
import pickle
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')

# Import our node attribute analyzer
from node_attribute_analyzer import NodeAttributeAnalyzer

class RGCNLinkPredictor:
    """
    Interface for link prediction using trained R-GCN model
    """
    
    def __init__(self, model_path: str = 'best_rgcn_model.pt', 
                 embeddings_path: str = 'nkb_rgcn_embeddings.pt',
                 hetero_data_path: str = 'improved_hetero_data.pt',
                 rdf_file_path: str = 'mappings/NKB_RDF_V3.ttl'):
        """
        Initialize the link predictor
        
        Args:
            model_path: Path to trained R-GCN model
            embeddings_path: Path to extracted embeddings
            hetero_data_path: Path to HeteroData object
            rdf_file_path: Path to original RDF file for attribute analysis
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing R-GCN Link Predictor on {self.device}")
        
        # Load HeteroData
        print("Loading heterogeneous data...")
        self.data = torch.load(hetero_data_path, weights_only=False)
        self.data = self.data.to(self.device)
        
        # Load embeddings
        print("Loading trained embeddings...")
        self.embeddings = torch.load(embeddings_path, weights_only=False)
        
        # Create node mappings
        self._create_node_mappings()
        
        # Load model architecture (we'll recreate it)
        self._load_model(model_path)
        
        # Initialize node attribute analyzer
        print("Initializing node attribute analyzer...")
        try:
            self.attribute_analyzer = NodeAttributeAnalyzer(rdf_file_path, hetero_data_path)
            self.has_attribute_analyzer = True
        except Exception as e:
            print(f"Warning: Could not initialize attribute analyzer: {e}")
            self.attribute_analyzer = None
            self.has_attribute_analyzer = False
        
        print("R-GCN Link Predictor initialized successfully")
        print(f"   - Total nodes: {sum(len(emb) for emb in self.embeddings.values()):,}")
        print(f"   - Node types: {len(self.embeddings)}")
        print(f"   - Edge types: {len(self.data.edge_types)}")
        print(f"   - Attribute analysis: {'Available' if self.has_attribute_analyzer else 'Unavailable'}")
    
    def _create_node_mappings(self):
        """Create mappings between node types and global indices"""
        self.node_to_global = {}
        self.global_to_node = {}
        self.node_type_ranges = {}
        
        global_idx = 0
        for node_type in self.data.node_types:
            num_nodes = self.data[node_type].num_nodes
            start_idx = global_idx
            end_idx = global_idx + num_nodes
            
            self.node_type_ranges[node_type] = (start_idx, end_idx)
            
            # Create bidirectional mappings
            for local_idx in range(num_nodes):
                self.node_to_global[(node_type, local_idx)] = global_idx
                self.global_to_node[global_idx] = (node_type, local_idx)
                global_idx += 1
        
        self.total_nodes = global_idx
        print(f"Created mappings for {self.total_nodes:,} nodes across {len(self.node_type_ranges)} types")
    
    def _load_model(self, model_path: str):
        """Load the trained model (just for architecture reference)"""
        try:
            # We don't actually need the full model for link prediction
            # since we have the embeddings, but we'll load it for completeness
            print(f"Model architecture reference loaded from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load model file: {e}")
            print("Using embeddings-only prediction (recommended)")
    
    def get_node_embedding(self, node_type: str, node_idx: int) -> torch.Tensor:
        """
        Get embedding for a specific node
        
        Args:
            node_type: Type of the node (e.g., 'material', 'assay')
            node_idx: Index of the node within its type
            
        Returns:
            Node embedding tensor
        """
        if node_type not in self.embeddings:
            raise ValueError(f"Unknown node type: {node_type}")
        
        if node_idx >= len(self.embeddings[node_type]):
            raise ValueError(f"Node index {node_idx} out of range for type {node_type}")
        
        return self.embeddings[node_type][node_idx]
    
    def predict_link_probability(self, node1_type: str, node1_idx: int, 
                                node2_type: str, node2_idx: int) -> float:
        """
        Predict the probability of a link between two nodes
        
        Args:
            node1_type: Type of first node
            node1_idx: Index of first node
            node2_type: Type of second node  
            node2_idx: Index of second node
            
        Returns:
            Link probability (0-1)
        """
        # Get embeddings
        emb1 = self.get_node_embedding(node1_type, node1_idx)
        emb2 = self.get_node_embedding(node2_type, node2_idx)
        
        # Compute similarity (cosine similarity -> probability)
        similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        probability = torch.sigmoid(similarity).item()
        
        return probability
    
    def find_similar_nodes(self, node_type: str, node_idx: int, 
                          target_type: Optional[str] = None, 
                          top_k: int = 10) -> List[Tuple[str, int, float]]:
        """
        Find the most similar nodes to a given node
        
        Args:
            node_type: Type of the query node
            node_idx: Index of the query node
            target_type: If specified, only search within this node type
            top_k: Number of similar nodes to return
            
        Returns:
            List of (node_type, node_idx, similarity_score) tuples
        """
        query_emb = self.get_node_embedding(node_type, node_idx)
        similarities = []
        
        # Search target types
        search_types = [target_type] if target_type else self.embeddings.keys()
        
        for search_type in search_types:
            if search_type == node_type:
                # Skip the query node itself
                type_embeddings = self.embeddings[search_type]
                sims = F.cosine_similarity(query_emb.unsqueeze(0), type_embeddings)
                
                for idx, sim in enumerate(sims):
                    if idx != node_idx:  # Skip self
                        similarities.append((search_type, idx, sim.item()))
            else:
                # Different type, compute all similarities
                type_embeddings = self.embeddings[search_type]
                sims = F.cosine_similarity(query_emb.unsqueeze(0), type_embeddings)
                
                for idx, sim in enumerate(sims):
                    similarities.append((search_type, idx, sim.item()))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:top_k]
    
    def predict_links_batch(self, node_pairs: List[Tuple[str, int, str, int]]) -> List[float]:
        """
        Predict link probabilities for multiple node pairs
        
        Args:
            node_pairs: List of (node1_type, node1_idx, node2_type, node2_idx) tuples
            
        Returns:
            List of link probabilities
        """
        probabilities = []
        
        print(f"Predicting links for {len(node_pairs)} node pairs...")
        for i, (n1_type, n1_idx, n2_type, n2_idx) in enumerate(node_pairs):
            if i % 1000 == 0 and i > 0:
                print(f"   Processed {i}/{len(node_pairs)} pairs")
            
            prob = self.predict_link_probability(n1_type, n1_idx, n2_type, n2_idx)
            probabilities.append(prob)
        
        return probabilities
    
    def get_node_info(self, node_type: str, node_idx: int, include_attributes: bool = False) -> Dict:
        """
        Get information about a specific node
        
        Args:
            node_type: Type of the node
            node_idx: Index of the node
            include_attributes: Whether to include detailed RDF attributes
            
        Returns:
            Dictionary with node information
        """
        embedding = self.get_node_embedding(node_type, node_idx)
        global_idx = self.node_to_global.get((node_type, node_idx))
        
        basic_info = {
            'node_type': node_type,
            'node_idx': node_idx,
            'global_idx': global_idx,
            'embedding_dim': embedding.shape[0],
            'embedding_norm': torch.norm(embedding).item(),
            'embedding_mean': embedding.mean().item(),
            'embedding_std': embedding.std().item()
        }
        
        # Add detailed attributes if available and requested
        if include_attributes and self.has_attribute_analyzer:
            try:
                detailed_info = self.attribute_analyzer.get_node_details(node_type, node_idx)
                basic_info.update({
                    'rdf_attributes': detailed_info,
                    'semantic_summary': self.attribute_analyzer.display_node_summary(node_type, node_idx)
                })
            except Exception as e:
                basic_info['attribute_error'] = str(e)
        
        return basic_info
    
    def get_enhanced_node_summary(self, node_type: str, node_idx: int) -> str:
        """
        Get an enhanced summary of node information including RDF attributes
        
        Args:
            node_type: Type of the node
            node_idx: Index of the node
            
        Returns:
            Formatted string with comprehensive node information
        """
        if not self.has_attribute_analyzer:
            return f"Node {node_type}[{node_idx}] - Attribute analysis unavailable"
        
        try:
            return self.attribute_analyzer.display_node_summary(node_type, node_idx)
        except Exception as e:
            return f"Node {node_type}[{node_idx}] - Error retrieving attributes: {e}"
    
    def predict_and_validate_link(self, node1_type: str, node1_idx: int, 
                                 node2_type: str, node2_idx: int) -> Dict:
        """
        Predict link probability and provide comprehensive validation
        
        Args:
            node1_type: Type of first node
            node1_idx: Index of first node
            node2_type: Type of second node
            node2_idx: Index of second node
            
        Returns:
            Dictionary with prediction and validation results
        """
        # Get basic prediction
        probability = self.predict_link_probability(node1_type, node1_idx, node2_type, node2_idx)
        
        result = {
            'prediction': {
                'probability': probability,
                'confidence': 'High' if probability > 0.8 else 'Medium' if probability > 0.6 else 'Low'
            },
            'nodes': {
                'source': f"{node1_type}[{node1_idx}]",
                'destination': f"{node2_type}[{node2_idx}]"
            }
        }
        
        # Add comprehensive validation if attribute analyzer is available
        if self.has_attribute_analyzer:
            try:
                validation = self.attribute_analyzer.validate_link_prediction(
                    node1_type, node1_idx, node2_type, node2_idx, probability
                )
                result['validation'] = validation
            except Exception as e:
                result['validation_error'] = str(e)
        
        return result
    
    def explore_node_type(self, node_type: str, sample_size: int = 10) -> pd.DataFrame:
        """
        Explore nodes of a specific type
        
        Args:
            node_type: Type of nodes to explore
            sample_size: Number of random nodes to sample
            
        Returns:
            DataFrame with node information
        """
        if node_type not in self.embeddings:
            raise ValueError(f"Unknown node type: {node_type}")
        
        num_nodes = len(self.embeddings[node_type])
        sample_indices = np.random.choice(num_nodes, min(sample_size, num_nodes), replace=False)
        
        node_info = []
        for idx in sample_indices:
            info = self.get_node_info(node_type, int(idx))
            node_info.append(info)
        
        return pd.DataFrame(node_info)
    
    def predict_missing_links(self, edge_type: Tuple[str, str, str], 
                            threshold: float = 0.8, 
                            max_predictions: int = 1000) -> pd.DataFrame:
        """
        Predict missing links for a specific edge type
        
        Args:
            edge_type: Tuple of (source_type, relation, target_type)
            threshold: Minimum probability threshold for predictions
            max_predictions: Maximum number of predictions to return
            
        Returns:
            DataFrame with predicted links
        """
        src_type, relation, dst_type = edge_type
        
        if src_type not in self.embeddings or dst_type not in self.embeddings:
            raise ValueError(f"Unknown node types in edge type: {edge_type}")
        
        print(f"Predicting missing links for {src_type} -> {dst_type}")
        
        # Get existing edges to avoid predicting them
        existing_edges = set()
        if edge_type in self.data.edge_types:
            edge_index = self.data[edge_type].edge_index
            for i in range(edge_index.size(1)):
                src_idx, dst_idx = edge_index[:, i].tolist()
                existing_edges.add((src_idx, dst_idx))
        
        # Sample node pairs to evaluate
        src_nodes = min(100, len(self.embeddings[src_type]))
        dst_nodes = min(100, len(self.embeddings[dst_type]))
        
        predictions = []
        evaluated = 0
        
        for src_idx in range(src_nodes):
            for dst_idx in range(dst_nodes):
                if (src_idx, dst_idx) not in existing_edges:
                    prob = self.predict_link_probability(src_type, src_idx, dst_type, dst_idx)
                    
                    if prob >= threshold:
                        predictions.append({
                            'src_type': src_type,
                            'src_idx': src_idx,
                            'dst_type': dst_type,
                            'dst_idx': dst_idx,
                            'relation': relation,
                            'probability': prob
                        })
                
                evaluated += 1
                if evaluated >= max_predictions:
                    break
            
            if evaluated >= max_predictions:
                break
        
        # Sort by probability
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        print(f"Found {len(predictions)} high-confidence predictions")
        return pd.DataFrame(predictions[:max_predictions])
    
    def export_embeddings(self, output_path: str = 'rgcn_embeddings_export.csv'):
        """
        Export all embeddings to CSV for external analysis
        
        Args:
            output_path: Path to save the CSV file
        """
        print(f"Exporting embeddings to {output_path}")
        
        export_data = []
        for node_type, embeddings in self.embeddings.items():
            for idx, embedding in enumerate(embeddings):
                row = {
                    'node_type': node_type,
                    'node_idx': idx,
                    'global_idx': self.node_to_global.get((node_type, idx), -1)
                }
                
                # Add embedding dimensions
                for dim_idx, value in enumerate(embedding.cpu().numpy()):
                    row[f'dim_{dim_idx}'] = value
                
                export_data.append(row)
        
        df = pd.DataFrame(export_data)
        df.to_csv(output_path, index=False)
        print(f"Exported {len(export_data):,} node embeddings")
    
    def interactive_search(self):
        """
        Interactive command-line interface for link prediction
        """
        print("\n" + "="*60)
        print("INTERACTIVE R-GCN LINK PREDICTION")
        print("="*60)
        print("\nAvailable node types:")
        for i, node_type in enumerate(self.embeddings.keys(), 1):
            num_nodes = len(self.embeddings[node_type])
            print(f"  {i:2d}. {node_type:<15} ({num_nodes:,} nodes)")
        
        print("\nCommands:")
        print("  1. predict <type1> <idx1> <type2> <idx2>  - Predict link probability")
        print("  2. similar <type> <idx> [target_type] [k]  - Find similar nodes")
        print("  3. info <type> <idx> [detailed]            - Get node information")
        print("  4. explore <type> [sample_size]            - Explore node type")
        print("  5. missing <src_type> <relation> <dst_type> - Find missing links")
        print("  6. export [filename]                       - Export embeddings")
        print("  7. validate <type1> <idx1> <type2> <idx2>  - Comprehensive link validation")
        print("  8. summary <type> <idx>                    - Enhanced node summary with attributes")
        print("  9. attributes <type> <idx>                 - Detailed RDF attributes")
        print("  10. features <type> <idx>                  - Comprehensive feature and attribute analysis")
        print("  11. quit                                   - Exit")
        
        while True:
            try:
                command = input("\n> ").strip().split()
                if not command:
                    continue
                
                if command[0] == 'quit':
                    break
                elif command[0] == 'predict' and len(command) >= 5:
                    prob = self.predict_link_probability(command[1], int(command[2]), 
                                                       command[3], int(command[4]))
                    print(f"Link probability: {prob:.4f}")
                
                elif command[0] == 'similar' and len(command) >= 3:
                    target_type = command[3] if len(command) > 3 else None
                    k = int(command[4]) if len(command) > 4 else 10
                    similar = self.find_similar_nodes(command[1], int(command[2]), target_type, k)
                    
                    print(f"\nTop {len(similar)} similar nodes:")
                    for node_type, idx, sim in similar:
                        print(f"  {node_type}[{idx}]: {sim:.4f}")
                
                elif command[0] == 'info' and len(command) >= 3:
                    include_detailed = len(command) > 3 and command[3].lower() == 'detailed'
                    info = self.get_node_info(command[1], int(command[2]), include_attributes=include_detailed)
                    print(f"\nNode Information:")
                    for key, value in info.items():
                        if key == 'rdf_attributes':
                            print(f"  RDF Attributes: [Complex object - use 'attributes' command for details]")
                        elif key == 'semantic_summary':
                            print(f"  Semantic Summary:\n{value}")
                        else:
                            print(f"  {key}: {value}")
                
                elif command[0] == 'explore' and len(command) >= 2:
                    sample_size = int(command[2]) if len(command) > 2 else 10
                    df = self.explore_node_type(command[1], sample_size)
                    print(f"\n{df}")
                
                elif command[0] == 'missing' and len(command) >= 4:
                    edge_type = (command[1], command[2], command[3])
                    df = self.predict_missing_links(edge_type)
                    print(f"\n{df.head(20)}")
                
                elif command[0] == 'export':
                    filename = command[1] if len(command) > 1 else 'rgcn_embeddings_export.csv'
                    self.export_embeddings(filename)
                
                elif command[0] == 'validate' and len(command) >= 5:
                    result = self.predict_and_validate_link(
                        command[1], int(command[2]), command[3], int(command[4])
                    )
                    print(f"\nLink Validation Results:")
                    print(f"Prediction: {result['prediction']['probability']:.4f} ({result['prediction']['confidence']} confidence)")
                    
                    if 'validation' in result:
                        val = result['validation']
                        print(f"Validation Confidence: {val['prediction']['confidence_level']}")
                        print(f"Recommendation: {val['validation_summary']['recommendation']}")
                        
                        # Show key evidence
                        if val['validation_summary']['key_evidence']:
                            print(f"Evidence: {', '.join(val['validation_summary']['key_evidence'])}")
                        
                        # Show node summaries
                        print(f"\nSource Node: {val['source_node']['labels'][:2] if val['source_node']['labels'] else 'No labels'}")
                        print(f"Destination Node: {val['destination_node']['labels'][:2] if val['destination_node']['labels'] else 'No labels'}")
                
                elif command[0] == 'summary' and len(command) >= 3:
                    summary = self.get_enhanced_node_summary(command[1], int(command[2]))
                    print(f"\nEnhanced Node Summary:\n{summary}")
                
                elif command[0] == 'attributes' and len(command) >= 3:
                    if not self.has_attribute_analyzer:
                        print("Attribute analysis not available")
                    else:
                        try:
                            details = self.attribute_analyzer.get_node_details(command[1], int(command[2]))
                            print(f"\nDetailed RDF Attributes for {command[1]}[{command[2]}]:")
                            print(f"URI: {details.get('uri', 'Unknown')}")
                            
                            # Show labels
                            labels = details.get('labels', [])
                            if labels:
                                print(f"Labels: {', '.join(labels[:5])}")
                            
                            # Show categories
                            categories = details.get('semantic_analysis', {}).get('semantic_categories', [])
                            if categories:
                                print(f"Categories: {', '.join(categories)}")
                            
                            # Show connectivity
                            conn = details.get('connectivity', {})
                            print(f"Connectivity: {conn.get('total_degree', 0)} total ({conn.get('out_degree', 0)} out, {conn.get('in_degree', 0)} in)")
                            
                            # Show top properties
                            props = details.get('properties', {})
                            if props:
                                print(f"Properties ({len(props)} total):")
                                for prop, values in list(props.items())[:3]:
                                    prop_name = prop.split('/')[-1] if '/' in prop else prop
                                    print(f"  - {prop_name}: {len(values)} value(s)")
                        
                        except Exception as e:
                            print(f"Error retrieving attributes: {e}")
                
                elif command[0] == 'features' and len(command) >= 3:
                    if not self.has_attribute_analyzer:
                        print("Feature analysis not available - attribute analyzer not loaded")
                    else:
                        try:
                            comprehensive_analysis = self.attribute_analyzer.display_comprehensive_analysis(
                                command[1], int(command[2])
                            )
                            print(f"\n{comprehensive_analysis}")
                        except Exception as e:
                            print(f"Error performing comprehensive analysis: {e}")
                
                else:
                    print("Invalid command. Type 'quit' to exit.")
                    
            except Exception as e:
                print(f"Error: {e}")
        
        print("Goodbye!")


def main():
    """
    Main function - demonstrates link prediction capabilities
    """
    print("R-GCN Link Prediction Demo")
    print("="*50)
    
    try:
        # Initialize predictor
        predictor = RGCNLinkPredictor()
        
        # Demo 1: Single link prediction
        print("\nDEMO 1: Single Link Prediction")
        print("-" * 40)
        
        # Example: Predict link between material[0] and assay[0]
        if 'material' in predictor.embeddings and 'assay' in predictor.embeddings:
            prob = predictor.predict_link_probability('material', 0, 'assay', 0)
            print(f"Link probability between material[0] and assay[0]: {prob:.4f}")
        
        # Demo 2: Find similar nodes
        print("\nDEMO 2: Finding Similar Nodes")
        print("-" * 40)
        
        if 'material' in predictor.embeddings:
            similar = predictor.find_similar_nodes('material', 0, top_k=5)
            print("Top 5 nodes similar to material[0]:")
            for node_type, idx, sim in similar:
                print(f"  {node_type}[{idx}]: similarity = {sim:.4f}")
        
        # Demo 3: Explore node types
        print("\nDEMO 3: Node Type Exploration")
        print("-" * 40)
        
        for node_type in list(predictor.embeddings.keys())[:3]:  # First 3 types
            print(f"\n{node_type.upper()} nodes:")
            df = predictor.explore_node_type(node_type, sample_size=3)
            print(df[['node_idx', 'embedding_norm', 'embedding_mean']].to_string(index=False))
        
        # Demo 4: Interactive mode
        print("\nStarting Interactive Mode...")
        print("   (Type 'quit' to exit)")
        predictor.interactive_search()
        
    except FileNotFoundError as e:
        print(f"Error: Required files not found: {e}")
        print("   Make sure you've run proper_rgcn_hetero.py first to generate:")
        print("   - best_rgcn_model.pt")
        print("   - nkb_rgcn_embeddings.pt") 
        print("   - improved_hetero_data.pt")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
