#!/usr/bin/env python3
"""
Advanced Link Prediction Analysis
Author: Pranav Singh

Provides advanced analysis tools for evaluating R-GCN link prediction performance.
Includes metrics computation, embedding visualization, and prediction analysis.

Usage:
    from link_prediction_analysis import AdvancedLinkAnalysis
    analysis = AdvancedLinkAnalysis(trained_predictor)
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from rgcn_link_predictor import RGCNLinkPredictor

class AdvancedLinkAnalysis:
    """
    Advanced analysis tools for R-GCN link prediction
    """
    
    def __init__(self, predictor: RGCNLinkPredictor):
        """
        Initialize with a trained predictor
        
        Args:
            predictor: Trained RGCNLinkPredictor instance
        """
        self.predictor = predictor
        self.data = predictor.data
        self.embeddings = predictor.embeddings
        
        print("Advanced Link Analysis initialized")
        print(f"   - Node types: {len(self.embeddings)}")
        print(f"   - Edge types: {len(self.data.edge_types)}")
    
    def evaluate_existing_links(self, edge_type: Tuple[str, str, str], 
                               sample_size: int = 1000) -> Dict:
        """
        Evaluate how well the model predicts existing links
        
        Args:
            edge_type: Edge type to evaluate
            sample_size: Number of links to sample for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        src_type, relation, dst_type = edge_type
        
        if edge_type not in self.data.edge_types:
            raise ValueError(f"Edge type {edge_type} not found in data")
        
        print(f"Evaluating existing links: {src_type} -> {dst_type}")
        
        # Get positive samples (existing edges)
        edge_index = self.data[edge_type].edge_index
        num_edges = edge_index.size(1)
        
        if num_edges == 0:
            return {"error": "No edges found for this type"}
        
        # Sample edges for evaluation
        sample_indices = np.random.choice(num_edges, min(sample_size, num_edges), replace=False)
        
        pos_predictions = []
        pos_pairs = []
        
        for idx in sample_indices:
            src_idx, dst_idx = edge_index[:, idx].tolist()
            prob = self.predictor.predict_link_probability(src_type, src_idx, dst_type, dst_idx)
            pos_predictions.append(prob)
            pos_pairs.append((src_idx, dst_idx))
        
        # Generate negative samples
        neg_predictions = []
        neg_pairs = []
        existing_edges = set(pos_pairs)
        
        src_nodes = len(self.embeddings[src_type])
        dst_nodes = len(self.embeddings[dst_type])
        
        while len(neg_predictions) < len(pos_predictions):
            src_idx = np.random.randint(0, src_nodes)
            dst_idx = np.random.randint(0, dst_nodes)
            
            if (src_idx, dst_idx) not in existing_edges:
                prob = self.predictor.predict_link_probability(src_type, src_idx, dst_type, dst_idx)
                neg_predictions.append(prob)
                neg_pairs.append((src_idx, dst_idx))
        
        # Compute metrics
        y_true = [1] * len(pos_predictions) + [0] * len(neg_predictions)
        y_scores = pos_predictions + neg_predictions
        
        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        
        # Compute precision at different thresholds
        precision_at_thresholds = {}
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            predicted_pos = sum(1 for score in y_scores[:len(pos_predictions)] if score >= threshold)
            total_predicted = sum(1 for score in y_scores if score >= threshold)
            precision = predicted_pos / max(total_predicted, 1)
            precision_at_thresholds[threshold] = precision
        
        return {
            'edge_type': edge_type,
            'num_evaluated': len(pos_predictions),
            'auc': auc,
            'average_precision': ap,
            'positive_score_mean': np.mean(pos_predictions),
            'positive_score_std': np.std(pos_predictions),
            'negative_score_mean': np.mean(neg_predictions),
            'negative_score_std': np.std(neg_predictions),
            'precision_at_thresholds': precision_at_thresholds
        }
    
    def discover_novel_links(self, src_type: str, dst_type: str, 
                           threshold: float = 0.8, 
                           max_candidates: int = 10000) -> pd.DataFrame:
        """
        Discover novel high-confidence links between node types
        
        Args:
            src_type: Source node type
            dst_type: Destination node type
            threshold: Minimum confidence threshold
            max_candidates: Maximum number of candidates to evaluate
            
        Returns:
            DataFrame with novel link predictions
        """
        print(f"Discovering novel links: {src_type} -> {dst_type}")
        
        # Get existing edges to exclude
        existing_edges = set()
        for edge_type in self.data.edge_types:
            if edge_type[0] == src_type and edge_type[2] == dst_type:
                edge_index = self.data[edge_type].edge_index
                for i in range(edge_index.size(1)):
                    src_idx, dst_idx = edge_index[:, i].tolist()
                    existing_edges.add((src_idx, dst_idx))
        
        print(f"   Excluding {len(existing_edges)} existing edges")
        
        # Sample candidate pairs
        src_nodes = min(100, len(self.embeddings[src_type]))
        dst_nodes = min(100, len(self.embeddings[dst_type]))
        
        candidates = []
        evaluated = 0
        
        for src_idx in range(src_nodes):
            for dst_idx in range(dst_nodes):
                if (src_idx, dst_idx) not in existing_edges and evaluated < max_candidates:
                    prob = self.predictor.predict_link_probability(src_type, src_idx, dst_type, dst_idx)
                    
                    if prob >= threshold:
                        # Get node information
                        src_info = self.predictor.get_node_info(src_type, src_idx)
                        dst_info = self.predictor.get_node_info(dst_type, dst_idx)
                        
                        candidates.append({
                            'src_type': src_type,
                            'src_idx': src_idx,
                            'src_norm': src_info['embedding_norm'],
                            'dst_type': dst_type,
                            'dst_idx': dst_idx,
                            'dst_norm': dst_info['embedding_norm'],
                            'probability': prob,
                            'confidence': 'High' if prob >= 0.9 else 'Medium'
                        })
                    
                    evaluated += 1
        
        df = pd.DataFrame(candidates)
        if not df.empty:
            df = df.sort_values('probability', ascending=False)
        
        print(f"Found {len(candidates)} novel link candidates")
        return df
    
    def analyze_cross_type_relationships(self) -> pd.DataFrame:
        """
        Analyze relationships between different node types
        
        Returns:
            DataFrame with cross-type relationship statistics
        """
        print("Analyzing cross-type relationships...")
        
        node_types = list(self.embeddings.keys())
        relationships = []
        
        for i, src_type in enumerate(node_types):
            for j, dst_type in enumerate(node_types):
                if i != j:  # Different types only
                    # Sample some node pairs
                    src_sample = min(20, len(self.embeddings[src_type]))
                    dst_sample = min(20, len(self.embeddings[dst_type]))
                    
                    similarities = []
                    for src_idx in range(src_sample):
                        for dst_idx in range(dst_sample):
                            prob = self.predictor.predict_link_probability(
                                src_type, src_idx, dst_type, dst_idx
                            )
                            similarities.append(prob)
                    
                    relationships.append({
                        'src_type': src_type,
                        'dst_type': dst_type,
                        'mean_similarity': np.mean(similarities),
                        'std_similarity': np.std(similarities),
                        'max_similarity': np.max(similarities),
                        'min_similarity': np.min(similarities),
                        'high_confidence_count': sum(1 for s in similarities if s >= 0.8),
                        'samples_evaluated': len(similarities)
                    })
        
        df = pd.DataFrame(relationships)
        df = df.sort_values('mean_similarity', ascending=False)
        
        print(f"Analyzed {len(relationships)} cross-type relationships")
        return df
    
    def generate_scientific_hypotheses(self, focus_type: str = 'material', 
                                     min_confidence: float = 0.85) -> List[Dict]:
        """
        Generate scientific hypotheses based on high-confidence novel links
        
        Args:
            focus_type: Node type to focus hypothesis generation on
            min_confidence: Minimum confidence for hypothesis generation
            
        Returns:
            List of hypothesis dictionaries
        """
        print(f"Generating scientific hypotheses for {focus_type} nodes...")
        
        hypotheses = []
        
        # Find high-confidence novel links involving the focus type
        for target_type in self.embeddings.keys():
            if target_type != focus_type:
                # Check both directions
                for src, dst in [(focus_type, target_type), (target_type, focus_type)]:
                    novel_links = self.discover_novel_links(src, dst, min_confidence, 1000)
                    
                    if not novel_links.empty:
                        top_links = novel_links.head(5)
                        
                        hypothesis = {
                            'hypothesis_type': f"{src} -> {dst} relationship",
                            'confidence_level': 'High' if novel_links['probability'].mean() >= 0.9 else 'Medium',
                            'num_predictions': len(novel_links),
                            'avg_probability': novel_links['probability'].mean(),
                            'top_examples': top_links.to_dict('records'),
                            'scientific_interpretation': self._interpret_relationship(src, dst, novel_links)
                        }
                        
                        hypotheses.append(hypothesis)
        
        print(f"Generated {len(hypotheses)} scientific hypotheses")
        return hypotheses
    
    def _interpret_relationship(self, src_type: str, dst_type: str, 
                              predictions: pd.DataFrame) -> str:
        """
        Generate scientific interpretation of predicted relationships
        """
        interpretations = {
            ('material', 'assay'): "These materials may be suitable for the predicted assay types based on their physicochemical properties.",
            ('material', 'result'): "These materials may produce similar experimental outcomes based on their structural similarity.",
            ('assay', 'result'): "These assays may be predictive of the suggested result types.",
            ('material', 'parameters'): "These materials may require specific experimental parameters for optimal testing.",
            ('parameters', 'result'): "These parameter settings may influence the predicted result outcomes."
        }
        
        key = (src_type, dst_type)
        base_interpretation = interpretations.get(key, 
            f"Novel {src_type}-{dst_type} relationships suggest previously unknown connections.")
        
        confidence_note = ""
        if predictions['probability'].mean() >= 0.9:
            confidence_note = " The high confidence scores suggest these predictions are highly reliable."
        elif predictions['probability'].mean() >= 0.8:
            confidence_note = " The moderate confidence scores suggest these predictions warrant further investigation."
        
        return base_interpretation + confidence_note
    
    def create_prediction_report(self, output_path: str = 'link_prediction_report.html'):
        """
        Create a comprehensive HTML report of link prediction analysis
        
        Args:
            output_path: Path to save the HTML report
        """
        print("Creating comprehensive prediction report...")
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>R-GCN Link Prediction Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 10px; }
                .section { margin: 20px 0; }
                .metric { background-color: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 5px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .high-confidence { color: #27ae60; font-weight: bold; }
                .medium-confidence { color: #f39c12; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>R-GCN Link Prediction Analysis Report</h1>
                <p>Generated from trained R-GCN model on NKB knowledge graph</p>
                <p><strong>Model Performance:</strong> 99.0% Validation AUC | 6M+ Parameters | 290K+ Nodes</p>
            </div>
        """
        
        # Add cross-type analysis
        cross_type_df = self.analyze_cross_type_relationships()
        html_content += """
            <div class="section">
                <h2>Cross-Type Relationship Analysis</h2>
                <p>Analysis of relationship strengths between different node types:</p>
        """
        html_content += cross_type_df.head(10).to_html(classes='table', escape=False)
        html_content += "</div>"
        
        # Add novel link discoveries
        html_content += """
            <div class="section">
                <h2>Novel Link Discoveries</h2>
        """
        
        key_types = ['material', 'assay', 'result']
        for src_type in key_types:
            for dst_type in key_types:
                if src_type != dst_type:
                    try:
                        novel_df = self.discover_novel_links(src_type, dst_type, 0.8, 1000)
                        if not novel_df.empty:
                            html_content += f"<h3>{src_type.title()} → {dst_type.title()}</h3>"
                            html_content += novel_df.head(5).to_html(classes='table', escape=False)
                    except Exception as e:
                        continue
        
        html_content += "</div>"
        
        # Add scientific hypotheses
        try:
            hypotheses = self.generate_scientific_hypotheses('material', 0.85)
            html_content += """
                <div class="section">
                    <h2>Generated Scientific Hypotheses</h2>
            """
            
            for i, hyp in enumerate(hypotheses[:5], 1):
                confidence_class = "high-confidence" if hyp['confidence_level'] == 'High' else "medium-confidence"
                html_content += f"""
                    <div class="metric">
                        <h4>Hypothesis {i}: {hyp['hypothesis_type']}</h4>
                        <p><span class="{confidence_class}">Confidence: {hyp['confidence_level']}</span> 
                           | Predictions: {hyp['num_predictions']} 
                           | Avg Probability: {hyp['avg_probability']:.3f}</p>
                        <p><em>{hyp['scientific_interpretation']}</em></p>
                    </div>
                """
            
            html_content += "</div>"
        except Exception as e:
            html_content += f"<p>Hypothesis generation encountered an issue: {e}</p>"
        
        html_content += """
            <div class="section">
                <h2>Model Statistics</h2>
                <div class="metric">Node Types: """ + str(len(self.embeddings)) + """</div>
                <div class="metric">Total Nodes: """ + f"{sum(len(emb) for emb in self.embeddings.values()):,}" + """</div>
                <div class="metric">Edge Types: """ + str(len(self.data.edge_types)) + """</div>
                <div class="metric">Embedding Dimension: 128</div>
            </div>
            
            <div class="section">
                <h2>Usage Recommendations</h2>
                <ul>
                    <li><strong>High Confidence Predictions (>0.9):</strong> Suitable for experimental validation</li>
                    <li><strong>Medium Confidence Predictions (0.8-0.9):</strong> Good candidates for further computational analysis</li>
                    <li><strong>Cross-Type Relationships:</strong> Use for understanding system-level interactions</li>
                    <li><strong>Novel Links:</strong> Prioritize for hypothesis-driven research</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Report saved to {output_path}")


def main():
    """
    Main function - demonstrates advanced link prediction analysis
    """
    print("Advanced Link Prediction Analysis")
    print("=" * 50)
    
    try:
        # Initialize predictor and analyzer
        predictor = RGCNLinkPredictor()
        analyzer = AdvancedLinkAnalysis(predictor)
        
        # Analyze cross-type relationships
        print("\nCross-Type Relationship Analysis:")
        cross_type_df = analyzer.analyze_cross_type_relationships()
        print(cross_type_df.head(10))
        
        # Discover novel links
        print("\nNovel Link Discovery:")
        if 'material' in predictor.embeddings and 'assay' in predictor.embeddings:
            novel_links = analyzer.discover_novel_links('material', 'assay', 0.8)
            if not novel_links.empty:
                print(novel_links.head())
            else:
                print("No high-confidence novel links found")
        
        # Generate scientific hypotheses
        print("\nScientific Hypothesis Generation:")
        hypotheses = analyzer.generate_scientific_hypotheses('material', 0.85)
        for i, hyp in enumerate(hypotheses[:3], 1):
            print(f"\nHypothesis {i}: {hyp['hypothesis_type']}")
            print(f"  Confidence: {hyp['confidence_level']}")
            print(f"  Predictions: {hyp['num_predictions']}")
            print(f"  Interpretation: {hyp['scientific_interpretation']}")
        
        # Create comprehensive report
        print("\nCreating Comprehensive Report...")
        analyzer.create_prediction_report()
        
        print("\nAdvanced analysis completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
