"""
Proper R-GCN Implementation for Heterogeneous Knowledge Graphs
============================================================

This implements a true Relational Graph Convolutional Network (R-GCN) 
using RGCNConv layers directly for the EPA NaKnowBase heterogeneous graph.

Key differences from previous attempts:
1. Uses RGCNConv directly (not with HeteroConv)
2. Properly handles heterogeneous graphs with multiple node/edge types
3. Creates a unified node representation that RGCNConv can work with
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData, Data
from torch_geometric.nn import RGCNConv, Linear
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pickle

class ProperRGCN(nn.Module):
    """
    Proper R-GCN implementation for heterogeneous graphs
    
    Architecture:
    1. Convert heterogeneous graph to homogeneous with relation types
    2. Use RGCNConv layers with proper relation handling
    3. Convert back to heterogeneous for link prediction
    """
    
    def __init__(self, 
                 hetero_data: HeteroData,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        
        self.hetero_data = hetero_data
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create mappings between heterogeneous and homogeneous representations
        self.node_type_to_range = {}
        self.node_type_to_indices = {}
        self.relation_to_id = {}
        self.id_to_relation = {}
        
        # Build the mappings
        self._build_mappings()
        
        # Total number of nodes and relations
        self.num_nodes = sum(hetero_data[nt].num_nodes for nt in hetero_data.node_types)
        self.num_relations = len(self.relation_to_id)
        
        print(f"R-GCN Architecture:")
        print(f"  - Total nodes: {self.num_nodes:,}")
        print(f"  - Total relations: {self.num_relations}")
        print(f"  - Hidden dim: {hidden_dim}")
        print(f"  - Layers: {num_layers}")
        
        # Node type embeddings (project different node types to same dimension)
        self.node_embeddings = nn.ModuleDict()
        for node_type in hetero_data.node_types:
            if hasattr(hetero_data[node_type], 'x') and hetero_data[node_type].x is not None:
                input_dim = hetero_data[node_type].x.size(1)
                self.node_embeddings[f"emb_{node_type}"] = nn.Linear(input_dim, hidden_dim)
            else:
                num_nodes = hetero_data[node_type].num_nodes
                self.node_embeddings[f"emb_{node_type}"] = nn.Embedding(num_nodes, hidden_dim)
        
        # R-GCN layers
        self.rgcn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # First layer: input -> hidden
                self.rgcn_layers.append(RGCNConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    num_relations=self.num_relations,
                    aggr='mean',
                    bias=True
                ))
            else:
                # Hidden layers: hidden -> hidden
                self.rgcn_layers.append(RGCNConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    num_relations=self.num_relations,
                    aggr='mean',
                    bias=True
                ))
        
        # Link prediction head
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def _build_mappings(self):
        """Build mappings between heterogeneous and homogeneous representations"""
        print("Building node and relation mappings...")
        
        # Build node type mappings
        current_idx = 0
        for node_type in self.hetero_data.node_types:
            num_nodes = self.hetero_data[node_type].num_nodes
            self.node_type_to_range[node_type] = (current_idx, current_idx + num_nodes)
            self.node_type_to_indices[node_type] = torch.arange(current_idx, current_idx + num_nodes)
            current_idx += num_nodes
            print(f"  - {node_type}: nodes {self.node_type_to_range[node_type][0]}-{self.node_type_to_range[node_type][1]-1}")
        
        # Build relation mappings
        relation_id = 0
        for edge_type in self.hetero_data.edge_types:
            src_type, rel_type, dst_type = edge_type
            relation_key = f"{src_type}__{rel_type}__{dst_type}"
            self.relation_to_id[relation_key] = relation_id
            self.id_to_relation[relation_id] = edge_type
            relation_id += 1
            print(f"  - Relation {relation_id-1}: {edge_type}")
    
    def _hetero_to_homo(self):
        """Convert heterogeneous graph to homogeneous for R-GCN processing"""
        
        # Create unified node features
        all_node_features = []
        
        for node_type in self.hetero_data.node_types:
            emb_key = f"emb_{node_type}"
            
            if hasattr(self.hetero_data[node_type], 'x') and self.hetero_data[node_type].x is not None:
                # Use existing features
                node_features = self.node_embeddings[emb_key](self.hetero_data[node_type].x)
            else:
                # Use learned embeddings
                num_nodes = self.hetero_data[node_type].num_nodes
                node_ids = torch.arange(num_nodes, device=next(self.parameters()).device)
                node_features = self.node_embeddings[emb_key](node_ids)
            
            all_node_features.append(node_features)
        
        # Concatenate all node features
        x = torch.cat(all_node_features, dim=0)
        
        # Create unified edge index and edge types
        all_edge_indices = []
        all_edge_types = []
        
        for edge_type in self.hetero_data.edge_types:
            src_type, rel_type, dst_type = edge_type
            
            if edge_type not in self.hetero_data.edge_index_dict:
                continue
                
            edge_index = self.hetero_data[edge_type].edge_index
            if edge_index.size(1) == 0:
                continue
            
            # Map to global node indices
            src_offset = self.node_type_to_range[src_type][0]
            dst_offset = self.node_type_to_range[dst_type][0]
            
            global_edge_index = edge_index.clone()
            global_edge_index[0] += src_offset  # Source nodes
            global_edge_index[1] += dst_offset  # Destination nodes
            
            # Get relation ID
            relation_key = f"{src_type}__{rel_type}__{dst_type}"
            relation_id = self.relation_to_id[relation_key]
            
            all_edge_indices.append(global_edge_index)
            all_edge_types.extend([relation_id] * edge_index.size(1))
        
        if all_edge_indices:
            edge_index = torch.cat(all_edge_indices, dim=1)
            edge_type = torch.tensor(all_edge_types, dtype=torch.long, device=x.device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
            edge_type = torch.empty((0,), dtype=torch.long, device=x.device)
        
        return x, edge_index, edge_type
    
    def forward(self, return_embeddings: bool = False):
        """Forward pass through R-GCN"""
        
        # Convert to homogeneous representation
        x, edge_index, edge_type = self._hetero_to_homo()
        
        # Apply R-GCN layers
        for i, rgcn_layer in enumerate(self.rgcn_layers):
            x = rgcn_layer(x, edge_index, edge_type)
            
            if i < len(self.rgcn_layers) - 1:  # Don't apply activation after last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        if return_embeddings:
            # Convert back to heterogeneous format
            embeddings_dict = {}
            for node_type in self.hetero_data.node_types:
                start_idx, end_idx = self.node_type_to_range[node_type]
                embeddings_dict[node_type] = x[start_idx:end_idx]
            return embeddings_dict
        
        return x
    
    def predict_links(self, edge_label_index: torch.Tensor, edge_type: tuple) -> torch.Tensor:
        """Predict link probabilities for given edge indices"""
        
        # Get embeddings
        embeddings_dict = self.forward(return_embeddings=True)
        
        src_type, _, dst_type = edge_type
        
        if src_type not in embeddings_dict or dst_type not in embeddings_dict:
            return torch.zeros(edge_label_index.size(1), device=edge_label_index.device)
        
        # Get source and destination embeddings
        src_embeddings = embeddings_dict[src_type][edge_label_index[0]]
        dst_embeddings = embeddings_dict[dst_type][edge_label_index[1]]
        
        # Concatenate and predict
        edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=-1)
        predictions = self.link_predictor(edge_embeddings).squeeze()
        
        return predictions

class RGCNTrainer:
    """Trainer for the proper R-GCN model"""
    
    def __init__(self, model: ProperRGCN, data: HeteroData, config: dict):
        self.model = model
        self.data = data
        self.config = config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['learning_rate'], 
            weight_decay=config['weight_decay']
        )
        self.criterion = nn.BCELoss()
        
        # Learning rate scheduler for overfitting prevention
        if config.get('lr_scheduler', False):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # Maximize validation AUC
                factor=config.get('lr_decay_factor', 0.8),
                patience=config.get('lr_decay_patience', 10),
                min_lr=1e-6
            )
        else:
            self.scheduler = None
        
        # Training history with more detailed tracking
        self.train_losses = []
        self.val_aucs = []
        self.val_aps = []
        self.train_aucs = []  # Track training AUC to monitor overfitting
        self.learning_rates = []
        self.best_epoch = 0
        
    def create_splits(self):
        """Create train/validation splits"""
        print("Creating data splits...")
        
        self.train_data = HeteroData()
        self.val_data = HeteroData()
        self.test_data = HeteroData()  # Add test set for final evaluation
        
        # Copy node data to all splits
        for node_type in self.data.node_types:
            for split_data in [self.train_data, self.val_data, self.test_data]:
                split_data[node_type].x = self.data[node_type].x
                split_data[node_type].num_nodes = self.data[node_type].num_nodes
        
        # Select edge types for training
        edge_type_counts = [(et, self.data[et].edge_index.size(1)) for et in self.data.edge_types]
        edge_type_counts.sort(key=lambda x: x[1], reverse=True)
        
        self.selected_edge_types = []
        for et, count in edge_type_counts[:self.config['num_edge_types']]:
            if count >= self.config['min_edges_per_type']:
                self.selected_edge_types.append(et)
        
        print(f"Selected {len(self.selected_edge_types)} edge types:")
        
        # Split edges with train/val/test splits
        train_ratio = self.config.get('train_split', 0.7)
        val_ratio = self.config.get('val_split', 0.15)
        test_ratio = self.config.get('test_split', 0.15)
        
        for edge_type in self.selected_edge_types:
            edge_index = self.data[edge_type].edge_index
            num_edges = edge_index.size(1)
            
            # Calculate split sizes
            train_size = int(num_edges * train_ratio)
            val_size = int(num_edges * val_ratio)
            test_size = num_edges - train_size - val_size
            
            perm = torch.randperm(num_edges, device=edge_index.device)
            
            train_idx = perm[:train_size]
            val_idx = perm[train_size:train_size + val_size]
            test_idx = perm[train_size + val_size:]
            
            self.train_data[edge_type].edge_index = edge_index[:, train_idx]
            self.val_data[edge_type].edge_index = edge_index[:, val_idx]
            self.test_data[edge_type].edge_index = edge_index[:, test_idx]
            
            print(f"  - {edge_type}: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
    
    def create_negative_samples(self, edge_type: tuple, pos_edge_index: torch.Tensor, num_neg: int = None):
        """Create negative samples"""
        src_type, _, dst_type = edge_type
        
        if num_neg is None:
            num_neg = pos_edge_index.size(1)
        
        src_num_nodes = self.data[src_type].num_nodes
        dst_num_nodes = self.data[dst_type].num_nodes
        
        src_neg = torch.randint(0, src_num_nodes, (num_neg,), device=self.device)
        dst_neg = torch.randint(0, dst_num_nodes, (num_neg,), device=self.device)
        
        return torch.stack([src_neg, dst_neg], dim=0)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for edge_type in self.selected_edge_types:
            if edge_type not in self.train_data.edge_types:
                continue
                
            pos_edge_index = self.train_data[edge_type].edge_index
            if pos_edge_index.size(1) == 0:
                continue
            
            # Create negative samples
            neg_edge_index = self.create_negative_samples(edge_type, pos_edge_index)
            
            # Combine positive and negative edges
            edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            edge_labels = torch.cat([
                torch.ones(pos_edge_index.size(1), device=self.device),
                torch.zeros(neg_edge_index.size(1), device=self.device)
            ], dim=0)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Update model's hetero_data to use training data
            original_data = self.model.hetero_data
            self.model.hetero_data = self.train_data
            
            predictions = self.model.predict_links(edge_label_index, edge_type)
            
            # Restore original data
            self.model.hetero_data = original_data
            
            # Handle size mismatch
            min_size = min(predictions.size(0), edge_labels.size(0))
            if min_size > 0:
                predictions = predictions[:min_size]
                edge_labels = edge_labels[:min_size]
                
                loss = self.criterion(predictions, edge_labels)
                loss.backward()
                
                # Gradient clipping to prevent overfitting
                if self.config.get('gradient_clipping', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['gradient_clipping']
                    )
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def evaluate(self, data_split='val'):
        """Evaluate model on validation or training data"""
        self.model.eval()
        all_aucs = []
        all_aps = []
        
        # Select data split
        if data_split == 'val':
            eval_data = self.val_data
        elif data_split == 'train':
            eval_data = self.train_data
        else:
            eval_data = self.test_data
        
        with torch.no_grad():
            for edge_type in self.selected_edge_types[:5]:  # Evaluate on top 5
                if edge_type not in eval_data.edge_types:
                    continue
                    
                pos_edge_index = eval_data[edge_type].edge_index
                if pos_edge_index.size(1) < 5:
                    continue
                
                neg_edge_index = self.create_negative_samples(edge_type, pos_edge_index)
                
                edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
                edge_labels = torch.cat([
                    torch.ones(pos_edge_index.size(1)),
                    torch.zeros(neg_edge_index.size(1))
                ], dim=0).cpu().numpy()
                
                # Update model's hetero_data to use evaluation data
                original_data = self.model.hetero_data
                self.model.hetero_data = eval_data
                
                predictions = self.model.predict_links(edge_label_index, edge_type)
                predictions = predictions.cpu().numpy()
                
                # Restore original data
                self.model.hetero_data = original_data
                
                # Handle size mismatch
                min_size = min(len(predictions), len(edge_labels))
                if min_size > 0 and len(set(edge_labels[:min_size])) > 1:
                    try:
                        auc = roc_auc_score(edge_labels[:min_size], predictions[:min_size])
                        ap = average_precision_score(edge_labels[:min_size], predictions[:min_size])
                        all_aucs.append(auc)
                        all_aps.append(ap)
                    except:
                        pass
        
        avg_auc = np.mean(all_aucs) if all_aucs else 0.5
        avg_ap = np.mean(all_aps) if all_aps else 0.5
        
        if data_split == 'val':
            self.val_aucs.append(avg_auc)
            self.val_aps.append(avg_ap)
        elif data_split == 'train':
            self.train_aucs.append(avg_auc)
            
        return avg_auc, avg_ap
    
    def train(self):
        """Main training loop with comprehensive overfitting monitoring"""
        print(f"\\nStarting R-GCN training for {self.config['num_epochs']} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("\\nOverfitting Prevention Measures:")
        print(f"  - Dropout: {self.model.dropout}")
        print(f"  - Weight Decay: {self.config['weight_decay']}")
        print(f"  - Gradient Clipping: {self.config.get('gradient_clipping', 'None')}")
        print(f"  - Early Stopping Patience: {self.config.get('early_stopping_patience', 'None')}")
        print(f"  - LR Scheduler: {self.config.get('lr_scheduler', False)}")
        print(f"  - Train/Val/Test Split: {self.config.get('train_split', 0.8):.1f}/{self.config.get('val_split', 0.1):.1f}/{self.config.get('test_split', 0.1):.1f}")
        
        best_val_auc = 0
        patience_counter = 0
        epochs_without_improvement = 0
        
        for epoch in tqdm(range(self.config['num_epochs']), desc="Training R-GCN"):
            # Training
            train_loss = self.train_epoch()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Evaluation
            if epoch % self.config['eval_every'] == 0:
                val_auc, val_ap = self.evaluate('val')
                train_auc, _ = self.evaluate('train')  # Monitor training performance
                
                # Calculate overfitting gap
                overfitting_gap = train_auc - val_auc
                
                print(f"Epoch {epoch:3d}: Loss={train_loss:.4f}, Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}, Gap={overfitting_gap:.4f}, LR={current_lr:.6f}")
                
                # Check for improvement
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    self.best_epoch = epoch
                    patience_counter = 0
                    epochs_without_improvement = 0
                    
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_rgcn_model.pt')
                    
                    # Save checkpoint if enabled
                    if self.config.get('save_checkpoints', False):
                        checkpoint = {
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'val_auc': val_auc,
                            'train_auc': train_auc,
                            'config': self.config
                        }
                        torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')
                        
                    print(f"  ✓ New best model saved! (AUC: {val_auc:.4f})")
                else:
                    patience_counter += 1
                    epochs_without_improvement += 1
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    self.scheduler.step(val_auc)
                    new_lr = self.optimizer.param_groups[0]['lr']
                    if new_lr != old_lr:
                        print(f"  📉 Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
                
                # Early stopping check
                early_stopping_patience = self.config.get('early_stopping_patience', float('inf'))
                if patience_counter >= early_stopping_patience:
                    print(f"\\n🛑 Early stopping triggered after {early_stopping_patience} epochs without improvement")
                    print(f"Best validation AUC: {best_val_auc:.4f} at epoch {self.best_epoch}")
                    break
                
                # Overfitting warning
                if overfitting_gap > 0.15:  # 15% gap indicates potential overfitting
                    print(f"  ⚠️  Large train-val gap detected: {overfitting_gap:.4f}")
                    
                if overfitting_gap > 0.25:  # 25% gap indicates severe overfitting
                    print(f"  🚨 Severe overfitting detected! Consider:")
                    print(f"     - Increasing dropout (current: {self.model.dropout})")
                    print(f"     - Increasing weight decay (current: {self.config['weight_decay']})")
                    print(f"     - Reducing model complexity")
        
        # Final test evaluation
        print(f"\\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)
        
        # Load best model
        self.model.load_state_dict(torch.load('best_rgcn_model.pt'))
        
        # Evaluate on all splits
        train_auc, train_ap = self.evaluate('train')
        val_auc, val_ap = self.evaluate('val')
        test_auc, test_ap = self.evaluate('test')
        
        final_overfitting_gap = train_auc - test_auc
        
        print(f"Train AUC: {train_auc:.4f}, AP: {train_ap:.4f}")
        print(f"Val AUC:   {val_auc:.4f}, AP: {val_ap:.4f}")
        print(f"Test AUC:  {test_auc:.4f}, AP: {test_ap:.4f}")
        print(f"\\nOverfitting Analysis:")
        print(f"  - Train-Test Gap: {final_overfitting_gap:.4f}")
        
        if final_overfitting_gap < 0.05:
            print(f"  ✅ Excellent generalization (gap < 5%)")
        elif final_overfitting_gap < 0.10:
            print(f"  ✅ Good generalization (gap < 10%)")
        elif final_overfitting_gap < 0.15:
            print(f"  ⚠️  Mild overfitting (gap < 15%)")
        else:
            print(f"  🚨 Significant overfitting (gap > 15%)")
        
        return {
            'best_val_auc': best_val_auc,
            'test_auc': test_auc,
            'test_ap': test_ap,
            'train_auc': train_auc,
            'final_overfitting_gap': final_overfitting_gap,
            'best_epoch': self.best_epoch,
            'train_losses': self.train_losses,
            'val_aucs': self.val_aucs,
            'val_aps': self.val_aps,
            'train_aucs': self.train_aucs,
            'learning_rates': self.learning_rates
        }

def main():
    """Test the proper R-GCN implementation"""
    
    print("=" * 80)
    print("PROPER R-GCN FOR HETEROGENEOUS KNOWLEDGE GRAPHS")
    print("=" * 80)
    
    # Configuration - Scaled up for full NKB representation with overfitting prevention
    config = {
        'num_epochs': 200,        # More epochs but with early stopping
        'hidden_dim': 128,        # Larger embeddings for richer representations
        'num_layers': 4,          # Deeper network for complex relationships
        'dropout': 0.4,           # Higher dropout for regularization
        'learning_rate': 0.005,   # Lower LR for stability with larger model
        'weight_decay': 1e-3,     # Stronger L2 regularization
        'num_edge_types': 25,     # More edge types for full representation
        'min_edges_per_type': 20, # Lower threshold to include more relations
        'eval_every': 5,          # Evaluate more frequently for early stopping
        'extract_embeddings': True,  # Enable embedding extraction
        
        # Overfitting prevention
        'early_stopping_patience': 20,  # Stop if no improvement for 20 evals
        'lr_scheduler': True,           # Learning rate decay
        'lr_decay_factor': 0.8,         # Decay factor
        'lr_decay_patience': 10,        # Decay LR if no improvement
        'gradient_clipping': 1.0,       # Clip gradients to prevent explosion
        'test_split': 0.15,            # Larger test set
        'val_split': 0.15,             # Larger validation set
        'train_split': 0.7,            # Smaller training set
        'monitor_train_val_gap': True,  # Monitor train/val performance gap
        'save_checkpoints': True       # Save model checkpoints
    }
    
    print(f"Configuration: {config}")
    print()
    
    # Load data
    print("Loading heterogeneous data...")
    try:
        data = torch.load('improved_hetero_data.pt', weights_only=False)
    except:
        from torch_geometric.data.storage import BaseStorage
        torch.serialization.add_safe_globals([BaseStorage])
        data = torch.load('improved_hetero_data.pt')
    
    print(f"Data loaded: {len(data.node_types)} node types, {len(data.edge_types)} edge types")
    
    # Initialize R-GCN model
    model = ProperRGCN(
        hetero_data=data,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    # Initialize trainer
    trainer = RGCNTrainer(model, data, config)
    trainer.create_splits()
    
    # Train
    results = trainer.train()
    
    print("\\n" + "=" * 80)
    print("R-GCN TRAINING COMPLETED!")
    print("=" * 80)
    print(f"Best Validation AUC: {results['best_val_auc']:.4f}")
    print("Model saved as: best_rgcn_model.pt")
    
    # Extract and save embeddings
    if config['extract_embeddings']:
        print("\\n" + "=" * 50)
        print("EXTRACTING LEARNED EMBEDDINGS")
        print("=" * 50)
        
        # Load best model
        model.load_state_dict(torch.load('best_rgcn_model.pt'))
        model.eval()
        
        # Extract embeddings for all node types
        with torch.no_grad():
            embeddings_dict = model.forward(return_embeddings=True)
        
        # Save embeddings
        embedding_stats = {}
        for node_type, embeddings in embeddings_dict.items():
            embedding_stats[node_type] = {
                'shape': embeddings.shape,
                'mean_norm': float(torch.norm(embeddings, dim=1).mean()),
                'std_norm': float(torch.norm(embeddings, dim=1).std())
            }
            print(f"  - {node_type}: {embeddings.shape} embeddings, avg norm: {embedding_stats[node_type]['mean_norm']:.3f}")
        
        # Save embeddings to file
        torch.save(embeddings_dict, 'nkb_rgcn_embeddings.pt')
        
        # Save embedding statistics
        with open('embedding_stats.json', 'w') as f:
            json.dump(embedding_stats, f, indent=2)
        
        print(f"\\n✓ Embeddings saved to: nkb_rgcn_embeddings.pt")
        print(f"✓ Statistics saved to: embedding_stats.json")
        
        # Analyze embedding quality
        print("\\n" + "=" * 50)
        print("EMBEDDING ANALYSIS")
        print("=" * 50)
        
        # Calculate embedding similarities within each node type
        for node_type in ['material', 'assay', 'result']:  # Focus on key entity types
            if node_type in embeddings_dict:
                emb = embeddings_dict[node_type]
                if emb.size(0) > 1:
                    # Compute pairwise cosine similarities
                    emb_norm = F.normalize(emb, p=2, dim=1)
                    similarities = torch.mm(emb_norm, emb_norm.t())
                    
                    # Remove diagonal (self-similarities)
                    mask = torch.eye(similarities.size(0), dtype=torch.bool)
                    off_diagonal = similarities[~mask]
                    
                    avg_sim = float(off_diagonal.mean())
                    std_sim = float(off_diagonal.std())
                    
                    print(f"  - {node_type}: avg similarity = {avg_sim:.3f} ± {std_sim:.3f}")
        
        # Cross-type similarity analysis
        print("\\nCross-type similarities:")
        key_types = ['material', 'assay', 'result', 'parameters']
        available_types = [t for t in key_types if t in embeddings_dict]
        
        for i, type1 in enumerate(available_types):
            for type2 in available_types[i+1:]:
                emb1 = F.normalize(embeddings_dict[type1], p=2, dim=1)
                emb2 = F.normalize(embeddings_dict[type2], p=2, dim=1)
                
                # Sample embeddings if too many
                if emb1.size(0) > 100:
                    idx1 = torch.randperm(emb1.size(0))[:100]
                    emb1 = emb1[idx1]
                if emb2.size(0) > 100:
                    idx2 = torch.randperm(emb2.size(0))[:100]
                    emb2 = emb2[idx2]
                
                cross_sim = torch.mm(emb1, emb2.t()).mean()
                print(f"  - {type1} ↔ {type2}: {float(cross_sim):.3f}")
        
        print("\\n✓ Embedding analysis completed!")
        
        # Create embedding visualization data
        print("\\nPreparing embedding visualization data...")
        
        # Sample embeddings for visualization
        vis_data = {}
        for node_type, embeddings in embeddings_dict.items():
            if embeddings.size(0) > 0:
                # Sample up to 500 nodes per type for visualization
                n_sample = min(500, embeddings.size(0))
                if n_sample < embeddings.size(0):
                    indices = torch.randperm(embeddings.size(0))[:n_sample]
                    sampled_emb = embeddings[indices]
                else:
                    sampled_emb = embeddings
                
                vis_data[node_type] = {
                    'embeddings': sampled_emb.cpu().numpy(),
                    'node_type': node_type,
                    'count': n_sample
                }
        
        # Save visualization data
        with open('embedding_visualization_data.pkl', 'wb') as f:
            pickle.dump(vis_data, f)
        
        print(f"✓ Visualization data saved to: embedding_visualization_data.pkl")
        print("  Use this for t-SNE, UMAP, or PCA visualization")
    
    print("\\n" + "=" * 80)
    print("FULL NKB R-GCN ANALYSIS COMPLETED!")
    print("=" * 80)
    print("Files created:")
    print("  - best_rgcn_model.pt: Trained R-GCN model")
    print("  - nkb_rgcn_embeddings.pt: Node embeddings for all types")
    print("  - embedding_stats.json: Embedding statistics")
    print("  - embedding_visualization_data.pkl: Data for visualization")
    print(f"\\nModel Performance:")
    print(f"  - Best Validation AUC: {results['best_val_auc']:.4f}")
    print(f"  - Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Hidden Dimensions: {config['hidden_dim']}")
    print(f"  - Network Depth: {config['num_layers']} layers")
    print(f"  - Relations Modeled: {model.num_relations}")

if __name__ == "__main__":
    main()
