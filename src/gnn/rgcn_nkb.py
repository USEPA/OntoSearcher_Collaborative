#!/usr/bin/env python3
"""
Optimized RGCN Training - Load data once, train efficiently
Modified for NKB dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging

from pykeen.triples import TriplesFactory
from torch_geometric.nn import RGCNConv, Linear
import rdflib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeepRGCN(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=120, 
                 num_layers=2, dropout=0.3, regularizer_weight=0.01):
        super(DeepRGCN, self).__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Entity embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        
        # RGCN layers
        self.rgcn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.rgcn_layers.append(
                RGCNConv(embedding_dim, embedding_dim, num_relations, num_blocks=4)
            )
        
        # Relation embeddings
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(num_layers)
        ])
        
        # Scoring layers
        self.score_head = Linear(embedding_dim, embedding_dim)
        self.score_tail = Linear(embedding_dim, embedding_dim)
        
        self.regularizer_weight = regularizer_weight
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
    
    def forward(self, edge_index, edge_type):
        x = self.entity_embeddings.weight
        
        for i, (rgcn_layer, layer_norm) in enumerate(zip(self.rgcn_layers, self.layer_norms)):
            x_new = rgcn_layer(x, edge_index, edge_type)
            if i > 0:
                x_new = x_new + x  # Residual
            x = layer_norm(x_new)
            x = F.relu(x)
            x = self.dropout(x)
        
        return x
    
    def score_hrt(self, heads, relations, tails, entity_embeddings):
        h_emb = entity_embeddings[heads]
        r_emb = self.relation_embeddings(relations)
        t_emb = entity_embeddings[tails]
        
        h_transformed = self.score_head(h_emb)
        t_transformed = self.score_tail(t_emb)
        
        scores = torch.sum(h_transformed * r_emb * t_transformed, dim=1)
        return scores
    
    def compute_loss(self, positive_scores, negative_scores):
        positive_loss = -F.logsigmoid(positive_scores).mean()
        negative_loss = -F.logsigmoid(-negative_scores).mean()
        
        loss = positive_loss + negative_loss
        
        reg_loss = (
            torch.norm(self.entity_embeddings.weight) + 
            torch.norm(self.relation_embeddings.weight)
        ) * self.regularizer_weight
        
        return loss + reg_loss

def load_nkb_data_once():
    """Load NKB data ONCE from pre-extracted triples factory"""
    logger.info("=== LOADING NKB DATA (ONE TIME ONLY) ===")
    
    # Load pre-extracted TriplesFactory
    tf_file = "nkb_triples_factory.pt"
    
    if not os.path.exists(tf_file):
        logger.error(f"TriplesFactory file not found: {tf_file}")
        logger.info("Available files in current directory:")
        for item in os.listdir("."):
            logger.info(f"  {item}")
        raise FileNotFoundError(f"Could not find {tf_file}")
    
    logger.info(f"Loading TriplesFactory from: {tf_file}")
    tf = torch.load(tf_file)
    logger.info(f"Loaded TriplesFactory successfully")
    
    # Split the dataset
    logger.info("Splitting dataset into train/test/validation...")
    training, testing, validation = tf.split([0.7, 0.2, 0.1], random_state=42)
    
    # Build graph
    logger.info("Building PyTorch Geometric graph...")
    train_triples = training.mapped_triples
    heads = train_triples[:, 0]
    relations = train_triples[:, 1] 
    tails = train_triples[:, 2]
    
    edge_index = torch.stack([
        torch.cat([heads, tails]),
        torch.cat([tails, heads])
    ])
    edge_type = torch.cat([relations, relations])
    
    logger.info(f"Data loading complete!")
    logger.info(f"Training: {len(train_triples):,} triples")
    logger.info(f"Entities: {training.num_entities:,}")
    logger.info(f"Relations: {training.num_relations}")
    logger.info(f"Graph edges: {len(edge_index[0]):,}")
    
    return training, testing, validation, edge_index, edge_type

def train_optimized_rgcn():
    """Train RGCN with data loaded once"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")
    
    # Memory optimization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # LOAD DATA ONCE
    training, testing, validation, edge_index, edge_type = load_nkb_data_once()
    
    # Move to GPU
    logger.info("Moving graph to GPU...")
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
    train_triples = training.mapped_triples.to(device)
    
    # Create model
    logger.info("Creating RGCN model...")
    model = DeepRGCN(
        num_entities=training.num_entities,
        num_relations=training.num_relations,
        embedding_dim=120,
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # Training parameters
    num_epochs = 10  # Reduced from 50 - model is learning fast enough
    batch_size = 512  # Increased from 256 for faster training
    batch_size = 256
    
    logger.info("=== STARTING OPTIMIZED RGCN TRAINING FOR NKB ===")
    start_time = time.time()
    
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_losses = []
        
        # Shuffle training data
        perm = torch.randperm(len(train_triples))
        train_triples_shuffled = train_triples[perm]
        
        num_batches = (len(train_triples_shuffled) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(train_triples_shuffled))
            batch_triples = train_triples_shuffled[start_idx:end_idx]
            
            heads, relations, tails = batch_triples[:, 0], batch_triples[:, 1], batch_triples[:, 2]
            
            # RGCN forward pass (this is fast - 0.4 seconds)
            entity_embeddings = model(edge_index, edge_type)
            
            # Scoring
            pos_scores = model.score_hrt(heads, relations, tails, entity_embeddings)
            
            # Negative sampling
            batch_size_actual = len(batch_triples)
            neg_heads = torch.randint(0, training.num_entities, (batch_size_actual,), device=device)
            neg_tails = torch.randint(0, training.num_entities, (batch_size_actual,), device=device)
            
            use_head_corruption = torch.rand(batch_size_actual, device=device) < 0.5
            neg_batch_heads = torch.where(use_head_corruption, neg_heads, heads)
            neg_batch_tails = torch.where(use_head_corruption, tails, neg_tails)
            
            neg_scores = model.score_hrt(neg_batch_heads, relations, neg_batch_tails, entity_embeddings)
            
            # Loss and backprop
            loss = model.compute_loss(pos_scores, neg_scores)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Progress within epoch
            if batch_idx % 500 == 0:
                logger.info(f"  Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.6f}")
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch:3d}/{num_epochs}: Loss={avg_loss:.6f}, Time={epoch_time:.1f}s")
        
        # Save checkpoint every epoch (safety save)
        if epoch % 1 == 0:  # Save every epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'losses': losses,
                'entity_to_id': training.entity_to_id,
                'relation_to_id': training.relation_to_id
            }
            os.makedirs("rgcn_nkb_checkpoints", exist_ok=True)
            torch.save(checkpoint, f"rgcn_nkb_checkpoints/rgcn_nkb_checkpoint_epoch_{epoch}.pt")
            logger.info(f"✓ Checkpoint saved: rgcn_nkb_checkpoints/rgcn_nkb_checkpoint_epoch_{epoch}.pt")
            
            # Also save current embeddings
            model.eval()
            with torch.no_grad():
                current_embeddings = model(edge_index, edge_type)
            torch.save(current_embeddings.cpu(), f"rgcn_nkb_checkpoints/rgcn_nkb_embeddings_epoch_{epoch}.pt")
            model.train()
            
            # Keep only last 3 checkpoints to save disk space
            if epoch >= 3:
                old_checkpoint = f"rgcn_nkb_checkpoints/rgcn_nkb_checkpoint_epoch_{epoch-3}.pt"
                old_embeddings = f"rgcn_nkb_checkpoints/rgcn_nkb_embeddings_epoch_{epoch-3}.pt"
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
                if os.path.exists(old_embeddings):
                    os.remove(old_embeddings)
        
        # Simple validation every 10 epochs
        if epoch % 10 == 0 and epoch > 0:
            scheduler.step(avg_loss)
    
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    training_time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    
    logger.info(f"Training completed in: {training_time_str}")
    
    # Save results
    os.makedirs("results_nkb", exist_ok=True)
    
    # Final embeddings
    model.eval()
    with torch.no_grad():
        final_entity_embeddings = model(edge_index, edge_type)
    
    # Save everything
    torch.save(model.state_dict(), "results_nkb/rgcn_nkb_model.pt")
    torch.save(final_entity_embeddings.cpu(), "results_nkb/rgcn_nkb_entity_embeddings.pt")
    torch.save(model.relation_embeddings.weight.detach().cpu(), "results_nkb/rgcn_nkb_relation_embeddings.pt")
    torch.save(training.entity_to_id, "results_nkb/rgcn_nkb_entity_to_id.pt")
    torch.save(training.relation_to_id, "results_nkb/rgcn_nkb_relation_to_id.pt")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('RGCN NKB Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('results_nkb/rgcn_nkb_training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save summary
    with open("results_nkb/rgcn_nkb_training_summary.txt", 'w') as f:
        f.write("RGCN NKB Training Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Training Time: {training_time_str}\n")
        f.write(f"Model: RGCN NKB (2-layer)\n")
        f.write(f"Embedding Dimension: 120\n")
        f.write(f"Entities: {training.num_entities:,}\n")
        f.write(f"Relations: {training.num_relations}\n")
        f.write(f"Training Triples: {len(training.mapped_triples):,}\n")
        f.write(f"Final Loss: {losses[-1]:.6f}\n")
        f.write(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    logger.info("=== TRAINING COMPLETE ===")
    logger.info("Results saved in results_nkb/ directory")

def main():
    try:
        train_optimized_rgcn()
        print("SUCCESS: Optimized RGCN training for NKB completed!")
    except Exception as e:
        print(f"ERROR: Training failed - {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
    
    
    
