# OntoSearcher_Collaborative

A comprehensive framework for semantic knowledge graph analysis and link prediction on federal nanomaterial safety datasets. This project implements Relational Graph Convolutional Networks (R-GCN) for discovering novel relationships in RDF knowledge graphs from NIOSH, EPA, and CPSC nanomaterial databases.

## Overview

This repository provides tools for:
- **RDF Knowledge Graph Processing**: Convert RDF/TTL files to graph representations
- **Graph Neural Network Training**: R-GCN implementation for heterogeneous graphs
- **Link Prediction**: Discover novel relationships between nanomaterials, assays, and results
- **Embedding Analysis**: Generate and analyze learned node embeddings
- **Schema Visualization**: Generate publication-ready schema diagrams

## Repository Structure

```
OntoSearcher_Collaborative/
├── src/                          # Source code
│   ├── gnn/                      # Graph Neural Network implementations
│   │   ├── rgcn_nkb.py          # Optimized R-GCN training for NKB
│   │   ├── proper_rgcn_hetero.py # R-GCN with heterogeneous graph support
│   │   ├── pyg_hetero_rgcn.py   # PyTorch Geometric HeteroData R-GCN
│   │   ├── rgcn_link_predictor.py # Link prediction interface
│   │   ├── link_prediction_analysis.py # Advanced analysis tools
│   │   ├── rdf_to_rgcn_analysis.py # RDF to R-GCN pipeline
│   │   ├── query_specific_node.py # Node querying utilities
│   │   └── test_node_analyzer.py # Node attribute testing
│   └── converters/               # Data conversion utilities
│       ├── improved_rdf_hetero_converter.py # RDF to HeteroData
│       └── rdf_to_networkx_focused.py # RDF to NetworkX
├── notebooks/                    # Jupyter notebooks
│   ├── gnn/                      # GNN experimentation
│   │   ├── gnn_nkb.ipynb        # NKB GNN training
│   │   ├── gnn_pred_nkb.ipynb   # Prediction analysis
│   │   └── kg_embeddings.ipynb  # Embedding visualization
│   ├── schema/                   # Schema exploration
│   │   └── schema2.ipynb        # Schema analysis
│   ├── experiments/              # Experimental work
│   │   └── llmexperiment.ipynb  # LLM integration experiments
│   └── ontosearcher/             # OntoSearcher notebooks
│       ├── cpsc.ipynb           # CPSC data processing
│       └── test.ipynb           # Testing utilities
├── mappings/                     # RDF data files
│   ├── NKB_RDF_V3.ttl           # NIOSH Knowledge Base RDF
│   ├── cpsc_database.ttl        # CPSC nanomaterial database
│   └── niosh_rdf_tutV2c.ttl     # NIOSH tutorial RDF
├── schemas/                      # Generated schema outputs
│   ├── nkb/                     # NKB schema files
│   ├── cpsc/                    # CPSC schema files
│   └── niosh/                   # NIOSH schema files
├── results/                      # Prediction results
├── docs/                         # Documentation
└── lib/                          # Frontend visualization libraries
```

## Key Components

### 1. R-GCN Link Prediction Pipeline

The core of this project is an R-GCN implementation for link prediction on heterogeneous knowledge graphs.

**Training (`src/gnn/rgcn_nkb.py`)**:
- 2-layer R-GCN with 120-dimensional embeddings
- Margin-based loss with negative sampling
- Checkpoint saving and learning rate scheduling

```python
# Train R-GCN on NKB data
python src/gnn/rgcn_nkb.py
```

**Heterogeneous R-GCN (`src/gnn/proper_rgcn_hetero.py`)**:
- Proper handling of multiple node types (material, assay, result, etc.)
- Multiple relation types with separate weight matrices
- Early stopping and overfitting prevention

```python
# Train with comprehensive overfitting prevention
python src/gnn/proper_rgcn_hetero.py
```

### 2. Link Prediction Interface

The `RGCNLinkPredictor` class provides a high-level interface for predictions:

```python
from src.gnn.rgcn_link_predictor import RGCNLinkPredictor

predictor = RGCNLinkPredictor()

# Predict link probability
prob = predictor.predict_link_probability('material', 0, 'assay', 5)

# Find similar nodes
similar = predictor.find_similar_nodes('material', 0, top_k=10)

# Interactive exploration
predictor.interactive_search()
```

### 3. RDF Data Conversion

Convert RDF knowledge graphs to PyTorch Geometric HeteroData:

```python
from src.converters.improved_rdf_hetero_converter import ImprovedRDFToHeteroData

converter = ImprovedRDFToHeteroData(nx_graph)
hetero_data = converter.create_hetero_data_with_all_types()
```

**Node Types Supported**:
- Entity nodes: material, assay, result, parameters, additive, medium, publication
- Concept nodes: NCIT, NPO, OBO ontology terms
- Blank nodes: Property containers (node bags)

### 4. Advanced Analysis

Generate scientific hypotheses from link predictions:

```python
from src.gnn.link_prediction_analysis import AdvancedLinkAnalysis

analyzer = AdvancedLinkAnalysis(predictor)

# Analyze cross-type relationships
relationships = analyzer.analyze_cross_type_relationships()

# Generate hypotheses
hypotheses = analyzer.generate_scientific_hypotheses('material', min_confidence=0.85)

# Create HTML report
analyzer.create_prediction_report('link_prediction_report.html')
```

## Data Sources

### EPA Nanotechnology Knowledge Base (NKB)
- 1.3M+ RDF triples
- Entities: materials, assays, results, parameters, publications
- Ontology mappings to NCIT, NPO, OBO, ENM

### CPSC Nanomaterial Database
- Consumer product nanomaterial data
- Risk assessment and perception scores
- Product categorization

### /NIOSH Exposure Data
- Experimental exposure parameters
- Subject information (species, strain, cell types)
- Physical and chemical properties

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/OntoSearcher_Collaborative.git
cd OntoSearcher_Collaborative

# Install dependencies
pip install torch torch_geometric rdflib networkx pandas scikit-learn matplotlib seaborn pykeen tqdm

# For PyTorch Geometric, follow: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
```

## Quick Start

### 1. Convert RDF to Graph
```bash
python src/converters/improved_rdf_hetero_converter.py
```
Creates: `improved_hetero_data.pt`, `networkx_graph.pkl`

### 2. Train R-GCN Model
```bash
python src/gnn/proper_rgcn_hetero.py
```
Creates: `best_rgcn_model.pt`, `nkb_rgcn_embeddings.pt`

### 3. Run Link Prediction
```bash
python src/gnn/rgcn_link_predictor.py
```
Interactive mode for exploring predictions.

### 4. Generate Analysis Report
```bash
python src/gnn/link_prediction_analysis.py
```
Creates: `link_prediction_report.html`

## Model Performance

| Metric | Value |
|--------|-------|
| Validation AUC | 99.0% |
| Total Parameters | 6M+ |
| Node Types | 10+ |
| Edge Types | 25+ |
| Total Nodes | 290K+ |

## Output Files

| File | Description |
|------|-------------|
| `best_rgcn_model.pt` | Trained R-GCN model weights |
| `nkb_rgcn_embeddings.pt` | Node embeddings for all types |
| `improved_hetero_data.pt` | PyTorch Geometric HeteroData |
| `networkx_graph.pkl` | NetworkX graph representation |
| `embedding_stats.json` | Embedding statistics |
| `link_prediction_report.html` | Comprehensive analysis report |

## Key Features

- **Heterogeneous Graph Support**: Handles multiple node and edge types
- **Ontology-Aware**: Preserves semantic relationships from NCIT, NPO, OBO
- **Overfitting Prevention**: Early stopping, dropout, gradient clipping
- **Interactive Exploration**: Command-line interface for predictions
- **Scientific Hypothesis Generation**: Automated interpretation of findings

## License

This project is part of the National Nanotechnology Coordination Office (NNCo) research efforts.

## Authors

**Pranav Singh, Dr. Holly Mortensen**

## Acknowledgments

- NIOSH Nanotechnology Research Center
- EPA Office of Research and Development
- Consumer Product Safety Commission (CPSC)

