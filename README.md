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
│   ├── rag/                      # Nanotoxicology RAG (OpenAI or local LLM)
│   │   ├── llm_backends.py      # OpenAI + Transformers (Llama) backends
│   │   ├── nanotoxicology_rag.py # RAG logic (Neo4j + LLM)
│   │   └── cli.py               # CLI: ask, interactive
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

### 5. Nanotoxicology RAG CLI (local first: Ollama, or OpenAI)

The same RAG logic as `notebooks/experiments/llmexperiment.ipynb` is available as a CLI. **For local use (no cloud, no heavy Python model loading)**, the default is **Ollama**: data stays on your machine and you avoid transformers/torch/protobuf issues.

**Recommended local setup:**
1. Install [Ollama](https://ollama.com) and start it.
2. Pull a small model: `ollama pull tinyllama` (or `ollama pull llama2` for better quality if you have RAM).
3. From the repo root (with Neo4j running and the graph loaded):

```bash
# Single question (uses Ollama + tinyllama by default)
python -m src.rag.cli ask "What consumer products contain silver nanoparticles?"

# Interactive Q&A
python -m src.rag.cli interactive

# Use a different Ollama model
python -m src.rag.cli ask "What products contain silver?" --model llama2
```

**Requirements for local (Ollama)**: Only Neo4j + Ollama; no `OPENAI_API_KEY` and no `torch`/`transformers` in Python for the default path.

**Using OpenAI (sends data to the cloud):**
```bash
export OPENAI_API_KEY=your_key
python -m src.rag.cli ask "What products contain silver?" --backend openai
```

**Optional dependencies**: For `--backend transformers` (load a Hugging Face model in-process), install `pip install -r requirements-rag.txt`. Not required for Ollama.

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

**For the Nanotoxicology RAG CLI (local, no cloud):**
```bash
pip install neo4j
# Install Ollama from https://ollama.com and run: ollama pull tinyllama
```
No `torch` or `transformers` are required for the default RAG path (Ollama). For `--backend openai` add `pip install openai` and set `OPENAI_API_KEY`.

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

### 5. Run the Nanotoxicology RAG CLI (question answering over the knowledge graph)

RAG runs **locally by default** (Ollama); no API key and no large Python model loading.

1. **Install Ollama** from [ollama.com](https://ollama.com) and start it (Ollama runs in the background once installed).
2. **Pull a small model** (from any terminal):
   ```bash
   ollama pull tinyllama
   ```
   For better quality if you have enough RAM: `ollama pull llama2`
3. **Ensure Neo4j is running** with the nanotoxicology graph loaded (as used in the LLM experiment notebook).
4. **From the repo root**, run the CLI:
   ```bash
   # Single question (default: Ollama + tinyllama)
   python -m src.rag.cli ask "What consumer products contain silver nanoparticles?"

   # Interactive Q&A (type exit to quit)
   python -m src.rag.cli interactive

   # Use a different Ollama model
   python -m src.rag.cli ask "What products contain silver?" --model llama2
   ```

**Optional:** Use OpenAI instead (sends data to the cloud; requires `OPENAI_API_KEY`):
```bash
export OPENAI_API_KEY=your_key
python -m src.rag.cli ask "What products contain silver?" --backend openai
```

**Troubleshooting:** If you see "Ollama request failed", ensure Ollama is running and you have pulled a model (`ollama pull tinyllama`). For Neo4j connection errors, check `--neo4j-uri`, `--neo4j-user`, and `--neo4j-password` (or set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`).

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

