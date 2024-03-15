# Automated BIM Model Annotation via Graph Neural Networks (GNN) - Python

This component of the project applies Graph Neural Networks (GNN) to train  Building Information Modeling (BIM) data. It processes input files to construct graph representations of BIM elements, which are then used to train and infer the GNN models.

## Description

The pipeline processes input files to generate node and edge data suitable for GNNs. 

## Graph Generation and Preprocessing

Graphs are generated from input files using the `parse_graphml_to_pyg` function, converting them into PyTorch Geometric Data format. This ensures proper structuring of node features and edge connections for GNN training.

## Model Architecture

Our GNN model, implemented in the `GCNModel` class, uses two layers of GCNConv, designed for processing graph-structured data. The model takes node features as input and produces annotations for BIM elements as output.

## Training Procedure

Training involves the following steps:

1. Splitting the data into training, validation, and testing sets.
2. Initializing the GCN model with specified input and output dimensions.
3. Training the model across several epochs and evaluating against a validation set.
4. Employing early stopping to mitigate overfitting.
5. Conducting final evaluations on a test set.

## Installation

Please ensure the following dependencies are installed:

- PyTorch
- PyTorch Geometric
- NetworkX
- Scikit-learn
- Pandas
- Matplotlib (for visualization)

Install the necessary Python packages using pip:

```sh
pip install torch torch-geometric networkx scikit-learn pandas matplotlib
