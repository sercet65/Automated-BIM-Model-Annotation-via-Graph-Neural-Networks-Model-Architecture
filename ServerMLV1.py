import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx 
from torch.nn import Linear
from sklearn.model_selection import train_test_split
import networkx as nx
import pandas as pd
import os
from visualize_graph import plot_training_validation, plot_test_accuracy
# 2. GraphML Parsing Function
def parse_graphml_to_pyg(filepath):
    G = nx.read_graphml(filepath)
    
    # Step 1: Identify all unique attributes across all nodes
    all_attributes = set()
    for _, node_data in G.nodes(data=True):
        all_attributes.update(node_data.keys())
    
    # Optional: Specify attributes that you know are categorical or not useful for features
    ignore_attributes = {'guid', 'element_type', 'info_string', 'label_guid', 'marker_guid', 'embedded_door_guid', 'embedded_in_wall_guid', 'zone_stamp_guid', 'room_name', 'room_number'}

    # Ensure that each node has all attributes, setting missing ones to a default value
    for _, node_data in G.nodes(data=True):
        for attr in all_attributes:
            if attr not in node_data:
                # Set default value based on attribute type (0 for numerical, "none" or similar for categorical)
                node_data[attr] = 0.0 if attr not in ignore_attributes else "none"
            else:
                # Attempt to convert numerical attributes to float
                if attr not in ignore_attributes:
                    try:
                        node_data[attr] = float(node_data[attr])
                    except ValueError:
                        node_data[attr] = 0.0  # Default for non-convertible values
    
    # Filter out ignored attributes and convert graph to PyTorch Geometric Data
    numerical_attributes = [attr for attr in all_attributes if attr not in ignore_attributes]
    numerical_attributes.sort()  # Sort attributes to ensure consistent order

    node_features = [[node_data.get(attr, 0.0) for attr in numerical_attributes] for _, node_data in G.nodes(data=True)]
    labels = [node_data.get('label_type', 0) for _, node_data in G.nodes(data=True)]

    features_tensor = torch.tensor(node_features, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    data = from_networkx(G)
    data.x = features_tensor
    data.y = labels_tensor
    
    # Add train_mask, val_mask, and test_mask attributes
    num_nodes = len(data.y)
    train_indices = torch.randperm(num_nodes) < 0.6 * num_nodes  # 60% for training
    val_indices = (torch.randperm(num_nodes) >= 0.6 * num_nodes) & (torch.randperm(num_nodes) < 0.8 * num_nodes)  # 20% for validation
    test_indices = ~train_indices & ~val_indices  # Remaining for testing
    data.train_mask = train_indices
    data.val_mask = val_indices
    data.test_mask = test_indices
    return data

# 3. Model Definitions
class GCNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# 4. Training and Evaluation Functions
def train(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, criterion, test_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            total_correct += (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
            total_samples += data.test_mask.sum().item()
    return total_correct / total_samples


# 6. Main Execution Block
if __name__ == "__main__":
    # Load and preprocess the data
    filepaths = [
        r'C:\Users\serve\OneDrive\Desktop\Phython\3.InputML\output_graph_1.graphml',
        r'C:\Users\serve\OneDrive\Desktop\Phython\3.InputML\output_graph_2.graphml',
        r'C:\Users\serve\OneDrive\Desktop\Phython\3.InputML\output_graph_3.graphml',
        r'C:\Users\serve\OneDrive\Desktop\Phython\3.InputML\output_graph_4.graphml',
        r'C:\Users\serve\OneDrive\Desktop\Phython\3.InputML\output_graph_5.graphml',
        r'C:\Users\serve\OneDrive\Desktop\Phython\3.InputML\output_graph_6.graphml',
        r'C:\Users\serve\OneDrive\Desktop\Phython\3.InputML\output_graph_7.graphml',

        r'C:\Users\serve\OneDrive\Desktop\Phython\3.InputML\output_graph_10.graphml',
        r'C:\Users\serve\OneDrive\Desktop\Phython\3.InputML\output_graph_11.graphml',
        r'C:\Users\serve\OneDrive\Desktop\Phython\3.InputML\output_graph_12.graphml',
      
        

        # Add paths for the rest of the files here
    ]
    
    # Load graphs
    all_data = [parse_graphml_to_pyg(filepath) for filepath in filepaths]

    # Use the first 2 for training, next 2 for validation, and last for testing (if needed)
    train_loader = DataLoader(all_data[:6], batch_size=1, shuffle=True)
    val_loader = DataLoader(all_data[6:8], batch_size=1, shuffle=True)
    test_loader = DataLoader(all_data[8:10], batch_size=1, shuffle=True)

    # Determine input and output dimensions based on your data
    input_dim = len(all_data[0].x[0])  # Assuming the input features have the same dimension for all nodes
    output_dim = 4  # 4 different label types

    # Initialize model, optimizer, and criterion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNModel(input_dim=input_dim, hidden_dim=64, output_dim=output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    train_losses = []
    val_accuracies = []
    for epoch in range(1, 100):
        # Train
        model.train()
        epoch_loss = 0  # Variable to accumulate loss for the current epoch
        num_batches = 0  # Variable to count the number of batches processed in the current epoch
        for iteration, data in enumerate(train_loader, 1):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()  # Accumulate loss for the current batch
            num_batches += 1
        
        # Calculate the average training loss for the current epoch
        avg_epoch_loss = epoch_loss / num_batches
        train_losses.append(avg_epoch_loss)  # Append the average loss to the list

        
         # Validate
        val_accuracy = evaluate(model, criterion, val_loader, device)
        val_accuracies.append(val_accuracy)  # Append validation accuracy to the list

        # Print training loss and validation accuracy for each epoch
        print(f"Epoch {epoch}, Average Training Loss: {avg_epoch_loss}, Validation Accuracy: {val_accuracy}")

        plot_training_validation(train_losses, val_accuracies, 'C:\\Users\\serve\\OneDrive\\Desktop\\Phython\\4.OutputML\\training_validation_plot.png')

    # Test
    test_accuracy = evaluate(model, criterion, test_loader, device)
    print(f"Final Test Accuracy: {test_accuracy}")
    
    # After training, make predictions on the testing set and save the predicted labels to CSV files
    output_folder = r'C:\Users\serve\OneDrive\Desktop\Phython\4.OutputML'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, data in enumerate(all_data):
    # Make predictions using the trained model
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation during inference
            data = data.to(device)  # Transfer data to the appropriate device (GPU or CPU)
            logits = model(data.x, data.edge_index)  # Use the trained model to generate logits
        predictions = logits.argmax(dim=1).cpu().numpy()

        # Extract node IDs
        label_types = data.y.cpu().numpy()  # label types are stored in 'y' attribute

    

            # Save predictions to CSV file
        predictions_output_path = os.path.join(output_folder, f'predictions_output_graph_{i+1}.csv')
        predictions_df = pd.DataFrame({'label_type': label_types, 'predicted_label': predictions})
        predictions_df.to_csv(predictions_output_path, index=False)
        
        
        # Plot and save test accuracy
        plot_test_accuracy(test_accuracy, 'C:\\Users\\serve\\OneDrive\\Desktop\\Phython\\4.OutputML\\test_accuracy_plot.png')