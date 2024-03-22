import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from torch_geometric.utils import from_networkx 
from torch.nn import Linear
import networkx as nx
import pandas as pd
import os
from visualize_graph import plot_training_validation, plot_test_accuracy, plot_confusion_matrix, plot_prediction_distribution
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(666)
np.random.seed(666)


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
    numerical_attributes = [attr for attr in all_attributes if attr not in ignore_attributes and attr != 'label_type']
    numerical_attributes.sort()  # Sort attributes to ensure consistent order

    node_features = [[node_data.get(attr, 0.0) for attr in numerical_attributes] for _, node_data in G.nodes(data=True)]
    labels = [node_data.get('label_type', 0) for _, node_data in G.nodes(data=True)]

    features_tensor = torch.tensor(node_features, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    data = from_networkx(G)
    data.x = features_tensor
    data.y = labels_tensor
    

    return data

# 3. Model Definitions
class GATModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=1):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1)  # Reduce output to output_dim

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)  # Dropout layer added
        x = self.conv2(x, edge_index)
        return x

# 4. Training and Evaluation Functions
def train(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0
    total_samples = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        # Add L2 regularization (weight decay)
        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss += 0.001 * l2_reg  # Hyperparameter lambda = 0.001
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.y.size(0)
        total_samples += data.y.size(0)
    return total_loss / total_samples

def evaluate(model, criterion, loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)  # Predicted labels
            total_correct += (pred == data.y).sum().item()
            total_samples += data.y.size(0)  # Add the number of nodes in this batch to the total
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.y.size(0)
    accuracy = total_correct / total_samples  # Calculate the accuracy
    avg_loss = total_loss / total_samples
    return accuracy, avg_loss


# 6. Main Execution Block
if __name__ == "__main__":
    # Load and preprocess the data
    filepaths = [
        r'C:\Users\ge26rix\Desktop\Python\3.InputML\output_graph_1.graphml',
        r'C:\Users\ge26rix\Desktop\Python\3.InputML\output_graph_2.graphml',
        r'C:\Users\ge26rix\Desktop\Python\3.InputML\output_graph_3.graphml',
        r'C:\Users\ge26rix\Desktop\Python\3.InputML\output_graph_4.graphml',
        r'C:\Users\ge26rix\Desktop\Python\3.InputML\output_graph_5.graphml',
        r'C:\Users\ge26rix\Desktop\Python\3.InputML\output_graph_6.graphml',
        r'C:\Users\ge26rix\Desktop\Python\3.InputML\output_graph_7.graphml',
        r'C:\Users\ge26rix\Desktop\Python\3.InputML\output_graph_8.graphml',

        r'C:\Users\ge26rix\Desktop\Python\3.InputML\output_graph_9.graphml',
        r'C:\Users\ge26rix\Desktop\Python\3.InputML\output_graph_10.graphml',
        r'C:\Users\ge26rix\Desktop\Python\3.InputML\output_graph_11.graphml',
        

        # Add paths for the rest of the files here
    ]
    
        # Load graphs
    all_data = [parse_graphml_to_pyg(filepath) for filepath in filepaths]

    # Shuffle the data indices
    num_graphs = len(all_data)
    indices = list(range(num_graphs))
    np.random.shuffle(indices)

    # Split the indices into training, validation, and testing sets
    train_indices = indices[:int(0.6 * num_graphs)]
    val_indices = indices[int(0.6 * num_graphs):int(0.8 * num_graphs)]
    test_indices = indices[int(0.8 * num_graphs):]

    # Create DataLoader for the entire dataset
    all_loader = DataLoader(all_data, batch_size=1, shuffle=True)

    # Create DataLoader for training, validation, and testing sets
    train_loader = DataLoader([all_data[i] for i in train_indices], batch_size=1, shuffle=True)
    val_loader = DataLoader([all_data[i] for i in val_indices], batch_size=1, shuffle=False)
    test_loader = DataLoader([all_data[i] for i in test_indices], batch_size=1, shuffle=False)

    # Determine input and output dimensions based on your data
    input_dim = len(all_data[0].x[0])  # Assuming the input features have the same dimension for all nodes
    output_dim = 4  # 4 different label types

    # Initialize model, optimizer, and criterion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GATModel(input_dim=input_dim, hidden_dim=64, output_dim=output_dim, heads=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    val_accuracies = []
    final_epoch = 0 
    for epoch in range(1, 300):
        # Train
        model.train()
        epoch_train_loss = 0  # Variable to accumulate loss for the current epoch
        num_batches = 0  # Variable to count the number of batches processed in the current epoch
        for iteration, data in enumerate(train_loader, 1):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out, data.y)  
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * data.num_graphs  # Update total loss
            num_batches += 1
        
        # Calculate the average training loss for the current epoch
        avg_epoch_train_loss  = epoch_train_loss  / len(train_loader.dataset)  # Divide by total number of graphs
        train_losses.append(avg_epoch_train_loss)  # Append the average loss to the list

        
         # Validate
        val_accuracy, epoch_val_loss = evaluate(model, criterion, val_loader, device)
        val_accuracies.append(val_accuracy)  # Append validation accuracy to the list
        val_losses.append(epoch_val_loss)

        # Print training loss and validation accuracy for each epoch
        print(f"Epoch {epoch}, Average Training Loss: {avg_epoch_train_loss}, Validation Accuracy: {val_accuracy}, Validation Loss: {epoch_val_loss}")
        output_folder = r'C:\Users\ge26rix\Desktop\Python\5.OutputML_GAT'
        
        final_epoch = epoch

        training_validation_plot_path = os.path.join(output_folder, 'training_validation_plot.png')
        plot_training_validation(train_losses, val_losses, training_validation_plot_path,  title="Training and Validation Metrics for GAT Model")
    # Test
    test_accuracy = evaluate(model, criterion, test_loader, device)
    test_accuracy_value = test_accuracy[0]
    print(f"Final Test Accuracy: {test_accuracy_value}")
    
    # After training, make predictions on the testing set and save the predicted labels to CSV files
    output_folder = r'C:\Users\ge26rix\Desktop\Python\5.OutputML_GAT'
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
        
        prediction_distribution_output_path = os.path.join(output_folder, f'prediction_distribution_graph_{i+1}.png')
        plot_prediction_distribution(predictions, label_types, prediction_distribution_output_path, title='Prediction Distribution of Label Types for GAT Model')
        
        # Plot and save test accuracy
        test_accuracy_plot_path = os.path.join(output_folder, 'test_accuracy_plot.png')
        plot_test_accuracy(test_accuracy_value, final_epoch, test_accuracy_plot_path, title="Test Accuracy for GAT Model")
        # Calculate and plot confusion matrix
        y_true = []
        y_pred = []

        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index)
                pred = out.argmax(dim=1)
                y_true.extend(data.y.cpu().numpy())  
                y_pred.extend(pred.cpu().numpy())  

        # Convert true and predicted labels to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Define label names if available
        label_names = ['No Label', 'Wall with Dimension', 'Door Label', 'Zone Stamp']  

        # Plot confusion matrix
        confusion_matrix_plot_path = os.path.join(output_folder, 'confusion_matrix.png')
        plot_confusion_matrix(y_true, y_pred, ['No Label', 'Wall with Dimension', 'Door Label', 'Zone Stamp'], confusion_matrix_plot_path, title="Confusion Matrix for GAT Model")