import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from torch_geometric.utils import from_networkx 
from torch.nn import Linear
import networkx as nx
import pandas as pd
import os
from visualize_graph import plot_training_validation, plot_test_accuracy, plot_confusion_matrix, plot_prediction_distribution, plot_validation_accuracy, plot_lr_schedule, plot_class_accuracies
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set random seed for reproducibility
torch.manual_seed(100)
np.random.seed(100)

# 2. GraphML Parsing Function
def parse_graphml_to_pyg(filepath):
    G = nx.read_graphml(filepath)
    
    # Optional: Specify attributes that you know are not useful for features
    ignore_attributes = {'guid', 'info_string', 'label_guid', 'marker_guid', 'embedded_door_guid', 'embedded_in_wall_guid', 'zone_stamp_guid'}

    # Step 1: Identify all unique attributes across all nodes
    all_attributes = set()
    for _, node_data in G.nodes(data=True):
        all_attributes.update(node_data.keys())
    
    
    # Ensure that each node has all attributes, setting missing ones to a default value
    for _, node_data in G.nodes(data=True):
        for attr in all_attributes:
            if attr in ignore_attributes:
                # If the attribute is in the ignore list, ensure it is not included in the node features
                if attr in node_data:
                    del node_data[attr]
            else:
                # Ensure every node has the attribute, with a default value if missing
                node_data.setdefault(attr, 0.0 if attr not in ignore_attributes else "none")
                # Attempt to convert numerical attributes to float
                if attr not in ignore_attributes:
                    try:
                        node_data[attr] = float(node_data[attr])
                    except ValueError:
                        node_data[attr] = 0.0  # Default for non-convertible values

    # Filter out ignored attributes and convert graph to PyTorch Geometric Data
    numerical_attributes = [attr for attr in all_attributes if attr not in ignore_attributes and attr != 'label_type']
    numerical_attributes.sort()  # Sort attributes to ensure consistent order

    # One-hot encode 'element_type' attribute
    element_types = set(node_data['element_type'] for _, node_data in G.nodes(data=True))
    element_type_dict = {element_type: idx for idx, element_type in enumerate(element_types)}
    for _, node_data in G.nodes(data=True):
        element_type = node_data['element_type']
        one_hot = [0] * len(element_types)
        one_hot[element_type_dict[element_type]] = 1
        # Flatten one-hot encoding to a single numerical value
        element_type_numerical = one_hot.index(1) if 1 in one_hot else 0
        node_data['element_type'] = element_type_numerical

        room_numbers = set(node_data['room_number'] for _, node_data in G.nodes(data=True))
        room_number_dict = {room_number: idx for idx, room_number in enumerate(room_numbers)}
        for _, node_data in G.nodes(data=True):
            room_number = node_data['room_number']
            one_hot_room_number = [0] * len(room_numbers)
            one_hot_room_number[room_number_dict[room_number]] = 1
            room_number_numerical = one_hot_room_number.index(1) if 1 in one_hot_room_number else 0
            node_data['room_number'] = room_number_numerical

        room_names = set(node_data['room_name'] for _, node_data in G.nodes(data=True))
        room_name_dict = {room_name: idx for idx, room_name in enumerate(room_names)}
        for _, node_data in G.nodes(data=True):
            room_name = node_data['room_name']
            one_hot_room_name = [0] * len(room_names)
            one_hot_room_name[room_name_dict[room_name]] = 1
            room_name_numerical = one_hot_room_name.index(1) if 1 in one_hot_room_name else 0
            node_data['room_name'] = room_name_numerical

    # Concatenate numerical attributes with 'element_type, room_number, room_name'
    node_features = [[node_data.get(attr, 0.0) for attr in numerical_attributes] + [node_data['element_type']]+[node_data['room_number']]+[node_data['room_name']] for _, node_data in G.nodes(data=True)]
    labels = [node_data['label_type'] for _, node_data in G.nodes(data=True) if 'label_type' in node_data]
   
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
    
     # Specify the directory where your input files are located
    input_directory_1 = r'C:\Users\serve\OneDrive\Desktop\Python\3.InputML\2.Sample Projects by floor'
    input_directory_2 = r'C:\Users\serve\OneDrive\Desktop\Python\3.InputML\3.Server Sample Projects'
    test_data_directory = r'C:\Users\serve\OneDrive\Desktop\Python\3.InputML\4.Test data'
    validation_data_directory = r'C:\Users\serve\OneDrive\Desktop\Python\3.InputML\5.Validation data'

    #C:\Users\serve\OneDrive\Desktop\Python\3.InputML\1.Sample Projects
    #C:\Users\serve\OneDrive\Desktop\Python\3.InputML\2.Sample Projects by floor
    #C:\Users\serve\OneDrive\Desktop\Python\3.InputML\3.Server Sample Projects
    # Get a list of all files in the input directory
    file_paths_1 = [os.path.join(input_directory_1, file) for file in os.listdir(input_directory_1) if file.endswith('.graphml')]
    file_paths_2 = [os.path.join(input_directory_2, file) for file in os.listdir(input_directory_2) if file.endswith('.graphml')]
    test_file_paths = [os.path.join(test_data_directory, file) for file in os.listdir(test_data_directory) if file.endswith('.graphml')]
    validation_file_paths = [os.path.join(validation_data_directory, file) for file in os.listdir(validation_data_directory) if file.endswith('.graphml')]

    # Split data into training, validation, and testing sets

    file_paths = file_paths_1 + file_paths_2
        # Load graphs
    all_data = [parse_graphml_to_pyg(filepath) for filepath in file_paths]
    test_data = [parse_graphml_to_pyg(filepath) for filepath in test_file_paths]
    validation_data = [parse_graphml_to_pyg(filepath) for filepath in validation_file_paths]
    print("Loading the following test files:")
    for file_path in test_file_paths:
        file_name = os.path.basename(file_path)  # Extracts the file name from the full path
        print(file_name)

    print("Loading the following validation files:")
    for file_path in validation_file_paths:
        file_name = os.path.basename(file_path)  # Extracts the file name from the full path
        print(file_name)
    
    
    # Define the number of files
    num_graphs = len(all_data)

    # Create shuffled indices
    indices = np.random.permutation(num_graphs)

    # Calculate the number of graphs for each split
    
    num_train = num_graphs

    # Generate shuffled indices and split them
    indices = np.random.permutation(num_graphs)
    train_indices = indices[:num_train]
    
    # Create DataLoader for the entire dataset
    all_loader = DataLoader(all_data, batch_size=16, shuffle=True)

    # Create DataLoader for training, validation, and testing sets
    train_loader = DataLoader([all_data[i] for i in train_indices], batch_size=16, shuffle=True)
    val_loader = DataLoader(validation_data, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False) 
    # Determine input and output dimensions based on your data
    input_dim = len(all_data[0].x[0])  # Assuming the input features have the same dimension for all nodes
    output_dim = 5  # 5 different label types

# Calculate class weights based on the frequency of each class in the training data
    class_weights = []
    for label_type in range(output_dim):
        class_count = sum([(data.y == label_type).any().item() for data in train_loader.dataset])
        if class_count == 0:
            class_weight = 0  # Assign a default value or handle the case accordingly
        else:
            class_weight = len(train_loader.dataset) / (output_dim * class_count)
        class_weights.append(class_weight)


    # Initialize model, optimizer, and criterion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GATModel(input_dim=input_dim, hidden_dim=256, output_dim=output_dim, heads=24).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float))
    
    # After initializing your optimizer
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=30, verbose=True)

    # Training loop 
    train_losses = []
    val_losses = []
    val_accuracies = []
    final_epoch = 0 
    learning_rates = [] 
    for epoch in range(1, 750):
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

        # Update learning rate scheduler at the end of each epoch
        scheduler.step(epoch_val_loss)
        
        learning_rates.append(optimizer.param_groups[0]['lr'])

        # Early stopping condition if necessary
        if scheduler.num_bad_epochs > scheduler.patience:
            print("Early stopping triggered.")
            break


        # Print training loss and validation accuracy for each epoch
        print(f"Epoch {epoch}, Average Training Loss: {avg_epoch_train_loss}, Validation Accuracy: {val_accuracy}, Validation Loss: {epoch_val_loss}")
        output_folder = r'C:\Users\serve\OneDrive\Desktop\Python\5.OutputML_GAT'
        
        final_epoch = epoch

        training_validation_plot_path = os.path.join(output_folder, 'training_validation_plot.png')
        plot_training_validation(train_losses, val_losses, training_validation_plot_path,  title="Training and Validation Metrics for GAT Model")

        validation_accuracy_plot_path = os.path.join(output_folder, 'validation_accuracy_plot.png')
        plot_validation_accuracy(val_accuracies, validation_accuracy_plot_path, title="Validation Accuracy for GAT Model")

        lr_schedule_plot_path = os.path.join(output_folder, 'learning_rate_schedule.png')
        plot_lr_schedule(learning_rates, lr_schedule_plot_path, title="Learning Rate Schedule for GAT Model")

    # Test
    test_accuracy = evaluate(model, criterion, test_loader, device)
    test_accuracy_value = test_accuracy[0]
    print(f"Final Test Accuracy: {test_accuracy_value}")
    
    # After training, make predictions on the testing set and save the predicted labels to CSV files
    output_folder = r'C:\Users\serve\OneDrive\Desktop\Python\5.OutputML_GAT'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

     
    # Loop through test files and process predictions
    for file_path, test_graph in zip(test_file_paths, test_data):
        # Retrieve the index of the test graph
        file_idx = int(os.path.basename(file_path).split('_')[2].split('.')[0])

        # Prepare DataLoader for the current test graph
        test_loader_single = DataLoader([test_graph], batch_size=16, shuffle=False)

        # Process predictions for the current test graph
        for i, data in enumerate(test_loader_single):
            # Retrieve the index of the test graph in all_data
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # Disable gradient computation during inference
                data = data.to(device)  # Transfer data to the appropriate device (GPU or CPU)
                logits = model(data.x, data.edge_index)  # Use the trained model to generate logits
            predictions = logits.argmax(dim=1).cpu().numpy()

            # Extract node IDs and graph indices for each node
            label_types = data.y.cpu().numpy()  # label types are stored in 'y' attribute
            graph_indices = data.batch.cpu().numpy()  # batch attribute contains the graph indices for each node

            # Save predictions to CSV file for the current test graph
            predictions_output_path = os.path.join(output_folder, f'predictions_output_graph_{file_idx}.csv')
            predictions_df = pd.DataFrame({'label_type': label_types, 'predicted_label': predictions})
            predictions_df.to_csv(predictions_output_path, index=False)

            # Generate prediction distribution plot for the current test graph
            prediction_distribution_output_path = os.path.join(output_folder, f'prediction_distribution_graph_{file_idx}.png')
            plot_prediction_distribution(predictions, label_types, prediction_distribution_output_path, title=f'Prediction Distribution of Label Types for GAT Model')
                        # Plot and save test accuracy
            test_accuracy_plot_path = os.path.join(output_folder, 'test_accuracy_plot.png')
            plot_test_accuracy(test_accuracy_value , final_epoch, test_accuracy_plot_path, title="Final Test Accuracy for GAT Model")


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
            label_names = ['No Label', 'Wall with Dimension', 'Connected Label', 'Door Marker', 'Zone Stamp']  


            # Plot confusion matrix
            confusion_matrix_plot_path = os.path.join(output_folder, 'confusion_matrix.png')
            plot_confusion_matrix(y_true, y_pred, ['No Label', 'Wall with Dimension', 'Connected Label', 'Door Marker', 'Zone Stamp'], confusion_matrix_plot_path, title="Confusion Matrix for GAT Model")

                    # Calculate and save class accuracies
            accuracy_histogram_path = os.path.join(output_folder, 'class_accuracies_histogram_GAT.png')
            plot_class_accuracies(y_true, y_pred, label_names, accuracy_histogram_path, title="Per-Class Accuracies for GAT Model")

# Copyright statement:
# The code produced herein is part of the master thesis conducted at the Technical University of Munich and should be used with proper citation.
# All rights reserved.
# Happy coding! by Server Çeter       
            