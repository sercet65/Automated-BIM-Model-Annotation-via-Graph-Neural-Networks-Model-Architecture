import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

def plot_training_validation(train_losses, val_losses, save_path, title=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_validation_accuracy(val_accuracies, save_path, title=None):
    plt.figure(figsize=(10, 6))
    plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_test_accuracy(test_accuracy_value, final_epoch, output_path, title):
    plt.figure(figsize=(5, 4))
    plt.bar(['Test Accuracy'], [test_accuracy_value], color='steelblue', width=0.3)
    plt.text(0, test_accuracy_value, f'Epoch: {final_epoch}', ha='center', va='bottom')
    plt.title(title, fontsize=14)
    plt.ylim(0, 1)
    plt.gca().yaxis.grid(False)  # Remove gridlines
    plt.gca().xaxis.grid(False)
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, output_path, title=None, normalize=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))  # Specify the labels parameter
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # We are setting the ticks based on the length of the classes array now
    ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Highlight the correct predictions along the diagonal
            color = "white" if cm[i, j] > thresh else "black"
            text = format(cm[i, j], fmt)
            ax.text(j, i, text,
                    ha="center", va="center",
                    color=color)
            # Add red border around diagonal cells
            if i == j:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
def plot_prediction_distribution(predictions, label_types, output_path, title):
    # Count the frequency of each predicted label type
        
    predictions = np.array(predictions)

    label_types = np.unique(label_types)

    label_counts = {label: np.sum(predictions == label) for label in label_types}

    # Sort labels based on their count
    labels = list(label_counts.keys())
    frequencies = [label_counts[label] for label in labels]

    # Create a bar plot of predicted label frequencies
    plt.figure(figsize=(8, 6))
    plt.bar(labels, frequencies, color='royalblue')
    plt.xlabel('Label Type', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title(title, fontsize=16)
    plt.xticks(labels, rotation=45)  # Rotate labels for better readability if needed
    plt.grid(axis='y')
    plt.tight_layout()

    # Save the plot to the specified output path
    plt.savefig(output_path)
    plt.close()

def plot_lr_schedule(learning_rates, output_path, title=None):
    epochs = range(1, len(learning_rates) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, learning_rates, marker='o', linestyle='-')
    plt.title(title, fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.yscale('log')  # If you have very small changes in lr, this can help to visualize
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_class_accuracies(y_true, y_pred, classes, output_path, title=None, colors=None):
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    # Calculate accuracies
    accuracies = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(12, 8))  # Adjust figure size as needed
    if colors is None:
        colors = ['steelblue', 'lightcoral', 'lightsteelblue', 'darkseagreen', 'silver']  # Define shades of blue
    bars = plt.bar(classes, accuracies, color=colors)
    plt.xlabel('Class', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    if title:
        plt.title(title, fontsize=20)
    plt.xticks(rotation=90, ha="center", fontsize=14)  # Rotate labels for better readability
    plt.yticks(fontsize=14)
    
    # Optionally, add text labels on each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')  # Adjust text alignment if necessary
    
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
    plt.savefig(output_path)
    plt.close()

# Copyright statement:
# The code produced herein is part of the master thesis conducted at the Technical University of Munich and should be used with proper citation.
# All rights reserved.
# Happy coding! by Server Çeter 