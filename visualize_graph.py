import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_training_validation(train_losses, val_losses, save_path, title=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_validation_accuracy(val_accuracies, save_path, title=None):
    plt.figure(figsize=(10, 6))
    plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_test_accuracy(test_accuracy_value, final_epoch, output_path, title):
    plt.figure(figsize=(6, 4))
    plt.bar(['Test Accuracy'], [test_accuracy_value ])
    plt.text(0, test_accuracy_value , f'Epoch: {final_epoch}', ha='center', va='bottom')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(axis='y')
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

    fmt = 'd' if not normalize else '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
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
    plt.bar(labels, frequencies, color='skyblue')
    plt.xlabel('Label Type')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.xticks(labels, rotation=45)  # Rotate labels for better readability if needed
    plt.grid(axis='y')
    plt.tight_layout()

    # Save the plot to the specified output path
    plt.savefig(output_path)
    plt.close()

        