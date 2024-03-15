import matplotlib.pyplot as plt

def plot_training_validation(train_losses, val_accuracies, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training and Validation Metrics')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_test_accuracy(test_accuracy, output_path):
    plt.figure(figsize=(6, 4))
    plt.bar(['Test Accuracy'], [test_accuracy])
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.savefig(output_path)
    plt.close()

    output_path = r'C:\Users\serve\OneDrive\Desktop\Phython\4.OutputML'
