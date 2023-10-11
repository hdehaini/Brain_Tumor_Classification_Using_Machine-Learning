import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset


data = pd.read_csv('Brain_Tumor.csv')

# Rename the 'Standard Deviation' column to 'StdDev'
data.rename(columns={'Standard Deviation': 'StdDev'}, inplace=True)

# Separate features and target
X = data[['Mean', 'Variance', 'StdDev', 'Skewness', 'Kurtosis', 'Contrast', 'Energy', 'ASM', 'Entropy', 'Homogeneity', 'Dissimilarity', 'Correlation', 'Coarseness']].values
y = data['Class'].values


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


class FeatureClassifier(nn.Module):
    def __init__(self, input_size):
        super(FeatureClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


model = FeatureClassifier(input_size=X_train.shape[1])
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    """
    Train the machine learning model.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
        optimizer (optim.Optimizer): Optimization algorithm (e.g., Adam).
        num_epochs (int): Number of training epochs (default: 10).
    """
    for epoch in range(num_epochs):
        model.train()
        # Training loop here
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation after each epoch
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, labels)
                # Calculate and log validation metrics here
        val_loss /= len(val_loader)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

        # You can compute additional validation metrics here, e.g., accuracy, precision, recall, F1-score, etc.
        # Example: Compute accuracy
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {accuracy:.2f}%')

        # You can add more metrics and logging as needed.

    print('Training finished.')



def validate_model(model, val_loader, criterion):
    """
    Validate the machine learning model.

    Args:
        model (nn.Module): The PyTorch model to be validated.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).

    Returns:
        float: Validation loss.
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, labels)
    
        val_loss /= len(val_loader)

    # Calculate and log validation metrics here
    print(f'Validation Loss: {val_loss:.4f}')

    # You can compute additional validation metrics here, e.g., accuracy, precision, recall, F1-score, etc.
    # Example: Compute accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')

    # You can add more metrics and logging as needed.

    return val_loss




def test_model(model, test_loader):
    """
    Test the machine learning model on a test dataset.

    Args:
        model (nn.Module): The PyTorch model to be tested.
        test_loader (DataLoader): DataLoader for the test dataset.

    Returns:
        dict: Dictionary of test metrics (e.g., accuracy, precision, recall).
    """
    model.eval()
    
    # Initialize variables for computing test metrics
    correct = 0
    total = 0
    all_predicted = []
    all_actual = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predicted.extend(predicted.tolist())
            all_actual.extend(labels.tolist())

    accuracy = 100 * correct / total
    # Calculate additional test metrics, e.g., precision, recall, F1-score
    # Example: Calculate precision, recall, and F1-score
    from sklearn.metrics import precision_score, recall_score, f1_score

    precision = precision_score(all_actual, all_predicted)
    recall = recall_score(all_actual, all_predicted)
    f1 = f1_score(all_actual, all_predicted)

    test_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Test Precision: {precision:.2f}')
    print(f'Test Recall: {recall:.2f}')
    print(f'Test F1-Score: {f1:.2f}')

    return test_metrics



def save_model(model, path):
    """
    Save the trained model to a file.

    Args:
        model (nn.Module): The PyTorch model to be saved.
        path (str): Path to the file where the model will be saved.
    """
    torch.save(model.state_dict(), path)


def load_model(model, path):
    """
    Load a pre-trained model from a file.

    Args:
        model (nn.Module): The PyTorch model to be loaded.
        path (str): Path to the file where the model is saved.
    """
    model.load_state_dict(torch.load(path))
    model.eval()

def predict_image(model, image_path):
    """
    Make predictions on a single image using the trained model.

    Args:
        model (nn.Module): The PyTorch model for inference.
        image_path (str): Path to the image to be predicted.

    Returns:
        float: Predicted class or probability.
    """
    model.eval()

    # Define image transformations (adjust to match the preprocessing used during training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to match model input size
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
    ])

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(image)

    # Assuming it's a binary classification (0 for non-tumor, 1 for tumor)
    predicted_class = 1 if output.item() > 0.5 else 0

    return predicted_class


def plot_training_metrics(train_metrics, val_metrics):
    """
    Plot training and validation metrics using Matplotlib.

    Args:
        train_metrics (dict): Training metrics (e.g., loss, accuracy).
        val_metrics (dict): Validation metrics (e.g., loss, accuracy).
    """
    epochs = range(1, len(train_metrics['loss']) + 1)

    # Plot training and validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_metrics['loss'], label='Training Loss')
    plt.plot(epochs, val_metrics['loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_metrics['accuracy'], label='Training Accuracy')
    plt.plot(epochs, val_metrics['accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Assuming you have your training, validation, and test data tensors
train_data_tensor = torch.tensor(X_train, dtype=torch.float32)
train_labels_tensor = torch.tensor(y_train, dtype=torch.float32)

val_data_tensor = torch.tensor(X_val, dtype=torch.float32)
val_labels_tensor = torch.tensor(y_val, dtype=torch.float32)

test_data_tensor = torch.tensor(X_test, dtype=torch.float32)
test_labels_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define batch sizes
batch_size = 32

# Create data loaders
train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(val_data_tensor, val_labels_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
test_model(model, test_loader)
save_model(model, 'brain_tumor_model.pth')
