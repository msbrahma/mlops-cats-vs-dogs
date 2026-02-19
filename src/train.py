
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score
from src.data_loader import get_data_loaders
from src.model import SimpleCNN

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, _ = get_data_loaders("data/processed")

    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 3

    mlflow.set_experiment("Cats_vs_Dogs_Experiment")

    with mlflow.start_run():

        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("optimizer", "Adam")

        for epoch in range(epochs):

            model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)

            model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_accuracy = accuracy_score(all_labels, all_preds)

            print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} Val Accuracy: {val_accuracy:.4f}")

            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

        torch.save(model.state_dict(), "models/cnn_model.pt")
        mlflow.log_artifact("models/cnn_model.pt")
        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    train()
