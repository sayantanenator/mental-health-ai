# src/training/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import os
from datetime import datetime


class MentalHealthTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.setup_experiment_tracking()

    def setup_experiment_tracking(self):
        """Setup MLflow for experiment tracking"""
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("mental-health-multimodal")
        mlflow.start_run(
            run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        # Log parameters
        mlflow.log_params({
            "learning_rate": self.config.training.learning_rate,
            "batch_size": self.config.training.batch_size,
            "epochs": self.config.training.epochs,
            "model": self.config.model.text_model,
        })

    def setup_model(self):
        """Initialize model, optimizer, and loss function"""
        from src.models.model import MultiModalMentalHealthModel

        self.model = MultiModalMentalHealthModel(self.config).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate
        )

        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=2, factor=0.5
        )

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            self.optimizer.zero_grad()

            # Move data to device
            text_input = {k: v.to(self.device)
                          for k, v in batch['text_input'].items()}
            audio_features = batch['audio_features'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            outputs = self.model(text_input, audio_features)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f'Epoch: {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        """Validate model performance"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                text_input = {k: v.to(self.device)
                              for k, v in batch['text_input'].items()}
                audio_features = batch['audio_features'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(text_input, audio_features)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = total_loss / len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_predictions) * 100

        return val_loss, val_accuracy, all_predictions, all_labels

    def train(self, train_loader, val_loader):
        """Main training loop"""
        print("Starting training...")
        self.setup_model()

        best_accuracy = 0

        for epoch in range(self.config.training.epochs):
            # Training
            train_loss = self.train_epoch(train_loader, epoch)

            # Validation
            val_loss, val_accuracy, _, _ = self.validate(val_loader)

            # Update learning rate
            self.scheduler.step(val_accuracy)

            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            }, step=epoch)

            print(f'Epoch {epoch+1}/{self.config.training.epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Val Accuracy: {val_accuracy:.2f}%')

            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self.save_model()

        print(f"Training completed. Best accuracy: {best_accuracy:.2f}%")
        return best_accuracy

    def save_model(self):
        """Save model checkpoint"""
        os.makedirs("models", exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.data
        }

        torch.save(checkpoint, "models/best_model.pth")
        mlflow.log_artifact("models/best_model.pth")
        print("Model saved.")


def main():
    """Main training function"""
    from src.utils.config import Config
    from src.data.dataset import DataManager

    config = Config()
    data_manager = DataManager(config)
    data_manager.initialize_processors()

    # Load dataset
    dataset = data_manager.load_dataset("data/processed/training_data.csv")

    # Create data loaders
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=config.training.batch_size, shuffle=False)

    # Train model
    trainer = MentalHealthTrainer(config)
    best_accuracy = trainer.train(train_loader, val_loader)

    return best_accuracy


if __name__ == "__main__":
    main()
