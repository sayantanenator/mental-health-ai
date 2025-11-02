# models/training/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import os
from src.model.architecture import MultiModalMentalHealthModel
from src.utils.config import Config


class MentalHealthTrainer:
    def __init__(self, config_path="config/default.yaml"):
        self.config = Config(config_path)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.setup_mlflow()

    def setup_mlflow(self):
        """Setup MLflow for experiment tracking"""
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("mental-health-detection")

    def setup_model(self):
        """Initialize model, optimizer, and loss function"""
        self.model = MultiModalMentalHealthModel(self.config).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=0.01
        )

        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):
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
                    f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        return total_loss / len(dataloader)

    def validate(self, dataloader):
        """Validate model performance"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                text_input = {k: v.to(self.device)
                              for k, v in batch['text_input'].items()}
                audio_features = batch['audio_features'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(text_input, audio_features)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)

        return avg_loss, accuracy

    def train(self, train_loader, val_loader):
        """Main training loop"""
        self.setup_model()

        best_accuracy = 0
        patience_counter = 0

        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                "learning_rate": self.config.training.learning_rate,
                "batch_size": self.config.training.batch_size,
                "epochs": self.config.training.epochs,
                "model": self.config.model.text_model
            })

            for epoch in range(self.config.training.epochs):
                # Training
                train_loss = self.train_epoch(train_loader, epoch)

                # Validation
                val_loss, val_accuracy = self.validate(val_loader)

                print(f"Epoch {epoch+1}/{self.config.training.epochs}")
                print(
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

                # Log metrics
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy
                }, step=epoch)

                # Early stopping
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    patience_counter = 0
                    self.save_model("best_model.pth")
                else:
                    patience_counter += 1

                if patience_counter >= self.config.training.early_stopping_patience:
                    print("Early stopping triggered!")
                    break

            # Save final model
            self.save_model("final_model.pth")
            mlflow.log_artifact("final_model.pth")

    def save_model(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.data
        }
        torch.save(checkpoint, filename)


if __name__ == "__main__":
    trainer = MentalHealthTrainer()
    # You would add your data loading logic here
    # trainer.train(train_loader, val_loader)
