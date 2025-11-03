# models/training/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import mlflow
import os
import time
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json


class MentalHealthTrainer:
    def __init__(self, config_path="config/default.yaml"):
        self.config = Config(config_path)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Training on: {self.device}")

        self.setup_experiment_tracking()
        self.setup_directories()

        # Training state
        self.best_accuracy = 0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def setup_experiment_tracking(self):
        """Setup MLflow for experiment tracking"""
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("mental-health-detection")

        # Start MLflow run
        self.run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.start_run(run_name=self.run_name)

        # Log parameters
        mlflow.log_params({
            "learning_rate": self.config.training.learning_rate,
            "batch_size": self.config.training.batch_size,
            "epochs": self.config.training.epochs,
            "model": self.config.model.text_model,
            "device": str(self.device)
        })

    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.config.training.save_dir, exist_ok=True)
        os.makedirs("logs/training", exist_ok=True)
        os.makedirs("models/evaluation", exist_ok=True)

    def setup_model(self, train_loader):
        """Initialize model with proper weight initialization"""
        from src.models.architecture import MultiModalMentalHealthModel

        self.model = MultiModalMentalHealthModel(self.config).to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)

        print(f"📊 Model Parameters:")
        print(f"   Total: {total_params:,}")
        print(f"   Trainable: {trainable_params:,}")

        # Optimizer with different learning rates for different parts
        text_params = [p for n, p in self.model.named_parameters(
        ) if 'text_encoder' in n and p.requires_grad]
        other_params = [p for n, p in self.model.named_parameters(
        ) if 'text_encoder' not in n and p.requires_grad]

        self.optimizer = torch.optim.AdamW([
            {'params': text_params, 'lr': self.config.training.learning_rate / 10},
            {'params': other_params, 'lr': self.config.training.learning_rate}
        ], weight_decay=0.01)

        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=2, factor=0.5, verbose=True
        )

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

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

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch: {epoch} [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f} LR: {current_lr:.2e}')

        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total

        return epoch_loss, epoch_accuracy

    def validate(self, val_loader, epoch):
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

        # Calculate additional metrics
        class_report = classification_report(
            all_labels, all_predictions, output_dict=True)
        conf_matrix = confusion_matrix(all_labels, all_predictions)

        return val_loss, val_accuracy, class_report, conf_matrix, all_predictions, all_labels

    def save_checkpoint(self, epoch, val_accuracy, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_accuracy': val_accuracy,
            'config': self.config.data,
            'timestamp': datetime.now().isoformat()
        }

        if is_best:
            filename = f"best_model_epoch_{epoch}_acc_{val_accuracy:.2f}.pth"
            self.best_accuracy = val_accuracy
        else:
            filename = f"checkpoint_epoch_{epoch}.pth"

        save_path = os.path.join(self.config.training.save_dir, filename)
        torch.save(checkpoint, save_path)

        # Also save as latest
        torch.save(checkpoint, os.path.join(
            self.config.training.save_dir, "latest_model.pth"))

        print(f"💾 Saved checkpoint: {save_path}")
        return save_path

    def train(self, train_loader, val_loader):
        """Main training loop"""
        print("🎯 Starting Training...")
        self.setup_model(train_loader)

        patience_counter = 0

        for epoch in range(self.config.training.epochs):
            start_time = time.time()

            # Training phase
            train_loss, train_accuracy = self.train_epoch(train_loader, epoch)

            # Validation phase
            val_loss, val_accuracy, class_report, conf_matrix, _, _ = self.validate(
                val_loader, epoch)

            epoch_time = time.time() - start_time

            # Update learning rate
            self.scheduler.step(val_accuracy)

            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

            # Print progress
            print(f"\n📈 Epoch {epoch+1}/{self.config.training.epochs}")
            print(f"   Time: {epoch_time:.2f}s")
            print(
                f"   Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            print(f"   Best Val Acc: {self.best_accuracy:.2f}%")

            # Log to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "epoch_time": epoch_time
            }, step=epoch)

            # Save best model
            if val_accuracy > self.best_accuracy:
                self.save_checkpoint(epoch, val_accuracy, is_best=True)
                patience_counter = 0

                # Log classification report for best model
                for class_name, metrics in class_report.items():
                    if isinstance(metrics, dict):
                        for metric_name, value in metrics.items():
                            mlflow.log_metric(
                                f"best_{class_name}_{metric_name}", value)
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.config.training.early_stopping_patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break

            # Save regular checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, val_accuracy)

        # Save final model
        self.save_checkpoint(self.config.training.epochs, val_accuracy)

        # Log training history
        self.save_training_history()

        print(
            f"\n✅ Training completed! Best accuracy: {self.best_accuracy:.2f}%")

        return self.best_accuracy

    def save_training_history(self):
        """Save training history to file"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_accuracy': self.best_accuracy,
            'timestamp': datetime.now().isoformat()
        }

        history_path = os.path.join(
            "logs/training", f"history_{self.run_name}.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        mlflow.log_artifact(history_path)
