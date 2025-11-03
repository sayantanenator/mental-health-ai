# src/training/evaluation.py
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime


class ModelEvaluator:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

    def evaluate(self, test_loader):
        """Comprehensive model evaluation"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for batch in test_loader:
                text_input = {k: v.to(self.device)
                              for k, v in batch['text_input'].items()}
                audio_features = batch['audio_features'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(text_input, audio_features)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # Calculate metrics
        results = self.calculate_metrics(
            all_predictions, all_labels, all_probabilities)
        self.generate_plots(all_predictions, all_labels, all_probabilities)
        self.save_results(results)

        return results

    def calculate_metrics(self, predictions, labels, probabilities):
        """Calculate evaluation metrics"""
        accuracy = np.mean(np.array(predictions) == np.array(labels))
        class_report = classification_report(
            labels, predictions, output_dict=True)
        conf_matrix = confusion_matrix(labels, predictions)

        # Calculate ROC-AUC for each class
        roc_auc = {}
        for i in range(self.config.model.num_classes):
            roc_auc[f"class_{i}"] = roc_auc_score(
                (np.array(labels) == i).astype(int),
                np.array(probabilities)[:, i]
            )

        return {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'roc_auc': roc_auc,
            'timestamp': datetime.now().isoformat()
        }

    def generate_plots(self, predictions, labels, probabilities):
        """Generate evaluation plots"""
        os.makedirs("models/evaluation", exist_ok=True)

        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        conf_matrix = confusion_matrix(labels, predictions)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('models/evaluation/confusion_matrix.png')
        plt.close()

    def save_results(self, results):
        """Save evaluation results"""
        with open('models/evaluation/evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print("Evaluation results saved.")
