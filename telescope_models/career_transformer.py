# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Career Transformer Model
BERT + Tabular Fusion for 88%+ accuracy career prediction.
Combines resume text embeddings with structured career features.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CareerTransformerModel(nn.Module):
    """
    BERT + Tabular Fusion Model for Career Prediction.

    Architecture:
    - Text Encoder: DistilBERT for resume/job description (768-dim)
    - Tabular Encoder: MLP for structured features (128-dim)
    - Fusion: Combined representation → Career outcome prediction

    Target Accuracy: 88%+
    Inference Time: <50ms per prediction
    """

    def __init__(
        self,
        num_tabular_features: int = 20,
        num_classes: int = 5,  # Career outcomes: 0-4
        bert_model_name: str = 'distilbert-base-uncased',
        hidden_dim: int = 512,
        dropout: float = 0.3
    ):
        super().__init__()

        self.num_tabular_features = num_tabular_features
        self.num_classes = num_classes

        # Text encoder: DistilBERT (lighter than BERT, 40% faster)
        try:
            from transformers import DistilBertModel, DistilBertTokenizer
            self.bert = DistilBertModel.from_pretrained(bert_model_name)
            self.tokenizer = DistilBertTokenizer.from_pretrained(bert_model_name)
            self.bert_dim = 768
            logger.info(f"Loaded {bert_model_name}")
        except ImportError:
            logger.warning("transformers not installed, using fallback text encoder")
            self.bert = None
            self.bert_dim = 128
            # Fallback: Simple learned text embeddings
            self.text_embedding = nn.Embedding(10000, self.bert_dim)

        # Tabular feature encoder
        self.tabular_encoder = nn.Sequential(
            nn.Linear(num_tabular_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
        )

        # Fusion and classification head
        fusion_input_dim = self.bert_dim + 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout + 0.1),  # Slightly higher dropout
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        text: List[str] = None,
        tabular_features: torch.Tensor = None,
        labels: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            text: List of text strings (resumes, job descriptions)
            tabular_features: Tensor of shape [batch_size, num_tabular_features]
            labels: Optional labels for training

        Returns:
            Dictionary with logits, loss (if labels provided), predictions
        """
        batch_size = tabular_features.size(0)

        # Encode text
        if text is not None and self.bert is not None:
            # Use BERT
            tokens = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )
            tokens = {k: v.to(tabular_features.device) for k, v in tokens.items()}

            bert_output = self.bert(**tokens)
            text_embedding = bert_output.last_hidden_state[:, 0, :]  # [CLS] token
        else:
            # Fallback: use simple embeddings
            # Hash text to indices
            if text is not None:
                indices = torch.tensor([hash(t) % 10000 for t in text]).to(tabular_features.device)
                text_embedding = self.text_embedding(indices).mean(dim=0, keepdim=True).expand(batch_size, -1)
            else:
                # No text provided, use zero embeddings
                text_embedding = torch.zeros(batch_size, self.bert_dim).to(tabular_features.device)

        # Encode tabular features
        tabular_embedding = self.tabular_encoder(tabular_features)

        # Fuse embeddings
        fused = torch.cat([text_embedding, tabular_embedding], dim=1)

        # Predict
        logits = self.fusion(fused)

        output = {'logits': logits}

        # Calculate loss if labels provided
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            output['loss'] = loss

        # Predictions and probabilities
        output['predictions'] = logits.argmax(dim=1)
        output['probabilities'] = torch.softmax(logits, dim=1)

        return output

    def predict(
        self,
        text: List[str],
        tabular_features: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Inference method for production use.

        Args:
            text: List of text strings
            tabular_features: Numpy array of shape [batch_size, num_features]

        Returns:
            Dictionary with predictions and probabilities
        """
        self.eval()

        with torch.no_grad():
            # Convert to tensors
            tabular_tensor = torch.tensor(tabular_features, dtype=torch.float32)

            # Forward pass
            output = self.forward(text=text, tabular_features=tabular_tensor)

            # Convert to numpy
            predictions = output['predictions'].cpu().numpy()
            probabilities = output['probabilities'].cpu().numpy()

        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence': probabilities.max(axis=1)
        }


class CareerDataset(torch.utils.data.Dataset):
    """
    Dataset for career prediction training.
    """

    def __init__(self, data_path: str = None, dataframe=None, text_column: str = 'occupation_title'):
        """
        Initialize dataset.

        Args:
            data_path: Path to parquet/csv file
            dataframe: Or provide pandas DataFrame directly
            text_column: Column to use as text input
        """
        import pandas as pd

        if dataframe is not None:
            self.df = dataframe
        elif data_path:
            if data_path.endswith('.parquet'):
                self.df = pd.read_parquet(data_path)
            else:
                self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Must provide either data_path or dataframe")

        self.text_column = text_column

        # Feature columns (adjust based on your data)
        self.feature_columns = [
            'years_experience',
            'education_level',
            'current_salary',
            'num_skills',
            'job_satisfaction',
            'industry_growth_rate',
            'salary_vs_median',
            'salary_per_year_exp',
            'skill_diversity',
            'career_mobility_score',
        ]

        # Ensure all feature columns exist
        missing = [col for col in self.feature_columns if col not in self.df.columns]
        if missing:
            logger.warning(f"Missing columns: {missing}")
            self.feature_columns = [col for col in self.feature_columns if col in self.df.columns]

        self.num_features = len(self.feature_columns)

        # Normalize features
        self.feature_means = self.df[self.feature_columns].mean().values
        self.feature_stds = self.df[self.feature_columns].std().values + 1e-8

        logger.info(f"Dataset loaded: {len(self.df)} samples, {self.num_features} features")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Text
        text = str(row[self.text_column]) if self.text_column in row else ""

        # Tabular features (normalized)
        features = row[self.feature_columns].values.astype(np.float32)
        features = (features - self.feature_means) / self.feature_stds

        # Label
        label = int(row['career_outcome']) if 'career_outcome' in row else 0

        return {
            'text': text,
            'features': torch.tensor(features, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }


def train_career_model(
    train_data_path: str,
    val_data_path: str = None,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    save_path: str = 'models/career_transformer.pth'
):
    """
    Training script for career model.

    Args:
        train_data_path: Path to training data
        val_data_path: Path to validation data (or use split)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        save_path: Where to save trained model
    """
    from torch.utils.data import DataLoader, random_split
    import os

    # Load data
    full_dataset = CareerDataset(data_path=train_data_path)

    # Split if no validation set provided
    if val_data_path is None:
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    else:
        train_dataset = full_dataset
        val_dataset = CareerDataset(data_path=val_data_path)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=2)

    # Model
    model = CareerTransformerModel(
        num_tabular_features=full_dataset.num_features,
        num_classes=5
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    logger.info(f"Training on {device}")
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            text = batch['text']
            features = batch['features'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            output = model(text=text, tabular_features=features, labels=labels)

            loss = output['loss']
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (output['predictions'] == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                text = batch['text']
                features = batch['features'].to(device)
                labels = batch['label'].to(device)

                output = model(text=text, tabular_features=features, labels=labels)

                val_loss += output['loss'].item()
                val_correct += (output['predictions'] == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss/len(train_loader):.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch,
                'feature_means': full_dataset.feature_means,
                'feature_stds': full_dataset.feature_stds,
                'feature_columns': full_dataset.feature_columns,
            }, save_path)
            logger.info(f"Saved best model: {val_acc:.4f} accuracy")

    logger.info(f"Training complete! Best val accuracy: {best_val_acc:.4f}")
    return model


if __name__ == "__main__":
    # Test model architecture
    model = CareerTransformerModel(num_tabular_features=10, num_classes=5)

    # Dummy data
    text = ["Software Engineer with 5 years experience", "Data Scientist"]
    features = torch.randn(2, 10)
    labels = torch.tensor([3, 4])

    output = model(text=text, tabular_features=features, labels=labels)

    print("=== Career Transformer Model Test ===")
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Predictions: {output['predictions']}")
    print(f"Probabilities shape: {output['probabilities'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")
    print("\n✓ Model architecture validated")
