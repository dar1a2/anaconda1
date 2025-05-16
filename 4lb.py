import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
data = pd.read_csv('dataset_simple.csv')
features = torch.FloatTensor(StandardScaler().fit_transform(data.iloc[:, :2].values))
labels = torch.FloatTensor(data.iloc[:, 2].values).reshape(-1, 1)  # binary labels: 0 or 1

# Define neural network
class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Model parameters
input_dim = features.shape[1]
hidden_dim = 100
output_dim = 1

# Initialize model, loss function, optimizer
model = NeuralNet(input_dim, hidden_dim, output_dim)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(features)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Evaluation
with torch.no_grad():
    outputs = model(features)
    predictions = (outputs > 0.5).float()
    accuracy = (predictions == labels).float().mean()
    print(f"\Точность: {accuracy.item() * 100:.2f}%")

    # Final classification using midpoint threshold
    max_prob = float(torch.max(outputs))
    min_prob = float(torch.min(outputs))
    threshold = (max_prob + min_prob) / 2

    final_preds = torch.Tensor(np.where(outputs >= threshold, 1, 0).reshape(-1, 1))
    classification_error = torch.sum(torch.abs(labels - final_preds)) / 2
    print('\nОшибка (Кол-во несовпавших ответов)):')
    print(classification_error)