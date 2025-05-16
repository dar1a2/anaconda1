import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Загрузка и нормализация данных
data = pd.read_csv('dataset_simple.csv')
X = StandardScaler().fit_transform(data.iloc[:, :2].values)
y = data.iloc[:, 2].values.reshape(-1, 1)

features = torch.tensor(X, dtype=torch.float32)
labels = torch.tensor(y, dtype=torch.float32)

# Определение архитектуры нейросети
class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

# Параметры модели
input_size = features.shape[1]
hidden_size = 100
output_size = 1

model = NeuralNet(input_size, hidden_size, output_size)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    predictions = model(features)
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch:4d}, Loss: {loss.item():.4f}')

# Оценка модели
with torch.no_grad():
    outputs = model(features)
    predicted_classes = (outputs > 0.5).float()
    accuracy = (predicted_classes == labels).float().mean()
    print(f'\nТочность: {accuracy.item() * 100:.2f}%')

    # Альтернативная классификация с порогом по середине
    max_prob = outputs.max().item()
    min_prob = outputs.min().item()
    threshold = (max_prob + min_prob) / 2

    adjusted_preds = (outputs >= threshold).float()
    classification_errors = torch.sum(torch.abs(labels - adjusted_preds)) / 2
    print('\nОшибка(Кол-во несовпавших ответов)):')
    print(int(classification_errors.item()))