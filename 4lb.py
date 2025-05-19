import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Загрузка и подготовка данных
data = pd.read_csv('dataset_simple.csv')
features = data.iloc[:, :2].values
labels = data.iloc[:, 2].values.reshape(-1, 1)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X_tensor = torch.tensor(features_scaled, dtype=torch.float32)
y_tensor = torch.tensor(labels, dtype=torch.float32)

# Определение модели
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Параметры модели
input_dim = X_tensor.shape[1]
hidden_dim = 100
output_dim = 1

model = SimpleClassifier(input_dim, hidden_dim, output_dim)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
num_epochs = 1000
for ep in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    
    if ep % 100 == 0:
        print(f'Эпоха {ep}, Потери: {loss.item():.4f}')

# Оценка точности
with torch.no_grad():
    predictions = model(X_tensor)
    predicted_classes = (predictions >= 0.5).float()
    accuracy = (predicted_classes == y_tensor).float().mean()
    print(f'\nТочность классификации: {accuracy.item() * 100:.2f}%')

# Классификация по новому порогу
max_val = torch.max(predictions).item()
min_val = torch.min(predictions).item()
custom_threshold = (max_val + min_val) / 2

final_preds = (predictions >= custom_threshold).float()
mismatches = (y_tensor != final_preds).sum() / 2
print('\nЧисло ошибок (несовпадений):')
print(int(mismatches.item()))