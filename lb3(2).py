# -*- coding: utf-8 -*-
"""

"""
import pandas as pd 
import numpy as np
import torch 
import torch.nn as nn

# Загрузка набора данных
data = pd.read_csv('data.csv')

# Разделение признаков и меток
target_values = data.iloc[:, 4:].values
target_values = np.where(target_values == "Iris-setosa", 1, -1)
feature_values = data.iloc[:, :4].values

# Преобразование данных в тензоры
inputs = torch.Tensor(feature_values)
targets = torch.Tensor(target_values)

# Линейная модель без активации
model = nn.Linear(4, 1)

# Начальные веса и смещения
print('w ', model.weight)
print('b ', model.bias)

# Функция потерь и оптимизатор
loss_function = nn.MSELoss()
trainer = torch.optim.SGD(model.parameters(), lr=0.001)

# Первый прямой проход
prediction_initial = model(inputs)
loss_initial = loss_function(prediction_initial, targets)
print('Ошибка: ', loss_initial.item())

# Обратное распространение
loss_initial.backward()
print('dl/dw: ', model.weight.grad)
print('dl/db: ', model.bias.grad)

# Шаг оптимизации
trainer.step()

# Цикл обучения
training = True
step = 1
while training:
    prediction = model(inputs)
    loss_value = loss_function(prediction, targets)
    print('Ошибка на шаге ' + str(step) + ': ', loss_value.item())
    loss_value.backward()
    trainer.step()
    step += 1
    if loss_value.item() < 1:
        training = False

# Порог для классификации
max_value = torch.max(prediction).item()
min_value = torch.min(prediction).item()
threshold = (max_value + min_value) / 2

# Ввод данных пользователем
print("Введите 4 параметра для определения класса цветка.")
running = True
while running:
    f1 = float(input("Признак 1: "))
    f2 = float(input("Признак 2: "))
    f3 = float(input("Признак 3: "))
    f4 = float(input("Признак 4: "))
    sample = torch.Tensor([f1, f2, f3, f4])
    result = model(sample)

    print("\nПредсказанный класс: ", end='')
    if result.item() >= threshold:
        print("Iris-setosa")
    else:
        print("Iris-versicolor")

    answer = input("Завершить? Введите y: ")
    if answer.lower() in ["y", "yes"]:
        running = False