# -*- coding: utf-8 -*-
"""

"""

import torch
from random import randint

# Генерация случайного числа от 1 до 10, создание тензора с этим числом
tensor_init = torch.randint(1, 10, (1,), dtype=torch.int64)
print("Исходный тензор: ", tensor_init.item())

# Преобразуем тензор в тип float32 и включаем отслеживание изменений
tensor_init = tensor_init.to(dtype=torch.float32)
tensor_init.requires_grad = True  # Включение отслеживания градиента

exp_power = 2
tensor_squared = tensor_init ** exp_power
print("Тензор возведенный в степень exp_power = 2: ", tensor_squared)

# Умножаем тензор на случайное значение
random_factor = randint(1, 10)
tensor_scaled = tensor_squared * random_factor
print(f"Тензор, умноженный на случайное значение random_factor = {random_factor}: ", tensor_scaled)

# Применяем экспоненту к тензору
tensor_exp = torch.exp(tensor_scaled)
print("Экспонента от тензора: ", tensor_exp)

# Вычисление градиента
tensor_exp.backward()
print("Продифференцированный тензор: ", tensor_init.grad)
print("Значение градиента без экспоненты: ", "{:f}".format(tensor_init.grad.item()))