import random

# Создаем список из случайных чисел
numbers = [random.randint(1, 100) for _ in range(10)]  # Список из 10 случайных чисел от 1 до 100

# 2. Создаем цикл для суммирования четных чисел
summ = 0
for num in numbers:
    if num % 2 == 0:  # Проверка на четность
        summ += num

# 3. Выводим сумму
print(f"Список чисел:{numbers}")
print(f"Сумма четных чисел в списке: {summ}")