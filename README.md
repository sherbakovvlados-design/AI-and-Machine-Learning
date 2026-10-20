# AI-and-Machine-Learning

# ЗАВДАННЯ 1. Один крок для однієї машини

print("===== ЗАВДАННЯ 1 =====")

x = 0.97
y = 8.7
w = 5
lr = 0.1

# Початковий прогноз і помилка
prediction = w * x
error = (prediction - y) ** 2

# Обчислення крутизни
gradient = 2 * (prediction - y) * x

# Оновлення ваги
new_w = w - lr * gradient

# Новий прогноз і помилка
new_prediction = new_w * x
new_error = (new_prediction - y) ** 2

print("Початковий прогноз:", prediction)
print("Початкова помилка:", error)
print("Крутизна:", gradient)
print("Нова вага:", new_w)
print("Новий прогноз:", new_prediction)
print("Нова помилка:", new_error)
print("Стало краще:", new_error < error)

# ЗАВДАННЯ 2. Чи згодні машини?

print("\n===== ЗАВДАННЯ 2 =====")

# Дані двох машин
cars = {
    "Fiat 128": (0.85, 8.1),
    "Pontiac Catalina": (2.01, 16.8)
}

# Перевіряємо дві ваги
for w in [5, 9]:
    print("\nВага w =", w)

    gradients = []

    for name, (x, y) in cars.items():
        prediction = w * x
        gradient = 2 * (prediction - y) * x

        gradients.append(gradient)

        print(name)
        print("Прогноз:", round(prediction, 4))
        print("Крутизна:", round(gradient, 4))

    # Порівнюємо напрямки крутизни
    if gradients[0] * gradients[1] > 0:
        print("Машини показують один напрямок")
    elif gradients[0] * gradients[1] < 0:
        print("Машини показують різні напрямки")
    else:
        print("Одна з крутизн дорівнює нулю")

# ЗАВДАННЯ 3. Розмір кроку

print("\n===== ЗАВДАННЯ 3 =====")

# Функція помилки
def loss(w):
    return (w - 8) ** 2

# Перевірка різних швидкостей навчання
for lr in [0.1, 0.5, 0.9, 1.0, 1.1]:

    w = 12

    print("\nШвидкість навчання:", lr)

    # Виконуємо 10 кроків
    for step in range(10):
        gradient = 2 * (w - 8)
        w = w - lr * gradient

        print(
            "Крок:", step + 1,
            "Вага:", round(w, 4),
            "Помилка:", round(loss(w), 4)
        )

print("\nГраничне значення lr = 1")
print("При lr = 1 вага коливається між 12 і 4")
print("При lr > 1 модель розходиться")

# ЗАВДАННЯ 4. Backpropagation
print("\n===== ЗАВДАННЯ 4 =====")

x = 1
y = 6
w1 = 2
w2 = 2
lr = 0.05

# 1. Прямий прохід
h = x * w1
prediction = h * w2
error = (prediction - y) ** 2

print("\nПрямий прохід")
print("h =", h)
print("Прогноз =", prediction)
print("Помилка =", error)

# 2. Зворотний прохід
d_prediction = 2 * (prediction - y)
gradient_w2 = d_prediction * h
d_h = d_prediction * w2
gradient_w1 = d_h * x

print("\nЗворотний прохід")
print("Вплив прогнозу:", d_prediction)
print("Крутизна по w2:", gradient_w2)
print("Вплив h:", d_h)
print("Крутизна по w1:", gradient_w1)

# 3. Перевірка крутизни
epsilon = 0.001

# Перевіряємо w1
test_prediction_w1 = x * (w1 + epsilon) * w2
test_error_w1 = (test_prediction_w1 - y) ** 2
check_w1 = (test_error_w1 - error) / epsilon

# Перевіряємо w2
test_prediction_w2 = x * w1 * (w2 + epsilon)
test_error_w2 = (test_prediction_w2 - y) ** 2
check_w2 = (test_error_w2 - error) / epsilon

print("\nПеревірка крутизни")
print("Перевірка w1:", round(check_w1, 4))
print("Перевірка w2:", round(check_w2, 4))

# 4. Один крок навчання
new_w1 = w1 - lr * gradient_w1
new_w2 = w2 - lr * gradient_w2

new_h = x * new_w1
new_prediction = new_h * new_w2
new_error = (new_prediction - y) ** 2

print("\nПісля навчання")
print("Нова вага w1:", new_w1)
print("Нова вага w2:", new_w2)
print("Новий прогноз:", new_prediction)
print("Нова помилка:", round(new_error, 4))
print("Помилка зменшилась:", new_error < error)

# ЗАВДАННЯ 5. Нульові ваги

print("\n===== ЗАВДАННЯ 5 =====")

x = 1
y = 6
w1 = 0
w2 = 0
lr = 0.05

# Прямий прохід
h = x * w1
prediction = h * w2
error = (prediction - y) ** 2

# Обчислення крутизни
d_prediction = 2 * (prediction - y)
gradient_w2 = d_prediction * h
gradient_w1 = d_prediction * w2 * x

print("Прогноз:", prediction)
print("Помилка:", error)
print("Крутизна w1:", gradient_w1)
print("Крутизна w2:", gradient_w2)

# Оновлення ваг
w1 = w1 - lr * gradient_w1
w2 = w2 - lr * gradient_w2

print("Нова вага w1:", w1)
print("Нова вага w2:", w2)

print("\nВисновок:")
print("Обидві крутизни дорівнюють нулю.")
print("Ваги не змінюються, тому навчання зупиняється.")
print("Випадкові початкові ваги допомагають")
print("нейромережі почати навчання.")
