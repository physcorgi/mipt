# generate_data.py
import numpy as np
import pandas as pd

# Параметры сетки
nx = 100  # шагов по X
ny = 80   # шагов по Y
x_min, x_max = 0, 10
y_min, y_max = 0, 8

X = np.linspace(x_min, x_max, nx)
Y = np.linspace(y_min, y_max, ny)
X_grid, Y_grid = np.meshgrid(X, Y)

# Комбинируем формы рельефа:
# 1. Основа — синусоидальные волны (холмы)
Z = 5 * np.sin(X_grid) * np.cos(Y_grid)

# 2. Добавим котловину (озеро)
center_x, center_y = 6, 4
depth = 8
radius = 3
dist = np.sqrt((X_grid - center_x)**2 + (Y_grid - center_y)**2)
Z -= depth * np.exp(-dist / radius)  # плавная впадина

# 3. Добавим овраг (долина)
Z -= 4 * np.exp(-((X_grid - 3)**2) / 4)  # продольная впадина

# 4. Мелкий шум — как неровности почвы
np.random.seed(42)
Z += np.random.normal(0, 0.3, Z.shape)

# Всё в таблицу
X_flat = X_grid.flatten()
Y_flat = Y_grid.flatten()
Z_flat = Z.flatten()

df = pd.DataFrame({
    'X': X_flat,
    'Y': Y_flat,
    'Z': Z_flat
})

# Сохраняем
df.to_csv('relief_data.csv', index=False)
print("✅ Данные сохранены в 'relief_data.csv'")
print(f"📏 Размер: {len(df)} точек ({nx}×{ny})")
