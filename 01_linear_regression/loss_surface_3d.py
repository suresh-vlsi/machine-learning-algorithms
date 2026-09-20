import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ============================================================
# 1. Training Data
# ============================================================

X = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y = np.array([35, 42, 51, 58, 66, 73], dtype=float)


# ============================================================
# 2. Loss Function
# ============================================================

def calculate_mse(b0, b1):

    y_pred = b0 + b1 * X

    return np.mean((y - y_pred) ** 2)


# ============================================================
# 3. Gradient Descent
# ============================================================

b0 = 0.0
b1 = 0.0

learning_rate = 0.01
epochs = 1000

b0_history = []
b1_history = []
loss_history = []


for epoch in range(epochs):

    # Prediction
    y_pred = b0 + b1 * X

    # Loss
    mse = np.mean((y - y_pred) ** 2)

    # Store history
    b0_history.append(b0)
    b1_history.append(b1)
    loss_history.append(mse)

    # Number of samples
    n = len(X)

    # Gradients
    db0 = (-2 / n) * np.sum(y - y_pred)

    db1 = (-2 / n) * np.sum(
        X * (y - y_pred)
    )

    # Parameter update
    b0 = b0 - learning_rate * db0

    b1 = b1 - learning_rate * db1


# ============================================================
# 4. Convert History to Arrays
# ============================================================

b0_history = np.array(b0_history)
b1_history = np.array(b1_history)
loss_history = np.array(loss_history)


# ============================================================
# 5. Parameter Grid
# ============================================================

b0_values = np.linspace(0, 40, 100)

b1_values = np.linspace(0, 12, 100)

B0, B1 = np.meshgrid(
    b0_values,
    b1_values
)


# ============================================================
# 6. Calculate Loss Surface
# ============================================================

MSE = np.zeros_like(B0)


for i in range(B0.shape[0]):

    for j in range(B0.shape[1]):

        MSE[i, j] = calculate_mse(
            B0[i, j],
            B1[i, j]
        )


# ============================================================
# 7. Create 3D Figure
# ============================================================

fig = plt.figure(figsize=(11, 8))

ax = fig.add_subplot(
    111,
    projection="3d"
)


# ============================================================
# 8. Plot Loss Surface
# ============================================================

surface = ax.plot_surface(
    B0,
    B1,
    MSE,
    alpha=0.7
)


# ============================================================
# 9. Plot Gradient Descent Path
# ============================================================

ax.plot(
    b0_history,
    b1_history,
    loss_history,
    linewidth=3,
    label="Gradient Descent Path"
)


# ============================================================
# 10. Starting Point
# ============================================================

ax.scatter(
    b0_history[0],
    b1_history[0],
    loss_history[0],
    s=100,
    label="Starting Point"
)


# ============================================================
# 11. Final Point
# ============================================================

ax.scatter(
    b0_history[-1],
    b1_history[-1],
    loss_history[-1],
    s=100,
    label="Final Point"
)


# ============================================================
# 12. Labels
# ============================================================

ax.set_xlabel("b0 (Intercept)")

ax.set_ylabel("b1 (Slope)")

ax.set_zlabel("MSE / Loss")

ax.set_title(
    "3D Loss Surface and Gradient Descent"
)


ax.legend()


# ============================================================
# 13. Display
# ============================================================

plt.show()