import numpy as np
import matplotlib.pyplot as plt


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

    mse = np.mean((y - y_pred) ** 2)

    return mse


# ============================================================
# 3. Gradient Descent
# ============================================================

b0 = 0.0
b1 = 0.0

learning_rate = 0.01
epochs = 1000


# Store the path taken by Gradient Descent

b0_history = []
b1_history = []
loss_history = []


for epoch in range(epochs):

    # ----------------------------------------
    # Prediction
    # ----------------------------------------

    y_pred = b0 + b1 * X


    # ----------------------------------------
    # Calculate loss
    # ----------------------------------------

    mse = np.mean((y - y_pred) ** 2)


    # ----------------------------------------
    # Store current position
    # ----------------------------------------

    b0_history.append(b0)
    b1_history.append(b1)
    loss_history.append(mse)


    # ----------------------------------------
    # Calculate gradients
    # ----------------------------------------

    n = len(X)

    db0 = (-2 / n) * np.sum(y - y_pred)

    db1 = (-2 / n) * np.sum(
        X * (y - y_pred)
    )


    # ----------------------------------------
    # Update parameters
    # ----------------------------------------

    b0 = b0 - learning_rate * db0

    b1 = b1 - learning_rate * db1


# ============================================================
# 4. Final Result
# ============================================================

print("Gradient Descent Complete")

print("Final b0 =", b0)
print("Final b1 =", b1)

final_mse = calculate_mse(b0, b1)

print("Final MSE =", final_mse)


# ============================================================
# 5. Convert History to NumPy Arrays
# ============================================================

b0_history = np.array(b0_history)

b1_history = np.array(b1_history)

loss_history = np.array(loss_history)


# ============================================================
# 6. Create Parameter Grid
# ============================================================

b0_values = np.linspace(
    0,
    40,
    200
)

b1_values = np.linspace(
    0,
    12,
    200
)


B0, B1 = np.meshgrid(
    b0_values,
    b1_values
)


# ============================================================
# 7. Calculate MSE Across the Entire Parameter Space
# ============================================================

MSE = np.zeros_like(B0)


for i in range(B0.shape[0]):

    for j in range(B0.shape[1]):

        MSE[i, j] = calculate_mse(
            B0[i, j],
            B1[i, j]
        )


# ============================================================
# 8. Create Contour Plot
# ============================================================

plt.figure(figsize=(10, 7))


# Draw loss contours

contours = plt.contour(
    B0,
    B1,
    MSE,
    levels=30
)


# Label contour lines

plt.clabel(
    contours,
    inline=True,
    fontsize=8
)


# ============================================================
# 9. Plot Gradient Descent Path
# ============================================================

plt.plot(
    b0_history,
    b1_history,
    marker=".",
    markersize=2,
    label="Gradient Descent Path"
)


# ============================================================
# 10. Starting Point
# ============================================================

plt.scatter(
    b0_history[0],
    b1_history[0],
    s=100,
    label="Starting Point"
)


# ============================================================
# 11. Final Point
# ============================================================

plt.scatter(
    b0_history[-1],
    b1_history[-1],
    s=100,
    label="Final Point"
)


# ============================================================
# 12. Labels
# ============================================================

plt.xlabel("b0 (Intercept)")

plt.ylabel("b1 (Slope)")

plt.title(
    "Loss Surface and Gradient Descent"
)

plt.legend()

plt.grid(True)


# ============================================================
# 13. Display
# ============================================================

plt.show()