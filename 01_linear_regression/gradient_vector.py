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

def mse(b0, b1):

    y_pred = b0 + b1 * X

    return np.mean((y - y_pred) ** 2)


# ============================================================
# 3. Gradient Function
# ============================================================

def gradient(b0, b1):

    y_pred = b0 + b1 * X

    n = len(X)

    db0 = (-2 / n) * np.sum(y - y_pred)

    db1 = (-2 / n) * np.sum(
        X * (y - y_pred)
    )

    return db0, db1


# ============================================================
# 4. Choose a Point on the Loss Surface
# ============================================================

b0 = 10.0
b1 = 5.0


# ============================================================
# 5. Calculate Loss
# ============================================================

current_loss = mse(b0, b1)


# ============================================================
# 6. Calculate Gradient
# ============================================================

db0, db1 = gradient(b0, b1)


print("Current point:")
print("b0 =", b0)
print("b1 =", b1)

print("\nCurrent MSE =", current_loss)

print("\nGradient:")
print("dJ/db0 =", db0)
print("dJ/db1 =", db1)


# ============================================================
# 7. Parameter Grid
# ============================================================

b0_values = np.linspace(0, 40, 200)

b1_values = np.linspace(0, 12, 200)

B0, B1 = np.meshgrid(
    b0_values,
    b1_values
)


# ============================================================
# 8. Calculate Loss Surface
# ============================================================

MSE = np.zeros_like(B0)


for i in range(B0.shape[0]):

    for j in range(B0.shape[1]):

        MSE[i, j] = mse(
            B0[i, j],
            B1[i, j]
        )


# ============================================================
# 9. Plot Contours
# ============================================================

plt.figure(figsize=(10, 7))

contours = plt.contour(
    B0,
    B1,
    MSE,
    levels=30
)

plt.clabel(
    contours,
    inline=True,
    fontsize=8
)


# ============================================================
# 10. Plot Current Point
# ============================================================

plt.scatter(
    b0,
    b1,
    s=120,
    label="Current Point"
)


# ============================================================
# 11. Plot Gradient Vector
# ============================================================

plt.quiver(
    b0,
    b1,
    db0,
    db1,
    angles="xy",
    scale_units="xy",
    scale=100,
    label="Gradient"
)


# ============================================================
# 12. Plot Negative Gradient
# ============================================================

plt.quiver(
    b0,
    b1,
    -db0,
    -db1,
    angles="xy",
    scale_units="xy",
    scale=100,
    label="-Gradient (Descent)"
)


# ============================================================
# 13. Labels
# ============================================================

plt.xlabel("b0 (Intercept)")

plt.ylabel("b1 (Slope)")

plt.title(
    "Gradient and Negative Gradient"
)

plt.legend()

plt.grid(True)


# ============================================================
# 14. Display
# ============================================================

plt.show()