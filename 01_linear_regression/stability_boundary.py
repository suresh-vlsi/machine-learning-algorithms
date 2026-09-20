import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Training Data
# ============================================================

X = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y = np.array([35, 42, 51, 58, 66, 73], dtype=float)

n = len(X)


# ============================================================
# Loss
# ============================================================

def mse(b0, b1):

    y_pred = b0 + b1 * X

    return np.mean((y - y_pred) ** 2)


# ============================================================
# Gradient
# ============================================================

def gradient(b0, b1):

    y_pred = b0 + b1 * X

    error = y_pred - y

    db0 = (2 / n) * np.sum(error)

    db1 = (2 / n) * np.sum(X * error)

    return db0, db1


# ============================================================
# Gradient Descent
# ============================================================

def gradient_descent(alpha, iterations):

    b0 = 0.0
    b1 = 0.0

    losses = []

    for i in range(iterations):

        loss = mse(b0, b1)

        losses.append(loss)

        db0, db1 = gradient(b0, b1)

        b0 = b0 - alpha * db0
        b1 = b1 - alpha * db1

    return losses


# ============================================================
# Hessian
# ============================================================

H = (2 / n) * np.array([
    [n, np.sum(X)],
    [np.sum(X), np.sum(X ** 2)]
])


# ============================================================
# Eigenvalues
# ============================================================

eigenvalues = np.linalg.eigvalsh(H)

lambda_max = np.max(eigenvalues)

alpha_max = 2 / lambda_max


print("===================================")
print("Stability Boundary")
print("===================================")

print("Largest eigenvalue =", lambda_max)

print("Theoretical limit =", alpha_max)


# ============================================================
# Learning Rates Around Boundary
# ============================================================

learning_rates = [
    0.04,
    0.05,
    0.06,
    0.062,
    0.06256,
    0.063,
    0.065,
    0.07
]


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(10, 6))


for alpha in learning_rates:

    losses = gradient_descent(
        alpha,
        100
    )

    plt.semilogy(
        losses,
        label=f"α = {alpha}"
    )


plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("Iteration")

plt.ylabel("MSE Loss")

plt.title(
    "Gradient Descent Near Stability Boundary"
)

plt.legend()

plt.grid(True)

plt.show()
