import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# DATA
# ============================================================

X = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y = np.array([35, 42, 51, 58, 66, 73], dtype=float)

n = len(X)


# ============================================================
# LOSS
# ============================================================

def mse(b0, b1):
    y_pred = b0 + b1 * X
    return np.mean((y - y_pred) ** 2)


# ============================================================
# GRADIENT
# ============================================================

def gradients(b0, b1):
    y_pred = b0 + b1 * X
    error = y_pred - y

    db0 = (2 / n) * np.sum(error)
    db1 = (2 / n) * np.sum(error * X)

    return db0, db1


# ============================================================
# ADAGRAD
# ============================================================

b0 = 0.0
b1 = 0.0

alpha = 0.5
epsilon = 1e-8

G_b0 = 0.0
G_b1 = 0.0

iterations = 1000

losses = []

for i in range(iterations):

    db0, db1 = gradients(b0, b1)

    # Accumulate squared gradients
    G_b0 += db0 ** 2
    G_b1 += db1 ** 2

    # Adaptive updates
    b0 -= (alpha / (np.sqrt(G_b0) + epsilon)) * db0
    b1 -= (alpha / (np.sqrt(G_b1) + epsilon)) * db1

    loss = mse(b0, b1)
    losses.append(loss)


# ============================================================
# RESULTS
# ============================================================

print("=" * 50)
print("AdaGrad")
print("=" * 50)

print(f"Final b0 = {b0}")
print(f"Final b1 = {b1}")
print(f"Final MSE = {losses[-1]}")


# ============================================================
# LOSS VS ITERATION
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(losses)

plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("AdaGrad - Loss vs Iteration")

plt.grid(True)
plt.show()


# ============================================================
# REGRESSION RESULT
# ============================================================

y_pred = b0 + b1 * X

plt.figure(figsize=(9, 6))

plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred, label="AdaGrad Regression Line")

plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("AdaGrad Linear Regression")

plt.legend()
plt.grid(True)

plt.show()