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
# RMSPROP
# ============================================================

b0 = 0.0
b1 = 0.0

alpha = 0.01
beta = 0.9
epsilon = 1e-8

S_b0 = 0.0
S_b1 = 0.0

iterations = 1000

losses = []


for i in range(iterations):

    db0, db1 = gradients(b0, b1)

    # Exponential moving average of squared gradients
    S_b0 = beta * S_b0 + (1 - beta) * db0 ** 2
    S_b1 = beta * S_b1 + (1 - beta) * db1 ** 2

    # Adaptive update
    b0 -= alpha * db0 / (np.sqrt(S_b0) + epsilon)
    b1 -= alpha * db1 / (np.sqrt(S_b1) + epsilon)

    losses.append(mse(b0, b1))


# ============================================================
# RESULTS
# ============================================================

print("=" * 50)
print("RMSProp")
print("=" * 50)

print(f"Final b0 = {b0}")
print(f"Final b1 = {b1}")
print(f"Final MSE = {losses[-1]}")


# ============================================================
# LOSS CURVE
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(losses)

plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("RMSProp - Loss vs Iteration")

plt.grid(True)
plt.show()


# ============================================================
# REGRESSION
# ============================================================

y_pred = b0 + b1 * X

plt.figure(figsize=(9, 6))

plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred, label="RMSProp Regression Line")

plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("RMSProp Linear Regression")

plt.legend()
plt.grid(True)

plt.show()