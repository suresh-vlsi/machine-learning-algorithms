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
# 3. Gradient
# ============================================================

def gradient(b0, b1):

    y_pred = b0 + b1 * X

    error = y_pred - y

    n = len(X)

    db0 = (2 / n) * np.sum(error)

    db1 = (2 / n) * np.sum(X * error)

    return db0, db1


# ============================================================
# 4. Gradient Descent
# ============================================================

def gradient_descent(alpha, iterations):

    # Initial parameters
    b0 = 0.0
    b1 = 0.0

    loss_history = []

    for i in range(iterations):

        # Calculate current loss
        loss = mse(b0, b1)

        loss_history.append(loss)

        # Calculate gradient
        db0, db1 = gradient(b0, b1)

        # Gradient Descent update
        b0 = b0 - alpha * db0
        b1 = b1 - alpha * db1

    return b0, b1, loss_history


# ============================================================
# 5. Learning Rates
# ============================================================

learning_rates = [
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2
]

iterations = 30


# ============================================================
# 6. Run Gradient Descent
# ============================================================

results = {}

for alpha in learning_rates:

    b0, b1, loss_history = gradient_descent(
        alpha,
        iterations
    )

    results[alpha] = {
        "b0": b0,
        "b1": b1,
        "loss": loss_history
    }

    print("-----------------------------------")
    print("Learning Rate =", alpha)
    print("Final b0 =", b0)
    print("Final b1 =", b1)
    print("Final MSE =", loss_history[-1])


# ============================================================
# 7. Plot Loss vs Iteration
# ============================================================

plt.figure(figsize=(10, 6))

for alpha in learning_rates:

    plt.plot(
        results[alpha]["loss"],
        marker="o",
        label=f"α = {alpha}"
    )

plt.xlabel("Iteration")
plt.ylabel("MSE Loss")

plt.title("Gradient Descent - First 30 Iterations")

plt.legend()

plt.grid(True)

plt.show()