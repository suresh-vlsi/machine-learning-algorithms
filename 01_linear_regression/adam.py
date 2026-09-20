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
# ADAM
# ============================================================

b0 = 0.0
b1 = 0.0

alpha = 0.1

beta1 = 0.9
beta2 = 0.999

epsilon = 1e-8

# First moments
m_b0 = 0.0
m_b1 = 0.0

# Second moments
v_b0 = 0.0
v_b1 = 0.0

iterations = 1000

losses = []


for t in range(1, iterations + 1):

    # --------------------------------------------------------
    # Gradient
    # --------------------------------------------------------

    db0, db1 = gradients(b0, b1)


    # --------------------------------------------------------
    # First moment
    # --------------------------------------------------------

    m_b0 = beta1 * m_b0 + (1 - beta1) * db0

    m_b1 = beta1 * m_b1 + (1 - beta1) * db1


    # --------------------------------------------------------
    # Second moment
    # --------------------------------------------------------

    v_b0 = beta2 * v_b0 + (1 - beta2) * db0 ** 2

    v_b1 = beta2 * v_b1 + (1 - beta2) * db1 ** 2


    # --------------------------------------------------------
    # Bias correction
    # --------------------------------------------------------

    m_hat_b0 = m_b0 / (1 - beta1 ** t)

    m_hat_b1 = m_b1 / (1 - beta1 ** t)

    v_hat_b0 = v_b0 / (1 - beta2 ** t)

    v_hat_b1 = v_b1 / (1 - beta2 ** t)


    # --------------------------------------------------------
    # Parameter update
    # --------------------------------------------------------

    b0 -= alpha * m_hat_b0 / (
        np.sqrt(v_hat_b0) + epsilon
    )

    b1 -= alpha * m_hat_b1 / (
        np.sqrt(v_hat_b1) + epsilon
    )


    # --------------------------------------------------------
    # Store loss
    # --------------------------------------------------------

    losses.append(mse(b0, b1))


# ============================================================
# RESULTS
# ============================================================

print("=" * 50)
print("Adam")
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
plt.title("Adam - Loss vs Iteration")

plt.grid(True)
plt.show()


# ============================================================
# REGRESSION RESULT
# ============================================================

y_pred = b0 + b1 * X

plt.figure(figsize=(9, 6))

plt.scatter(X, y, label="Actual Data")

plt.plot(
    X,
    y_pred,
    label="Adam Regression Line"
)

plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")

plt.title("Adam Linear Regression")

plt.legend()
plt.grid(True)

plt.show()