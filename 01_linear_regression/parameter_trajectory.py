import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PARAMETER TRAJECTORY:
# ORIGINAL vs SCALED FEATURES
# ============================================================

X = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y = np.array([35, 42, 51, 58, 66, 73], dtype=float)

n = len(X)


# ============================================================
# 1. Feature scaling
# ============================================================

mean_X = np.mean(X)
std_X = np.std(X)

X_scaled = (X - mean_X) / std_X


# ============================================================
# 2. Gradient Descent
# ============================================================

def gradient_descent(X, y, alpha, iterations):

    b0 = 0.0
    b1 = 0.0

    b0_history = []
    b1_history = []
    losses = []

    for i in range(iterations):

        # Prediction
        y_pred = b0 + b1 * X

        # Error
        error = y_pred - y

        # MSE
        mse = np.mean(error ** 2)

        # Store current point
        b0_history.append(b0)
        b1_history.append(b1)
        losses.append(mse)

        # Gradients
        db0 = (2 / n) * np.sum(error)
        db1 = (2 / n) * np.sum(error * X)

        # Update
        b0 = b0 - alpha * db0
        b1 = b1 - alpha * db1

    return (
        np.array(b0_history),
        np.array(b1_history),
        np.array(losses),
        b0,
        b1
    )


# ============================================================
# 3. Run gradient descent
# ============================================================

iterations = 100

# Original feature
b0_orig, b1_orig, loss_orig, final_b0_orig, final_b1_orig = \
    gradient_descent(
        X,
        y,
        alpha=0.01,
        iterations=iterations
    )


# Scaled feature
b0_scaled, b1_scaled, loss_scaled, final_b0_scaled, final_b1_scaled = \
    gradient_descent(
        X_scaled,
        y,
        alpha=0.1,
        iterations=iterations
    )


# ============================================================
# 4. Print results
# ============================================================

print("=" * 60)
print("PARAMETER TRAJECTORY")
print("=" * 60)

print("\nOriginal feature:")
print(f"Final b0 = {final_b0_orig:.6f}")
print(f"Final b1 = {final_b1_orig:.6f}")
print(f"Final MSE = {loss_orig[-1]:.6f}")

print("\nScaled feature:")
print(f"Final b0 = {final_b0_scaled:.6f}")
print(f"Final b1 = {final_b1_scaled:.6f}")
print(f"Final MSE = {loss_scaled[-1]:.6f}")


# ============================================================
# 5. Find optimal parameters analytically
# ============================================================

X_matrix = np.column_stack((np.ones(n), X))

theta_original = np.linalg.inv(
    X_matrix.T @ X_matrix
) @ X_matrix.T @ y

X_scaled_matrix = np.column_stack(
    (np.ones(n), X_scaled)
)

theta_scaled = np.linalg.inv(
    X_scaled_matrix.T @ X_scaled_matrix
) @ X_scaled_matrix.T @ y


print("\n" + "=" * 60)
print("ANALYTICAL OPTIMUM")
print("=" * 60)

print("\nOriginal:")
print(f"b0* = {theta_original[0]:.6f}")
print(f"b1* = {theta_original[1]:.6f}")

print("\nScaled:")
print(f"b0* = {theta_scaled[0]:.6f}")
print(f"b1* = {theta_scaled[1]:.6f}")


# ============================================================
# 6. Parameter trajectory
# ============================================================

plt.figure(figsize=(10, 7))

plt.plot(
    b0_orig,
    b1_orig,
    marker="o",
    markersize=3,
    label="Original Feature"
)

plt.plot(
    b0_scaled,
    b1_scaled,
    marker="o",
    markersize=3,
    label="Scaled Feature"
)


# Original optimum
plt.scatter(
    theta_original[0],
    theta_original[1],
    s=150,
    marker="*",
    label="Original Optimum"
)


# Scaled optimum
plt.scatter(
    theta_scaled[0],
    theta_scaled[1],
    s=150,
    marker="*",
    label="Scaled Optimum"
)


# Starting point
plt.scatter(
    0,
    0,
    s=100,
    marker="x",
    label="Starting Point"
)


plt.xlabel("b0 (Intercept)")
plt.ylabel("b1 (Slope)")

plt.title(
    "Gradient Descent Parameter Trajectory"
)

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 7. Loss comparison
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    loss_orig,
    label="Original Feature"
)

plt.plot(
    loss_scaled,
    label="Scaled Feature"
)

plt.xlabel("Iteration")
plt.ylabel("MSE Loss")

plt.title(
    "Loss Convergence: Original vs Scaled"
)

plt.legend()
plt.grid(True)

plt.show()