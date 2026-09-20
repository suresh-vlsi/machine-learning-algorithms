import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# FEATURE SCALING AND CONDITIONING IN LINEAR REGRESSION
# ============================================================

print("=" * 60)
print("FEATURE SCALING AND CONDITIONING")
print("=" * 60)


# ------------------------------------------------------------
# 1. Dataset
# ------------------------------------------------------------

X = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y = np.array([35, 42, 51, 58, 66, 73], dtype=float)

n = len(X)


# ------------------------------------------------------------
# 2. Standardize the feature
# ------------------------------------------------------------

mean_X = np.mean(X)
std_X = np.std(X)

X_scaled = (X - mean_X) / std_X


print("\nOriginal feature:")
print(X)

print("\nMean:")
print(mean_X)

print("\nStandard deviation:")
print(std_X)

print("\nScaled feature:")
print(X_scaled)


# ------------------------------------------------------------
# 3. Construct Hessian for original feature
# ------------------------------------------------------------

H_original = (2 / n) * np.array([
    [n, np.sum(X)],
    [np.sum(X), np.sum(X ** 2)]
])


# ------------------------------------------------------------
# 4. Construct Hessian for scaled feature
# ------------------------------------------------------------

H_scaled = (2 / n) * np.array([
    [n, np.sum(X_scaled)],
    [np.sum(X_scaled), np.sum(X_scaled ** 2)]
])


print("\n" + "=" * 60)
print("HESSIAN MATRICES")
print("=" * 60)

print("\nOriginal Hessian:")
print(H_original)

print("\nScaled Hessian:")
print(H_scaled)


# ------------------------------------------------------------
# 5. Eigenvalues
# ------------------------------------------------------------

eigen_original = np.linalg.eigvalsh(H_original)
eigen_scaled = np.linalg.eigvalsh(H_scaled)


print("\n" + "=" * 60)
print("EIGENVALUES")
print("=" * 60)

print("\nOriginal Hessian eigenvalues:")
print(eigen_original)

print("\nScaled Hessian eigenvalues:")
print(eigen_scaled)


# ------------------------------------------------------------
# 6. Condition number
# ------------------------------------------------------------

condition_original = (
    np.max(eigen_original) /
    np.min(eigen_original)
)

condition_scaled = (
    np.max(eigen_scaled) /
    np.min(eigen_scaled)
)


print("\n" + "=" * 60)
print("CONDITION NUMBER")
print("=" * 60)

print(f"\nOriginal condition number = {condition_original:.4f}")
print(f"Scaled condition number   = {condition_scaled:.4f}")


# ------------------------------------------------------------
# 7. Gradient-descent stability limits
# ------------------------------------------------------------

lambda_max_original = np.max(eigen_original)
lambda_max_scaled = np.max(eigen_scaled)

alpha_max_original = 2 / lambda_max_original
alpha_max_scaled = 2 / lambda_max_scaled


print("\n" + "=" * 60)
print("GRADIENT DESCENT STABILITY LIMIT")
print("=" * 60)

print(
    f"\nOriginal alpha < {alpha_max_original:.6f}"
)

print(
    f"Scaled alpha   < {alpha_max_scaled:.6f}"
)


# ------------------------------------------------------------
# 8. Gradient Descent function
# ------------------------------------------------------------

def gradient_descent(X, y, alpha, iterations):

    b0 = 0.0
    b1 = 0.0

    losses = []
    b0_history = []
    b1_history = []

    for iteration in range(iterations):

        # Prediction
        y_pred = b0 + b1 * X

        # Error
        error = y_pred - y

        # MSE
        mse = np.mean(error ** 2)

        # Store history
        losses.append(mse)
        b0_history.append(b0)
        b1_history.append(b1)

        # Gradients
        db0 = (2 / len(X)) * np.sum(error)
        db1 = (2 / len(X)) * np.sum(error * X)

        # Parameter update
        b0 = b0 - alpha * db0
        b1 = b1 - alpha * db1

    return (
        b0,
        b1,
        np.array(losses),
        np.array(b0_history),
        np.array(b1_history)
    )


# ------------------------------------------------------------
# 9. Run GD on original feature
# ------------------------------------------------------------

alpha_original = 0.01

(
    b0_original,
    b1_original,
    losses_original,
    b0_hist_original,
    b1_hist_original
) = gradient_descent(
    X,
    y,
    alpha_original,
    1000
)


# ------------------------------------------------------------
# 10. Run GD on scaled feature
# ------------------------------------------------------------

alpha_scaled = 0.1

(
    b0_scaled,
    b1_scaled,
    losses_scaled,
    b0_hist_scaled,
    b1_hist_scaled
) = gradient_descent(
    X_scaled,
    y,
    alpha_scaled,
    1000
)


# ------------------------------------------------------------
# 11. Print final results
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("GRADIENT DESCENT RESULTS")
print("=" * 60)

print("\nOriginal feature:")
print(f"b0 = {b0_original:.6f}")
print(f"b1 = {b1_original:.6f}")
print(f"Final MSE = {losses_original[-1]:.6f}")

print("\nScaled feature:")
print(f"b0 = {b0_scaled:.6f}")
print(f"b1 = {b1_scaled:.6f}")
print(f"Final MSE = {losses_scaled[-1]:.6f}")


# ============================================================
# PLOT 1: LOSS VS ITERATION
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    losses_original,
    label="Original Feature"
)

plt.plot(
    losses_scaled,
    label="Scaled Feature"
)

plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Feature Scaling - Loss vs Iteration")

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# PLOT 2: LOG LOSS
# ============================================================

plt.figure(figsize=(10, 6))

plt.semilogy(
    losses_original,
    label="Original Feature"
)

plt.semilogy(
    losses_scaled,
    label="Scaled Feature"
)

plt.xlabel("Iteration")
plt.ylabel("MSE Loss (log scale)")
plt.title("Feature Scaling - Convergence Comparison")

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# PLOT 3: ORIGINAL DATA AND REGRESSION
# ============================================================

y_pred_original = (
    b0_original +
    b1_original * X
)

plt.figure(figsize=(10, 6))

plt.scatter(
    X,
    y,
    label="Actual Data"
)

plt.plot(
    X,
    y_pred_original,
    label="Original Feature Regression"
)

plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")

plt.title("Linear Regression Using Original Feature")

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# PLOT 4: SCALED FEATURE REGRESSION
# ============================================================

y_pred_scaled = (
    b0_scaled +
    b1_scaled * X_scaled
)

plt.figure(figsize=(10, 6))

plt.scatter(
    X_scaled,
    y,
    label="Actual Data"
)

plt.plot(
    X_scaled,
    y_pred_scaled,
    label="Scaled Feature Regression"
)

plt.xlabel("Scaled Hours Studied")
plt.ylabel("Exam Score")

plt.title("Linear Regression Using Scaled Feature")

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(
    f"\nOriginal condition number : {condition_original:.4f}"
)

print(
    f"Scaled condition number   : {condition_scaled:.4f}"
)

print(
    f"\nOriginal stability limit : {alpha_max_original:.6f}"
)

print(
    f"Scaled stability limit   : {alpha_max_scaled:.6f}"
)

print("\nFeature scaling changes the geometry")
print("of the optimization problem.")