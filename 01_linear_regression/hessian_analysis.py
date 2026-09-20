import numpy as np


# ============================================================
# Training Data
# ============================================================

X = np.array([1, 2, 3, 4, 5, 6], dtype=float)

n = len(X)


# ============================================================
# Hessian of MSE
# ============================================================

H = (2 / n) * np.array([
    [n, np.sum(X)],
    [np.sum(X), np.sum(X ** 2)]
])


print("===================================")
print("Hessian Analysis")
print("===================================")

print("\nHessian H:")
print(H)


# ============================================================
# Eigenvalues
# ============================================================

eigenvalues = np.linalg.eigvalsh(H)

print("\nEigenvalues:")
print(eigenvalues)


# ============================================================
# Maximum Eigenvalue
# ============================================================

lambda_max = np.max(eigenvalues)

print("\nLargest eigenvalue:")
print(lambda_max)


# ============================================================
# Theoretical Stability Limit
# ============================================================

alpha_max = 2 / lambda_max

print("\nTheoretical stability limit:")
print("alpha <", alpha_max)


# ============================================================
# Test Learning Rates
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

print("\n===================================")
print("Learning Rate Classification")
print("===================================")

for alpha in learning_rates:

    if alpha < alpha_max:
        print(
            f"alpha = {alpha:<6} -> theoretically stable"
        )

    elif alpha == alpha_max:
        print(
            f"alpha = {alpha:<6} -> stability boundary"
        )

    else:
        print(
            f"alpha = {alpha:<6} -> theoretically unstable"
        )
        