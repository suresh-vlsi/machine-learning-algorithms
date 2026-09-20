import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. DATASET
# ============================================================

X = np.array([1, 2, 3, 4, 5, 6], dtype=float)

y = np.array([35, 42, 51, 58, 66, 73], dtype=float)

n = len(X)


# ============================================================
# 2. MODEL
# ============================================================

def predict(X, b0, b1):
    """
    Linear regression model:

        y_hat = b0 + b1*x
    """
    return b0 + b1 * X


# ============================================================
# 3. LOSS FUNCTION
# ============================================================

def mse(X, y, b0, b1):
    """
    Mean Squared Error:

             1
        J = --- sum((y_hat - y)^2)
             n
    """

    y_pred = predict(X, b0, b1)

    return np.mean((y_pred - y) ** 2)


# ============================================================
# 4. GRADIENT
# ============================================================

def compute_gradient(X, y, b0, b1):
    """
    Derivatives of MSE:

        dJ/db0 = (2/n) * sum(y_hat - y)

        dJ/db1 = (2/n) * sum((y_hat - y)*x)
    """

    y_pred = predict(X, b0, b1)

    error = y_pred - y

    db0 = (2 / n) * np.sum(error)

    db1 = (2 / n) * np.sum(error * X)

    return db0, db1


# ============================================================
# 5. VANILLA GRADIENT DESCENT
# ============================================================

def gradient_descent(
    X,
    y,
    learning_rate=0.001,
    iterations=1000
):

    # Initial parameters
    b0 = 0.0
    b1 = 0.0

    # Store history
    losses = []

    b0_history = []
    b1_history = []

    velocity_history = []

    for iteration in range(iterations):

        # ----------------------------------------------------
        # Current loss
        # ----------------------------------------------------

        current_loss = mse(X, y, b0, b1)

        losses.append(current_loss)

        b0_history.append(b0)
        b1_history.append(b1)

        # ----------------------------------------------------
        # Compute gradient
        # ----------------------------------------------------

        db0, db1 = compute_gradient(
            X,
            y,
            b0,
            b1
        )

        # ----------------------------------------------------
        # Gradient descent update
        # ----------------------------------------------------

        b0 = b0 - learning_rate * db0

        b1 = b1 - learning_rate * db1

        # Vanilla GD has no velocity
        velocity_history.append(0.0)

    return (
        b0,
        b1,
        losses,
        b0_history,
        b1_history,
        velocity_history
    )


# ============================================================
# 6. MOMENTUM GRADIENT DESCENT
# ============================================================

def momentum_gradient_descent(
    X,
    y,
    learning_rate=0.001,
    beta=0.9,
    iterations=1000
):

    # --------------------------------------------------------
    # Initial parameters
    # --------------------------------------------------------

    b0 = 0.0
    b1 = 0.0

    # --------------------------------------------------------
    # Initial velocity
    # --------------------------------------------------------

    v0 = 0.0
    v1 = 0.0

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    losses = []

    b0_history = []
    b1_history = []

    velocity_history = []

    for iteration in range(iterations):

        # ----------------------------------------------------
        # Current loss
        # ----------------------------------------------------

        current_loss = mse(X, y, b0, b1)

        losses.append(current_loss)

        b0_history.append(b0)
        b1_history.append(b1)

        # ----------------------------------------------------
        # Gradient
        # ----------------------------------------------------

        db0, db1 = compute_gradient(
            X,
            y,
            b0,
            b1
        )

        # ----------------------------------------------------
        # Momentum equations
        #
        # v(k+1) = beta*v(k) - alpha*gradient
        #
        # theta(k+1) = theta(k) + v(k+1)
        # ----------------------------------------------------

        v0 = beta * v0 - learning_rate * db0

        v1 = beta * v1 - learning_rate * db1

        # ----------------------------------------------------
        # Parameter update
        # ----------------------------------------------------

        b0 = b0 + v0

        b1 = b1 + v1

        # ----------------------------------------------------
        # Store velocity magnitude
        # ----------------------------------------------------

        velocity = np.sqrt(v0**2 + v1**2)

        velocity_history.append(velocity)

    return (
        b0,
        b1,
        losses,
        b0_history,
        b1_history,
        velocity_history
    )


# ============================================================
# 7. TRAIN BOTH MODELS
# ============================================================

learning_rate = 0.001

beta = 0.9

iterations = 1000


# Vanilla GD

(
    gd_b0,
    gd_b1,
    gd_losses,
    gd_b0_history,
    gd_b1_history,
    gd_velocity
) = gradient_descent(
    X,
    y,
    learning_rate=learning_rate,
    iterations=iterations
)


# Momentum GD

(
    mom_b0,
    mom_b1,
    mom_losses,
    mom_b0_history,
    mom_b1_history,
    mom_velocity
) = momentum_gradient_descent(
    X,
    y,
    learning_rate=learning_rate,
    beta=beta,
    iterations=iterations
)


# ============================================================
# 8. FINAL RESULTS
# ============================================================

print("=" * 60)
print("MOMENTUM GRADIENT DESCENT")
print("=" * 60)

print("\nDataset:")
print("X =", X)
print("y =", y)

print("\nHyperparameters:")
print("Learning rate =", learning_rate)
print("Momentum beta =", beta)
print("Iterations =", iterations)


print("\n" + "-" * 60)
print("VANILLA GRADIENT DESCENT")
print("-" * 60)

print("Final b0 =", gd_b0)
print("Final b1 =", gd_b1)
print("Final MSE =", gd_losses[-1])


print("\n" + "-" * 60)
print("MOMENTUM GRADIENT DESCENT")
print("-" * 60)

print("Final b0 =", mom_b0)
print("Final b1 =", mom_b1)
print("Final MSE =", mom_losses[-1])


# ============================================================
# 9. FINAL PREDICTIONS
# ============================================================

gd_predictions = predict(
    X,
    gd_b0,
    gd_b1
)

mom_predictions = predict(
    X,
    mom_b0,
    mom_b1
)


print("\n" + "-" * 60)
print("PREDICTIONS")
print("-" * 60)

print("\nActual:")
print(y)

print("\nGD Predictions:")
print(gd_predictions)

print("\nMomentum Predictions:")
print(mom_predictions)


# ============================================================
# 10. PLOT 1
# LOSS VS ITERATION
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    gd_losses,
    label="Vanilla Gradient Descent"
)

plt.plot(
    mom_losses,
    label="Momentum Gradient Descent"
)

plt.xlabel("Iteration")

plt.ylabel("MSE Loss")

plt.title(
    "Gradient Descent vs Momentum - Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 11. PLOT 2
# REGRESSION LINE COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    X,
    y,
    label="Actual Data"
)

plt.plot(
    X,
    gd_predictions,
    label="Vanilla GD"
)

plt.plot(
    X,
    mom_predictions,
    label="Momentum GD"
)

plt.xlabel("Hours Studied")

plt.ylabel("Exam Score")

plt.title(
    "Vanilla GD vs Momentum GD"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 12. PLOT 3
# PARAMETER TRAJECTORY
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    gd_b0_history,
    gd_b1_history,
    label="Vanilla GD"
)

plt.plot(
    mom_b0_history,
    mom_b1_history,
    label="Momentum GD"
)

plt.scatter(
    gd_b0_history[0],
    gd_b1_history[0],
    marker="o",
    s=100,
    label="Starting Point"
)

plt.scatter(
    gd_b0,
    gd_b1,
    marker="*",
    s=200,
    label="GD Final"
)

plt.scatter(
    mom_b0,
    mom_b1,
    marker="*",
    s=200,
    label="Momentum Final"
)

plt.xlabel("b0 (Intercept)")

plt.ylabel("b1 (Slope)")

plt.title(
    "Parameter Trajectory: GD vs Momentum"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 13. PLOT 4
# VELOCITY MAGNITUDE
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    gd_velocity,
    label="Vanilla GD"
)

plt.plot(
    mom_velocity,
    label="Momentum GD"
)

plt.xlabel("Iteration")

plt.ylabel("Velocity Magnitude")

plt.title(
    "Velocity During Optimization"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 14. COMPARE LOSS AFTER DIFFERENT ITERATIONS
# ============================================================

checkpoints = [
    0,
    1,
    5,
    10,
    20,
    50,
    100,
    200,
    500,
    999
]

print("\n" + "=" * 60)
print("LOSS COMPARISON")
print("=" * 60)

print(
    f"{'Iteration':>10}"
    f"{'GD Loss':>20}"
    f"{'Momentum Loss':>20}"
)

for i in checkpoints:

    print(
        f"{i:>10}"
        f"{gd_losses[i]:>20.6f}"
        f"{mom_losses[i]:>20.6f}"
    )


# ============================================================
# 15. FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

print("""
Vanilla Gradient Descent:
    theta(k+1) = theta(k) - alpha * gradient

Momentum Gradient Descent:
    v(k+1)     = beta*v(k) - alpha*gradient
    theta(k+1) = theta(k) + v(k)

Momentum remembers previous updates.

Therefore:

    beta = 0
        -> ordinary gradient descent

    beta > 0
        -> momentum is introduced

    beta close to 1
        -> stronger memory

Momentum can accelerate movement along directions
where the gradient consistently points the same way.
""")