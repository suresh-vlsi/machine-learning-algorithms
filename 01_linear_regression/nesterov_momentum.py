import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# NESTEROV ACCELERATED GRADIENT (NAG)
# Linear Regression from Scratch
#
# Comparison:
#   1. Vanilla Gradient Descent
#   2. Momentum Gradient Descent
#   3. Nesterov Accelerated Gradient
# ============================================================


# ============================================================
# 1. TRAINING DATA
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
        J = --- sum (y_hat - y)^2
             n
    """

    y_pred = predict(X, b0, b1)

    error = y_pred - y

    return np.mean(error ** 2)


# ============================================================
# 4. GRADIENT
# ============================================================

def gradients(X, y, b0, b1):
    """
    Gradient of MSE:

        dJ/db0 = (2/n) sum(y_hat - y)

        dJ/db1 = (2/n) sum((y_hat - y)x)
    """

    y_pred = predict(X, b0, b1)

    error = y_pred - y

    db0 = (2 / n) * np.sum(error)

    db1 = (2 / n) * np.sum(error * X)

    return db0, db1


# ============================================================
# 5. VANILLA GRADIENT DESCENT
# ============================================================

def vanilla_gradient_descent(
    X,
    y,
    alpha=0.001,
    iterations=1000
):
    """
    Standard gradient descent:

        theta(k+1)
        =
        theta(k) - alpha * gradient
    """

    b0 = 0.0
    b1 = 0.0

    loss_history = []

    b0_history = []
    b1_history = []

    for iteration in range(iterations):

        # ----------------------------------------
        # Calculate gradient at current position
        # ----------------------------------------

        db0, db1 = gradients(
            X,
            y,
            b0,
            b1
        )

        # ----------------------------------------
        # Parameter update
        # ----------------------------------------

        b0 = b0 - alpha * db0

        b1 = b1 - alpha * db1

        # ----------------------------------------
        # Calculate loss
        # ----------------------------------------

        loss = mse(
            X,
            y,
            b0,
            b1
        )

        # Store history

        loss_history.append(loss)

        b0_history.append(b0)

        b1_history.append(b1)

    return (
        b0,
        b1,
        np.array(loss_history),
        np.array(b0_history),
        np.array(b1_history)
    )


# ============================================================
# 6. MOMENTUM GRADIENT DESCENT
# ============================================================

def momentum_gradient_descent(
    X,
    y,
    alpha=0.001,
    beta=0.9,
    iterations=1000
):
    """
    Momentum Gradient Descent:

        v(k+1)
        =
        beta*v(k)
        -
        alpha*gradient(theta(k))

        theta(k+1)
        =
        theta(k) + v(k+1)
    """

    b0 = 0.0
    b1 = 0.0

    # Initial velocities

    v_b0 = 0.0
    v_b1 = 0.0

    loss_history = []

    b0_history = []
    b1_history = []

    for iteration in range(iterations):

        # ----------------------------------------
        # Gradient at current position
        # ----------------------------------------

        db0, db1 = gradients(
            X,
            y,
            b0,
            b1
        )

        # ----------------------------------------
        # Momentum update
        # ----------------------------------------

        v_b0 = (
            beta * v_b0
            -
            alpha * db0
        )

        v_b1 = (
            beta * v_b1
            -
            alpha * db1
        )

        # ----------------------------------------
        # Parameter update
        # ----------------------------------------

        b0 = b0 + v_b0

        b1 = b1 + v_b1

        # ----------------------------------------
        # Loss
        # ----------------------------------------

        loss = mse(
            X,
            y,
            b0,
            b1
        )

        loss_history.append(loss)

        b0_history.append(b0)

        b1_history.append(b1)

    return (
        b0,
        b1,
        np.array(loss_history),
        np.array(b0_history),
        np.array(b1_history)
    )


# ============================================================
# 7. NESTEROV ACCELERATED GRADIENT
# ============================================================

def nesterov_gradient_descent(
    X,
    y,
    alpha=0.001,
    beta=0.9,
    iterations=1000
):
    """
    Nesterov Accelerated Gradient:

    STEP 1:
        Look ahead:

        theta_lookahead
        =
        theta + beta*v

    STEP 2:
        Calculate gradient at look-ahead point:

        g =
        gradient(theta_lookahead)

    STEP 3:
        Update velocity:

        v(k+1)
        =
        beta*v(k)
        -
        alpha*g

    STEP 4:
        Update parameters:

        theta(k+1)
        =
        theta(k) + v(k+1)
    """

    # Initial parameters

    b0 = 0.0
    b1 = 0.0

    # Initial velocities

    v_b0 = 0.0
    v_b1 = 0.0

    loss_history = []

    b0_history = []
    b1_history = []

    for iteration in range(iterations):

        # ==================================================
        # STEP 1: LOOK AHEAD
        # ==================================================

        lookahead_b0 = b0 + beta * v_b0

        lookahead_b1 = b1 + beta * v_b1

        # ==================================================
        # STEP 2: GRADIENT AT LOOK-AHEAD POSITION
        # ==================================================

        db0, db1 = gradients(
            X,
            y,
            lookahead_b0,
            lookahead_b1
        )

        # ==================================================
        # STEP 3: UPDATE VELOCITY
        # ==================================================

        v_b0 = (
            beta * v_b0
            -
            alpha * db0
        )

        v_b1 = (
            beta * v_b1
            -
            alpha * db1
        )

        # ==================================================
        # STEP 4: UPDATE PARAMETERS
        # ==================================================

        b0 = b0 + v_b0

        b1 = b1 + v_b1

        # ==================================================
        # CALCULATE LOSS AT ACTUAL PARAMETERS
        # ==================================================

        loss = mse(
            X,
            y,
            b0,
            b1
        )

        loss_history.append(loss)

        b0_history.append(b0)

        b1_history.append(b1)

    return (
        b0,
        b1,
        np.array(loss_history),
        np.array(b0_history),
        np.array(b1_history)
    )


# ============================================================
# 8. EXPERIMENT PARAMETERS
# ============================================================

alpha = 0.001

beta = 0.9

iterations = 1000


# ============================================================
# 9. RUN VANILLA GD
# ============================================================

(
    vanilla_b0,
    vanilla_b1,
    vanilla_losses,
    vanilla_b0_history,
    vanilla_b1_history
) = vanilla_gradient_descent(
    X,
    y,
    alpha=alpha,
    iterations=iterations
)


# ============================================================
# 10. RUN MOMENTUM GD
# ============================================================

(
    momentum_b0,
    momentum_b1,
    momentum_losses,
    momentum_b0_history,
    momentum_b1_history
) = momentum_gradient_descent(
    X,
    y,
    alpha=alpha,
    beta=beta,
    iterations=iterations
)


# ============================================================
# 11. RUN NESTEROV GD
# ============================================================

(
    nesterov_b0,
    nesterov_b1,
    nesterov_losses,
    nesterov_b0_history,
    nesterov_b1_history
) = nesterov_gradient_descent(
    X,
    y,
    alpha=alpha,
    beta=beta,
    iterations=iterations
)


# ============================================================
# 12. PRINT FINAL RESULTS
# ============================================================

print("=" * 65)

print("NESTEROV ACCELERATED GRADIENT")
print("LINEAR REGRESSION COMPARISON")

print("=" * 65)

print(f"\nLearning rate α = {alpha}")

print(f"Momentum β = {beta}")

print(f"Iterations = {iterations}")


# ------------------------------------------------------------
# Vanilla GD
# ------------------------------------------------------------

print("\n" + "-" * 65)

print("VANILLA GRADIENT DESCENT")

print("-" * 65)

print(
    f"Final b0  = {vanilla_b0:.8f}"
)

print(
    f"Final b1  = {vanilla_b1:.8f}"
)

print(
    f"Final MSE = {vanilla_losses[-1]:.8f}"
)


# ------------------------------------------------------------
# Momentum
# ------------------------------------------------------------

print("\n" + "-" * 65)

print("MOMENTUM GRADIENT DESCENT")

print("-" * 65)

print(
    f"Final b0  = {momentum_b0:.8f}"
)

print(
    f"Final b1  = {momentum_b1:.8f}"
)

print(
    f"Final MSE = {momentum_losses[-1]:.8f}"
)


# ------------------------------------------------------------
# Nesterov
# ------------------------------------------------------------

print("\n" + "-" * 65)

print("NESTEROV ACCELERATED GRADIENT")

print("-" * 65)

print(
    f"Final b0  = {nesterov_b0:.8f}"
)

print(
    f"Final b1  = {nesterov_b1:.8f}"
)

print(
    f"Final MSE = {nesterov_losses[-1]:.8f}"
)


# ============================================================
# 13. PLOT 1
# LOSS VS ITERATION
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    vanilla_losses,
    label="Vanilla Gradient Descent"
)

plt.plot(
    momentum_losses,
    label="Momentum Gradient Descent"
)

plt.plot(
    nesterov_losses,
    label="Nesterov Accelerated Gradient"
)

plt.xlabel("Iteration")

plt.ylabel("MSE Loss")

plt.title(
    "Vanilla GD vs Momentum vs Nesterov"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 14. PLOT 2
# FIRST 100 ITERATIONS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    vanilla_losses[:100],
    marker="o",
    markersize=2,
    label="Vanilla GD"
)

plt.plot(
    momentum_losses[:100],
    marker="o",
    markersize=2,
    label="Momentum GD"
)

plt.plot(
    nesterov_losses[:100],
    marker="o",
    markersize=2,
    label="Nesterov GD"
)

plt.xlabel("Iteration")

plt.ylabel("MSE Loss")

plt.title(
    "First 100 Iterations - Optimization Comparison"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 15. PLOT 3
# PARAMETER TRAJECTORIES
# ============================================================

plt.figure(figsize=(10, 7))

plt.plot(
    vanilla_b0_history,
    vanilla_b1_history,
    label="Vanilla GD"
)

plt.plot(
    momentum_b0_history,
    momentum_b1_history,
    label="Momentum GD"
)

plt.plot(
    nesterov_b0_history,
    nesterov_b1_history,
    label="Nesterov GD"
)


# Starting point

plt.scatter(
    0,
    0,
    marker="o",
    s=100,
    label="Starting Point"
)


# Final points

plt.scatter(
    vanilla_b0,
    vanilla_b1,
    marker="x",
    s=100
)

plt.scatter(
    momentum_b0,
    momentum_b1,
    marker="x",
    s=100
)

plt.scatter(
    nesterov_b0,
    nesterov_b1,
    marker="x",
    s=100
)


plt.xlabel("b0 (Intercept)")

plt.ylabel("b1 (Slope)")

plt.title(
    "Parameter Trajectories"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 16. PLOT 4
# FINAL REGRESSION LINES
# ============================================================

plt.figure(figsize=(10, 6))

# Actual data

plt.scatter(
    X,
    y,
    label="Actual Data"
)


# Smooth x-axis

x_line = np.linspace(
    X.min(),
    X.max(),
    100
)


# Vanilla line

y_vanilla = predict(
    x_line,
    vanilla_b0,
    vanilla_b1
)

plt.plot(
    x_line,
    y_vanilla,
    label="Vanilla GD"
)


# Momentum line

y_momentum = predict(
    x_line,
    momentum_b0,
    momentum_b1
)

plt.plot(
    x_line,
    y_momentum,
    label="Momentum GD"
)


# Nesterov line

y_nesterov = predict(
    x_line,
    nesterov_b0,
    nesterov_b1
)

plt.plot(
    x_line,
    y_nesterov,
    label="Nesterov GD"
)


plt.xlabel("Hours Studied")

plt.ylabel("Exam Score")

plt.title(
    "Final Regression Lines"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 17. CONVERGENCE COMPARISON
# ============================================================

print("\n" + "=" * 65)

print("CONVERGENCE COMPARISON")

print("=" * 65)

print(
    f"Vanilla GD final MSE   = "
    f"{vanilla_losses[-1]:.8f}"
)

print(
    f"Momentum GD final MSE  = "
    f"{momentum_losses[-1]:.8f}"
)

print(
    f"Nesterov GD final MSE  = "
    f"{nesterov_losses[-1]:.8f}"
)


# ============================================================
# 18. LOSS AT SELECTED ITERATIONS
# ============================================================

print("\n" + "=" * 65)

print("LOSS AT SELECTED ITERATIONS")

print("=" * 65)

selected_iterations = [
    0,
    1,
    5,
    10,
    25,
    50,
    100,
    500,
    999
]


print(
    "\nIteration"
    "      Vanilla GD"
    "      Momentum GD"
    "      Nesterov GD"
)

print("-" * 65)


for i in selected_iterations:

    print(
        f"{i:8d}"
        f"{vanilla_losses[i]:18.6f}"
        f"{momentum_losses[i]:18.6f}"
        f"{nesterov_losses[i]:18.6f}"
    )


# ============================================================
# 19. MATHEMATICAL SUMMARY
# ============================================================

print("\n" + "=" * 65)

print("MATHEMATICAL SUMMARY")

print("=" * 65)

print("""
VANILLA GRADIENT DESCENT:

    theta(k+1)
    =
    theta(k) - alpha * grad J(theta(k))


MOMENTUM:

    v(k+1)
    =
    beta*v(k)
    -
    alpha*grad J(theta(k))

    theta(k+1)
    =
    theta(k) + v(k+1)


NESTEROV:

    theta_lookahead
    =
    theta(k) + beta*v(k)

    g
    =
    grad J(theta_lookahead)

    v(k+1)
    =
    beta*v(k)
    -
    alpha*g

    theta(k+1)
    =
    theta(k) + v(k+1)


KEY DIFFERENCE:

    Momentum:
        gradient is evaluated at current position.

    Nesterov:
        gradient is evaluated at the look-ahead position.
""")