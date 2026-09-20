import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# MOMENTUM GRADIENT DESCENT
# Experiment: Effect of beta (momentum coefficient)
# ============================================================

# ------------------------------------------------------------
# 1. Training Data
# ------------------------------------------------------------

X = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y = np.array([35, 42, 51, 58, 66, 73], dtype=float)

n = len(X)


# ------------------------------------------------------------
# 2. Model
# ------------------------------------------------------------

def predict(X, b0, b1):
    """
    Linear regression model:

        y_hat = b0 + b1*x
    """
    return b0 + b1 * X


# ------------------------------------------------------------
# 3. Loss Function
# ------------------------------------------------------------

def mse(X, y, b0, b1):
    """
    Mean Squared Error:

             1
        J = --- sum (y_hat - y)^2
             n
    """

    y_pred = predict(X, b0, b1)

    return np.mean((y_pred - y) ** 2)


# ------------------------------------------------------------
# 4. Gradient
# ------------------------------------------------------------

def gradients(X, y, b0, b1):
    """
    Gradients of MSE.

        dJ/db0 = (2/n) sum(y_hat - y)

        dJ/db1 = (2/n) sum((y_hat - y)x)
    """

    y_pred = predict(X, b0, b1)

    error = y_pred - y

    db0 = (2 / n) * np.sum(error)

    db1 = (2 / n) * np.sum(error * X)

    return db0, db1


# ------------------------------------------------------------
# 5. Momentum Gradient Descent
# ------------------------------------------------------------

def momentum_gradient_descent(
    X,
    y,
    alpha=0.001,
    beta=0.9,
    iterations=1000
):
    """
    Momentum Gradient Descent

        v_b0 = beta*v_b0 - alpha*dJ/db0
        v_b1 = beta*v_b1 - alpha*dJ/db1

        b0 = b0 + v_b0
        b1 = b1 + v_b1
    """

    # Initial parameters
    b0 = 0.0
    b1 = 0.0

    # Initial velocities
    v_b0 = 0.0
    v_b1 = 0.0

    # Store history
    loss_history = []
    b0_history = []
    b1_history = []

    for iteration in range(iterations):

        # ----------------------------------------
        # Calculate gradients
        # ----------------------------------------

        db0, db1 = gradients(X, y, b0, b1)

        # ----------------------------------------
        # Momentum update
        # ----------------------------------------

        v_b0 = beta * v_b0 - alpha * db0

        v_b1 = beta * v_b1 - alpha * db1

        # ----------------------------------------
        # Parameter update
        # ----------------------------------------

        b0 = b0 + v_b0

        b1 = b1 + v_b1

        # ----------------------------------------
        # Calculate loss
        # ----------------------------------------

        current_loss = mse(X, y, b0, b1)

        # Store history
        loss_history.append(current_loss)
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
# 6. Experiment Parameters
# ============================================================

alpha = 0.001

betas = [
    0.0,
    0.5,
    0.9,
    0.99
]

iterations = 1000


# ============================================================
# 7. Run Experiments
# ============================================================

results = {}

for beta in betas:

    b0, b1, losses, b0_history, b1_history = (
        momentum_gradient_descent(
            X,
            y,
            alpha=alpha,
            beta=beta,
            iterations=iterations
        )
    )

    results[beta] = {
        "b0": b0,
        "b1": b1,
        "losses": losses,
        "b0_history": b0_history,
        "b1_history": b1_history
    }


# ============================================================
# 8. Print Results
# ============================================================

print("=" * 60)
print("MOMENTUM GRADIENT DESCENT - BETA EXPERIMENT")
print("=" * 60)

print(f"\nLearning rate alpha = {alpha}")
print(f"Iterations = {iterations}")

print("\nTraining data:")
print("X =", X)
print("y =", y)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

for beta in betas:

    result = results[beta]

    print(f"\nBeta = {beta}")

    print(f"Final b0  = {result['b0']:.6f}")

    print(f"Final b1  = {result['b1']:.6f}")

    print(f"Final MSE = {result['losses'][-1]:.6f}")


# ============================================================
# 9. Plot 1 - Loss vs Iteration
# ============================================================

plt.figure(figsize=(10, 6))

for beta in betas:

    losses = results[beta]["losses"]

    plt.plot(
        range(iterations),
        losses,
        label=f"β = {beta}"
    )

plt.xlabel("Iteration")
plt.ylabel("MSE Loss")

plt.title(
    "Momentum Gradient Descent - Effect of β"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 10. Plot 2 - First 100 Iterations
# ============================================================

plt.figure(figsize=(10, 6))

for beta in betas:

    losses = results[beta]["losses"]

    plt.plot(
        range(100),
        losses[:100],
        marker="o",
        markersize=2,
        label=f"β = {beta}"
    )

plt.xlabel("Iteration")
plt.ylabel("MSE Loss")

plt.title(
    "Momentum Gradient Descent - First 100 Iterations"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 11. Plot 3 - Parameter Trajectory
# ============================================================

plt.figure(figsize=(10, 6))

for beta in betas:

    b0_history = results[beta]["b0_history"]

    b1_history = results[beta]["b1_history"]

    plt.plot(
        b0_history,
        b1_history,
        label=f"β = {beta}"
    )

    # Mark starting point
    plt.scatter(
        b0_history[0],
        b1_history[0],
        marker="o"
    )

    # Mark final point
    plt.scatter(
        b0_history[-1],
        b1_history[-1],
        marker="x"
    )


plt.xlabel("b0 (Intercept)")

plt.ylabel("b1 (Slope)")

plt.title(
    "Parameter Trajectory for Different β Values"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 12. Plot 4 - Final Regression Lines
# ============================================================

plt.figure(figsize=(10, 6))

# Actual training data
plt.scatter(
    X,
    y,
    label="Actual Data"
)

# Plot regression line for each beta

x_line = np.linspace(
    X.min(),
    X.max(),
    100
)

for beta in betas:

    b0 = results[beta]["b0"]

    b1 = results[beta]["b1"]

    y_line = predict(
        x_line,
        b0,
        b1
    )

    plt.plot(
        x_line,
        y_line,
        label=f"β = {beta}"
    )


plt.xlabel("Hours Studied")

plt.ylabel("Exam Score")

plt.title(
    "Final Regression Lines for Different β"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 13. Compare Final Loss
# ============================================================

print("\n" + "=" * 60)
print("FINAL LOSS COMPARISON")
print("=" * 60)

for beta in betas:

    final_loss = results[beta]["losses"][-1]

    print(
        f"β = {beta:<4} "
        f"Final MSE = {final_loss:.8f}"
    )


# ============================================================
# 14. Interpretation
# ============================================================

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

print("""
β controls how much previous velocity is retained.

β = 0
    -> No momentum
    -> Equivalent to vanilla gradient descent

β = 0.5
    -> Moderate momentum

β = 0.9
    -> Strong momentum
    -> Usually moves faster toward the minimum

β = 0.99
    -> Very strong momentum
    -> Can cause overshooting and oscillation

Momentum update:

    v(k+1) = β v(k) - α ∇J(θ(k))

    θ(k+1) = θ(k) + v(k+1)

The experiment shows the trade-off between:

    fast convergence
            vs.
    excessive momentum / oscillation
""")