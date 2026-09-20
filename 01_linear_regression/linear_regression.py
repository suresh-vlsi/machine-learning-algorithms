import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

print("Linear Regression from Scratch")
print("Gradient Descent Animation")


# ==========================================
# 1. Training Data
# ==========================================

X = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y = np.array([35, 42, 51, 58, 66, 73], dtype=float)


# ==========================================
# 2. Initial Parameters
# ==========================================

b0 = 0.0
b1 = 0.0

learning_rate = 0.001
epochs = 1000


# ==========================================
# 3. Store Training History
# ==========================================

b0_history = []
b1_history = []
loss_history = []


# ==========================================
# 4. Gradient Descent
# ==========================================

for epoch in range(epochs):

    # Predictions
    y_pred = b0 + b1 * X

    # Mean Squared Error
    mse = np.mean((y - y_pred) ** 2)

    # Store current state
    b0_history.append(b0)
    b1_history.append(b1)
    loss_history.append(mse)

    # Number of samples
    n = len(X)

    # Gradients
    db0 = (-2 / n) * np.sum(y - y_pred)

    db1 = (-2 / n) * np.sum(X * (y - y_pred))

    # Update parameters
    b0 = b0 - learning_rate * db0
    b1 = b1 - learning_rate * db1


# ==========================================
# 5. Print Final Model
# ==========================================

print("\nTraining Complete")

print("Final b0 =", b0)
print("Final b1 =", b1)

final_predictions = b0 + b1 * X

final_mse = np.mean((y - final_predictions) ** 2)

print("Final MSE =", final_mse)


# ==========================================
# 6. Create Animation Figure
# ==========================================

fig, ax = plt.subplots(figsize=(9, 6))


# Plot actual data

ax.scatter(
    X,
    y,
    label="Actual Data"
)


# Create empty regression line

line, = ax.plot(
    [],
    [],
    linewidth=2,
    label="Regression Line"
)


# ==========================================
# 7. Configure Graph
# ==========================================

ax.set_xlim(0.5, 6.5)
ax.set_ylim(0, 80)

ax.set_xlabel("Hours Studied")
ax.set_ylabel("Exam Score")

ax.set_title("Gradient Descent Learning")


ax.legend()


# ==========================================
# 8. Information Text
# ==========================================

info_text = ax.text(
    0.05,
    0.95,
    "",
    transform=ax.transAxes,
    verticalalignment="top"
)


# ==========================================
# 9. Animation Function
# ==========================================

def update(frame):

    current_b0 = b0_history[frame]

    current_b1 = b1_history[frame]

    current_mse = loss_history[frame]

    # Calculate current predictions

    current_predictions = (
        current_b0 +
        current_b1 * X
    )

    # Update regression line

    line.set_data(
        X,
        current_predictions
    )

    # Update information

    info_text.set_text(
        f"Iteration: {frame}\n"
        f"b0 = {current_b0:.4f}\n"
        f"b1 = {current_b1:.4f}\n"
        f"MSE = {current_mse:.4f}"
    )

    return line, info_text


# ==========================================
# 10. Create Animation
# ==========================================

animation = FuncAnimation(
    fig,
    update,
    frames=range(0, epochs, 5),
    interval=30,
    repeat=False
)


plt.show()