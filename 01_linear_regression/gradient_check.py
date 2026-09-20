import numpy as np


# ============================================================
# Training Data
# ============================================================

X = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y = np.array([35, 42, 51, 58, 66, 73], dtype=float)


# ============================================================
# Loss Function
# ============================================================

def mse(b0, b1):

    y_pred = b0 + b1 * X

    return np.mean((y - y_pred) ** 2)


# ============================================================
# Gradient
# ============================================================

def gradient(b0, b1):

    y_pred = b0 + b1 * X

    n = len(X)

    db0 = (-2 / n) * np.sum(y - y_pred)

    db1 = (-2 / n) * np.sum(
        X * (y - y_pred)
    )

    return db0, db1


# ============================================================
# Starting Point
# ============================================================

b0 = 10.0
b1 = 5.0


# ============================================================
# Calculate Gradient
# ============================================================

db0, db1 = gradient(b0, b1)

current_loss = mse(b0, b1)


print("===================================")
print("Gradient Check")
print("===================================")

print("\nStarting parameters:")
print("b0 =", b0)
print("b1 =", b1)

print("\nCurrent MSE:")
print(current_loss)

print("\nGradient:")
print("dJ/db0 =", db0)
print("dJ/db1 =", db1)


# ============================================================
# Step Size
# ============================================================

alpha = 0.001


# ============================================================
# Move in +Gradient Direction
# ============================================================

b0_plus = b0 + alpha * db0
b1_plus = b1 + alpha * db1

loss_plus = mse(
    b0_plus,
    b1_plus
)


# ============================================================
# Move in -Gradient Direction
# ============================================================

b0_minus = b0 - alpha * db0
b1_minus = b1 - alpha * db1

loss_minus = mse(
    b0_minus,
    b1_minus
)


# ============================================================
# Results
# ============================================================

print("\n-----------------------------------")

print("\nMove in +Gradient direction:")

print("New b0 =", b0_plus)
print("New b1 =", b1_plus)
print("New MSE =", loss_plus)


print("\nMove in -Gradient direction:")

print("New b0 =", b0_minus)
print("New b1 =", b1_minus)
print("New MSE =", loss_minus)


# ============================================================
# Interpretation
# ============================================================

print("\n===================================")
print("Interpretation")
print("===================================")

if loss_plus > current_loss:
    print("+Gradient increased the loss.")

if loss_minus < current_loss:
    print("-Gradient decreased the loss.")

print("\nTherefore:")

print("Gradient  -> uphill")

print("-Gradient -> downhill")