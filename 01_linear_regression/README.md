# Linear Regression & Optimization from Scratch

A from-scratch implementation and mathematical study of **Linear Regression, Gradient Descent, Gradient Checking, Loss Surfaces, Learning-Rate Stability, Feature Scaling, Momentum, Nesterov Accelerated Gradient, AdaGrad, RMSProp, and Adam**.

The purpose of this project is to connect the mathematics of optimization directly to executable Python experiments rather than treating optimizers as black boxes.

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Dataset](#2-dataset)
- [3. Linear Regression](#3-linear-regression)
- [4. Mean Squared Error](#4-mean-squared-error)
- [5. Gradient](#5-gradient)
- [6. Gradient Descent](#6-gradient-descent)
- [7. Loss Curves and Convergence](#7-loss-curves-and-convergence)
- [8. Loss Surface](#8-loss-surface)
- [9. Gradient Direction](#9-gradient-direction)
- [10. Gradient Checking](#10-gradient-checking)
- [11. Learning Rate](#11-learning-rate)
- [12. Hessian and Eigenvalues](#12-hessian-and-eigenvalues)
- [13. Stability Boundary](#13-stability-boundary)
- [14. Feature Scaling](#14-feature-scaling)
- [15. Condition Number](#15-condition-number)
- [16. Parameter Trajectory](#16-parameter-trajectory)
- [17. Momentum](#17-momentum)
- [18. Momentum Beta](#18-momentum-beta)
- [19. Nesterov Accelerated Gradient](#19-nesterov-accelerated-gradient)
- [20. AdaGrad](#20-adagrad)
- [21. RMSProp](#21-rmsprop)
- [22. Adam](#22-adam)
- [23. Algorithm Comparison](#23-algorithm-comparison)
- [24. Project Structure](#24-project-structure)
- [25. Installation](#25-installation)
- [26. Commands](#26-commands)
- [27. What Was Implemented](#27-what-was-implemented)
- [28. Conclusion](#28-conclusion)

---

## 1. Project Overview

The project develops optimization progressively:

```text
Linear Regression
      |
      v
Mean Squared Error
      |
      v
Gradient
      |
      v
Gradient Descent
      |
      +--------------------+
      |                    |
      v                    v
Learning Rate        Gradient Checking
      |
      v
Hessian + Eigenvalues
      |
      v
Stability
      |
      v
Feature Scaling + Conditioning
      |
      v
Momentum
      |
      v
Nesterov
      |
      v
AdaGrad
      |
      v
RMSProp
      |
      v
Adam
```

The project connects **multivariable calculus, linear algebra, numerical optimization, and machine learning**.

---

## 2. Dataset

The working example is a simple one-feature regression problem:

```text
Hours Studied     Exam Score
1                 35
2                 42
3                 51
4                 58
5                 66
6                 73
```

Therefore:

```math
X = [1,2,3,4,5,6], \qquad
 y = [35,42,51,58,66,73].
```

---

## 3. Linear Regression

The model is:

```math
\hat{y}_i = b_0 + b_1 x_i
```

where:

- $b_0$ = intercept
- $b_1$ = slope
- $x_i$ = input feature
- $y_i$ = target
- $\hat y_i$ = prediction

Parameter vector:

```math
\theta =
\begin{bmatrix}
 b_0 \\
 b_1
\end{bmatrix}.
```

---

## 4. Mean Squared Error

The project uses Mean Squared Error (MSE):

```math
J(b_0,b_1)
=
\frac{1}{n}
\sum_{i=1}^{n}
(\hat y_i-y_i)^2.
```

Substituting the model:

```math
J(b_0,b_1)
=
\frac{1}{n}
\sum_{i=1}^{n}
(b_0+b_1x_i-y_i)^2.
```

Optimization objective:

```math
\boxed{\min_{b_0,b_1} J(b_0,b_1)}.
```

Residual:

```math
e_i = \hat y_i-y_i.
```

Therefore:

```math
J = \frac{1}{n}\sum_{i=1}^{n}e_i^2.
```

---

## 5. Gradient

The gradient is the vector of partial derivatives:

```math
\nabla J =
\begin{bmatrix}
\dfrac{\partial J}{\partial b_0} \\
\dfrac{\partial J}{\partial b_1}
\end{bmatrix}.
```

### Derivative with respect to $b_0$

```math
\frac{\partial J}{\partial b_0}
=
\frac{2}{n}
\sum_{i=1}^{n}(\hat y_i-y_i).
```

### Derivative with respect to $b_1$

```math
\frac{\partial J}{\partial b_1}
=
\frac{2}{n}
\sum_{i=1}^{n}(\hat y_i-y_i)x_i.
```

Therefore:

```math
\boxed{
\nabla J=
\begin{bmatrix}
\dfrac{2}{n}\sum_i(\hat y_i-y_i) \\
\dfrac{2}{n}\sum_i(\hat y_i-y_i)x_i
\end{bmatrix}}
```

---

## 6. Gradient Descent

The core update is:

```math
\boxed{
\theta_{t+1}
=
\theta_t-\alpha\nabla J(\theta_t)
}
```

For the two parameters:

```math
\boxed{
 b_0^{(t+1)} = b_0^{(t)}
 - \alpha\frac{\partial J}{\partial b_0}
}
```

```math
\boxed{
 b_1^{(t+1)} = b_1^{(t)}
 - \alpha\frac{\partial J}{\partial b_1}
}
```

The negative sign is what makes the method descend the objective.

---

## 7. Loss Curves and Convergence

The loss history records:

```math
J_0,J_1,J_2,\ldots,J_T.
```

A successful run generally moves toward a lower loss:

```text
Loss
 ^
 |\
 | \
 |  \
 |   \________
 |
 +--------------------> Iteration
```

The project records and plots the entire history instead of hiding the optimization process.

---

## 8. Loss Surface

With two parameters, the objective is a surface:

```math
J = J(b_0,b_1).
```

A contour plot shows equal-loss curves. A 3D plot shows:

```math
z = J(b_0,b_1).
```

The Gradient Descent path is:

```math
(b_0^{(0)},b_1^{(0)})
\rightarrow
(b_0^{(1)},b_1^{(1)})
\rightarrow \cdots \rightarrow
(b_0^{(T)},b_1^{(T)}).
```

---

## 9. Gradient Direction

The gradient points toward the direction of steepest local increase:

```math
\boxed{\nabla J = \text{steepest ascent direction}}.
```

Therefore:

```math
\boxed{-\nabla J = \text{steepest descent direction}}.
```

For a small displacement $\Delta\theta$:

```math
J(\theta+\Delta\theta)
\approx
J(\theta)+\nabla J(\theta)^T\Delta\theta.
```

Choose:

```math
\Delta\theta=-\alpha\nabla J.
```

Then:

```math
\Delta J
\approx
-\alpha\nabla J^T\nabla J
=
-\alpha\lVert\nabla J\rVert^2
\le 0.
```

So, for sufficiently small positive $\alpha$, the negative-gradient direction locally decreases the loss.

---

## 10. Gradient Checking

The analytical gradient can be independently checked using central finite differences:

```math
\boxed{
\frac{\partial J}{\partial \theta}
\approx
\frac{
J(\theta+\epsilon)-J(\theta-\epsilon)
}{2\epsilon}}
```

The desired result is:

```math
\boxed{
\nabla J_{\text{analytical}}
\approx
\nabla J_{\text{numerical}}
}
```

This is useful for detecting derivative and implementation mistakes.

---

## 11. Learning Rate

The learning rate is $\alpha$ and controls step size:

```math
\Delta\theta=-\alpha\nabla J.
```

### Small $\alpha$

- Small steps
- Stable but potentially slow convergence

### Appropriate $\alpha$

- Efficient convergence

### Excessive $\alpha$

- Overshooting
- Oscillation
- Divergence
- Very large numerical values

---

## 12. Hessian and Eigenvalues

The Hessian is:

```math
\boxed{H=\nabla^2J(\theta)}.
```

For this two-parameter experiment:

```math
\boxed{
H=
\begin{bmatrix}
2 & 7 \\
7 & 30.3333
\end{bmatrix}}
```

The eigenvalues found experimentally are approximately:

```math
\lambda_1\approx0.36494,
\qquad
\lambda_2\approx31.96839.
```

Hence:

```math
\boxed{\lambda_{\max}\approx31.96839}.
```

The eigenvalues describe the curvature scales of the quadratic objective.

---

## 13. Stability Boundary

For this quadratic objective, the Gradient Descent stability condition is:

```math
\boxed{
0<\alpha<\frac{2}{\lambda_{\max}}
}
```

Using the measured largest eigenvalue:

```math
\alpha < \frac{2}{31.96839}
\approx 0.06256.
```

So the experiment gives the approximate stability boundary:

```math
\boxed{\alpha<0.06256}.
```

Learning rates above the boundary can generate instability and divergence.

---

## 14. Feature Scaling

Standardization is:

```math
\boxed{x' = \frac{x-\mu}{\sigma}}.
```

For this dataset:

```math
\mu=3.5
```

and approximately:

```math
X'=[-1.464,-0.878,-0.293,0.293,0.878,1.464].
```

With standardized $X$, the Hessian becomes approximately:

```math
H'=
\begin{bmatrix}
2&0\\
0&2
\end{bmatrix}.
```

Thus the eigenvalues become approximately $2$ and $2$.

Feature scaling therefore changes the geometry of the optimization problem and can allow larger stable learning rates.

---

## 15. Condition Number

The condition number of the Hessian is:

```math
\boxed{\kappa(H)=\frac{\lambda_{\max}}{\lambda_{\min}}}.
```

For the original coordinates:

```math
\kappa(H)
\approx
\frac{31.96839}{0.36494}
\approx87.6.
```

For the standardized feature:

```math
\boxed{\kappa(H')=1}.
```

A large condition number corresponds to very different curvature scales and can make Gradient Descent follow inefficient trajectories.

---

## 16. Parameter Trajectory

The project tracks:

```math
\theta_t=
\begin{bmatrix}
 b_0^{(t)}\\
 b_1^{(t)}
\end{bmatrix}
```

and plots the path in $(b_0,b_1)$ parameter space.

This shows optimization geometry directly rather than only showing the scalar loss value.

Scaling changes the numerical coordinates of the optimum, but the transformed model can represent the same regression relationship.

---

## 17. Momentum

Vanilla Gradient Descent:

```math
\theta_{t+1}
=
\theta_t-\alpha g_t.
```

Momentum introduces velocity:

```math
\boxed{
v_t=\beta v_{t-1}-\alpha g_t
}
```

and then:

```math
\boxed{
\theta_{t+1}=\theta_t+v_t
}
```

The term $\beta v_{t-1}$ provides memory of previous updates.

When:

```math
\beta=0
```

Momentum reduces to ordinary Gradient Descent.

---

## 18. Momentum Beta

The coefficient $\beta$ controls the amount of previous velocity retained.

The project compares:

```math
\beta\in\{0,0.5,0.9,0.99\}.
```

Increasing $\beta$ increases inertia. Larger inertia can accelerate motion in consistent directions but can also create overshooting and oscillation.

---

## 19. Nesterov Accelerated Gradient

Nesterov evaluates the gradient after moving to a look-ahead position:

```math
\boxed{
\theta_{\text{lookahead}}
=
\theta_t+\beta v_t
}
```

Then:

```math
\boxed{
g_t=\nabla J(\theta_{\text{lookahead}})
}
```

Velocity update:

```math
\boxed{
v_{t+1}=\beta v_t-\alpha g_t}
```

Parameter update:

```math
\boxed{
\theta_{t+1}=\theta_t+v_{t+1}
}
```

The distinguishing idea is:

```text
Momentum  -> gradient at current position
Nesterov  -> gradient at look-ahead position
```

---

## 20. AdaGrad

AdaGrad accumulates squared gradients:

```math
\boxed{G_t=G_{t-1}+g_t^2}.
```

Update:

```math
\boxed{
\theta_{t+1}
=
\theta_t-
\frac{\alpha}{\sqrt{G_t}+\epsilon}g_t
}
```

This produces parameter-specific adaptive step sizes.

The limitation is that $G_t$ never decreases, so the effective learning rate can become very small after long training.

---

## 21. RMSProp

RMSProp replaces permanent accumulation with an exponential moving average:

```math
\boxed{
S_t=eta S_{t-1}+(1-\beta)g_t^2
}
```

Update:

```math
\boxed{
\theta_{t+1}
=
\theta_t-
\frac{\alpha}{\sqrt{S_t}+\epsilon}g_t
}
```

Recent squared gradients receive more influence than old squared gradients.

---

## 22. Adam

Adam combines a first-moment estimate with a second-moment estimate.

First moment:

```math
\boxed{
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t}
```

Second moment:

```math
\boxed{
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2}
```

Bias correction:

```math
\boxed{
\hat m_t=\frac{m_t}{1-\beta_1^t}
}
```

```math
\boxed{
\hat v_t=\frac{v_t}{1-\beta_2^t}
}
```

Final update:

```math
\boxed{
\theta_{t+1}
=
\theta_t-
\alpha\frac{\hat m_t}
{\sqrt{\hat v_t}+\epsilon}
}
```

Typical values:

```math
\beta_1=0.9,\qquad
\beta_2=0.999,\qquad
\epsilon=10^{-8}.
```

---

## 23. Algorithm Comparison

| Algorithm | Main idea |
|---|---|
| Gradient Descent | Current gradient |
| Momentum | Gradient + velocity history |
| Nesterov | Look-ahead gradient + momentum |
| AdaGrad | Accumulated squared gradients |
| RMSProp | Moving average of squared gradients |
| Adam | First + second moments + bias correction |

Conceptual progression:

```text
Gradient Descent
      |
      v
Momentum
      |
      v
Nesterov
      |
      +--------------+
      |              |
      v              v
   AdaGrad        RMSProp
      |              |
      +------+-------+
             |
             v
            Adam
```

---

## 24. Project Structure

```text
01_linear_regression/
│
├── README.md
│
├── linear_regression.py
├── loss_surface.py
├── loss_surface_3d.py
├── gradient_vector.py
├── gradient_check.py
├── learning_rate_experiment.py
├── hessian_analysis.py
├── stability_boundary.py
├── feature_scaling.py
├── parameter_trajectory.py
├── momentum_gradient_descent.py
├── momentum_beta_experiment.py
├── nesterov_momentum.py
├── adagrad.py
├── rmsprop.py
├── adam.py
│
├── plots/
│   ├── 3D Loss Surface and Gradient Descent.png
│   ├── AdaGrad - Loss vs Iteration.png
│   ├── Adam-loss vs Iteration.png
│   ├── Feature Scaling - Loss vs Iteration.png
│   ├── Gradient and Negative Gradient.png
│   ├── Gradient Descent - First 30 Iterations.png
│   ├── Gradient Descent Learning.png
│   ├── Gradient Descent Near Stability Boundary.png
│   ├── Gradient Descent Parameter Trajectory.png
│   ├── Gradient Descent vs Momentum - Loss.png
│   ├── Momentum Gradient Descent - Effect of beta.png
│   ├── RMSProp - Loss vs Iteration.png
│   └── Vanilla GD vs Momentum vs Nesterov.png
│
└── docs/
    └── Linear Regression / Optimization report files
```

---

## 25. Installation

Verify Python:

```powershell
python --version
```

Install dependencies:

```powershell
pip install numpy matplotlib
```

No Scikit-learn dependency is required for the core implementations in this section.

---

## 26. Commands

From PowerShell:

```powershell
cd C:\Users\Lenovo\machine-learning-algorithms\01_linear_regression
```

Run the scripts:

```powershell
python linear_regression.py
python loss_surface.py
python loss_surface_3d.py
python gradient_vector.py
python gradient_check.py
python learning_rate_experiment.py
python hessian_analysis.py
python stability_boundary.py
python feature_scaling.py
python parameter_trajectory.py
python momentum_gradient_descent.py
python momentum_beta_experiment.py
python nesterov_momentum.py
python adagrad.py
python rmsprop.py
python adam.py
```

### Git commands

From the repository root:

```powershell
cd C:\Users\Lenovo\machine-learning-algorithms
git status
git add .
git commit -m "Add linear regression and optimization from scratch"
git branch -M main
git remote -v
git push -u origin main
```

---

## 27. What Was Implemented

Implemented manually using Python, NumPy, and Matplotlib:

- Linear Regression
- Mean Squared Error
- Analytical gradients
- Numerical gradient checking
- Gradient Descent
- Loss tracking
- 2D loss contours
- 3D loss surface
- Gradient vector visualization
- Learning-rate experiments
- Hessian calculation
- Eigenvalue analysis
- Stability boundary experiments
- Feature scaling
- Condition-number analysis
- Parameter trajectories
- Momentum
- Momentum beta experiments
- Nesterov Accelerated Gradient
- AdaGrad
- RMSProp
- Adam

The project therefore exposes the mechanics of optimization rather than hiding them inside a library optimizer.

---

## 28. Conclusion

The complete conceptual progression is:

```text
Loss
  |
  v
Gradient
  |
  v
Gradient Descent
  |
  +--> Learning Rate
  |
  +--> Hessian / Eigenvalues / Stability
  |
  +--> Feature Scaling / Conditioning
  |
  +--> Momentum
  |       |
  |       v
  |    Nesterov
  |
  +--> AdaGrad
  |
  +--> RMSProp
  |
  +--> Adam
```

The central objective of the project is to understand not only **what** an optimizer does, but **why** its updates behave as they do.

### Optimization Complete

```text
✓ Linear Regression
✓ MSE
✓ Gradient
✓ Gradient Descent
✓ Loss Curves
✓ Loss Surfaces
✓ Gradient Geometry
✓ Gradient Checking
✓ Learning Rate
✓ Hessian
✓ Eigenvalues
✓ Stability Boundary
✓ Feature Scaling
✓ Condition Number
✓ Parameter Trajectory
✓ Momentum
✓ Momentum Beta
✓ Nesterov
✓ AdaGrad
✓ RMSProp
✓ Adam
```

**End of Optimization.**
