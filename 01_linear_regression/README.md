# Linear Regression & Optimization from Scratch

A from-scratch implementation and mathematical study of **Linear
Regression, Gradient Descent, Gradient Checking, Loss Surfaces,
Learning-Rate Stability, Feature Scaling, Momentum, Nesterov Accelerated
Gradient, AdaGrad, RMSProp, and Adam**.

## Learning Path

``` text
Linear Regression
       |
       v
Loss Function
       |
       v
Gradient
       |
       v
Gradient Descent
       |
       v
Learning Rate
       |
       v
Hessian + Stability
       |
       v
Feature Scaling
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

## 1. Project Overview

This project studies Linear Regression and optimization from first
principles. The core algorithms are implemented directly using Python
and NumPy rather than treating optimizers as black boxes.

The project connects:

-   Linear algebra
-   Multivariable calculus
-   Numerical optimization
-   Numerical stability
-   Machine learning

## 2. Learning Objectives

-   Understand Linear Regression mathematically.
-   Implement Mean Squared Error manually.
-   Derive and implement analytical gradients.
-   Implement Gradient Descent from scratch.
-   Visualize loss surfaces and optimization trajectories.
-   Verify gradients numerically.
-   Study learning-rate effects and stability.
-   Understand Hessian curvature and eigenvalues.
-   Understand feature scaling and conditioning.
-   Implement Momentum and Nesterov.
-   Implement AdaGrad, RMSProp, and Adam.
-   Compare optimization methods experimentally.

## 3. Linear Regression

The model is:

\[ `\boxed{\hat y=b_0+b_1x}`{=tex} \]

where `b0` is the intercept and `b1` is the slope.

For parameters

\[ `\theta`{=tex}=
```{=tex}
\begin{bmatrix}
b_0\\
b_1
\end{bmatrix}
```
\]

the objective is to find parameters that minimize prediction error.

## 4. Mean Squared Error

The loss function is:

\[ `\boxed{
J(b_0,b_1)=
\frac{1}{n}
\sum_{i=1}^{n}
(\hat y_i-y_i)^2
}`{=tex} \]

Substituting the model:

\[ `\boxed{
J(b_0,b_1)=
\frac{1}{n}
\sum_{i=1}^{n}
(b_0+b_1x_i-y_i)^2
}`{=tex} \]

The optimization problem is:

\[ `\boxed{\min_{b_0,b_1}J(b_0,b_1)}`{=tex} \]

## 5. Gradient

The gradient is:

\[ `\nabla `{=tex}J=
```{=tex}
\begin{bmatrix}
\frac{\partial J}{\partial b_0}\\
\frac{\partial J}{\partial b_1}
\end{bmatrix}
```
\]

with

\[ `\boxed{
\frac{\partial J}{\partial b_0}
=
\frac{2}{n}\sum(\hat y-y)
}`{=tex} \]

and

\[ `\boxed{
\frac{\partial J}{\partial b_1}
=
\frac{2}{n}\sum(\hat y-y)x
}`{=tex} \]

## 6. Gradient Descent

The fundamental update is:

\[ `\boxed{
\theta_{t+1}
=
\theta_t-\alpha\nabla J(\theta_t)
}`{=tex} \]

where `alpha` is the learning rate.

The gradient points toward increasing loss, so the negative gradient
points toward decreasing loss:

\[ `\boxed{
\nabla J=\text{uphill},\qquad
-\nabla J=\text{downhill}
}`{=tex} \]

## 7. Loss Curves and Loss Surfaces

The project visualizes:

-   MSE versus iteration.
-   2D contour loss surfaces.
-   3D loss surfaces.
-   Gradient and negative-gradient vectors.
-   Parameter trajectories.

The two-parameter loss is:

\[ J=J(b_0,b_1) \]

The optimization path can therefore be observed directly in parameter
space.

## 8. Gradient Checking

The analytical gradient is verified using the central finite-difference
approximation:

\[ `\boxed{
\frac{\partial J}{\partial\theta}
\approx
\frac{
J(\theta+\epsilon)-J(\theta-\epsilon)
}{
2\epsilon
}
}`{=tex} \]

The goal is:

\[ `\boxed{
\nabla J_{analytical}
\approx
\nabla J_{numerical}
}`{=tex} \]

This provides an independent check of derivative implementations.

## 9. Learning Rate

The learning rate controls the size of each update.

### Small learning rate

Stable but potentially slow.

### Appropriate learning rate

Efficient convergence.

### Large learning rate

May cause:

-   Overshooting
-   Oscillation
-   Divergence
-   Very large loss
-   Numerical overflow

Therefore:

\[
`\boxed{\text{Learning rate is a stability-critical hyperparameter}}`{=tex}
\]

## 10. Hessian

The Hessian is:

\[ `\boxed{H=\nabla^2J(\theta)}`{=tex} \]

For the Linear Regression experiment, the Hessian was approximately:

\[ H=
```{=tex}
\begin{bmatrix}
2 & 7\\
7 & 30.3333
\end{bmatrix}
```
\]

with eigenvalues approximately:

\[ `\lambda`{=tex}\_1`\approx0.3649`{=tex} \]

\[ `\lambda`{=tex}\_2`\approx31.9684`{=tex} \]

The largest eigenvalue determines the most restrictive curvature
direction.

## 11. Stability Boundary

For a quadratic objective, Gradient Descent is stable when:

\[ `\boxed{
0<\alpha<\frac{2}{\lambda_{max}}
}`{=tex} \]

Using:

\[ `\lambda`{=tex}\_{max}`\approx31.9684`{=tex} \]

gives:

\[ `\boxed{\alpha<0.06256}`{=tex} \]

The stability-boundary experiment verifies the relationship between
Hessian curvature and maximum stable learning rate.

## 12. Feature Scaling

Standardization is:

\[ `\boxed{
x_{scaled}=\frac{x-\mu}{\sigma}
}`{=tex} \]

Feature scaling can make the optimization geometry better conditioned
and can significantly improve convergence.

The project visualizes the difference through loss curves and parameter
trajectories.

## 13. Parameter Trajectory

The parameters are tracked as:

\[ (b_0,b_1) \]

at every iteration.

This shows how the optimizer moves through parameter space rather than
only showing whether the loss decreases.

## 14. Momentum

Momentum introduces a velocity term:

\[ `\boxed{
v_t=\beta v_{t-1}-\alpha g_t
}`{=tex} \]

followed by:

\[ `\boxed{
\theta_{t+1}=\theta_t+v_t
}`{=tex} \]

Momentum incorporates historical gradient information and can accelerate
movement along consistent directions.

## 15. Momentum Beta Experiment

The coefficient `beta` controls how strongly previous velocity is
retained.

Values explored include:

\[ `\beta=0`{=tex},;0.5,;0.9,;0.99 \]

When:

\[ `\beta=0`{=tex} \]

the method reduces to ordinary Gradient Descent.

Large momentum can also produce oscillations or overshooting depending
on the learning rate and curvature.

## 16. Nesterov Accelerated Gradient

Nesterov first evaluates a look-ahead position:

\[ `\boxed{
\theta_{lookahead}
=
\theta_t+\beta v_t
}`{=tex} \]

Then:

\[ `\boxed{
g_t=\nabla J(\theta_{lookahead})
}`{=tex} \]

followed by:

\[ `\boxed{
v_{t+1}=\beta v_t-\alpha g_t
}`{=tex} \]

and:

\[ `\boxed{
\theta_{t+1}=\theta_t+v_{t+1}
}`{=tex} \]

Conceptually:

``` text
Current position
      |
      v
Predict momentum movement
      |
      v
Look ahead
      |
      v
Calculate gradient
      |
      v
Correct movement
```

## 17. AdaGrad

AdaGrad accumulates squared gradients:

\[ `\boxed{
G_t=G_{t-1}+g_t^2
}`{=tex} \]

and updates:

\[ `\boxed{
\theta_{t+1}
=
\theta_t-
\frac{\alpha}{\sqrt{G_t}+\epsilon}g_t
}`{=tex} \]

This creates parameter-specific adaptive learning rates.

### Advantage

Large or frequently occurring gradients receive progressively smaller
effective updates.

### Limitation

The accumulator only grows, so the effective learning rate can
eventually become extremely small.

## 18. RMSProp

RMSProp uses an exponentially weighted average of squared gradients:

\[ `\boxed{
S_t=
\beta S_{t-1}
+
(1-\beta)g_t^2
}`{=tex} \]

The update is:

\[ `\boxed{
\theta_{t+1}
=
\theta_t-
\frac{\alpha}{\sqrt{S_t}+\epsilon}g_t
}`{=tex} \]

Unlike AdaGrad, old gradients gradually lose influence.

## 19. Adam

Adam combines momentum-like first-moment estimation with RMSProp-like
second-moment estimation.

First moment:

\[ `\boxed{
m_t=
\beta_1m_{t-1}
+
(1-\beta_1)g_t
}`{=tex} \]

Second moment:

\[ `\boxed{
v_t=
\beta_2v_{t-1}
+
(1-\beta_2)g_t^2
}`{=tex} \]

Bias correction:

\[ `\boxed{
\hat m_t=
\frac{m_t}{1-\beta_1^t}
}`{=tex} \]

\[ `\boxed{
\hat v_t=
\frac{v_t}{1-\beta_2^t}
}`{=tex} \]

Final update:

\[ `\boxed{
\theta_{t+1}
=
\theta_t
-
\alpha
\frac{\hat m_t}
{\sqrt{\hat v_t}+\epsilon}
}`{=tex} \]

Typical values:

\[ `\beta`{=tex}\_1=0.9,`\qquad`{=tex}
`\beta`{=tex}\_2=0.999,`\qquad`{=tex} `\epsilon=10`{=tex}\^{-8} \]

## 20. Optimization Algorithm Comparison

  Algorithm          Main Idea
  ------------------ ------------------------------------------
  Gradient Descent   Current gradient
  Momentum           Gradient + velocity history
  Nesterov           Momentum + look-ahead gradient
  AdaGrad            Accumulated squared gradients
  RMSProp            Moving average of squared gradients
  Adam               First + second moments + bias correction

Conceptually:

``` text
Gradient Descent
      |
      v
Momentum
      |
      v
Nesterov
      |
      +----------------+
      |                |
      v                v
   AdaGrad          RMSProp
      |                |
      +-------+--------+
              |
              v
             Adam
```

The major ideas are:

\[ `\boxed{\text{Momentum}\rightarrow\text{direction/history}}`{=tex} \]

\[
`\boxed{\text{AdaGrad/RMSProp}\rightarrow\text{adaptive step size}}`{=tex}
\]

\[
`\boxed{\text{Adam}\rightarrow\text{direction + adaptive step size}}`{=tex}
\]

## 21. Project Structure

``` text
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
└── adam.py
```

## 22. Requirements

-   Python 3.x
-   NumPy
-   Matplotlib

The core optimization algorithms do not require Scikit-learn.

## 23. Installation

Verify Python:

``` bash
python --version
```

Install dependencies:

``` bash
pip install numpy matplotlib
```

## 24. Running the Experiments

Open the project directory:

``` powershell
cd C:\Users\Lenovo\machine-learning-algorithms\01_linear_regression
```

Run individual experiments:

``` powershell
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

## 25. Key Mathematical Results

### Linear Regression

\[ `\boxed{\hat y=b_0+b_1x}`{=tex} \]

### MSE

\[ `\boxed{
J=
\frac{1}{n}
\sum_{i=1}^{n}
(\hat y_i-y_i)^2
}`{=tex} \]

### Gradient Descent

\[ `\boxed{
\theta_{t+1}
=
\theta_t-\alpha\nabla J
}`{=tex} \]

### Numerical Gradient

\[ `\boxed{
\frac{\partial J}{\partial\theta}
\approx
\frac{
J(\theta+\epsilon)-J(\theta-\epsilon)
}{
2\epsilon
}
}`{=tex} \]

### Stability

\[ `\boxed{
0<\alpha<\frac{2}{\lambda_{max}}
}`{=tex} \]

### Momentum

\[ `\boxed{
v_t=\beta v_{t-1}-\alpha g_t
}`{=tex} \]

\[ `\boxed{
\theta_{t+1}=\theta_t+v_t
}`{=tex} \]

### AdaGrad

\[ `\boxed{
G_t=G_{t-1}+g_t^2
}`{=tex} \]

### RMSProp

\[ `\boxed{
S_t=
\beta S_{t-1}
+
(1-\beta)g_t^2
}`{=tex} \]

### Adam

\[ `\boxed{
m_t=
\beta_1m_{t-1}
+
(1-\beta_1)g_t
}`{=tex} \]

\[ `\boxed{
v_t=
\beta_2v_{t-1}
+
(1-\beta_2)g_t^2
}`{=tex} \]

## 26. Important Observations

### Gradient Direction

\[ `\nabla `{=tex}J`\rightarrow`{=tex}`\text{uphill}`{=tex} \]

\[ -`\nabla `{=tex}J`\rightarrow`{=tex}`\text{downhill}`{=tex} \]

### Learning Rate

A small learning rate may be slow; a large one can become unstable.

### Hessian

The Hessian describes local curvature. Its largest eigenvalue controls
the most restrictive direction for Gradient Descent on a quadratic
objective.

### Feature Scaling

Scaling can improve conditioning and make optimization more efficient.

### Momentum

Momentum adds historical velocity to the current update.

### Nesterov

Nesterov evaluates the gradient at a look-ahead position.

### AdaGrad

AdaGrad adapts learning rates using accumulated squared gradients.

### RMSProp

RMSProp uses a moving average so old gradients gradually lose influence.

### Adam

Adam combines first-moment and second-moment estimates with bias
correction.

## 27. What Was Implemented From Scratch

-   Linear Regression
-   Mean Squared Error
-   Analytical gradients
-   Numerical gradients
-   Gradient Descent
-   Loss tracking
-   Loss surface visualization
-   3D loss-surface visualization
-   Gradient vector visualization
-   Gradient checking
-   Learning-rate experiments
-   Hessian analysis
-   Eigenvalue analysis
-   Stability analysis
-   Feature scaling
-   Parameter trajectory tracking
-   Momentum
-   Momentum beta experiments
-   Nesterov Accelerated Gradient
-   AdaGrad
-   RMSProp
-   Adam

## 28. Future Extensions

### Optimization

-   Mini-batch Gradient Descent
-   Stochastic Gradient Descent
-   AdamW
-   Nadam
-   AMSGrad
-   Learning-rate scheduling
-   Cosine decay
-   Exponential decay
-   Warm-up schedules

### Regression

-   Multiple Linear Regression
-   Polynomial Regression
-   Ridge Regression
-   Lasso Regression
-   Elastic Net

### Machine Learning

-   Logistic Regression
-   Perceptron
-   k-Nearest Neighbors
-   Decision Trees
-   Support Vector Machines
-   Neural Networks

## 29. Conclusion

The project develops optimization progressively:

\[ `\boxed{
\text{Loss}
\rightarrow
\text{Gradient}
\rightarrow
\text{Gradient Descent}
}`{=tex} \]

then:

\[ `\boxed{
\text{Learning Rate}
\rightarrow
\text{Hessian}
\rightarrow
\text{Stability}
}`{=tex} \]

then:

\[ `\boxed{
\text{Feature Scaling}
\rightarrow
\text{Momentum}
\rightarrow
\text{Nesterov}
}`{=tex} \]

and finally:

\[ `\boxed{
\text{AdaGrad}
\rightarrow
\text{RMSProp}
\rightarrow
\text{Adam}
}`{=tex} \]

The central objective is not merely to obtain a regression line, but to
understand **why optimization algorithms move the way they do** and to
connect the mathematics directly to executable Python experiments.

## Optimization Section Completed

``` text
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
✓ Parameter Trajectory
✓ Momentum
✓ Momentum Beta
✓ Nesterov
✓ AdaGrad
✓ RMSProp
✓ Adam
```

**End of Optimization.**
