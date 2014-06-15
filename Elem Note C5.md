## Basis Expansions and Regularization
 
### Basic Ideas
 
A *linear basis expansion* in X is formed as \\[f(X) = \sum^M_{m=1}{\beta_m h_m(X)},\\] where \\(h_m(X) : \mathbb{R}^p \mapsto \mathbb{R}\\) is the \\(m\\)th transformation of \\(X\\). And \\(h_m(X)\\) can be all kinds of non-linear transformation.
 
### Piecewise Polynomials and Splines
 
Generally, an order-\\(M\\) spline with knots \\(\xi_j\\), \\(j=1,...,K\\) is a piecewise-polynomial of order \\(M\\), and has continuous derivatives up to order \\(M-2\\). A cubic spline has \\(M=4\\). The general from for the truncated-power basis set would be \\[\begin{aligned} h_j(X)=X^{j-1},& j=1,...,M, \\\ h_{M+l}(X) = (X-\xi_l)^{M-1}_+, & l=1,...,K. \end{aligned}\\]
 
#### Natural Cubic Splines
 
A *natural cubic spline* adds additional constraints, namely that the function is linear beyond the boundary knots. This frees up four degrees of freedom (two constraints each in both boundary regions), which can be spent more profitably by sprinkling more knots in the interior region.
 
A natural cubic spline with \\(K\\) knots is represented by \\(K\\) basis functions:
\\[ N_1(X)= 1, \ N_2(X) = X, \ N_{k+2}(X) = d_k(X)-d_{K-1}(X), \\\
d_k(X) = \frac{(X-\xi_k)^3_+ - (X-\xi_K)^3_+}{\xi_K- \xi_k}\\]
 
### Smoothing Splines
 
This is used to avoid the knot selection problem completely by using a maximal set of knots. And the complexity of the fit is controlled by regularization.
 
The problem: among all functions \\(f(x)\\) with two continuous derivatives, find one that minimizes the penalized residual sum of squares
\\[ \mathrm{RSS}(f, \lambda) = \sum^N_{i=1}{\{y_i-f(x_i)\}}^2 + \lambda \int \{f''(t)\}^2 dt\\]
where \\(\lambda\\) is a fixed *smoothing parameter*.
 
Remarkably, it can be shown that equation above has an explicit, finite-dimensional, unique minimizer which is a natural cubic spline with knots at the unique values of the \\(x_i\\), \\(i=1,...,N\\).
 
#### Smoother Matrices
 
We can write the fitting values of a fitted smoothing spline as \\(\hat{f}(x) = \sum^N_{j=1} N_j(x) \hat{\theta}_j\\). And then \\[ \hat{\mathbf{f}} = \mathbf{N}( \mathbf{N}^T  \mathbf{N} + \lambda\mathbf{\Omega}_N)^{-1}  \mathbf{N}^T  \mathbf{y} =  \mathbf{S}_\lambda  \mathbf{y}\\]
 
And suppose \\(\mathbf{B}_\xi \\) is a \\(N \times M\\) matrix of \\(M\\) cubic-spline basis functions evaluated at the \\(N\\) training points \\(x_i\\), with knot sequence \\(\xi\\), and \\(M \ll N\\). The vector of fitted spline values is given by
\\[ \hat{\mathbf{f}} = \mathbf{B}_\xi ( \mathbf{B}^T_\xi \mathbf{B}_\xi)^{-1} \mathbf{B}^T_\xi \mathbf{y} = \mathbf{H}_\xi \mathbf{y} \\]
 
Then there are some similarities and differences between \\(\mathbf{S}_\lambda \\) and \\(\mathbf{H}_\xi\\) :

* Both are symmetric, positive semidefinite matrieces.
* \\(\mathbf{H}_\xi \mathbf{H}_\xi = \mathbf{H}_\xi \\) (idempotent), while \\( \mathbf{S}_\lambda \mathbf{S}_\lambda \preceq \mathbf{S}_\lambda\\) because of the *shrinking* nature of \\(\mathbf{S}_\lambda\\).
* \\(\mathbf{H}_\xi\\) has rank \\(M\\), while \\(\mathbf{S}_\lambda \\) has rank \\(N\\).
 
#### Degrees of Freedom
 
We define *effective degrees of freedom* of a smoothing spline to be \\[\mathrm{df}_\lambda = \operatorname{trace}(\mathbf{S}_\lambda ), \\] the sum of the diagonal elements of \\(\mathbf{S}_\lambda \\).
 
Since \\(\mathbf{S}_\lambda \\) is symmetric (and positive semidefinite), it has a real eigen-decomposition. The *Reinsch* form is \\[\mathbf{S}_\lambda = (\mathbf{I} + \lambda \mathbf{K})^{-1}\\]
And the eigen-decomposition of \\(\mathbf{S}_\lambda\\) is \\[\mathbf{S}_\lambda = \sum^N_{k=1}\rho_k(\lambda) \mathbf{u}_k \mathbf{u}_k^T \\\ with\ \rho_k(\lambda) = \frac{1}{1+\lambda d_k}\\]

* The \\(\mathbf{K}\\) is known as the *penalty matrix*, and does not depend on \\(\lambda\\).
* The eigenvectors of \\(\mathbf{S}_\lambda\\) are not affected by changes in \\(\lambda\\).
* \\(\mathbf{S}_\lambda \mathbf{y} = \sum^N_{k=1}  \mathbf{u}_k \rho_k(\lambda) \left \langle \mathbf{u}_k, \mathbf{y} \right \rangle\\). The denominator is shrinking using \\(\rho_k(\lambda)\\), while the basis-regression method has the components either left alone or shrunk to zero. For this reason, smoothing splines are referred to as *shrinking* smoothers, while regression splines are *projection* smoothers.
* The sequence of \\(\mathbf{u}_k\\), ordered by decreasing \\(\rho_k(\lambda)\\), appear to increase in complexity.
* The first two eigenvalues are *always* one. They correspond to the two-dimensional eigenspace of functions linear in \\(x\\), which are never shrunk.
* \\(\lambda\\) controls the rate of the eigenvalues \\(\rho_k(\lambda) = 1/ (1+\lambda d_k)\\) decrease to zero. \\(d_k\\) are the eigenvalues of the penalty matrix \\(\mathbf{K}\\).
* \\(\mathrm{df}_\lambda = \operatorname{trace}(\mathbf{S}_\lambda ) = \sum^N_{k=1} \rho_k(\lambda)  \\). For projection smoothers, all the eigenvalues are 1, each one corresponding to a dimension of the projection subspace.
 
### The Bias-Variance Tradeoff
 
the integrated squared prediction error (EPE) combins both bias and variance in a single summary :
\\[\begin{aligned}  \mathrm{EPE}(\hat{f}_\lambda) &= \mathrm{E}(Y-\hat{f}_\lambda(X))^2 \\\
   &= \mathrm{Var}(Y) + \mathrm{E}\left [ \mathrm{Bias}^2(\hat{f}_\lambda(X)) + \mathrm{Var}(\hat{f}_\lambda(X)) \right ] \\\
    &= \sigma^2 + \mathrm{MSE}(\hat{f}_\lambda) \end{aligned}\\]
 
N-fold (leave-one-out) cross-validation curve (CV) is :
\\[ \mathrm{CV}(\hat{f}_\lambda) = \frac{1}{N} \sum^N_{i=1} \left ( \frac{y_i - \hat{f}_\lambda(x_i)} {1-S_\lambda(i,i)} \right )^2 \\]
 
The EPE and CV curves have a similar shape. The CV curve is approximately as an estimate of the EPE curve.
 
### Nonparametric Logistic Regression
 
\\[ \mathbf{f}^{new} = \mathbf{N}( \mathbf{N}^T \mathbf{W}\mathbf{N} + \lambda\mathbf{\Omega})^{-1} \mathbf{N}^T \mathbf{W} \left ( \mathbf{f}^{old} + \mathbf{W}^{-1}(\mathbf{y} - \mathbf{p}) \right ) = \mathbf{S}_{\lambda, \omega} \mathbf{z}\\]
 
### Regularization and Reproducing Kernal Hilbert Spaces
 
A general class of regularization problems has the form \\[\underset{f \in \mathcal{H}}{\operatorname{min}}\left [ \sum^N_{i=1} L(y_i, f(x_i)) + \lambda J(f) \right ]\\] where \\(L(y_i, f(x_i))\\) is a loss function, \\(J(f)\\) is a penalty functional, and \\(\mathcal{H}\\) is a space of functions on which \\(J(f)\\) is  defined.
 
General penalty functionals have the form \\[J(f) = \int_{\mathbb{R}^d} \frac{|\tilde{f}(s)|^2} {\tilde{G}(s)} ds, \\] where \\(\tilde{f}\\) denotes the Fourier transform of \\(f\\), and \\(\tilde{G}\\) is some positive function that falls off to zero as \\(\|s\| \to \infty\\).  Penalty increases for high-frequency components of \\(f\\).
 
Under some additional assumptions the solultions have the form \\[f(X) = \sum^K_{k=1} \alpha_k \phi_k (X) + \sum^N_{i=1} \theta_i G(X-x_i), \\] where the \\(\phi_k\\) span the null space of the penalty functional \\(J\\), and \\(G\\) is the inverse Fourier transform of \\(\tilde{G}\\).
 
#### Spaces of Functions Generated by Kernels
 
A positive definite kernal \\(K(x,y)\\) can generate a space of functions \\(\mathcal{H}_K\\), which is called a *reproducing kernal Hilbert space* (RKHS).
 
### Wavelet Smoothing
 
Haar wavelets are simple to understand, but not smooth enough for most purposes. The symmlet wavelets are smoother.
 
### Computations for B-Splines
 
First we define the augmented knot sequence \\(\tau\\) such that
 
* \\(\tau_1 \le \tau_2 \le \dots \le \tau_M \le \xi_0\\);
* \\(\tau_{j+M}=\xi_j,\,j=1,\dotsc,K\\);
* \\(\xi_{K+1} \le \tau_{K+M+1} \le \tau_{K+M+2} \le \dots \le \tau_{K+2M}\\).
 
The actual values of these additional knots beyond the boundary are arbitrary.
 
Denote by \\(B_{i,m}(x)\\) the \\(i\\)th *B*-spline basis function of order \\(m\\) for the knot-sequence \\(\tau\\), \\(m \le M\\).
\\[ B_{i,1}(x) = \begin{cases} 1\; \; \; if\, \tau_i \le x \le \tau_{i+1} \\\ 0 \; \; \; otherwise \end{cases} \\] for \\(i=1,...,K+2M-1\\). Also known as Haar basis functions.
\\[ B_{i,m}(x) = \frac{x-\tau_i}{\tau_{i+m-1} - \tau_i} B_{i,m-1}(x) + \frac{\tau_{i+m} -x}{\tau_{i+m} - \tau_{i+1}} B_{i+1,m-1}(x)\\] for \\(i=1,...,K+2M-m\\).
 
Least squares computations with \\(N\\) observations and \\(K+M\\) variables (basis functions) take \\(O(N(K+M)^2 + (K+M)^3)\\) flops (floating point operations). If the \\(N\\) observations are sorted, the computational complexity can be reduced to \\(O(N)\\).
 
Computation for smoothing splines are converted to B-splines with manipulation on the dimension of B-spline basis functions.
