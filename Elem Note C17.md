## Undirected Graphical Models

### Introduction

*Undirected graph* is also known as *Markov random fields* or *Markov networks*.

The absence of an edge between two vertices means the corresponding random variables are conditionally independent, given the other variables.

### Markov Graphs and Their Properties

+ Pairwise Markov independencies of \\(\mathcal{G}\\) : if no edge joining \\(X\\) and \\(Y\\) \\(\iff\\) \\( X \bot Y | rest \\)
+ Global Markov properties of \\(\mathcal{G}\\) : if \\(C\\) separates \\(A\\) and \\(B\\) then \\( A \bot B | C \\)

### Undirected Graphical Models for Continuous Variables

The inverse covaiance matrix \\( \mathbf{\Theta} = \Sigma^{-1} \\) contains information about the *partial covariances* between the variables. If the \\(ij\\)th component is zero, the variables \\(i\\) and \\(j\\) are conditionally independent, given the other variables.

\\(\mathbf{\Theta}\\) captures all the second-order information needed to describe the conditional distribution of each node given the rest, and is the so-called "natural" parameter for the Gaussian graphical model.

#### Estimation of the Parameters when the Graph Structure is Known

Let \\(\mathbf{S}\\) be the empirical convariance matrix, the log-likelihood of the data can be written as 
\\[ \ell(\mathbf{\Theta}) = \log{\det{\mathbf{\Theta}}} - \mathrm{trace}(\mathbf{S\Theta}) \\]

The quantity \\(-\ell(\mathbf{\Theta})\\) is a convex function of \\(\mathbf{\Theta}\\).

**A Modified Regression Algorithm for Estimation of an Undirected Gaussian Graphical Model with Known Structure**

1. Initialize \\( \mathbf{W} = \mathbf{S} \\)
2. Repeat for \\( j=1,2,...,p \\) until convergence:
  a) Partition the matrix \\( \mathbf{W} \\) into part 1: all but the \\(j\\)th row and column, and part 2: the \\(j\\)th row and column.
  b) Solve \\( \mathbf{W}_{11}^* \beta^* - s_{12}^* = 0 \\) for the unconstrained edge parameters \\( \beta^* \\), using the reduced system of equations. Obtain \\( \hat{\beta} \\) by padding \\( \hat{\beta}^* \\) with zeros in the appropriate positions.
  c) Update \\( w_{12} = \mathbf{W}_{11} \hat{\beta} \\)
3. In the final cycle (for each \\(j\\)) solve for \\( \hat{\theta}_{12} = - \beta \cdot \hat{\theta}_{22} \\), with \\( 1/\hat{\theta}_{22} = s_{22} - w_{12}^T \hat{\beta} \\)

#### Estimation of the Graph Structure

**Graphical Lasso**

1. Initialize \\( \mathbf{W}=\mathbf{S} + \mathbf{\lambda I} \\). The diagonal of \\(\mathbf{W}\\) remains unchagned in what follows.
2. Repeat for \\( j=1,2,...,p,1,2,...,p,... \\) until convergence:
  a) Partition the matrix \\(\mathbf{W}\\) into part 1: all but the \\(j\\)th row and column, and part 2: the \\(j\\)th row and column.
  b) Solve the estimating equations \\( \mathbf{W}_{11} \beta - s_{12} + \lambda \cdot \mathrm{Sign}(\beta) = 0 \\) using the cyclical coordinate-descent algorithm for the modified lasso.
  c) Update \\(w_{12} = \mathbf{W}_{11} \hat{\beta}\\)
3. In the final cycle (for each \\(j\\)) solve for \\( \hat{\theta}_{12} = - \beta \cdot \hat{\theta}_{22} \\), with \\( 1/\hat{\theta}_{22} = w_{22} - w_{12}^T \hat{\beta} \\)

### Undirected Graphical Models for Discrete Variables

Pairwise Markov networks with binary variables are also called *Ising models* and *Boltzmann machines*, where the veritces are referred to as "nodes" or "units" and are binary-valued.

Denoting the binary valued variable at node \\(j\\) by \\(X_j\\), the Ising model for their joint probabilities is given by 
\\[ p(X, \mathbf{\Theta}) = \exp \left[ \sum_{(j,k) \in E} \theta_{jk} X_j X_k - \Phi(\mathbf{\Theta}) \right] \; for\; X \in \mathcal{X} \\]
with \\( \mathcal{X} = \{ 0,1 \}^p \\). As with the Gaussian model of the previous section, only pairwise interactions are modeled. \\( \Phi(\mathbf{\Theta}) \\) is the log of the partition function, and is defined by
\\[ \Phi({\mathbf{\Theta}}) = \log \sum_{x \in \mathcal{X}} \left[ \exp \left( \sum_{(j,k) \in E} \theta_{jk} x_j x_k \right) \right] \\]

The Ising model implies a logistic form for each node conditional on the others:
\\[ \mathrm{Pr}(X_j = 1|X_{-j}=x_{-j}) = \frac{1}{1+ \exp( -\theta_{j0} - \sum_{(j,k) \in E} \theta_{jk} x_k )} \\]

#### Estimation of the Parameters when the Graph Strucutre is Known

The maximum likelihood estimates simply match the estimated inner products between the nodes to their observed inner products. To find the maximum likelihood estimates, we can use gradient search or Newton methods. 

For smaller \\(p\\), a number of standard statistical approaches are available.

+ Poisson log-linear modeling: \\(O(p^4 2^p)\\), \\(p < 20\\)
+ Gradient descent: \\( O(p^2 2^{p-2}) \\), \\(p \le 30\\)
+ Iterative proportional fitting: One complete cycle costs the same as a gradient evaluation, but may be more efficient.

When \\(p\\) is large, other approaches have been used to approximate the gradient.

+ The mean field approximation.
+ Gibbs sampling.

#### Restricted Boltzmann Machines

A restricted Boltzmann machine (RBM) consists of one layer of visible units and one layer of hidden units with no connections within each layer. It has the same generic form as a single hidden layer neural network.

+ The neural network minimizes the error (cross-entropy) between the targets and their model predictions, conditional on the input features.
+ The RBM maximizes the log-likelihood for the joint distribution of all visible units -- the features and targets.

*Contrastive divergence*: The learning works well if we estimate the second expectation by starting the Markov chain at the data and only running for a few steps instead of to convergence. The idea is that when the parameters are far from the solution, it may be wasteful to iterate the Gibbs sampler to stationarity, as just a single iteration will reveal a good direction for moving the estimates.
