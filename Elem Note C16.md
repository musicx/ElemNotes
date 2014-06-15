## Ensemble Learning

### Introduction

Ensemble learning can be broken down into two tasks: developing a population of base learners from the training data, and then combining them to form the composite predictor. In this chapter we discuss boosting technology that goes a step further; it builds an ensemble model by conducting a regularized and supervised search in a high-dimensional space of weak learners.

### Boosting and Regularization Paths

#### Penalized Regression

The regularization has the form as :
\\[ \underset{\alpha}{\operatorname{min}} \left\{ \sum_{i=1}^N \left( y_i - \sum_{k=1}^K \alpha_k T_k (x_i) \right)^2 + \lambda \cdot J(\alpha) \right\} \\]
\\( J(\alpha) \\) is a function of the coefficients that generally penalizes larger values. Examples are 
\\[ \begin{align} J(\alpha) &= \sum_{k=1}^K |\alpha_k|^2 \;\;\;\;\;\; ridge\;regression \\
J(\alpha) &= \sum_{k=1}^K |\alpha_k| \;\;\;\;\;\;\; lasso \end{align} \\]

Owing to the very large number of basis functions \\( T_k \\), directly solving the fomular with lasso penalty is not possible. However, a feasible forward stagewise strategy exists that closely approximates the effect of the lasso.

Forward Stagewise Linear Regression

1. Initialize \\( \breve{\alpha}_k = 0 \\), \\(k=1, ..., K\\). Set \\( \epsilon > 0 \\) to some small constant, and \\( M \\) large.
2. For \\( m = 1\\) to \\(M\\) :
  a) \\( (\beta^*, k^*) = \mathrm{arg} \min_{\beta, k} \sum_{i=1}^N \left(y_i - \sum_{l=1}^K \breve{\alpha}_l T_l (x_i) - \beta T_k(x_i) \right)^2 \\)
  b) \\( \breve{\alpha}_{k^*} \gets \breve{\alpha}_{k^*} + \epsilon \cdot \mathrm{sign}{(\beta^*)} \\)
3. Output \\( f_M(x) = \sum_{k=1}^K \breve{\alpha}_k T_k(x) \\)

Tree boosting with shrinkage closely resembles the algorithm above, with the learning raet parameter \\( \nu \\) corresponding to \\( \epsilon \\)

With a small fraction of dominant variables, best subset approaches often work well. But with a moderate fraction of strong variables, it is well known that subset selection can be excessively greedy, often yielding poor results when compared to less aggressive strategies such as lasso or ridge regression. The dramatic improvements often seen when shrinkage is used with boosting are yet another confirmation of this approach.

#### The "Bet on Sparsity" Principle

Use a procedure that does well in sparse problems, since no probcedure does well in dense problems.

Qualifications:

+ For any given application, the degree of sparseness/denseness depends on the unknown true target function, and the chosen dictionary \\( \mathcal{T} \\).
+ The notion of sparse versus dense is relative to the size of the training data set and/or the noise-to-signal ratio (NSR). Larger training sets allow us to estimate coefficients with smaller standard errors. Likewise in situations with small NSR, we can identify more nonzero coefficients with a given sample size than in situations where the NSR is larger.
+ The size of the dictionary plays a role as well. Increasing the size of the dictionary may lead to a sparser representation of our function, but the search problem becomes more difficult leading to higher variance.
