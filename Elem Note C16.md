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

#### Regularization Paths

The sequence of boosted classifiers form an \\( L_1 \\)-regularized monotone path to a margin-maximizing solution.

Of course the margin-maximizing end of the path can be a very poor, overfit solution. Early stopping amounts to picking a point along the path, and should be done with the aid of a validation dataset.

### Learning Ensembles

The learning process can be broken down into two stages:

+ A finite dictionary \\( \mathcal{T}_L = \{ T_1(x), T_2(x), ..., T_M(x) \} \\) of basis functions in induced from the training data.
+ A family of functions \\( f_\lambda(x) \\) is built by fitting a lasso path in this dictionary:
\\[ \alpha(\lambda) = \mathrm{arg} \underset{\alpha}{\operatorname{min}} \sum_{i=1}^N L[ y_i, \alpha_0 + \sum_{m=1}^M \alpha_m T_m(x_i) ] + \lambda \sum_{m=1}^M |\alpha_m| \\]

#### Learning a Good Ensemble

A good collection of ensembles \\( \mathcal{T}_L \\) covers the space well in places where they are needed, and are sufficiently different from each other for the post-processor to be effective.

A measure of (lack of) relevance that uses the loss function to evaluate on the training data:
\\[ Q(\gamma) = \underset{c_0, c_1}{\operatorname{min}} \sum_{i=1}^N L(y_i, c_0 + c_1 b(x_i; \gamma)) \\]

If a single basis function were to be selected, it would be the global minimizer \\( \gamma^* = \mathrm{arg} \min_{\gamma \in \Gamma} Q(\gamma) \\). Introducing randomness in the selection of \\(\gamma\\) would necessarily produce less optimal values with \\( Q(\gamma) \le Q(\gamma^*) \\). A natural measure of the characteristic *width* \\( \sigma \\) of the sampling scheme \\( \mathcal{S} \\),
\\[ \sigma = \mathrm{E}_\mathcal{S} [Q(\gamma) - Q(\gamma^*)] \\]

+ \\( \sigma \\) too narrow suggests too many of the \\(b(x; \gamma_m)\\) look alike, and similar to \\(b(x; \gamma^*)\\);
+ \\( \sigma \\) too wide implies a large spread in the \\(b(x; \gamma_m)\\), but possibly consisting of many irrelevant cases.

**ISLE Ensemble Generation**

1. \\( f_0(x) = \mathrm{arg} \min_c \sum_{i=1}^N L(y_i, c)\\)
2. For \\( m = 1 \\) to \\(M\\) do
  a) \\( \gamma_m = \mathrm{arg} \min_\gamma \sum_{i \in S_m(\eta)} L(y_i, f_{m-1}(x_i) + b(x_i;\gamma))\\)
  b) \\( f_m(x) = f_{m-1}(x) + \nu b(x; \gamma_m) \\)
3. \\( \mathcal{T}_{ISLE} = \{ b(x;\gamma_1), b(x;\gamma_2), ..., b(x;\gamma_M) \} \\)

\\(S_m(\eta)\\) refers to a subsample of \\(N \cdot \eta \;(\eta \in (0, 1])\\) of the training observations, typically *without* replacement. Simulations suggests picking \\( \eta \le \frac{1}{2} \\), and for large \\(N\\) picking \\( \eta \sim 1 / \sqrt{N} \\). Reducing \\(\eta\\) increases the randomness, and hence the width \\(\sigma\\). The parameter \\(\nu \in [0,1]\\) introduces *memory* into the randomization process.

The authors recommend values \\( \nu=0.1 \\) and \\( \eta \le \frac{1}{2} \\), and call their combined procedure *Importance sampled learning ensemble* (ISLE).

#### Rule Ensembles

From a tree we can generate rules, which describe each node from the root to the leaves. Then all the rules from all the trees in a ensemble can be treated as a new ensemble. There are some advantages to doing so.

+ The space of models is enlarged, and can lead to improved performance.
+ Rules are easier to interpret than trees, so there is the potential for a simplified model.
+ It is often natural to augment \\( \mathcal{T}_{RULE} \\) by including each variable \\( X_j \\) separately as well, thus allowing the ensemble to model linear functions well.
