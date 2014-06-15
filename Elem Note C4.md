## Linear Methods for Classification

### Classification with Linear Regression

\\(\hat{\mathbf{Y}}\\) is a \\(N \times K\\) *indicator response matrix*.

\\[\hat{\mathbf{Y}}=\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}\\]

Then we define \\(\hat{\mathbf{B}}\\) so that \\(\hat{\mathbf{Y}}=\mathbf{X}\hat{\mathbf{B}}\\). And a new observation is classified as follows:

- compute the fitted output \\(\hat{f}(x)^T=(1,x^T)\hat{\mathbf{B}}\\), a \\(K\\) vector;
- identify the largest component and classify accordingly: \\[\hat{G}(x)=\operatorname{arg\,max}_{k \in \mathcal{G}} \hat{f}_k(x).\\]

A big problem of this is that when classes \\(K \ge 3\\), classes can be *masked* by others.

![regression](img\c4-1.png)

To resolve such worst-case scenarios, one would need general polynomial terms and cross-products of total degree \\(K-1\\), \\(O(p^{K-1})\\) terms in the \\(p\\)-dimensional input space.

### Linear Discriminant Analysis

We can model class density as multivariate Gaussian function \\[f_k(x)=\frac{1}{(2\pi)^{p/2} \left | \mathbf{\Sigma}_k \right |^{1/2}} e^{- \frac{1}{2} (x-\mu_k)^T \mathbf{\Sigma}^{-1}_k {(x - \mu_k)}}.\\]

\\(\mathrm{LDA}\\) is a special case when we assume that the classes have a common covariance matrix \\(\mathbf{\Sigma}_k = \mathbf{\Sigma}\ \forall{k}\\).

The *linear discriminant functions* is \\[\delta_k(x)=x^T\mathbf{\Sigma}^{-1}\mu_k - \frac{1}{2} \mu_k^T\mathbf{\Sigma}^{-1}\mu_k + \log{\pi_k}\\]. It is an equaivalent description of the decision rule, with \\(G(x)=\operatorname{arg\,max}_k\delta_k(x)\\).

Notice that there is no Gaussian assumption for the features as of now. \\(\mathrm{LDA}\\) can be exteded beyond the realm of Gaussian data. However, when classifying with \\(\delta_k\\), if fomula \\(x^T \hat{\mathbf{\Sigma}}^{-1}(\hat{\mu}_2 - \hat{\mu}_1) > {}^1/_2 (\hat{\mu}_2 + \hat{\mu}_1)^T\hat{\mathbf{\Sigma}}^{-1}(\hat{\mu}_2 - \hat{\mu}_1) - \log{(N_2/N_1)}\\) is used, Gaussian assumption is made in this step. So empirical cut-off should be chosen with other methods to avoid the Gaussian assumption in this case.

* \\(\hat{\pi}_k = N_k/N\\), where \\(N_k\\) is the number of class-\\(k\\) observations;
* \\(\hat{\mu}_k = \sum_{g_i=k}{x_i/N_k}\\);
* \\(\hat{\mathbf{\Sigma}} = \sum_{k=1}^K \sum_{g_i=k} {(x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T } /(N-K)\\).

If the \\(\mathbf{\Sigma}_k\\) are not assumed to be equal, then we get *quadratic discriminant functions* (\\(\mathrm{QDA}\\)) as \\[\delta_k(x) = - \frac{1}{2} \log{\left | \mathbf{\Sigma}_k \right |} - \frac{1}{2} (x-\mu_k)^T \mathbf{\Sigma}_k^{-1} (x-\mu_k) + \log{\pi_k}\\]

#### Regularized Discriminant Analysis

The regularized covariance matrices have the form \\[\hat{\mathbf{\Sigma}}_k(\alpha) = \alpha\hat{\mathbf{\Sigma}}_k + (1-\alpha)\hat{\mathbf{\Sigma}},\\]

And similar modifications allow \\(\hat{\mathbf{\Sigma}}\\) itself to be shrunk tward the scalar covariance, \\[\hat{\mathbf{\Sigma}}(\gamma) = \gamma\hat{\mathbf{\Sigma}} +(1-\gamma)\hat{\sigma}^2\mathbf{I}\\]

#### Computation for LDA

- Do a egien decomposition \\(\hat{\mathbf{\Sigma}} =\mathbf{UD}\mathbf{U}^T\\). And **sphere** the data so \\(X^{*} \leftarrow \mathbf{D}^{-\frac{1}{2}} \mathbf{U}^T X\\). The common covariance estimatie of \\(X^{*}\\) will now be the identidy.
- Classify to the closest class centroid in the transformed space, modulo the effect of the class prior probabilities \\(\pi_k\\).

#### Reduced-rank LDA

The \\(K\\) centroids in \\(p\\)-dimensional input space line in an affine subspace of dimension \\(\leq\ K-1\\). We can just project the \\(X^*\\) onto this centroid-spanning subspace \\(H_{K-1}\\), and make distance comparisons there. This is used as dimension reduction.

Finding the sequences of optimal subspaces for LDA involves the following steps:

* compute the \\(K \times p\\) matrix of class centroids \\(\mathbf{M}\\) and the common covariance matrix \\(\mathbf{W}\\) (for *within-class* covariance);
* compute \\(\mathbf{M}^{*}=\mathbf{M}\mathbf{W}^{-{}^1/_2}\\) using the eigen-decomposition of \\(\mathbf{W}\\)
* compute \\(\mathbf{B}^{*}\\), the covariance matrix of \\(\mathbf{M}^{*}\\) (\\(\mathbf{B}\\) for *between-class* covariance), and its eigen-decomposition \\(\mathbf{B}^{*}=\mathbf{V}^{*}\mathbf{D}_B\mathbf{V}^{*T}\\). The columns \\(v^{*}_l\\) for \\(\mathbf{V}^{*}\\) in sequence from first to last define the coordinates of the optimal subspaces.
* Combining all these operations the \\(l\\)th *discriminant variable* is given by \\(Z_l=v_l^TX\\) with \\(v_l=\mathbf{W}^{-{}^1/_2}v_l^{*}\\).

Fisher also get the same result with a different method, which does not make Gaussian assumption. And it can be writen as to maximize the *Rayleigh quotient*, \\[\underset{a}{\operatorname{max}} \frac{a^T\mathbf{B}a}{a^T\mathbf{W}a}.\\]

This is a generalized eigenvalue problem, with \\(a\\) given by the largest eigenvalue of \\(\mathbf{W}^{-1}\mathbf{B}\\). The optimal \\(a_i\\) is just identical as \\(v_i\\) in earlier equation. And the \\(a_l\\) are referred to as *discriminant coordinates*.

### Logistic Regression

Logistic regression models are usually fit by maximum likelihood, using the conditional likelihood of \\(G\\) given \\(X\\), which can be modeled with *multinomial* distribution.

For 2 classes problem, the log-likelihood can be written \\[\ell(\beta) = \sum^N_{i=1}\left \{y_i\beta^Tx_i - \log{(1+e^{\beta^Tx_i})} \right \}. \\]

To maximize the log-likelihood, we set its derivatives to zero. The score equations are \\[\frac{\partial\ell(\beta)}{\partial\beta}=\sum^N_{i=1}{x_i(y_i-p(x_i;\beta))}=0,\\]

To solve the equations above, we use the Newton-Raphson algorithm, which requires the second-derivative or Hessian matrix \\[\frac{\partial^2\ell(\beta)}{\partial\beta\partial\beta^T}=-\sum^N_{i=1}{x_ix_i^Tp(x_i;\beta)(1-p(x_i;\beta))}.\\]

Starting with \\(\beta^{old}\\), a single Nowton update is \\[\beta^{new}=\beta^{old} - \left ( \frac{\partial^2\ell(\beta)}{\partial\beta\partial\beta^T} \right )^{-1} \frac{\partial\ell(\beta)}{\partial\beta},\\] where the derivatives are evaluated at \\(\beta^{old}\\).

Define \\(\mathbf{p}\\) the vector of fitted probabilities with \\(i\\)th element \\(p(x_i;\beta^{old})\\) and \\(\mathbf{W}\\) a \\(N \times N\\) diagonal matrix of weights with \\(i\\)th diagonal element \\(p(x_i;\beta^{old})(1-p(x_i;\beta^{old}))\\). Then we have
\\[\frac{\partial\ell(\beta)}{\partial\beta}=\mathbf{X}^T(\mathbf{y}-\mathbf{p})\\]
\\[\frac{\partial^2\ell(\beta)}{\partial\beta\partial\beta^T}=-\mathbf{X}^T \mathbf{WX}\\]

The Newton step is thus \\[\beta^{new}=(\mathbf{X}^T \mathbf{WX})^{-1}\mathbf{X}^T \mathbf{W}\left ( \mathbf{X} \beta^{old} + \mathbf{W}^{-1}(\mathbf{y}-\mathbf{p}) \right ) \\
=(\mathbf{X}^T \mathbf{WX})^{-1}\mathbf{X}^T \mathbf{Wz}\\]

\\(\mathbf{z}\\) is known as the *adjusted response*.

This algorithm is referred to as *iteratively reweigted least squares* or \\(\mathrm{IRLS}\\), since each iteration solves the weighted least squares problem: \\[\beta^{new} \leftarrow \underset{\beta}{\operatorname{arg\ min}}(\mathbf{z}-\mathbf{X}\beta)^T\mathbf{W}(\mathbf{z}-\mathbf{X}\beta).\\]

#### Quadratic Approximations and Inference

- *Rao score test* can be used for inclusion of a term.
- *Wald test* can be used for exclusion of a term.
- These tests do NOT require iterative fitting, and are based on the current model fit.

#### Logistic Regression vs. LDA

The log-odds of these 2 models share the same format \\[ \log{\frac{\mathrm{Pr}(G=k|X=x)}{\mathrm{Pr}(G=K|X=x)}} = \beta_{k0} + \beta^T_k x\\]

However, logistic regression model is more general, making less assumptions. We can write the *joint density* of \\(X\\) and \\(G\\) as \\[\mathrm{Pr}(X,G=k)=\mathrm{Pr}(X)\mathrm{Pr}(G=k|X)\\]

Logistic regression model leaves the marginal density of \\(X\\) as an arbitrary density function. And in LDA model, the marginal density does play a role \\[\mathrm{Pr}(X)=\sum^K_{k=1}{\pi_k\phi(X;\mu_k,\Sigma)}\\]

This assumption can help to estimate the parameters more efficiently (lower variance). But with 30% more data, the conditional likelihood will do as well.

### Separating Hyperplanes

Separating hyperplanes have the format as \\[\{x:\hat{\beta}_0 + \hat{\beta}_1x_1 + \hat{\beta}_2 x_2 = 0 \}.\\]

Classifiers with such format, that compute a linear combination of the input features and return the sign, were called *perceptrons*.

#### Rosenblatt's Perceptron Learning Algorithm

If a response \\(y_i=1\\) is misclassified, then \\(x^T_i\beta + \beta_0 < 0\\). The goal is to minimize \\[D(\beta, \beta_0)=-\sum_{i \in \mathcal{M}}{y_i(x^T_i\beta + \beta_0)}\\]

where \\(\mathcal{M}\\) indexes the set of misclassified points.

The algorithm uses *stochastic gradient descent* to minimize this piecewise linear criterion.

#### Optimal Separating Hyperplanes

The *optimal separating hyperplane* separates the two classes and maximizes the distance to the closest point from either class. The problem is phrased as
\\[\underset{\beta, \beta_0, \|\beta\|=1}{\operatorname{max}}M \\\ \mathrm{subject\ to\ }y_i(x^T_i\beta+\beta_0) \geq M, i=1,...,N\\]

The solution is obtained by maximizing the Wolfe dual \\(L_D\\) in the positive orthant, a convex optimization problem.
\\[L_D = \sum^N_{i=1}{\alpha_i} - \frac{1}{2}\sum^N_{i=1}\sum^N_{k=1}{\alpha_i \alpha_k y_i y_k x^T_i x_k}\\]
\\[\mathrm{subject\ to\ } \alpha_i \geq 0 \mathrm{\ and\ } \sum^N_{i=1}{\alpha_i y_i} = 0\\]
\\[ \beta = \sum^N_{i=1}{\alpha_i y_i x_i}\\]
\\[ 0 = \sum^N_{i=1}{\alpha_i y_i}\\]

And the solution must satisfy the *Karush-Kuhn-Tucker* conditions
\\[\alpha_i [y_i(x^T_i \beta + \beta_0) -1] = 0 \forall i\\]


