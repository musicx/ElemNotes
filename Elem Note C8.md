## Model Inference and Averaging
 
### Bootstrap and Maximum Likelihood Methods
 
* Nonparametric bootstrap : sample with replacement from the training data.
* Parametric bootstrap : simulate new responses by adding Gaussian noise to the predicted value.
 
#### Maximum Likelihood Inference
 
We denote a probability density for our observations \\(z_i \sim g_\theta(z)\\), where \\(\theta\\) represents one or more unkonwn parameters that govern the distribution of \\(Z\\). This is called a *parametric model* for \\(Z\\).
 
Maximum likelihood is based on the *likelihood function*, given by \\[L(\theta ; \mathbf{Z}) = \prod^N_{i=1} g_\theta(z_i)\\] the probability of the boserved data under the model \\(g_\theta\\).
 
Denote the logarithm of \\(L(\theta ; \mathbf{Z})\\) by
\\[ \ell(\theta ; \mathbf{Z}) = \sum^N_{i=1} \ell(\theta ; z_i) = \sum^N_{i=1} \log{g_\theta(z_i)} \\]
As \\(\mathbf{Z}\\) is fixed, this is sometimes abbreviated as \\(\ell(\theta)\\).
 
The method of maximum likelihood chooses the value \\(\theta = \hat{\theta}\\) to maximize \\(\ell(\theta)\\).
 
The *information matrix* is \\[ \mathbf{I}(\theta) = - \sum^N_{i=1} \frac{\partial^2 \ell(\theta; z_i)}{\partial \theta \partial \theta^T} \\]
When \\( \mathbf{I}(\theta) \\) is evaluated at \\(\theta = \hat{\theta}\\), it is often called the *observation information*.
 
The *Fisher information* (or expected information) is \\[ \mathbf{i}(\theta) = \mathrm{E}_\theta [\mathbf{I} (\theta)] \\]
 
And let \\(\theta_0\\) denote the true value of \\(\theta\\).
 
The sampling distribution of the maximum likelihood estimator has a limiting normal distribution:
\\[ \hat{\theta} \to N(\theta_0 , \mathbf{i}(\theta_0)^{-1}) \\]
as \\(N \to \infty\\).
 
#### Bootstrap vs. Maximum Likelihood
 
In essence the bootstrap is a computer implementation of nonparametric or parametric maximum likelihood.
 
### Bayesian Methods
 
In the Bayesian approach to inference, we specify a sampling model \\(\mathrm{Pr}(\mathbf{Z} | \theta) \\) (density or probability mass function) for our data given the parameters, and a prior distribution for the parameters  \\(\mathrm{Pr}(\theta) \\) reflecting our knowledge about \\(\theta\\) before we see the data. We then compute the posterior distribution
\\[\mathrm{Pr}( \theta | \mathbf{Z} ) = \frac{\mathrm{Pr}(\mathbf{Z} | \theta) \cdot \mathrm{Pr}(\theta)} {\int \mathrm{Pr}(\mathbf{Z} | \theta) \cdot \mathrm{Pr}(\theta) d\theta}\\]
which represents our updated knowledge about \\(\theta\\) after we see the data.
 
The posterior distribution also provides the basis for predicting the values of a futre observation \\(z^{new}\\), via the *predictive distribution* :
\\[\mathrm{Pr}(z^{new} | \mathbf{Z}) = \int \mathrm{Pr}(z^{new} | \theta) \cdot \mathrm{Pr}(\theta | \mathbf{Z})  d\theta\\]
 
In constrast, the maximum likelihood approach would not account for the uncertainty in estimating \\(\theta\\).
 
### The EM Algorithm
 
#### EM Algorithm for Two-component Gaussian Mixture
 
1\. Take initial guesses for the parameters \\(\hat{\mu_1}\\), \\(\hat{\sigma_1}^2\\), \\(\hat{\mu_2}\\), \\(\hat{\sigma_2}^2\\), \\(\hat{\pi}\\). Then the log-likelihood would be
\\[ \ell(\theta; \mathbf{Z}) = \sum^N_{i=1} \log{[ (1-\pi) \phi_{\theta_1}(y_i) + \pi \phi_{\theta_2}(y_i) ]} \\]
where \\(\theta = (\mu, \sigma^2) \\).
 
2\. *Expectation Step* : compute the responsibilities
\\[ \hat{\gamma_i} = \frac {\hat{\pi} \phi_{\hat{\theta_2}}(y_i)} {(1-\hat{\pi}) \phi_{\hat{\theta_1}}(y_i) + \hat{\pi} \phi_{\hat{\theta_2}}(y_i)} , \; i=1,2,...,N. \\]
where \\(\gamma_i(\theta) = \mathrm{E}(\Delta_i | \theta, \mathbf{Z}) = \mathrm{Pr}(\Delta_i = 1 | \theta, \mathbf{Z})\\) is called the *responsibility* of model 2 for observation \\(i\\). \\(\Delta_i\\) is a latent parameter that if \\(\Delta_i = 1\\) then \\(Y_i\\) comes from model 2, otherwise it comes from model 1. And the log-likelihood would be
 \\[ \ell(\theta; \mathbf{Z, \Delta}) = \sum^N_{i=1} [ (1-\Delta_i) \log{\phi_{\theta_1}(y_i)} + \Delta_i \log{\phi_{\theta_2}(y_i)} ] \\\ + \sum^N_{i=1} [ (1-\Delta_i) \log{(1 - \pi)} + \Delta_i \log{\pi} ] \\]
 
3\. *Maximization Step* : compute the weighted means and variances :
\\[\begin{align} \hat{\mu_1} &= \frac{\sum^N_{i=1} (1-\hat{\gamma_i}) y_i} {\sum^N_{i=1} (1-\hat{\gamma_i})} \\\ \hat{\sigma_1}^2 &= \frac {\sum^N_{i=1} (1-\hat{\gamma_i}) (y_i - \hat{\mu_1})^2} {\sum^N_{i=1} (1-\hat{\gamma_i})}  \\\ \hat{\mu_2} &= \frac{\sum^N_{i=1} \hat{\gamma_i} y_i} {\sum^N_{i=1} \hat{\gamma_i}}  \\\ \hat{\sigma_2}^2 &= \frac {\sum^N_{i=1} \hat{\gamma_i} (y_i - \hat{\mu_1})^2} {\sum^N_{i=1} \hat{\gamma_i}} \end{align} \\]
and the mixing probability \\(\hat{\pi} = \sum^N_{i=1} \hat{\gamma_i} \, / \, N\\).
 
4\. Iterate steps 2 and 3 until convergence. 
 
#### The generaliezed EM Algorithm
 
1\. Start with initial guesses for the parameters \\(\hat{\theta}^{(0)}\\).
 
2\. *Expectation Step*: at the \\(j\\)th step, compute \\[Q(\theta', \hat{\theta}^{(j)}) = \mathrm{E}(\ell_0 (\theta'; \mathbf{T}) | \mathbf{Z}, \hat{\theta}^{(j)} ) \\]
as a function of the dummy argument \\(\theta'\\).
 
3\. *Maximization Step*: detemine the new estimate \\(\hat{\theta}^{(j+1)}\\) as the maximizer of \\(Q(\theta', \hat{\theta}^{(j)})\\) over \\(\theta'\\). In fact, we need only to find a value \\(\hat{\theta}^{(j+1)}\\) so that \\(Q(\hat{\theta}^{(j+1)}, \hat{\theta}^{(j)}) > Q(\hat{\theta}^{(j)}, \hat{\theta}^{(j)})\\). Such procedures are called *GEM (generalized EM)* algorithms.
 
4\. Iterate steps 2 and 3 until convergence.
 
### Markov Chain Monte Carlo for Sampling from the Posterior
 
Gibbs sampling, an MCMC procedure, samples from the conditional distributions to approximate the joint distribution.
 
The Gibbs Smapler :
1. Take some initial values \\(U_k^{(0)}\\), \\(k=1,2,...,K\\).
2. Repeat for \\(t=1,2,...,. \\) : For \\(k=1,2,...,K\\) generate \\(U_k^{(t)}\\) from \\(\mathrm{Pr} (U_k^{(t)} | U_1^{(t)}, ... , U_{k-1}^{(t)}, U_{k+1}^{(t-1)}, ..., U_K^{(t-1)}) \\).
3. Continue step 2 unitil the joint distribution of \\( ( U_1^{(t)}, U_2^{(t)}, ..., U_K^{(t)}  ) \\) *does not change*.
 
Gibbs sampling produces a Markov chain whose stationary distribution is the true joint distribution.
 
### Bagging
 
Suppose we fit a model to our training data \\(\mathbf{Z} = \{ (x_1, y_1), (x_2, y_2), ..., (x_N, y_N) \}\\), obtaining the prediction \\(\hat{f} (x)\\) at input \\(x\\). Bootstrap aggregation or *bagging* averages this prediction over a collection of bootstrap samples, thereby reducing its variance. For each bootstrap sample \\(\mathbf{Z}^{*b}\\), \\(b=1,2,...,B\\), we fit our model, giving prediction \\(\hat{f}^{*b}(x)\\). The bagging estimate is defined by
\\[ \hat{f}_{bag} (x) = \frac{1}{B} \sum^B_{b=1}\hat{f}^{*b}(x) \\]
 
This bagged estimate is a Monte Carlo estimate of the true bagging estimate, approaching it as \\(B \to \infty\\). And it will differ from the original estimate only when the latter is a nonlinear or adaptive function of the data.
 
Bagging can dramatically reduce the variance of unstable procedures like trees, leading to improved prediction, because averaging reduces variance and leaves bias unchanged.
 
The above argument does not hold for classification under 0-1 loss, because of the nonadditivity of bias and variance. Bagging a good classifier can make it better, and vice versa.
 
### Model Averaging and Stacking
 
#### Averaging
 
We have a set of candidate models \\(\mathcal{M}_m\\), \\(m=1,...,M\\) for our training set \\(\mathbf{Z}\\). Suppose \\(\zeta\\) is some quantity of interest, eg, a prediction \\(f(x)\\) at some fixed feature value \\(x\\). The posterior distribution of \\(\zeta\\) is
\\[ \mathrm{Pr}(\zeta | \mathbf{Z}) = \sum^M_{m=1}  \mathrm{Pr}(\zeta | \mathcal{M}_m, \mathbf{Z})   \mathrm{Pr}(\mathcal{M}_m | \mathbf{Z}) \\]
with posterior mean
\\[ \mathrm{E}(\zeta | \mathbf{Z}) = \sum^M_{m=1} \mathrm{E}(\zeta | \mathcal{M}_m, \mathbf{Z}) \mathrm{Pr}(\mathcal{M}_m | \mathbf{Z}) \\]
 
*Committee methods* take a simple unweighted average of the predictions from each model.
 
Combining models never makes things worse, at the population level. Of course the population linear regression is not available, and it is natural to replace it with the linear regression over the training set. But there are simple examples where this does not work well, possibly because model complexity is not taken into account.
 
#### Stacking
 
*Stacked generalization*, or *stacking*, is a way of avoiding giving unfairly high weight to models with higher complexity.
 
Let \\(\hat{f_m}^{-i}(x)\\) be the prediction at \\(x\\), using model \\(m\\), applied to the dataset with the \\(i\\)th training observation removed. The stacking weights are given by :
\\[ \hat{w}^{st} = \underset{w}{\operatorname{argmin}} \sum^N_{i=1} \left [ y_i - \sum^M_{m=1} w_m \hat{f_m}^{-i}(x_i) \right ]^2 \\]
The final prediction is \\(\sum_m{\hat{w_m}^{st}\hat{f_m}(x)}\\). Better results can be obtained by restricting the weights to be nonnegative, and to sum to 1.
 
The stacking idea is very general. One acn use any learning method to combine the models, and the weights could also depend on the input location \\(x\\).
 
### Stochastic Search: Bumping
 
*Bumping* uses bootstrap sampling to move randomly through model space, so that can avoid the model getting stuck in poor solutions of local minima.
 
We draw bootstrap samples and fit a model to each. We then choose the model that produces the smallest prediction error, averaged over the *original training set*. For squared error, we choose the model from bootstrap sample \\(\hat{b}\\) where
\\[ \hat{b} = \underset{b}{\operatorname{arg\,min}} \sum^N_{i=1} [y_i - \hat{f}^{*b}(x_i)]^2\\]
 
 
 
