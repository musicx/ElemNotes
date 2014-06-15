## Random Forest

### Introduction

Bagging or boostrap aggregation is a technique for reducing the variance of an estimated prediction function. Bagging seems to work especially well for high-variance, low-bias procedures, such as trees. Boosting appears to dominate bagging on most problems.

### Definition of Random Forest

Random Forest for Regression or Classification

1. For \\(b=1\\) to \\(B\\):

    a) Draw a bootstrap sample \\(\mathbf{Z}^*\\) of size \\(N\\) from the training data.

    b) Grow a random-forest tree \\(T_b\\) to the boostrapped data, by recursively repeating the following steps for each terminal node of the tree, until the minimum node size \\(n_{min}\\) is reached.
 
      - Select \\(m\\) variables at random from the \\(p\\) variables.
      - Pick the best variable / split-point among the \\(m\\).
      - Split the node into two daughter nodes.
2. Output the ensemble of trees \\(\{ T_b \}^B_1 \\).

To make a prediction at a new point \\(x\\):

+ *Regression*: \\( \hat{f}^B_{rf}(x) = \frac{1}{B} \sum^B_{b=1} T_b (x) \\).
+ *Classification*: Let \\(\hat{C}_b (x)\\) be the class prediction of the \\(b\\)th random-forest tree. Then \\(\hat{C}^B_{rf}(x)=majority\,vote\,\{\hat{C}_b (x)\}^B_1\\).

The size of the correlation of pairs of bagged trees limits the benefits of averaging. Random Forest improves the variance reduction of bagging by reducing the correlation between the trees, without increasing the variance too much.

Not all estimators can be improved by shaking up the data like this. It seems that highly nonlinear estimators, such as trees, benefit the most. For bootstrapped trees, \\(\rho\\) is typically small (<0.05), while \\(\sigma^2\\) is not much larger than the vaiance for the original tree. On the other hand, bagging does not change linear estimates, such as the sample mean; the pairwise correlation between bootstrapped means is about 50%.

### Details of Random Forest

The inventors make the following recommendations:

* For classification, the default value for \\(m\\) is \\(\lfloor \sqrt{p} \rfloor\\) and the minimum node size is one.
* For regression, the default value for \\(m\\) is \\(\lfloor p/3 \rfloor\\) and the minimum node size is five.

#### Out of Bag Samples

For each observation \\(z_i = (x_i, y_i)\\), construct its random forest predictor by averaging only those trees corresponding to bootstrap samples in which \\(z_i\\) did not appear.

#### Variable Importance

Variable importance can be constructed for random forests in exactly the same way as they were for gradient-boosted models. At each split in each tree, the improvement in the split-criterion is the importance measure attributed to the splitting variable, and is accumulated over all the trees in the forest separately for each variable.

Random forests use the OOB samples to construct a variable importance measure. When the \\(b\\)th tree is grown, the OOB samples are passed down the tree, and the prediction accuracy is recorded. Then the values for the \\(j\\)th variable are randomly permuted in the OOB samples, and the accuracy is again computed. The decrease in accuracy as a result of this permuting is averaged over all trees, and is used as a measure of the importance of variable \\(j\\) in the random forest.

#### Proximity Plots

For every tree, any pair of OOB observations sharing a terminal node has their proximity increased by one. The proximity matrix is then represented in two dimensions using multidimensional scaling. The proximity plot gives an indication of which observations are effectively close together in the eyes of the random forest classifier.

The plots tend to have a star shape, one arm per class, which is more pronounced the better the classification performance.

#### Random Forest and Overfitting

When the number of variables is large, but the fraction of relevant variables small, random forests are likely to perform pooly with small \\(m\\). 

Another claim is that random forests "cannot overfit" the data.


