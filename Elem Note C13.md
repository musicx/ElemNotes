##Prototype Methods and Nearest-Neighbors

### Prototype Methods

Prototype methods represent the training data by a set of points in feature space. These prototypes are typically not examples from the training sample. Each prototype has an associated class label, and classification of a query point \\(x\\) is made to the class of the closest prototype. "Closest" is usually defined by Euclidean distance in the feature space, after each feature has been standardized to have overall mean 0 and variance 1 in the training sample.

These methods can be very effective if the prototypes are well positioned to capture the distribution of each class. The main challange is to fiture out how many prototypes to use and where to put them.

#### K-means Clustering

One chooses the desired number of cluster centres, say \\(R\\), and the K-means procedure iteratively moves the centers to minimize the total within cluster variance.

Given an initial set of centers, the K-means algorithm alternates the two steps:

* for each center we identify the subset of training points (its cluster) taht is closer to it than any other center;
* the means of each feature for the data points in each cluster are computed, and this mean vector becomes the new center for that cluster.

these two steps are iterated until convergence.

To use K-means clustering for classification of labeled data into \\(K\\) classes, the steps are:

*  apply K-means clustering to the training data in each class separately, using \\(R\\) prototypes per class;
*  assign a class label to each of the \\(K*R\\) prototypes;
*  classify a new feature \\(x\\) to the class of the closest prototype.

Shortcoming: for each class, the other classes do not have a say in the positioning of the prototypes for that class.

#### Learning Vector Quantization

LVQ is an *online* algorithm. Observations are processed one at a time.

1. Choose \\(R\\) initial prototypes for each class : \\(m_1(k), m_2(k), ..., m_R(k)\\), \\(k=1,2,...,K\\), for example, by sampling \\(R\\) training points at random from each class.
2. Sample a training point \\(x_i\\) randomly (with replacement), and let \\((j, k)\\) index the closest prototype \\(m_j(k)\\) to \\(x_i\\).
    * a\. If \\(g_i = k\\) (i.e., they are in the same class), move the prototype towards the traing point:
    \\[ m_j(k) \gets m_j(k) + \epsilon(x_i - m_j(k))\\]
    where \\(\epsilon\\) is the *learning rate*. 
    * b\. If \\(g_i \neq k\\) (i.e., they are in different classes), move the prototype away from the training point:
    \\[ m_j(k) \gets m_j(k) - \epsilon(x_i - m_j(k))\\]
3. Repeat step 2, descreasing the learning rate \\(\epsilon\\) with each iteration towards zero.

#### Gaussian Mixtures

Each cluster is described in terms of a Gaussian density, which has a centroid, and a covariance matrix. 

* In the E-step, each observation is assigned a *responsibility* or weight for each cluster, based on the likelihood of each of the corresponding Gaussians. Observations will divide their weights to surrounding centroids according to the distance between them.
* In the M-step, each observation contributes to the weighted means (and covariances) for *every* cluster.

The Gaussian mixture model is often referred to as a *soft* clustering method, while K-means is *hard*.

### \\(k\\)-Nearest-Neighbor Classifiers

These classifiers are *memory-based*, and require no model to be fit. Given a query point \\(x_0\\), we find the \\(k\\) training points \\(x_{(r)}\\), \\(r=1,2,...,k\\) closest in distance to \\(x_0\\), and then classify using majority vote among the \\(k\\) neighbors.

### Adaptive Nearest-Neighbor Methods

The *discriminant adaptive nearest-neighbor* (DANN) metric at a query point \\(x_0\\) is defined by \\[D(x, x_0) = (x-x_0)^T \mathbf{\Sigma} (x-x_0) \\], where 
\\[ \begin{align} \mathbf{\Sigma} &= \mathbf{W}^{-1/2} [\mathbf{W}^{-1/2} \mathbf{B} \mathbf{W}^{-1/2} + \epsilon\mathbf{I} ]  \mathbf{W}^{-1/2} \\
&= \mathbf{W}^{-1/2} [ \mathbf{B}^{*} + \epsilon\mathbf{I} ] \mathbf{W}^{-1/2} \end{align} \\]

Here \\(\mathbf{W}\\) is the pooled within-class covariance matrix \\(\sum^K_{k=1} \pi_k \mathbf{W}_k\\) and \\(\mathbf{B}\\) is the between class covariance matrix \\(\sum^K_{k=1} \pi_k (\bar{x}_k - \bar{x})(\bar{x}_k - \bar{x})^T\\), with \\(\mathbf{W}\\) and \\(\mathbf{B}\\) computed using only the 50 nearest  neighbors around \\(x_0\\). After computation of the metric, it is used in a nearest-neighbor rule at \\(x_0\\).

This formula is simple in operation. It first spheres the data with respect to \\(\mathbf{W}\\), and then stretches the neighborhood in the zero-eigenvalue directions of \\(\mathbf{B}^{*}\\) (the between-matrix for the sphered data). 

The \\(\epsilon\\) parameter rounds the neighborhood, form an infinite strip to an ellipsoid, to avoid using points far away from the query point. The value of \\(\epsilon =1 \\) seems to work well in general.

#### Global Dimension Reduction for Nearest-Neighbors

Hastie and Tibshirani proposed a variation of the discriminant-adaptive nearest-neighbor method for global dimension reduction. At each training point \\(x_i\\), the between-centroids sum of squares matrix \\(\mathbf{B}_i\\) is computed, and then these matrices are averaged over all training points:

\\[ \bar{\mathbf{B}} = \frac{1}{N} \sum^N_{i=1} \mathbf{B}_i \\]

Let \\(e_1, e_2, ..., e_p\\) be the eigenvectors of the matrix \\(\bar{\mathbf{B}}\\), ordered from largest to smallest eigenvalue \\(\theta_k\\). Then these eigenvectors span the optimal subspaces for global subspace reduction. The derivation is based on the fact that the best rank-\\(L\\) approximation to \\(\bar{\mathbf{B}}\\), \\(\bar{\mathbf{B}}_{[L]} = \sum^L_{l=1} \theta_l e_l e_l^T\\), solves the least squares problem
\\[ \underset{rank(M)=L}{\operatorname{min}} \sum^N_{i=1} trace[(\mathbf{B}_i - \mathbf{M})^2] \\]
This formula can be seen as a way of finding the best approximating subspace of dimension \\(L\\) to a series of \\(N\\) subspaces by weighted least squares.

### Computational Consideration

One drawback of nearest-neighbor rules in general is the computational load, both in finding the neighbors and storing the entire training set.

Various *editing* and *condensing* procedures have been proposed to reduce the storage requirements. The idea is to isolate a subset of the training set that suffices for nearest-neighbor predictions, and throw away the remaining training data. Intuitively, it seems important to keep the training points that are near the decision boundaries and on the correct side of those boundaries.
