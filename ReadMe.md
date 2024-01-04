
This a classic image segmentation problem, which is part of the homework
assignments for the course ECE 271A at UCSD for Fall 2023.

This is a brief summary of the solution. Refer the pdf in each folder
for detailed explanation.

# **Core Problem**

Given an image of cheetah, we need to create a binary mask to segment
the image into foreground(cheetah) and background(grass).
<style>
td, th {
   border: none!important;
}
</style>
Cheetah | cheetah_mask
:--------: |:--------:
![](/Users/reyasadhu/Downloads/vertopal.com_Readme/images/media/image1.png) |![](/Users/reyasadhu/Downloads/vertopal.com_Readme/images/media/image2.png)
The mask on the right is given to check the performance of our model i.e compare our masks to this given mask.

# **Problem Formulation:**

For classification purposes, Bayesian discrimination rule (BDR) is
applied with minimum probability of error criteria. Representing each
pixel feature by x and each class by i,

$$Predicted\ class = \ i^{*}(x) = \begin{matrix}
argmax \\
i \\
\end{matrix}P_{Y|X}\left( i \middle| x \right) = \begin{matrix}
argmax \\
i \\
\end{matrix}P_{X|Y}\left( x \middle| i \right)P_{Y}(i)$$

Here i can be either cheetah or grass and for each pixel, x can be a
scalar or vector depending on which feature we are using. To extract the
features, the image was broken into 8×8 block images. A two-dimensional
discrete cosine transform (DCT) is performed on the blocks. To get a
one-dimensional feature vector, the resulting transformed block is
rearranged following zig-zag pattern.

# **Training Data:**

As a training data we are given a dataset consisting of 64 DCT
coefficients of cheetah and grass pixels. The file
***TrainingSamplesDCT_8.mat*** contains a training set of vectors
obtained from a similar image (stored as a matrix, each row is a
training vector) for each of the classes. There are two matrices,
***TrainsampleDCT_BG*** and ***TrainsampleDCT_FG*** for foreground and
background samples respectively.

# **Solution:**

## 1.  **Second Maximum Energy Method:**

The first method is very simple and uses only one feature of the sample
vector, which is called second maximum energy method. Unlike the others,
it does not explicitly assume a probabilistic model. Instead, the main
underlying assumption is that the two classes, the grass and the cheetah
have different textural patterns that would elicit better discrimination
between the two classes in the distribution of frequency components than
in the distribution of intensities in the spatial domain.

From the DCT vector, we choose the index with the second largest energy
or absolute value as a feature to compute the discriminating function
with. The class-conditional probabilities,
$P_{X|Y}\left( x \middle| i \right)$, based on the feature parameter x
as defined above, was estimated by constructing a histogram of such
feature values based on the training data given and normalizing it with
the total number training examples for each class. The probabilities for
the histogram bins are as such equivalent to the maximum likelihood
estimates for a multinomial likelihood with number of different possible
outcomes equal to the number of bins in the histogram.

Similarly, the class priors, $P_{Y}(i)$ were estimated by dividing the
number of class examples in the training data with the total number of
training examples.
<div style="text-align: center">
<img src="/Users/reyasadhu/Downloads/vertopal.com_Readme/images/media/image3.jpeg" width="600" height="300" class="center">
</div>

This method produces the following mask with a probability of error of 16.63%.
<div style="text-align: center">
<img src="/Users/reyasadhu/Downloads/vertopal.com_Readme/images/media/image4.jpg" width="350" height= "350" class="center">
</div>


## 2.  **Parametric Methods:**

In parametric approaches, our assumption is that we have an independent
and identically distributed sample
$D = \left\{ x_{1},x_{2},\ldots\ ,\ x_{N} \right\}.\ \ $ We assume the
samples $x_{i}$'s are drawn from some known probability density family,
$P_{X}$(x; θ), parameterized by the vector θ, for example Gaussian. If we
assume the density of individual sample, we know the likelihood of the
entire dataset as the samples are i.i.d.

$P_{T}(D;\theta) = \prod_{i = 1}^{N}{P_{X}\left( x_{i};\theta \right)}$

The main advantage of this method is that if we estimate the parameters
of the distribution: for example, mean and variance for a Gaussian
distribution, the whole distribution is known. We estimate the
parameters of the distribution from the training set, plug in these
estimates to the assumed model and get an estimated distribution, which
we then use for classification. There are two methods we will discuss to
estimate the parameters of a distribution: Maximum Likelihood (ML) and
Bayesian parameter estimation.

### 1.  **Maximum Likelihood (ML) Estimation:**

In ML estimation, we search for the value of θ that maximizes the likelihood of the sample. For convenience, we can maximize its log(.) in order to convert the product into a sum and to lead to further computational simplification. Then our classification problem can be summed up by these two equations,

$ i^{*}(x) = \begin{matrix}
argmax \\
i \\
\end{matrix}\left( \ \log{P_{X|Y}\left( x \middle| i;\theta_{i}^{*} \right)} + \log{P_{Y}(i)} \right) $

$ \theta_{i}^{*} = \begin{matrix}
argmax \\
\theta \\
\end{matrix}P_{T|Y}(D|i,\theta) $

When we apply ML into our problem, we assume that the class conditional densities are multivariate Gaussians of 64 dimensions. Then under ML assumption, the mean and variance of the Gaussian Distribution can be estimated by the sample mean and variance.

Given the training samples in ***TrainingSamplesDCT_8\_new.mat ,*** we estimate the mean and the variance of the likelihood function $P_{X|Y}\left( x \middle| i \right)$. The prior probability is the same as the previous method, as it was the maximum likelihood estimate. Here, using these estimations we can visualize the feature distribution for all 64 features. The plots will look like these.
<style>
td, th {
   border: none!important;
}
</style>
|   |    |
:--------: |:--------:
![](/Users/reyasadhu/Downloads/vertopal.com_Readme/images/media/image5.jpeg) |![](/Users/reyasadhu/Downloads/vertopal.com_Readme/images/media/image6.jpeg)

Now, the best features for the classification purpose will be where there is a considerable difference between $P_{X|Y}\left( X \middle| cheetah \right)$ and $P_{X|Y}(X|grass)$ for all x's. Except feature 1, all the other features overlap each other with an almost similar mean. So, we choose the distributions based on the spread(variance). By a visual inspection, we choose the best 8 features to be \[1,7,8,9,12,14,18,27\] and the worst 8 features to be \[3,4,5,59,60,62,63,64\].

By keeping the plots side by side , we can clearly see the difference.
<style>
td, th {
   border: none!important;
}
</style>
|   |    |
:--------: |:--------:
![](/Users/reyasadhu/Downloads/vertopal.com_Readme/images/media/image7.jpeg) | ![](/Users/reyasadhu/Downloads/vertopal.com_Readme/images/media/image8.jpeg)

For the best features, the two conditional distributions are clearly separated from each other, while in the worst features they are overlapping each other. Which means worst features are nearly same for both classes, and thus not reliable for classification purpose.

After creating the mask, we can calculate the probability of error.

With 64 features, P(error) = 8%

With 8 best features, P(error) = 5%
<style>
td, th {
   border: none!important;
}
</style>
|    |  |
:---------------------: |:--------------------:
![](/Users/reyasadhu/Downloads/vertopal.com_Readme/images/media/image9.jpg) | ![](/Users/reyasadhu/Downloads/vertopal.com_Readme/images/media/image10.jpg)


We can see using the 8 best features gives us a better result than using all the 64 features. Its because all the 64 features include those features which have a similar density function and thus cannot distinguish the classes. So, they skew the classification decision and give more error. This is a case of high dimensionality, where we should only consider the useful dimensions rather than all to get a better accuracy.

### 2.  **Bayesian Parametric Estimation:**

The main difference between Maximum Likelihood estimation and Bayesian estimation is how we look at the parameter θ (mean and variance for gaussian distribution) : a fixed value or a random variable. Bayesian estimation assumes that θ is a random variable with a prior density $P_{\theta}(\theta)$.

For our problem, we assume $P_{x|\theta}(x|\theta)$ to be G(x, µ, Σ). The parameter θ here is only µ because Σ is computed from the sample covariance, which is a plausible tweak to have Σ.

So, $P_{x|\theta}(x|\theta)$ =$P_{x|\theta}(x|µ)$ . In addition, we also assume the prior distribution, $P_{\theta}(\theta) = P_{µ}(µ)$ to be G(µ, $\mu_{0}$, $\Sigma_{0}$). The two parameters, $\mu_{0}$ and $\Sigma_{0}$, are given. By multiplying the likelihood and the prior, we can compute the posterior $P_{\theta|T}(µ|D)$ and thanks to a good property of the Gaussian distribution, $P_{\theta|T}(µ|D)$ is also a Gaussian distribution whose mean and variance can be calculated from $\mu_{0}$, $\Sigma_{0}$, Σ.

Then we can calculate the predictive distribution $P_{x|T}(x|D)$ or $P_{X|i}(x|i,D)$ for each class, which we can then plug into the BDR to get the classification.

$$P_{x|T}\left( x \middle| D \right) = \int_{}^{}{P_{x|\theta}\left( x \middle| \mu \right)P_{\theta|T}\left( \mu \middle| D \right)\ d\mu}$$

For our problem, we have used $\Sigma_{0}$ as a diagonal matrix,
$ {(\Sigma_{0})}_{ii} = \alpha w_{i} $, with given $\alpha$ and $w\ $
and we see the performance of the model by changing the value of $\alpha$. We also use two different strategies for the value of $\mu_{0}$.
For one strategy, the $\mu_{0}$ is different for the two classes , and for the second strategy its same for both classes. We apply these on four training dataset of different sizes.

### 3.  **MAP Estimation:**
The method of maximum a posteriori (MAP) estimation can be used to obtain unknown parameters that maximize the posterior probability density function. It is closely related to ML estimate but this method incorporates the prior distribution. This feature of MAP provides regularization of ML estimates. In MAP estimation, we assume that $P_{\theta|T}(\theta|D)$ has a narrow peak around its mode, thereby allowing us to avoid computing the integral in the Bayesian estimate.
 As a result, our decision function becomes,

$$i^{*}(x) = \begin{matrix}
argmax \\
i \\
\end{matrix}\left( \ \log{P_{X|Y}\left( x \middle| i;\theta_{i}^{MAP} \right)} + \log{P_{Y}(i)} \right)$$

$$\theta_{i}^{*} = \begin{matrix}
argmax \\
\theta \\
\end{matrix}P_{\theta|T}\left( \theta \middle| D,i \right) = \begin{matrix}
argmax \\
\theta \\
\end{matrix}P_{T|Y,\theta}\left( D \middle| i,\theta \right)P_{\theta|Y}\left( \theta \middle| i \right)\ $$

Refer the pdf on the ***Bayesian Parametric Estimation folder*** to understand the relative performance of these three parametric methods on different sized datasets and for different priors.

## 3.  **Expectation- Maximization (EM) algorithm:**

EM algorithm is a powerful algorithm for finding maximum likelihood
estimates with missing data. For our problem, we assume that the class
conditional probability is a weighted mixture of Gaussians with diagonal
covariance matrix.

$$P_{x|i}\left( x \middle| i \right) = \sum_{c = 1}^{C}{G\left( x,\mu_{c},\Sigma_{c} \right)\pi_{c}}$$

The problem here is that we do not know the component labels beforehand.
Thus, this is a problem of Expectation Maximization. We must first
assume the number of Gaussian components present and also the number of
features (dimensions) that we want to work in. After deciding these
numbers, we run the EM algorithm on the training data set to find
${\mu_{c},\Sigma_{c},\pi}_{c}$.. Once the Gaussians are known we use BDR
to classify the test data. The EM algorithm requires the initial
parameter estimates to be initialized. In this assignment, we will
initialize them at random and explore the effect of different
initializations on the final probability of error. In addition, we also
look at the effect that the dimensionality of the feature vector has on
the probability of error.

![](vertopal_4a86ab8f50ea4ec088e9268a9851090b/media/image11.emf)

We can observe that the error decreases with the increase in dimensions.
But also, after a certain value of dimensions it started to increase.
This is similar to the part 2 (Maximum Likelihood Estimation), where we
have seen best 8 features work better than all 64 features.

Also, for c=1, i.e., if we consider the likelihood to be made of only 1
gaussian, that has significantly more error than the other component
numbers. So, the likelihood is definitely not made of any 1 component.

For c=4,8, or 16, the error is the lowest. So, using one of these values
can give us the best segmentation result. However, we cannot tell the
precise optimal number for the number of components from this single
experiment.
