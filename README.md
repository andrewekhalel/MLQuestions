# Computer Vision Interview Questions
A collection of technical interview questions for computer vision engineering positions.

#### 1) Given stride S and kernel sizes  for each layer of a (1-dimensional) CNN, create a function to compute the [receptive field](https://www.quora.com/What-is-a-receptive-field-in-a-convolutional-neural-network) of a particular node in the network. This is just finding how many input nodes actually connect through to a neuron in a CNN. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

#### 2) Implement [connected components](http://aishack.in/tutorials/labelling-connected-components-example/) on an image/matrix. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]


#### 3) Implement a sparse matrix class in C++. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]


#### 4) Create a function to compute an [integral image](https://en.wikipedia.org/wiki/Summed-area_table), and create another function to get area sums from the integral image.[[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]


#### 5) How would you [remove outliers](https://en.wikipedia.org/wiki/Random_sample_consensus) when trying to estimate a flat plane from noisy samples? [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

#### 6) How does [CBIR](https://www.robots.ox.ac.uk/~vgg/publications/2013/arandjelovic13/arandjelovic13.pdf) work? [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

#### 7) How does image registration work? Sparse vs. dense [optical flow](http://www.ncorr.com/download/publications/bakerunify.pdf) and so on. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

#### 8) Describe how convolution works. What about if your inputs are grayscale vs RGB imagery? What determines the shape of the next layer? [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

#### 9) Talk me through how you would create a 3D model of an object from imagery and depth sensor measurements taken at all angles around the object. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

#### 10) Implement SQRT(const double & x) without using any special functions, just fundamental arithmetic. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

#### 11) Reverse a bitstring. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

#### 12) Implement non maximal suppression as efficiently as you can. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

#### 13) Reverse a linked list in place. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

#### 14) What's the trade-off between bias and variance? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]

If our model is too simple and has very few parameters then it may have high bias and low variance. On the other hand if our model has large number of parameters then it’s going to have high variance and low bias. So we need to find the right/good balance without overfitting and underfitting the data. [[src]](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229)

#### 15) What is gradient descent? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
[[Answer]](https://towardsdatascience.com/gradient-descent-in-a-nutshell-eaf8c18212f0)

#### 16) Explain over- and under-fitting and how to combat them? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
[[Answer]](https://towardsdatascience.com/overfitting-vs-underfitting-a-complete-example-d05dd7e19765)

#### 17) How do you combat the curse of dimensionality? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]

 - Manual Feature Selection
 - Principal Component Analysis (PCA)
 - Multidimensional Scaling
 - Locally linear embedding
[[src]](https://towardsdatascience.com/why-and-how-to-get-rid-of-the-curse-of-dimensionality-right-with-breast-cancer-dataset-7d528fb5f6c0)

#### 18) What is regularization, why do we use it, and give some examples of common methods? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
A technique that discourages learning a more complex or flexible model, so as to avoid the risk of overfitting. 
Examples
 - Ridge (L2 norm)
 - Lasso (L1 norm)
The obvious *disadvantage* of **ridge** regression, is model interpretability. It will shrink the coefficients for least important predictors, very close to zero. But it will never make them exactly zero. In other words, the *final model will include all predictors*. However, in the case of the **lasso**, the L1 penalty has the effect of forcing some of the coefficient estimates to be *exactly equal* to zero when the tuning parameter λ is sufficiently large. Therefore, the lasso method also performs variable selection and is said to yield sparse models.
[[src]](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a)

#### 19) Explain Principal Component Analysis (PCA)? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
[[Answer]](https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c)

#### 20) Why is ReLU better and more often used than Sigmoid in Neural Networks? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
Imagine a network with random initialized weights ( or normalised ) and almost 50% of the network yields 0 activation because of the characteristic of ReLu ( output 0 for negative values of x ). This means a fewer neurons are firing ( sparse activation ) and the network is lighter. [[src]](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0)

## Contributions
Contributions are most welcomed.
 1. Fork the repository.
 2. Commit your *questions* or *answers*.
 3. Open **pull request**.