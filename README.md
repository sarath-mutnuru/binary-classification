# binary-classification using LDA
Binary classification using LDA , PCA and Max likelihood estimation

Here we use Linear Discriminant Analysis to perform Binary classification.
The Data has faces and Non-faces in it.
We use Principal Component Analysis on the images and perform LDA on the newly obtained features
Userdefined functions are written for performing PCA

**Short Theory**

We basically do a MAP estimate g_hat = argmax P(Y=g|X)= argmax{ P(X|g) * P(Y=g) }
We now need to calculate P(X|g) which is generally called class conditional distribution
We model this as a Multivariate Gaussian and estimate the Mean and Co-Variance using Max likelihood estimation.
We model two such Gaussians one for each class.

**PCA**

We do PCA on the data and perform the above method on the newly obtained data.
We see there will be increase in accuracy( recall and precision)

The data is included.

main.py is the main code and rest are modules required.

**Requirements**

numpy,sklearn,opencv in python
