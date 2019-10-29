#!/usr/bin/env python
# coding: utf-8

# An unsupervised learning algorithm: application to the discrimination of seismic events and quarry blasts in the vicinity of Istanbul

# Dependencies


import os
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
import seaborn as sns
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
from sklearn import mixture
from matplotlib.colors import LogNorm

get_ipython().run_line_magic('matplotlib', 'inline')


# Reading data

def read__data():
    return pd.read_csv('.\\data\\eqdata2.csv')


table = read__data()

# Distribution of Data

print(table.head())

print(table.describe())

# Histograms

table.plot.hist(bins=30, alpha=0.75)

table.groupby(['y'])['c'].hist(bins=20, alpha=0.75)

table.groupby(['y'])['sr'].hist(bins=30, alpha=0.75)

table.groupby(['y'])['sp'].hist(bins=30, alpha=0.75)

table.groupby(['y'])['t'].hist(bins=10, alpha=0.75)

# Corelation


plt.scatter(table['c'][0:150], table['sp'][0:150], c='blue')
plt.scatter(table['c'][151:], table['sp'][151:], c='red')
plt.xlabel('Complexity');
plt.ylabel('S/P')
plt.show()

plt.scatter(table['c'][0:150], table['t'][0:150], c='blue')
plt.scatter(table['c'][151:], table['t'][151:], c='red')
plt.xlabel('Complexity');
plt.ylabel('Time')

plt.scatter(table['c'][0:150], table['sr'][0:150], c='blue')
plt.scatter(table['c'][151:], table['sr'][151:], c='red')
plt.xlabel('Complexity');
plt.ylabel('Spectral Ratio')

sns.set(style="ticks")
sns.pairplot(table, hue="y")

# Feature Importance with Tree Classifier

T = table.to_numpy()
X = T[:, 0:4]
y = T[:, -1]
forest = ExtraTreesClassifier(n_estimators=10,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

# Principal Component Analysis

T = table.to_numpy()
X = T[:, 0:4]
y = T[:, -1]
pca = decomposition.PCA(n_components=2)
pca.fit(X)
K = pca.transform(X)

print(pca.explained_variance_ratio_)

yy = np.expand_dims(y, axis=1)
yy.shape

T = np.hstack((K, yy))

dataset = pd.DataFrame({'Column1': T[:, 0], 'Column2': T[:, 1], 'Column3': T[:, 2], })
dataset.describe()

sns.set(style="ticks")
sns.pairplot(dataset, hue="Column3")

table['clog'] = np.log(table['c'])
table['splog'] = np.log(table['sp'])
table.drop(['sr', 't'], axis=1)

sns.set(style="ticks")
sns.pairplot(table, hue="y")

plt.scatter(table['clog'][0:150], table['sp'][0:150], c='blue')
plt.scatter(table['clog'][151:], table['sp'][151:], c='red')

plt.scatter(table['c'][0:150], table['splog'][0:150], c='blue')
plt.scatter(table['c'][151:], table['splog'][151:], c='red')

plt.scatter(table['clog'][0:150], table['splog'][0:150], c='blue')
plt.scatter(table['clog'][151:], table['splog'][151:], c='red')
plt.xlabel('log(Complexity)');
plt.ylabel('log(S/P)')
plt.show()

aab = table.groupby(['y'])['splog'].hist(bins=30, alpha=0.75)

aab = table.groupby(['y'])['clog'].hist(bins=20, alpha=0.75)

# Quarry Blasts
cqb = np.array(table['clog'][0:150])
spqb = np.array(table['splog'][0:150])
Xqb = np.concatenate((np.expand_dims(cqb, axis=1), np.expand_dims(spqb, axis=1)), axis=1)

# Earthquakes
ceq = np.array(table['clog'][150:])
speq = np.array(table['splog'][150:])
Xeq = np.concatenate((np.expand_dims(ceq, axis=1), np.expand_dims(speq, axis=1)), axis=1)

# All data
c = np.array(table['clog'])
sp = np.array(table['splog'])
X = np.concatenate((np.expand_dims(c, axis=1), np.expand_dims(sp, axis=1)), axis=1)

# covariance of all data
np.cov(ceq, speq)


def estimateGaussion(x):
    # The input X is the dataset with each n-dimensional data point in one row
    # The output is an n-dimensional vector mu, the mean of the data set
    # and the variances sigma^2, an n x 1 vector
    mu = x.mean(axis=0)
    sigma2 = x.std(axis=0)
    cov = np.cov(ceq, speq)
    return mu, sigma2, cov


mu, sigma, cov = estimateGaussion(Xeq)


def multivariateGaussian(X, mu, cov):
    Xm = X - mu
    Cov_det = np.linalg.det(cov)
    Cov_inv = np.linalg.inv(cov)
    N = (2 * np.pi) ** (-2 / 2) * Cov_det ** (-0.5)
    EN = np.exp(-0.5 * np.sum(np.multiply(np.matmul(Xm, Cov_inv), Xm), axis=1))
    return N * EN


p = multivariateGaussian(X, mu, cov)

N = 60
XX = np.linspace(0, 3.5, N)
YY = np.linspace(-0.5, 2, N)
XX, YY = np.meshgrid(XX, YY)

# Mean vector and covariance matrix
mu = np.array([ceq.mean(), speq.mean()])
Sigma = np.cov(ceq, speq)
# Sigma = np.diag(np.array([np.std(ceq),np.std(aeq)]))

# Pack X and Y into a single 3-dimensional array
pos = np.empty(XX.shape + (2,))
pos[:, :, 0] = XX
pos[:, :, 1] = YY


def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.
    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2 * np.pi) ** n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)

    return np.exp(-fac / 2) / N


# The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(pos, mu, Sigma)

# Create a surface plot and projected filled contour plot under it.
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(XX, YY, Z, rstride=2, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)

cset = ax.contourf(XX, YY, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

# Adjust the limits, ticks and view angle
ax.set_zlim(-0.15, 0.6)
ax.set_zticks(np.linspace(0, 0.5, 6))
ax.view_init(27, -21)

plt.show()

fig, ax = plt.subplots()
CS = ax.contour(XX, YY, Z, levels=[0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1], colors=['green'])
plt.scatter(table['clog'][0:150], table['splog'][0:150], c='blue')
plt.scatter(table['clog'][151:], table['splog'][151:], c='red')
plt.xlabel('log(Complexity)');
plt.ylabel('log(S/P)')
plt.show()

test = np.expand_dims(np.array([2, 1]), axis=0)
multivariateGaussian(test, mu, cov)

# Finding the best epsilon


bestEpsilon = 0
bestF1 = 0
F1 = 0

listss = np.linspace(p.min(), p.max(), 1000)
for epsilon in listss:
    pred = (p <= epsilon) * 1
    F1 = metrics.f1_score(y, pred)

    if F1 > bestF1:
        bestF1 = F1
        bestEpsilon = epsilon

pred = (p <= bestEpsilon) * 1
pred


def look_result(y, pred):
    df = {'f1': metrics.f1_score(y, pred), 'recall': metrics.recall_score(y, pred),
          'precision': metrics.precision_score(y, pred)}
    cm = metrics.confusion_matrix(y, pred)
    print(df)
    print(cm)
    return


look_result(y, pred)

bestEpsilon

fig, ax = plt.subplots()
CS = ax.contour(XX, YY, Z, levels=[bestEpsilon], colors=['green'])
plt.scatter(table['clog'][0:150], table['splog'][0:150], c='blue')
plt.scatter(table['clog'][151:], table['splog'][151:], c='red')
plt.scatter(X[pred == 1, 0], X[pred == 1, 1], s=2, c='red', marker='o')

# GaussianMixture from Sklearn


c = np.array(table['clog'])
sp = np.array(table['splog'])
Xtt = np.concatenate((np.expand_dims(c, axis=1), np.expand_dims(sp, axis=1)), axis=1)

clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(Xtt)

# display predicted scores by the model as a contour plot
x = np.linspace(-2., 3.5)
y = np.linspace(-3., 2.)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(Xtt[:, 0], Xtt[:, 1], .8)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()

test = np.expand_dims(np.array([2, 1]), axis=0)
attt = clf.predict_proba(test)
attt

multivariateGaussian(test, mu, cov)

attt = clf.predict(Xtt)
attt = attt ^ 1
attt

look_result(yy, attt)

# K-Means


kmeans = KMeans(n_clusters=2, random_state=0).fit(Xtt)
kttt = kmeans.labels_
kttt

look_result(yy, kttt)


# PCA


def read__data():
    return pd.read_csv('.\\data\\eqdata2.csv')


table = read__data()
table['clog'] = np.log(table['c'])
table['alog'] = np.log(table['a'])
table.drop(['y'], axis=1)

Xs = table.to_numpy()

Xs = table.to_numpy()
pca = decomposition.PCA(n_components=2, svd_solver='full')
pca.fit(Xs)
print(pca.explained_variance_ratio_)

Zs = pca.transform(Xs)

# c=np.array(table['clog'])
# a=np.array(table['alog'])
# Xtt=np.concatenate((np.expand_dims(c, axis=1),np.expand_dims(a, axis=1)), axis=1)
clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(Zs)

atts = clf.predict(Zs)
# atts= atts ^ 1

look_result(yy, atts)
