# -*- coding: utf-8 -*-
"""INFO523_final_project_Jaron_F.ipynb


# Principal Component Analysis (PCA)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import plotly.express as px
from sklearn.decomposition import PCA

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # covariance, function needs samples as columns
        cov = np.cov(X.T)

        # eigenvalues, eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # -> eigenvector v = [:,i] column vector, transpose for easier calculations
        # sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # store first n eigenvectors
        self.components = eigenvectors[0 : self.n_components]
        

    def transform(self, X):
        # project data
        #X = X - self.mean
        return np.dot(X, self.components.T)

"""# Iris Dataset"""

#2-D Plotly

# data = datasets.load_digits()
iris = datasets.load_iris()
X = iris.data
y = iris.target

y_label = []

for i in y:
  if i == 0:
    y_label.append('Setosa')
  elif i == 1:
    y_label.append('Versicolor')
  else:
    y_label.append('Virginica')

Species = pd.DataFrame(y_label,columns=['Species'])



# Project the data onto the 2 primary principal components
pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)


print("Shape of X:", X.shape)
print("Shape of transformed X:", X_projected.shape)


X_df = pd.DataFrame(X_projected,columns=['PC1','PC2'])
X_y_df = df_scores = pd.concat([X_df, Species], axis=1)

fig = px.scatter(X_y_df, x='PC1', y='PC2',color='Species')

fig.show()

# Testing 3-D Plotly

# data = datasets.load_digits()
iris = datasets.load_iris()
X = iris.data
y = iris.target

y_label = []

for i in y:
  if i == 0:
    y_label.append('Setosa')
  elif i == 1:
    y_label.append('Versicolor')
  else:
    y_label.append('Virginica')

Species = pd.DataFrame(y_label,columns=['Species'])

# Project the data onto the 2 primary principal components
pca = PCA(3)
pca.fit(X)
X_projected = pca.transform(X)

print("Shape of X:", X.shape)
print("Shape of transformed X:", X_projected.shape)


X_df = pd.DataFrame(X_projected,columns=['PC1','PC2','PC3'])
X_y_df = df_scores = pd.concat([X_df, Species], axis=1)

fig = px.scatter_3d(X_y_df, x='PC1', y='PC2', z='PC3',color='Species')

fig.show()

"""# Wine Dataset"""

#2-D Plotly

# data = datasets.load_digits()
wine = datasets.load_wine()
X = wine.data
y = wine.target


y_label = []

for i in y:
  if i == 0:
    y_label.append('Wine Type 1')
  elif i == 1:
    y_label.append('Wine Type 2')
  else:
    y_label.append('Wine Type 3')

WineType = pd.DataFrame(y_label,columns=['WineType'])

# Project the data onto the 2 primary principal components
pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)

print("Shape of X:", X.shape)
print("Shape of transformed X:", X_projected.shape)


X_df = pd.DataFrame(X_projected,columns=['PC1','PC2'])
X_y_df = df_scores = pd.concat([X_df, WineType], axis=1)

fig = px.scatter(X_y_df, x='PC1', y='PC2',color='WineType')

fig.show()

# Testing 3-D Plotly

# data = datasets.load_digits()
wine = datasets.load_wine()
X = wine.data
y = wine.target

y_label = []

for i in y:
  if i == 0:
    y_label.append('Wine Type 1')
  elif i == 1:
    y_label.append('Wine Type 2')
  else:
    y_label.append('Wine Type 3')

WineType = pd.DataFrame(y_label,columns=['WineType'])

# Project the data onto the 2 primary principal components
pca = PCA(3)
pca.fit(X)
X_projected = pca.transform(X)

print("Shape of X:", X.shape)
print("Shape of transformed X:", X_projected.shape)


X_df = pd.DataFrame(X_projected,columns=['PC1','PC2','PC3'])
X_y_df = df_scores = pd.concat([X_df, WineType], axis=1)
print(X_y_df)

fig = px.scatter_3d(X_y_df, x='PC1', y='PC2', z='PC3',color='WineType')

fig.show()

"""# Heart Disease Dataset"""

from google.colab import files
uploaded = files.upload()
heart_data= pd.read_csv('heart.csv',header=None,skiprows=1)

#2-D Plotly

X,y = heart_data.values[:,:-1], \
heart_data.values[:,-1]


y_label = []

for i in y:
  if i == 0:
    y_label.append('Healthy')
  else:
    y_label.append('Heart Disease')

HealthStatus = pd.DataFrame(y_label,columns=['HealthStatus'])

# Project the data onto the 2 primary principal components
pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)

print("Shape of X:", X.shape)
print("Shape of transformed X:", X_projected.shape)


X_df = pd.DataFrame(X_projected,columns=['PC1','PC2'])
X_y_df = df_scores = pd.concat([X_df, HealthStatus], axis=1)
print(X_y_df)

fig = px.scatter(X_y_df, x='PC1', y='PC2',color='HealthStatus')

fig.show()

# Testing 3-D Plotly

# data = datasets.load_digits()
X,y = heart_data.values[:,:-1], \
heart_data.values[:,-1]


y_label = []

for i in y:
  if i == 0:
    y_label.append('Healthy')
  else:
    y_label.append('Heart Disease')

HealthStatus = pd.DataFrame(y_label,columns=['HealthStatus'])

# Project the data onto the 2 primary principal components
pca = PCA(3)
pca.fit(X)
X_projected = pca.transform(X)

print("Shape of X:", X.shape)
print("Shape of transformed X:", X_projected.shape)


X_df = pd.DataFrame(X_projected,columns=['PC1','PC2','PC3'])
X_y_df = df_scores = pd.concat([X_df, HealthStatus], axis=1)
print(X_y_df)

fig = px.scatter_3d(X_y_df, x='PC1', y='PC2', z='PC3',color='HealthStatus')

fig.show()

"""# Breast Cancer Dataset"""

#2-D Plotly

# data = datasets.load_digits()
breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target



y_label = []

for i in y:
  if i == 0:
    y_label.append('Benign')
  else:
    y_label.append('Malignant')

CancerStatus = pd.DataFrame(y_label,columns=['CancerStatus'])

# Project the data onto the 2 primary principal components
pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)

print("Shape of X:", X.shape)
print("Shape of transformed X:", X_projected.shape)


X_df = pd.DataFrame(X_projected,columns=['PC1','PC2'])
X_y_df = df_scores = pd.concat([X_df, CancerStatus], axis=1)

fig = px.scatter(X_y_df, x='PC1', y='PC2',color='CancerStatus')

fig.show()

#3-D Plotly

# data = datasets.load_digits()
breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target



y_label = []

for i in y:
  if i == 0:
    y_label.append('Benign')
  else:
    y_label.append('Malignant')

CancerStatus = pd.DataFrame(y_label,columns=['CancerStatus'])

# Project the data onto the 3 primary principal components
pca = PCA(3)
pca.fit(X)
X_projected = pca.transform(X)

print("Shape of X:", X.shape)
print("Shape of transformed X:", X_projected.shape)


X_df = pd.DataFrame(X_projected,columns=['PC1','PC2','PC3'])
X_y_df = df_scores = pd.concat([X_df, CancerStatus], axis=1)

fig = px.scatter_3d(X_y_df, x='PC1', y='PC2', z='PC3',color='CancerStatus')

fig.show()
