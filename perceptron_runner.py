import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from perceptron import Perceptron
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)


y = df.iloc[0:100, 4].values # Pull out 100 class labels 
y = np.where(y == 'Iris-setosa', -1, 1) # assign each label to its integer class label (Versicolor is 1, Setosa is -1)
X = df.iloc[0:100, [0,2]].values # grab first and second column (sepal length, petal length) and assign then to a feature matrix to visualize them
plt.scatter(X[:50, 0], X[:50, 1], color = 'red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.show()
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()
breakpoint()
