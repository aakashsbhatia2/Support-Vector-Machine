# Support-Vector-Machine

You will first generate a small toy-dataset using the Scikit-learn1

library in Python.

Such a toy dataset can be quickly generates as follows:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
X0, y = make_blobs(n_samples=100, n_features = 2, centers=2,
cluster_std=1.05, random_state=10)

X1 = np.c_[np.ones((X0.shape[0])), X0] # add one to the x-values to incorporate bias

You can see what your sample looks like, using the plt.scatter() and plt.show() functions.

The next step is to create a data dictionary and see what the highest feature value is in your sample
dataset. You can create the data dictionary are follows:
positive_x =[]
negative_x =[]
for i,label in enumerate(y):
if label == 0:
negative_x.append(X[i])
else:
positive_x.append(X[i])
data_dict = {-1:np.array(negative_x), 1:np.array(positive_x)}


and then get the highest feature value
max_fval = float(’-inf’)

for y_i in data_dict:
if np.amax(data_dict[y_i]) > max_fval:
max_fval=np.amax(data_dict[y_i])


Your programming task for this assignment is to train a SVM classifier on your sample data.
Note that this is fundamentally an optimization problem, and you should use the gradient descent
algorithm to solve it (as discussed in lecture). It is easy to step across the minimum if you are not
adjusting your step size properly. To resolve that, I would advise you to progressively decrease the
step size using the highest feature value that you obtained earlier. Perhaps the easiest approach is
to simply iterate over the data dictionary using smaller and smaller step sizes by maintaining an
array of the form
step_sizes = [max_fval * 0.1, max_fval * 0.01, max_fval * 0.001, ... ]

I: Training SVM 25 points
Write a function called train(data dict), and train your SVM using this function on the first
80% of your sample data. You do not need to shuffle the data at this stage, since the sample data
generation code already performs shuffling.

II: Visualizing the data and the maximum-margin separator 10 points
Write a function called draw(), which will show the sample data along with the maximum-margin
separating hyperplane. An example of such a visualization is shown below:

III: Testing the fit of the classifier 5 points
Write another function called test, and test the remaining 20% of your sample data using the
hyperplane obtained through your train method. When you have obtained the result, simply add
the result in a README.txt file. The description should only include two things: the total number
of data points on which your test method was run, and how many of those data points ended up
being misclassified.

Notes
The programming language and library has been fixed for this homework. This was a conscious
decision, mainly because any graduate student of machine learning, data science, etc. should have
working knowledge of at least one or two of the most widely-used machine learning libraries.