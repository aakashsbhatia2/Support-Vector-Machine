#Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def create_data():
    X0, y = make_blobs(n_samples=100, n_features = 2, centers=2, cluster_std=1.05, random_state=10)
    X1 = np.c_[np.ones((X0.shape[0])), X0] # add one to the x-values to incorporate bias
    return X0, y, X1

def generate_scatterplot(X0, y):

    #Variable to store 1's and 0's as individual elements in list
    y_temp = []

    #Variables to store x and y values to be plotted (from X0) based on their labels (0 or 1)
    x_val_0 = []
    y_val_0 = []
    x_val_1 = []
    y_val_1 = []
    
    #loop through y to create y_temp
    for i in y:
        y_temp.append(i)

    #loop through X0 to create scatterplot related data
    for i in range(len(X0)):
        if y_temp[i] == 1:
            x_val_1.append(X0[i][0])
            y_val_1.append(X0[i][1])
        else:
            x_val_0.append(X0[i][0])
            y_val_0.append(X0[i][1])

    #Create scatterplot
    plt.scatter(x_val_0, y_val_0, x_val_1, y_val_1)
    plt.show()


def main():
    X0, y, X1 = create_data()
    generate_scatterplot(X0, y)

if __name__ == '__main__':
    main()
