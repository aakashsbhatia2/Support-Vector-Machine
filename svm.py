#Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import random

def create_data():
    X0, y = make_blobs(n_samples=100, n_features = 2, centers=2, cluster_std=1.05, random_state=10)
    X1 = np.c_[np.ones((X0.shape[0])), X0] # add one to the x-values to incorporate bias
    return X0, y, X1

def train_svm(samples_train,labels_train, T):
    theta = [0,0,0] 
    w = [0,0,0] 
    avg_w = [0,0,0] 

    for t in range(1,T):
        for i in range(len(w)):
            w[i] = w[i] - 0.02/t*theta[i]
            avg_w[i] += w[i]
        sample = int(random.uniform(0,len(samples_train)-1))
        x_i = samples_train[sample]
        if labels_train[sample] == 0:
            y_i = 0
        else:
            y_i = 1
        if y_i*np.dot(w,x_i) < 1:
            for i in range(len(w)):
                theta[i] += 0.01*y_i*x_i[i]
    for i in range(len(avg_w)): 
        avg_w[i] = avg_w[i]/(T) 
    print(avg_w)
    """avg_w=w
    print(avg_w)"""
    generate_scatterplot(samples_train, labels_train, avg_w)

def generate_scatterplot(samples_train, labels_train, avg_w):
    plt.scatter(samples_train[:, 1], samples_train[:, 2], c=labels_train);
    ax = plt.gca()
    xlim = ax.get_xlim()

    xx = np.linspace(xlim[0], xlim[1])

    yy = -(avg_w[1]/avg_w[2]) * xx - (avg_w[0]/avg_w[2])
    yy1 = 4-(avg_w[1]/avg_w[2]) * xx - (avg_w[0]/avg_w[2])
    yy2 = -4-(avg_w[1]/avg_w[2]) * xx - (avg_w[0]/avg_w[2])
    plt.plot(xx, yy, c = 'r')
    plt.plot(xx, yy1, linestyle='dashed', c = 'c')
    plt.plot(xx, yy2, linestyle='dashed', c = 'c')
    plt.show()

def split_data(data, labels):
    samples_train = []
    labels_train = []
    samples_test = []
    labels_test = []

    for i in range(len(data)):
        if i < len(data)*0.80:
            samples_train.append(data[i])
            labels_train.append(labels[i])
        else:
            samples_test.append(data[i])
            labels_test.append(labels[i])
            
    samples_train = np.array(samples_train)
    labels_train = np.array(labels_train)
    samples_test = np.array(samples_test)
    labels_test = np.array(labels_test)
    return samples_train, labels_train, samples_test, labels_test 

def main():
    X0, y, X1 = create_data()
    samples_train, labels_train, samples_test, labels_test = split_data(X1, y)
    train_svm(samples_train,labels_train, 100)

if __name__ == '__main__':
    main()
