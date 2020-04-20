#Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import random
import math

def create_data():
    X0, y = make_blobs(n_samples=100, n_features = 2, centers=2, cluster_std=1.05, random_state=10)
    
    positive_x =[]
    negative_x =[]
    for i,label in enumerate(y):
        if label == 0:
            negative_x.append(X0[i])
        else:
            positive_x.append(X0[i])
    data_dict = {-1:np.array(negative_x), 1:np.array(positive_x)}
    
    list_vals = []
    for i in data_dict:
        if i == -1:
            for val in data_dict[i]:
                val= val.tolist()
                val.append(-1)
                val.insert(0,1)
                list_vals.append(val)
        else:
            for val in data_dict[i]:
                val = val.tolist()
                val.append(1)
                val.insert(0,1)
                list_vals.append(val)
    data_final = np.array(list_vals)

    return data_final

def split_data(data_final):
    samples_test = data_final[60:]
    samples_train = data_final[:60]
    return samples_train, samples_test

def train_svm(samples_train, T):
    w = np.array([0,0,0])
    lmd = 0.01
    converged = False
    t = 0
    min_distance_negative = 0
    min_distance_positive = 0
    while(converged == False) and t < 1000000:
        
        t +=1
        sample = int(random.uniform(0,len(samples_train)-1))
        x_i = samples_train[sample][:-1]
        y_i = samples_train[sample][-1]
                
        n_t = 1/(t*lmd)
            
        if (y_i*np.dot(x_i,w)) < 1:
            w = (1-n_t*lmd)*w + (n_t*y_i*x_i)
        else:
            w = (1-n_t*lmd)*w 

        min_distance_positive, min_distance_negative, converged = check_support_vectors(samples_train, w)
    print("Number of iterations to Converge: ", t)
    generate_scatterplot(samples_train, min_distance_negative, min_distance_positive, w)

def check_support_vectors(samples_train, w):
    min_distance_positive = float('inf')
    min_distance_negative = float('inf')
    
    for i in samples_train:
        x1 = i[1]
        x2 = i[2]
        y = i[3]
        d = abs(w[1]*x1 + w[2]*x2 + w[0])/ (math.sqrt(w[1]**2) + (math.sqrt(w[2]**2)))
        if y == -1:
            if d<=min_distance_negative:
                min_distance_negative = d
        else:
            if d<=min_distance_positive:
                min_distance_positive = d

    errors = 0    
    for i in samples_train:
        prediction = np.dot(i[:-1], w)
        if prediction<0 and i[-1] == 1:
            errors +=1
        elif prediction>0 and i[-1] == -1:
            errors +=1
    if round(min_distance_positive,1) == round(min_distance_negative,1) and errors == 0:
        return round(min_distance_positive)+0.8, round(min_distance_negative)+0.8, True
    else:
        return 1,1,False

def generate_scatterplot(samples_train, min_distance_negative, min_distance_positive, w):
    plt.scatter(samples_train[:, 1], samples_train[:, 2], c=samples_train[:, 3]);
    ax = plt.gca()
    xlim = ax.get_xlim()

    xx = np.linspace(xlim[0], xlim[1])
    #y = mx + c which is same as w1x + w2y + b = 0
    #m = w1/w2 and c = b/w2
    yy = -(w[1]/w[2]) * xx - (w[0]/w[2])
    yy1 = min_distance_positive-(w[1]/w[2]) * xx - (w[0]/w[2])
    yy2 = -min_distance_negative-(w[1]/w[2]) * xx - (w[0]/w[2])
    plt.plot(xx, yy, c="r")
    plt.plot(xx, yy1, c="g", linestyle = "dashed")
    plt.plot(xx, yy2, c="g", linestyle = "dashed")
    plt.show()

def main():
    data_final = create_data()
    samples_train, samples_test = split_data(data_final)
    train_svm(samples_train, 100000)

if __name__ == '__main__':
    main()
