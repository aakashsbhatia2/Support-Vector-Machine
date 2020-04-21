"""

Imported Libraries

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import random
import math


def create_data():
    """

    Function to create data.
    - Here, X0 and y hold the initial points created by make_blobs
    - A np.array is created using X0 and y which contains data of the form [1, x1, x2, y] and is stored in data_final
    - This function returns data_final

    """
    X0, y = make_blobs(n_samples=100, n_features = 2, centers=2, cluster_std=1.05, random_state=10)
    X0 = X0.tolist()
    for i in range(len(X0)):
        X0[i].insert(0, 1)
        if y[i] == 0:
            X0[i].append(-1)
        else:
            X0[i].append(1)
    data_final = np.array(X0)
    
    return data_final

def split_data(data_final):

    """

    Function to split data.
    - The fina_data array is split into the training dataset and testing dataset
    - 80% train 
    - 20% test
    - The data is split AFTER shuffling [code in main function]

    """
    samples_test = data_final[80:]
    samples_train = data_final[:80]
    return samples_train, samples_test

def train(samples_train, T):
    """

    Function to train SVM model.
    - This function takes input as the training samples and number of epochs
    - While the algorithm has not converged (i.e. we have not found the optimal hyperplane)
    - A random point is select
    - If y*(<w,x>) <1: We update weight vector
    - Check if the support vector is on either side, equidistant and the hyperplane perfectly seperates the two clusters
    - terminate
    """
    w = np.array([0,0,0])
    lmd = 3
    converged = False
    t = 0
    min_distance_negative = 0
    min_distance_positive = 0
    while(converged == False) and t < T:
        
        t +=1
        sample = int(random.uniform(0,len(samples_train)-1))
        x_i = samples_train[sample][:-1]
        y_i = samples_train[sample][-1]
                
        n_t = 1/(t*lmd)
            
        if (y_i*np.dot(x_i,w)) < 1:
            w[1:] = (1-n_t*lmd)*w[1:] + (n_t*y_i*x_i[1:])

            #Calculating bias independantly
            w[0] = w[0] + (y_i*x_i[0])
        else:
            w = (1-n_t*lmd)*w

        min_distance_positive, min_distance_negative, converged = check_support_vectors(samples_train, w)
    print("=================\n")
    print("Number of iterations to Converge: ", t)
    draw(samples_train, samples_train, min_distance_negative, min_distance_positive, w,"Train Plot")
    return w, min_distance_negative, min_distance_positive

def check_support_vectors(samples_train, w):
    """

    Here, we identify the support vectors for each weight vectort and see if it is equidistant from w

    """

    min_distance_positive = 999.0
    min_distance_negative = 999.0
    
    for i in samples_train:
        x1 = i[1]
        x2 = i[2]
        y = i[3]
        try:
            d = abs(w[1]*x1 + w[2]*x2 + w[0])/ (math.sqrt(w[1]**2) + (math.sqrt(w[2]**2)))
            if y == -1:
                if d<=min_distance_negative:
                    min_distance_negative = d
            else:
                if d<=min_distance_positive:
                    min_distance_positive = d
        except: 
            pass

    if round(min_distance_positive,1) == round(min_distance_negative,1):
        return round(min_distance_positive)+0.6, round(min_distance_negative)+0.6, True
    else:
        return 1,1,False

def draw(samples_train, samples_test, min_distance_negative, min_distance_positive, w, plot_type):
    """

    Generating the scatter plot for the trained samples and corresponding tested samples

    """

    if plot_type == "Train Plot":
        plt.scatter(samples_train[:, 1], samples_train[:, 2], c=samples_train[:, 3],  edgecolor="black")
        ax = plt.gca()
        ax.set_facecolor('red')
        ax.patch.set_alpha(0.1)
        xlim = ax.get_xlim()

        xx = np.linspace(xlim[0], xlim[1])
        yy = -(w[1]/w[2]) * xx - (w[0]/w[2])
        yy1 = min_distance_positive-(w[1]/w[2]) * xx - (w[0]/w[2])
        yy2 = -min_distance_negative-(w[1]/w[2]) * xx - (w[0]/w[2])
        plt.plot(xx, yy, c="r")
        plt.plot(xx, yy1, c="g", linestyle = "dashed")
        plt.plot(xx, yy2, c="g", linestyle = "dashed")
        plt.show()
    if plot_type == "Test Plot":
        plt.scatter(samples_train[:, 1], samples_train[:, 2], c=(samples_train[:, 3]), edgecolor="black")
        plt.scatter(samples_test[:, 1], samples_test[:, 2], c='r', marker="D", edgecolor="black")
        ax = plt.gca()
        ax.set_facecolor('red')
        ax.patch.set_alpha(0.1)
        xlim = ax.get_xlim()

        xx = np.linspace(xlim[0], xlim[1])
        yy = -(w[1]/w[2]) * xx - (w[0]/w[2])
        yy1 = min_distance_positive-(w[1]/w[2]) * xx - (w[0]/w[2])
        yy2 = -min_distance_negative-(w[1]/w[2]) * xx - (w[0]/w[2])
        plt.plot(xx, yy, c="r")
        plt.plot(xx, yy1, c="g", linestyle = "dashed")
        plt.plot(xx, yy2, c="g", linestyle = "dashed")
        plt.show()
        

def test(sample_train, samples_test, w, min_distance_negative, min_distance_positive, plot_type):
    """

    Function to test SVM Model

    """
    errors = 0
    for i in samples_test:
        prediction = np.dot(i[:-1], w)
        if prediction<0 and i[-1] == 1:
            errors +=1
        elif prediction>0 and i[-1] == -1:
            errors +=1
    print("\nTotal Points trained: ", len(sample_train))
    print("\nTotal Points tested: ", len(samples_test))
    print("\nTotal Points Misclassified: ", errors)
    print("\n=================")
    draw(sample_train, samples_test, min_distance_negative, min_distance_positive, w, plot_type)
    


def main():

    #Generate dataset
    data_final = create_data()
    
    #Generate Train and Test splot
    samples_train, samples_test = split_data(data_final)

    #Calculate weight vector and Support vectors by training model. T = 150,000
    w, min_distance_negative, min_distance_positive = train(samples_train, T = 150000)

    #Test SVM Model
    test(samples_train, samples_test, w, min_distance_negative, min_distance_positive, "Test Plot")

if __name__ == '__main__':
    main()
