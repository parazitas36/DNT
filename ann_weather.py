import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import random

path = "weather.xlsx"
MAX_EPOCHS = 1000
lr = 10

def nonlin(x, deriv=False):
    if deriv:
        return x*(1-x)
    return 1/(1+np.exp(-x))

# Reads weather data from excel file
def readData(path):
    data = pd.read_excel(path)
    df = pd.DataFrame(data)
    perceptions = df['PRCP'].to_list()
    perceptions = norm(perceptions)
    tempMAXS = df['TMAX'].to_list()    
    tempMAXS = norm(tempMAXS)
    tempMINS = df['TMIN'].to_list()    
    tempMINS = norm(tempMINS)
    rains = df['RAIN'].to_list()    
    rains = [1 if val == 'True' or val == "TRUE" or val == True else 0 for val in rains]
    rainsTommorow = []
    for i in range(len(rains)-1):
        rainsTommorow.append(rains[i+1])
    rainsTommorow.append(True)

    return perceptions, tempMAXS, tempMINS, rains, rainsTommorow

# Creates input dataset / neurons
def createNeurons(perceptions, tempMAXS, tempMINS, rains, rainsTommorow):
    X = []
    y = []
    for i in range(len(perceptions)):
        X.append([
            perceptions[i],
            tempMAXS[i],
            tempMINS[i],
            rains[i]
            ])
        y.append(rainsTommorow[i])
    
    X = np.array(X)
    y = np.array([y]).T
    return X, y

# Predict output
def predict(X, weights):
    return np.dot(X, weights)

# Calculates and prints Mean Square Error 
def get_MSE(errors):
    MSE = np.square(errors)
    MSE = np.sum(MSE) / len(MSE)
    return MSE

# Finds and prints Median Absolute Deviation
def get_MAD(errors):
    MAD = np.abs(errors)
    MAD = np.median(MAD)
    return MAD

def norm(x):
    return (x - np.min(x))/(np.max(x)-np.min(x))

def model_training(X, weights, y):
    MSE = []
    MAD = []
    for i in range(MAX_EPOCHS):
        # Calculate output
        l1 = nonlin(np.dot(X, weights))
        # Calculate error
        error = y - l1
        delta = error*nonlin(l1, True)
        MSE.append(get_MSE(error))
        MAD.append(get_MAD(error))
        # Update weights
        weights += lr*np.dot(X.T, delta)
    return weights, MSE, MAD

def split_data(X, y, test_start_index, test_end_index):
    Xl = []
    yl = []
    Xt = []
    yt =[]
    for i in range(len(X)):
        if i >= test_start_index and i < test_end_index:
            Xt.append(X[i])
            yt.append(y[i])
        else:
            Xl.append(X[i])
            yl.append(y[i])
   
    Xl = np.array(Xl)
    Xt = np.array(Xt)
    yl = np.array(yl)
    yt = np.array(yt)

    return Xl, Xt, yl, yt

def main():
    perceptions, tempMAXS, tempMINS, rains, rainsTommorow = readData(path)
    X, y = createNeurons(perceptions, tempMAXS, tempMINS, rains, rainsTommorow)
    np.random.seed(1)
    #weights = np.random.random((X.shape[1], 1))
    chunkSize = int(len(X)/10)

    for i in range(10):
        Xl, Xt, yl, yt = split_data(X, y, i*chunkSize, i*chunkSize+chunkSize)
        weights = np.zeros((X.shape[1], 1))
        #print("Initial weights:", weights)
        weights, MSE, MAD = model_training(Xl, weights, yl)
        print("Updated weights",weights)
        results = []
        for j in range(len(Xt)):
            if nonlin(np.dot(Xt[j], weights)) == yt[j]:
                results.append(1)
        #row = X[i]
        #perceptions, tempMAXS, tempMINS, rains = row[0], row[1], row[2], row[3]
        #print("Actual: %.2f Predict: %.2f Inputs: [PRCP: %.2f | TMAX: %.2f | TMIN: %.2f | RAINS: %.2f]"%(
        #    y[i], nonlin(np.dot(X[i], weights)), perceptions, tempMAXS, tempMINS, rains))

        print('Iteration #%d Accuracy: %d/%d'%(i+1, len(results), len(yt)))
    # plot.plot(np.arange(0, MAX_EPOCHS, 1), MSE)
    # plot.title("MSE")
    # plot.show()
    # plot.plot(np.arange(0, MAX_EPOCHS, 1), MAD)
    # plot.title("MAD")
    # plot.show()

if __name__ == "__main__":
    main()