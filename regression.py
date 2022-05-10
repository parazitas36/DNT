import numpy as np
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression

n = 10 # Order number
model = LinearRegression()

class SunSpot:
    def __init__(self, year, spots):
        self.year = year
        self.spots = spots
    def __repr__(self):
        return "Year: "+str(self.year)+ " Spots: "+str(self.spots)

# Reads data from file
def readFile(path):
    file = open(path, "r")
    data = file.readlines()
    sunSpots = []

    for i in range(len(data)):
        line = data[i]
        parts = line.split("\t")
        sunSpot = SunSpot(int(parts[0]), int(parts[1].split("\n")[0]))
        sunSpots.append(sunSpot)
    return sunSpots

# Draws sunspots activity
def drawSunspotsActivity(data):
    years = [s.year for s in data]
    spots = [s.spots for s in data]
    plot.plot(years, spots)
    plot.xticks(np.arange(min(years), max(years)+1, 50))
    plot.title("Sunspots activity by year")
    plot.xlabel("Years")
    plot.ylabel("Sunspots count")
    plot.show()

# Draws 3D scatter plot
def draw3D(P, T):
    x1 = np.array(P[0])[0]
    x2 = np.array(P[1])[0]
    z = np.array(T)[0]
    fig = plot.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x1, x2, z)
    ax.set_xlabel('First input')
    ax.set_ylabel('Second input')
    ax.set_zlabel('Output')
    ax.set_title('3D scatter plot of inputs and output')
    plot.show()

# Predicted and actual results comparison
def draw_comparison(Tu, Tsu, x):
    Tu = np.array(Tu)[0]
    Tsu = np.array(Tsu)[0]
    plot.plot(x, Tu, label="Actual result")
    plot.plot(x, Tsu, label="Predicted result")
    plot.title('Actual and predicted results comparison')
    plot.legend()
    plot.show()

# Creates a vector of errors and represents it in graph and histogram
def error_vector(actual, prediction, years):
    e = actual - prediction
    e = np.array(e)[0]
    # Graph
    plot.plot(years, e)
    plot.title("Vector of errors")
    plot.xlabel("Year")
    plot.ylabel("Error")
    plot.show()
    # Histogram
    plot.hist(e)
    plot.title('Histogram of errors')
    plot.xlabel('Error')
    plot.ylabel('Count')
    plot.show()

    return e

# Calculates and prints Mean Square Error 
def get_MSE(errors):
    MSE = np.square(errors)
    MSE = np.sum(MSE) / len(MSE)
    print("MSE:", MSE)

# Finds and prints Median Absolute Deviation
def get_MAD(errors):
    MAD = np.abs(errors)
    MAD = np.median(MAD)
    print("MAD:", MAD)

# Splits data into learning and testing data collections
def splitData(data):
    spots = [d.spots for d in data]
    P = []
    T = []

    for i in range(len(spots)-n):
        P.append(spots[i:i+n])
        T.append(spots[i+n])
    return P, T

# Fit the model
def fit_model(x, y):
    # Adjust matrices for model.fit()
    x = x.T
    x = np.array(x)
    y = y.T
    y = np.array(y)
    model.fit(x, y)

# Get model results
def get_results(x, y):
    # Adjust matrices for model.score()
    x = x.T
    x = np.array(x)
    y = y.T
    y = np.array(y)
    r_sq = model.score(x, y)
    bias, weights = model.intercept_, model.coef_
    return r_sq, bias, weights

# Predicts result
def predict(x):
    x = x.T
    x = np.array(x)
    return model.predict(x)

def main():
    sunSpots = readFile("sunspot.txt")
    drawSunspotsActivity(sunSpots)
    P, T = splitData(sunSpots)
    P = np.matrix(P)
    T = np.matrix(T)
    P = P.T
    draw3D(P, T)

    # Split data to learning and verification data collections
    Pu = P[0:n, 0:200]
    Tu = T[0:n, 0:200]
    Pv = P[0:n, 200:]
    Tv = T[0:n, 200:]

    fit_model(Pu, Tu)
    r_sq, bias, weights = get_results(Pu, Tu)

    print('R^2:', r_sq)
    print('bias:', bias)
    print('weights:', weights)

    # Predictions with learning data
    Tsu = predict(Pu)
    Tsu = np.matrix(Tsu)
    Tsu = Tsu.T
    x_years = [sunSpots[i].year for i in range(n, 200+n)]
    draw_comparison(Tu,Tsu, x_years)

    # Predictions with verification data
    Tsu = predict(Pv)
    Tsu = np.matrix(Tsu)
    Tsu = Tsu.T
    x_years = [sunSpots[i].year for i in range(200+n, len(sunSpots))]
    draw_comparison(Tv,Tsu, x_years)

    # Predictions with whole data
    Tsu = predict(P)
    Tsu = np.matrix(Tsu)
    Tsu = Tsu.T
    x_years = [sunSpots[i].year for i in range(n, len(sunSpots))]
    errors = error_vector(T, Tsu, x_years)
    get_MSE(errors)
    get_MAD(errors)

if __name__ == "__main__":
    main()