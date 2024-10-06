import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
s = StandardScaler()

#Read dataset into data frame
df = pd.read_csv('MarathonData.csv')

#Populate all empty fields in the CrossTraining and Category columns
df.fillna({'CrossTraining': 0}, inplace=True)
df.fillna({'Category': df['Category'].mode()[0]}, inplace=True)

#Remove outliers
df = df[(df["sp4week"] < 14) & (df["sp4week"] > 10)].reset_index(drop=True)

#Load X (input) and Y(target values) numpy arrays
X = df[["sp4week", "km4week"]].values
Y = df[["MarathonTime"]].values.flatten()

#Checking all fields in the data is populated
print("Checking for NaN in X and Y:")
print(np.isnan(X).sum())
print(np.isnan(Y).sum())

#Run input data through the scikit-learn StandardScaler
X = s.fit_transform(X)

#Set initial values
b_init = 0
w_init = np.zeros(2)
iterations = 3000
learning_rate = 0.001

def compute_cost(x, y, w, b):
    m = x.shape[0]
    predictions = np.dot(x, w) + b
    errors = predictions - y
    total_cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return total_cost

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    predictions = np.dot(x, w) + b
    errors = predictions - y
    dj_dw = (1 / m) * np.dot(x.T, errors)
    dj_db = (1 / m) * np.sum(errors)
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, gradient_function, alpha, num_iters):
    w = w_in
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        #Every 100 iterations print the cost (to show the cost is decreasing in gradient descent)
        if i % 100 == 0:
            cost = compute_cost(x, y, w, b)
            print(f"Iteration {i}, Cost: {cost}")

        #Checks w and b have values
        if np.isnan(w).any() or np.isnan(b):
            print("Warning: NaN detected in weights or bias.")
            break

    return w, b

#Run gradient descent
w,b = gradient_descent(X ,Y, w_init, b_init, compute_gradient, learning_rate, iterations)
print("w,b found by gradient descent, w: ", w, "b ", b)

#Calculating the percentage of test predictions are within 5% of the target value
m = X.shape[0]
accuracy_counter = 0
for i in range(m):
    a = "No"
    prediction = np.dot(X[i], w) + b
    errorA = round(prediction, 2) - round(Y[i], 2)
    errorB = round(Y[i], 2) - round(prediction, 2)
    if (errorA <= 0.05 * Y[i] and errorA >= 0) or (errorB <= 0.05 * Y[i] and errorB >= 0):
        accuracy_counter += 1
        a = "Match"
    print(f"prediction: {prediction:0.2f}, target value: {Y[i]:0.2f}, match: {a}")

print("Accuracy: ", accuracy_counter/m)
