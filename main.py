import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
s = StandardScaler()

df = pd.read_csv('MarathonData.csv')

df['Category'].fillna(df['Category'].mode()[0], inplace=True)  # Filling with mode
df['CrossTraining'].fillna(0, inplace=True)  # Filling with 0
print(df.isnull().sum())

X = df[["sp4week", "km4week"]].values
Y = df[["MarathonTime"]].values.flatten()

print("Checking for NaN in X and Y:")
print(np.isnan(X).sum())
print(np.isnan(Y).sum())

X = s.fit_transform(X)

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

        if i % 100 == 0:
            cost = compute_cost(x, y, w, b)
            print(f"Iteration {i}, Cost: {cost}")

        if np.isnan(w).any() or np.isnan(b):
            print("Warning: NaN detected in weights or bias.")
            break

    return w, b

w,b = gradient_descent(X ,Y, w_init, b_init, compute_gradient, learning_rate, iterations)
print("w,b found by gradient descent, w: ", w, "b ", b)

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
