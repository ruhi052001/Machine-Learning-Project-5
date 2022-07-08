# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Lab 5 Part B
# reading the csv file
df = pd.read_csv("abalone.csv")
# splitting data into train and test data
train, test = train_test_split(df, test_size=0.30, random_state=42, shuffle=True)
# saving the training and testing datasets as csv files
train.to_csv('abalone-train.csv', index=False)
test.to_csv('abalone-test.csv', index=False)

# Question 1
print("Q1:")
# calculating the Pearson correlation coefficient of every attribute with the target attribute rings
input_var = df[df.columns[1:]].corr()['Rings'][:-1].idxmax()
print("The attribute which has the highest Pearson correlation coefficient with the target attribute Rings is",
      input_var)
lin_reg = LinearRegression().fit(np.array(train["Shell weight"]).reshape(-1, 1), train['Rings'])
# Question 1 Part a
# 2923 because its the length of training data
x = np.linspace(0, 1, 2923).reshape(-1, 1)
# finding the best fit line
ly = lin_reg.predict(x)
# plotting scatter plot
plt.scatter(train['Shell weight'], train['Rings'])
# plotting best fit line
plt.plot(x, ly, linewidth=3, color='orange')
plt.xlabel('Shell weight')
plt.ylabel('Rings')
plt.title('Best Fit Line')
plt.show()

# part b
print('part b:')
y_train_pred = lin_reg.predict(np.array(train["Shell weight"]).reshape(-1, 1))
rmse_train = (mse(train['Rings'], y_train_pred)) ** 0.5
print("The rmse for training data is", round(rmse_train, 3))

# part c
print('part c:')
y_test_pred = lin_reg.predict(np.array(test["Shell weight"]).reshape(-1, 1))
rmse_test = (mse(test['Rings'].to_numpy(), y_test_pred)) ** 0.5
print("The rmse for testing data is", round(rmse_test, 3))

# part d
plt.scatter(test['Rings'].to_numpy(), y_test_pred)
plt.xlabel('Actual Rings')
plt.ylabel('Predicted Rings')
plt.title('Univariate linear regression model')
plt.show()

# Question 2
print("---Q2---:")
X_train = train.iloc[:, :-1].values
Y_train = train.iloc[:, train.shape[1] - 1].values
X_test = test.iloc[:, :-1].values
Y_test = test.iloc[:, test.shape[1] - 1].values

# part a
print('part a:')
reg_train = LinearRegression().fit(X_train, Y_train)
rmse_train = (mse(Y_train, reg_train.predict(X_train))) ** 0.5
print("The rmse for training data is", round(rmse_train, 3))

# part b
print('part b:')
reg_test = LinearRegression().fit(X_test, Y_test)
rmse_test = (mse(Y_test, reg_test.predict(X_test))) ** 0.5
print("The rmse for testing data is", round(rmse_test, 3))

# part c
plt.scatter(Y_test, reg_test.predict(X_test))
plt.xlabel('Actual Rings')
plt.ylabel('Predicted Rings')
plt.title('Multivariate linear regression model')
plt.show()

# Question 3
print("---Q3---:")
P = [2, 3, 4, 5]
# part a
print('part a:')
X = np.array(train['Shell weight']).reshape(-1, 1)
RMSE = []
for p in P:
    poly_features = PolynomialFeatures(p)  # p is the degree
    x_poly = poly_features.fit_transform(X)
    reg = LinearRegression()
    reg.fit(x_poly, Y_train)
    Y_pred = reg.predict(x_poly)
    rmse = (mse(Y_train, Y_pred)) ** 0.5
    RMSE.append(rmse)
    print("The rmse for p=", p, 'is', round(rmse, 3))

# plotting bar graph of rmse vs degree of polynomial
plt.bar(P, RMSE)
plt.xlabel('p (degree of polynomial)')
plt.ylabel('RMSE(training data)')
plt.title("Univariate non-linear regression model")
plt.show()

# part b
print('part b:')
RMSE = []
X = np.array(test['Shell weight']).reshape(-1, 1)
Y_pred = []
for p in P:
    poly_features = PolynomialFeatures(p)  # p is the degree
    x_poly = poly_features.fit_transform(X)
    reg = LinearRegression()
    reg.fit(x_poly, Y_test)
    Y_pred = reg.predict(x_poly)
    rmse = (mse(Y_test, Y_pred)) ** 0.5
    RMSE.append(rmse)
    print("The rmse for p=", p, 'is', round(rmse, 3))

# plotting bar graph of rmse vs degree of polynomial
plt.bar(P, RMSE)
plt.xlabel('p (degree of polynomial)')
plt.ylabel('RMSE(test data)')
plt.title("Univariate non-linear regression model")
plt.show()

# part c
# value of p=5 has the lowest rmse
x_poly = PolynomialFeatures(5).fit_transform(x)
reg = LinearRegression()
reg.fit(x_poly, Y_train)
cy = reg.predict(x_poly)
plt.scatter(train['Shell weight'], train['Rings'])
plt.plot(np.linspace(0, 1, 2923), cy, linewidth=3, color='orange')
plt.xlabel('Shell weight')
plt.ylabel('Rings')
plt.title('Best Fit Curve')
plt.show()

# part d
# because the best degree of polynomial is 5 as p=5 has minimum rmse
plt.scatter(Y_test, Y_pred)
plt.xlabel('Actual Rings')
plt.ylabel('Predicted Rings')
plt.title('Univariate non-linear regression model')
plt.show()

# Question 4
print("---Q4---:")
# part a
print('part a:')
RMSE = []
for p in P:
    poly_features = PolynomialFeatures(p)  # p is the degree
    x_poly = poly_features.fit_transform(X_train)
    reg = LinearRegression()
    reg.fit(x_poly, Y_train)
    Y_pred = reg.predict(x_poly)
    rmse = (mse(Y_train, Y_pred)) ** 0.5
    RMSE.append(rmse)
    print("The rmse for p=", p, 'is', round(rmse, 3))

# plotting bar graph of rmse vs degree of polynomial
plt.bar(P, RMSE)
plt.xlabel('p (degree of polynomial)')
plt.ylabel('RMSE(training data)')
plt.title("Multivariate non-linear regression model")
plt.show()

# part b
print('part b:')
RMSE = []
Y_pred = []
for p in P:
    poly_features = PolynomialFeatures(p)  # p is the degree
    x_poly = poly_features.fit_transform(X_test)
    reg = LinearRegression()
    reg.fit(x_poly, Y_test)
    Y_pred = reg.predict(x_poly)
    rmse = (mse(Y_test, Y_pred)) ** 0.5
    RMSE.append(rmse)
    print("The rmse for p=", p, 'is', round(rmse, 3))
    # d
    # because the best degree of polynomial is 3 as p=3 has minimum rmse
    if p == 3:
        plt.scatter(Y_test, Y_pred)
        plt.xlabel('Actual Rings')
        plt.ylabel('Predicted Rings')
        plt.title('Univariate non-linear regression model')
        plt.show()

# plotting bar graph of rmse vs degree of polynomial
plt.bar(P, RMSE)
plt.xlabel('p (degree of polynomial)')
plt.ylabel('RMSE(test data)')
plt.title("Multivariate non-linear regression model")
plt.show()
