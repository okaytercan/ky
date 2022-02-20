# import required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# data loading
data = pd.read_csv('salary.csv')

x = data.iloc[:,1:2]
y = data.iloc[:,2:]
X = x.values
Y = y.values


# linear regression

lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()


# polynomial regression
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()


poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

# predictions

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))




# data scaling
sc1=StandardScaler()

x_scl = sc1.fit_transform(X)

sc2=StandardScaler()
y_scl = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))




svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_scl,y_scl)

plt.scatter(x_scl,y_scl,color='red')
plt.plot(x_scl,svr_reg.predict(x_scl),color='blue')
plt.show()

print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))

# decision tree
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z = X + 0.5
K = X - 0.4
plt.scatter(X,Y, color='red')
plt.plot(x,r_dt.predict(X), color='blue')

plt.plot(x,r_dt.predict(Z),color='green')
plt.plot(x,r_dt.predict(K),color='yellow')
plt.show()

print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))


