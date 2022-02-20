# import required libraries
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# data loading
data = pd.read_csv('salary.csv')

x = data.iloc[:,2:5]
y = data.iloc[:,5:]
X = x.values
Y = y.values

print(data.corr())

#linear regression

lin_reg = LinearRegression()
lin_reg.fit(X,Y)


model=sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())

print('Linear R2 value')    # R-squared
print(r2_score(Y, lin_reg.predict(X)))


#polynomial regression
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)


# predictions
print('poly OLS')
model2=sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print(model2.fit().summary())

print('Polynomial R2 value')
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))

# data scaling
sc1=StandardScaler()
x_scl = sc1.fit_transform(X)
sc2=StandardScaler()
y_scl = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))




svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_scl,y_scl)


print('SVR OLS')
model3=sm.OLS(svr_reg.predict(x_scl),x_scl)
print(model3.fit().summary())


print('SVR R2 value')
print(r2_score(y_scl, svr_reg.predict(x_scl)))

#Decision Tree Regresion
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)


print('Decision Tree OLS')
model4=sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())

print('Decision Tree R2 value')
print(r2_score(Y, r_dt.predict(X)))

#Random Forest Regresion

rf_reg=RandomForestRegressor(n_estimators = 10,random_state=0)
rf_reg.fit(X,Y.ravel())



print('Random Forest OLS')
model5=sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())



print('Random Forest R2 value')
print(r2_score(Y, rf_reg.predict(X)))


# summary R-squared values
print('\n\n-----------------------\n R-squared values \n-----------------------')
print('\nLinear R2 value')
print(r2_score(Y, lin_reg.predict(X)))

print('\nPolynomial R2 value')
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))

print('\nSVR R2 value')
print(r2_score(y_scl, svr_reg.predict(x_scl)))


print('\nDecision Tree R2 value')
print(r2_score(Y, r_dt.predict(X)))

print('\nRandom Forest R2 value')
print(r2_score(Y, rf_reg.predict(X)))