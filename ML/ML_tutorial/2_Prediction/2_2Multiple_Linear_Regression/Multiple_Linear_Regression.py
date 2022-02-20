# import required libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm 


# data loading
data = pd.read_csv('tennis.csv')

#encoder: converting type of columns to 'category' (Category -> Numeric)

# creating instance of labelencoder and assigning numerical values and storing in all column
dt = data.apply(preprocessing.LabelEncoder().fit_transform)

c = dt.iloc[:,:1]
# creating instance of one-hot-encoder
ohe = preprocessing.OneHotEncoder()
#label encoded values of c
c=ohe.fit_transform(c).toarray()
print(c)

weather = pd.DataFrame(data = c, index = range(14), columns=['overcast','rainy','sunny'])
r_data = pd.concat([weather,data.iloc[:,1:3]],axis = 1)     # r_data --> recent data
r_data = pd.concat([dt.iloc[:,-2:],r_data], axis = 1)



# Training and Test Sets: Splitting Data

x_train, x_test,y_train,y_test = train_test_split(r_data.iloc[:,:-1],r_data.iloc[:,-1:],test_size=0.33, random_state=0)

# model building (linear regression)

regressor = LinearRegression()
regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)

print("Prediction : \n",y_pred)

#backward elimination

X = np.append(arr = np.ones((14,1)).astype(int), values=r_data.iloc[:,:-1], axis=1 )
X_l = r_data.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog = r_data.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print("------- \n",r.summary())

r_data = r_data.iloc[:,1:]


X = np.append(arr = np.ones((14,1)).astype(int), values=r_data.iloc[:,:-1], axis=1 )
X_l = r_data.iloc[:,[0,1,2,3,4]].values
r_ols = sm.OLS(endog = r_data.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print("-------- \n",r.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)















    
    

