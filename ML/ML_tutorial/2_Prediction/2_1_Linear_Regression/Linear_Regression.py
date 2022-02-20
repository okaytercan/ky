# import required libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# data loading
data = pd.read_csv('sales.csv')

months = data[['Months']]
print(months)

sales = data[['Sales']]
print(sales)


# Training and Test Sets: Splitting Data

x_train, x_test,y_train,y_test = train_test_split(months,sales,test_size=0.33, random_state=0)

# model building (linear regression)

lr = LinearRegression()
lr.fit(x_train,y_train)

predict = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()


# generating visualizations with pyplot
plt.plot(x_train,y_train,label="Predict")
plt.plot(x_test,lr.predict(x_test),label="Original Data")

plt.title("Sales by Months")
plt.xlabel("Months")
plt.ylabel("Sales")
plt.legend()















    
    

