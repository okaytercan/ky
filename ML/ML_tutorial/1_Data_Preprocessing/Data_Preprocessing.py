# import required libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# data loading
data = pd.read_csv('data.csv')
print(data)


# missing values (Imputer object using the mean strategy and missing_values type for imputation)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

age = data.iloc[:,1:4].values
print("Original Data : \n",age)

# fitting the data to the imputer object
imputer = imputer.fit(age[:,1:4])
# imputing the data  
age[:,1:4] = imputer.transform(age[:,1:4])
print("Imputed Data : \n",age)

#encoder: converting type of columns to 'category' (Category -> Numeric)

country = data.iloc[:,0:1].values
print(country)

# creating instance of labelencoder
le = preprocessing.LabelEncoder()

# Assigning numerical values and storing in another column
country[:,0] = le.fit_transform(data.iloc[:,0])
print(country)

# creating instance of one-hot-encoder
ohe = preprocessing.OneHotEncoder()
# passing country column (label encoded values of country)
country = ohe.fit_transform(country).toarray()
print(country)


# convert a NumPy Array to Pandas Dataframe
result = pd.DataFrame(data=country, index = range(22), columns = ['fr','tr','us'])
print(result)

result2 = pd.DataFrame(data=age, index = range(22), columns = ['height','weight','age'])
print(result2)

sex = data.iloc[:,-1].values
print(sex)

result3 = pd.DataFrame(data = sex, index = range(22), columns = ['sex'])
print(result3)

# combining data
r=pd.concat([result,result2], axis=1)
print(r)

r2=pd.concat([r,result3], axis=1)
print(r2)

#Training and Test Sets: Splitting Data
x_train, x_test,y_train,y_test = train_test_split(r,result3,test_size=0.33, random_state=0)

# data scaling
sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)










