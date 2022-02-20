# required library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# data loading
data = pd.read_csv('data.csv')

print(data)

x = data.iloc[:,1:4].values # independent variables
y = data.iloc[:,4:].values # dependent variables
print("dependent variables : \n",y)

# splitting the dataset to train and test
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)


# data scaling
sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# logistic regression

logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print("Prediction : \n",y_pred)
print("Test : \n",y_test)


# test the performance of our model – Confusion Matrix
cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix : \n",cm)

# Performance measure – Accuracy
print ("Accuracy : ", accuracy_score(y_test, y_pred))


# K-Nearest-Neighbor model
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("K-NN : \n",cm)

# Support Vector Machine model -- kernel---> poly to rbf
svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)

y_pred_svm = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred_svm)
print("SVM : \n",cm)

# Naive Bayes model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred_gnb = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred_gnb)
print("GNB : \n",cm)

# Decision Tree Classifier model
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("DTC : \n",cm)


# Random Forest Classifier model
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print("RFC : \n",cm)





