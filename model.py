import numpy as np
import pandas as pd
import pickle

data=pd.read_csv('Social_Network_Ads.csv')
print(data.head())
print("\n")
print(data.dtypes)
print("\n")

X=data.iloc[:,0:2].values
y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)

print(knn.fit(X_train,y_train))

print(knn.predict([[41,200000]]))

pickle.dump(knn,open('model.pkl','wb'))