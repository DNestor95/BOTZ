from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
import pandas as pd
import csv
# use knn to determine if the data in data.csv in a bot or not with the feild Bot Label and show the overall likelyhood that anyone in the dataset is a bot
# load dataset
dataset = pd.read_csv("data.csv")
# print(dataset)
# split dataset into training and test data
X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:, :-1], dataset.iloc[:, -1], random_state=1)
# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)
# fitting the model
knn.fit(X_train, y_train)
# predict the response
pred = knn.predict(X_test)
# evaluate accuracy
print("accuracy: {}".format(knn.score(X_test, y_test)))



    