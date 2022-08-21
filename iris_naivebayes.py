import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("enjoy_sports.csv")
dataset.shape
dataset['Decision'].value_counts()
dataset.head()
dataset = dataset.drop("Day", axis=1)

p = {'Sunny': 1, 'Overcast': 2, 'Rain': 3}
dataset.Outlook = [p[item] for item in dataset.Outlook.astype(str)]

q = {'Hot': 4, 'Mild': 5, 'Cool': 6}
dataset.Temp = [q[item] for item in dataset.Temp.astype(str)]

r = {'High': 7, 'Normal': 8}
dataset.Humidity = [r[item] for item in dataset.Humidity.astype(str)]

s = {'Weak': 9, 'Strong': 10}
dataset.Wind = [s[item] for item in dataset.Wind.astype(str)]

#dataset.head()

x = dataset[['Outlook','Temp','Humidity','Wind']]
y = dataset[['Decision']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1)

final = GaussianNB()
final.fit(x_train, y_train)

y_pred = final.predict(x_test)

from sklearn import metrics
print("Accuracy is:",metrics.accuracy_score(y_test, y_pred)*100)
