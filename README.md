# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python.
2. Set variables for assigning dataset values.
3. Import LinearRegression from the sklearn.
4. Assign the points for representing the graph.
5. Predict the regression for marks by using the representation of graph
6. Compare the graphs and hence we obtain the LinearRegression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Rishabendran R    
RegisterNumber:  212219040121
*/
#implement simple linear regression model for predicting the marks scored
from typing import ValuesView
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('/content/student_scores.csv')

#assigning hours to X & scores to Y
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred = reg.predict(X_test)


plt.scatter(X_train,Y_train,color="green")
plt.plot(X_train,reg.predict(X_train),color="red")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,reg.predict(X_test),color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:
![simple linear regression model for predicting the marks scored](./images/exp2.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
