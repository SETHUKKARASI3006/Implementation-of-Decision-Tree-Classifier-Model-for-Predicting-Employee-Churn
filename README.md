# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SETHUKKARASI C
RegisterNumber:  212223230201
*/
```

```
import pandas as pd
```
<br>

```
data = pd.read_csv("Employee.csv")
```
<br>

```
data.head()
```
<br>

![out1](/o1.png)
<br>

```
data.info()
```
<br>

![out2](/o2.png)
<br>

```
data.isnull().sum()
```
<br>

![out3](/o3.png)

```
data["left"].value_counts()
```
<br>

![out4](/o4.png)

```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
<br>

![out5](/o5.png)
<br>

```
X = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","Work_accident","promotion_last_5years","salary"]]
X.head()
```
<br>

![out6](/o6.png)
<br>

```
y = data["left"]
y.head()
```
<br>

![out7](/o7.png)
<br>

```
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(X_train, Y_train)
```
<br>

![out8](/o8.png)
<br>

```
y_pred = dt.predict(X_test)
print(y_pred)
```
<br>

![out9](/o9.png)
<br>

```
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test,y_pred)
accuracy
```
<br>

![out10](/o10.png)
<br>

```
dt.predict([[0.5, 0.8, 9, 260, 6, 0, 2]])
```
<br>

![out11](/o11.png)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
