# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:07:30 2026

@author: syeda
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

df2=pd.read_csv( r"C:\Users\syeda\OneDrive\Desktop\Healthcare_Analysis\Healthcare_Model_Ready.csv",)

print(df2.dtypes)

#VIF 
X=df2.drop(columns=["Test Results"])
X_sample = X.sample(n=50000, random_state=42)

vif=pd.DataFrame()

vif["Feature"]=X.columns
vif["VIF"]=[
    variance_inflation_factor(X_sample.values,i)
    for i in range(X_sample.shape[1])
    ]

print(vif)

#Based of vif vals
df2=df2.drop(["DYear","Doa Year"],axis=1)

#Model 

y=df2["Test Results"]

X_train , X_test , y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
    )

model=LogisticRegression(
    random_state=42,
    max_iter=1000,
    )

model.fit(X_train,y_train)
y_pred=model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#Decision Model 
model=DecisionTreeClassifier()

param_grid={
    "max_depth":[3,7,9,10,15],
    "min_samples_split":[2,5,10],
    "criterion":['gini','entropy']
    }

grid_search=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="f1"
    )

grid_search.fit(X_train,y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)

best_tree=grid_search.best_estimator_
y_pred=best_tree.predict(X_test)

print("Accuracy: ",accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


plt.figure(figsize=(18,10))

plot_tree(
    best_tree,
    feature_names=X.columns,
    class_names=["Abnormal", "Normal", "Inconclusive"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=9
)

plt.show()




#Observation
#Model Comparison
#Logistic Regression Accuracy: 32.8%
#Decision Tree Accuracy: 34.1%
#Decision Tree was selected as the final model due to its slightly better performance.