# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 09:34:09 2020

@author: User
"""


import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
label_encoder=preprocessing.LabelEncoder()

#Reading dataset
Titanic_test1=pd.read_csv("Titanic_train.csv")

##fill in missing value

Titanic_test2=Titanic_test1.drop(["Name","Ticket","Cabin","PassengerId"], axis=1)
Titanic_test3=Titanic_test2.dropna()
Titanic_test=Titanic_test3.drop_duplicates()

## converting string to numerics
Titanic_test["Sex"]=label_encoder.fit_transform(Titanic_test["Sex"])
Titanic_test["Embarked"]=label_encoder.fit_transform(Titanic_test["Embarked"])


##applyinf RandomForest
ref_model=RandomForestClassifier(n_estimators=1000,max_features=2,oob_score=True)
#features=[ 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare', 'Embarked']
features=["Sex","Age","Fare"]
ref_model.fit(X=Titanic_test[features],y=Titanic_test["Survived"])

## printing OOB accuracy
print("OOB ACCURACY")
print(ref_model.oob_score_)