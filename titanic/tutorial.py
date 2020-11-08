# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 14:12:42 2020

@author: ciara
"""


import numpy as np 
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier



#print the files available in the path 
for dirname, _, filenames in os.walk('.'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        


#load the data 
train_data = pd.read_csv("C:/Users/ciara/OneDrive/Documents/GitHub/kaggle/titanic/data/train.csv")
train_data.head() 
print(train_data.shape) #891 rows

test_data = pd.read_csv("C:/Users/ciara/OneDrive/Documents/GitHub/kaggle/titanic/data/test.csv")
test_data.head()
print(test_data.shape) #418 rows






## Investigating the difference between female and male passengers. Looking at a single column
women = train_data.loc[train_data.Sex == 'female']["Survived"] #only the survival column for all the female patients
rate_women = sum(women)/len(women) #can use sum to get the percentage as survival is binary output 

#233 out of 314 women survived 
print("\n % of women who survived:", rate_women)


#repeat with male individuals 
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print(" % of men who survived:", rate_men)




## Using ML to consider several columns at once 
#random forest model - build several trees on sub samples of the data
y = train_data["Survived"] #labels for training data

#we only want to look at this subset of features
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features]) #get_dummies converts categorical variables to numerical 
X_test = pd.get_dummies(test_data[features]) #same for test data 

#initialise model, fit to training data, and predict on test data
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)


#create a file of output 
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('rf_submission.csv', index=False)