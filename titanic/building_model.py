# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 14:12:42 2020

@author: ciara
"""


import numpy as np 
import pandas as pd
import os




#print the files available in the path 
for dirname, _, filenames in os.walk('.'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        


#load the data 
train_data = pd.read_csv("C:/Users/ciara/OneDrive/Documents/GitHub/kaggle/titanic/data/train.csv")
print(train_data.head())

test_data = pd.read_csv("C:/Users/ciara/OneDrive/Documents/GitHub/kaggle/titanic/data/test.csv")
print(test_data.head())