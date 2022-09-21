# -*- coding: utf-8 -*-
"""
@author: Fadwa
"""


import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, r2_score

#load data 
url="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
df=pd.read_csv(url)
print(df.head())

#plot the Scatter plot to check the relationship between Hours and Scores
f,axs=plt.subplots(3,1,figsize=(10,10))
f.text(0.5, 0.04, "Hours", ha='center')
f.text(0.04, 0.5, "Scores", va='center', rotation='vertical')

axs[0].scatter(df["Hours"],df["Scores"])
print("Correlation =",df["Hours"].corr(df["Scores"]))

##prepare  data 
x=df["Hours"].values.reshape(25,1)
y=df["Scores"].values

#split data into traininig and test sets 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)

#train our model 
regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)