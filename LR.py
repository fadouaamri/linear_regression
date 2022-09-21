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

#plot the Scatter plot to check relationship between Hours and Scores
f,axs=plt.subplots(3,1,figsize=(10,10))
f.text(0.5, 0.04, "Hours", ha='center')
f.text(0.04, 0.5, "Scores", va='center', rotation='vertical')

axs[0].scatter(df["Hours"],df["Scores"])
print("Correlation =",df["Hours"].corr(df["Scores"]))