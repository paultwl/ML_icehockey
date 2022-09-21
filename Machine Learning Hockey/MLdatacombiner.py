import numpy as np                   # import numpy package under shorthand "np"
import pandas as pd                  # import pandas package under shorthand "pd"
import matplotlib.pyplot as plt
from nose.tools import assert_equal
from numpy.testing import assert_array_equal

# Regression import 


from sklearn.preprocessing import PolynomialFeatures    # function to generate polynomial and interaction features
from sklearn.linear_model import LinearRegression    # classes providing Linear Regression with ordinary squared error loss and Huber loss, respectively
from sklearn.metrics import mean_squared_error    # function to calculate mean squared error


playerData = pd.read_csv('MLplayerdata2.csv')   #read data fetched from eliteprospects ADD DATAFILE
U16sm_standings = pd.read_csv('U16sm_standings.csv')

Combined = pd.merge(playerData, U16sm_standings, how='left', left_on=[' team', ' season'], right_on=[' team', ' season']).dropna()

data = Combined.drop([' GP',' G',' A',' P',' PIM', ' +/-'], inplace = False, axis = 1) #drop unnecessary columns 


data.head(10) 


x = 0

while x > data.size:
    position = data.iloc[x, 2][1]
    data.iloc[x, 2] = position
    x = x + 1
    
data.head(30)
 
 
data.loc[data[' pos'] == 'L', ' pos'] = 2
data.loc[data[' pos'] == 'F', ' pos'] = 2
data.loc[data[' pos'] == 'R', ' pos'] = 2
data.loc[data[' pos'] == 'C', ' pos'] = 2
data.loc[data[' pos'] == 'D', ' pos'] = 1

data.head(20)
data.tail(5)
