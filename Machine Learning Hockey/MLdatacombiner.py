import posix
import numpy as np                   # import numpy package under shorthand "np"
import pandas as pd                  # import pandas package under shorthand "pd"
import matplotlib.pyplot as plt
#from nose.tools import assert_equal
from numpy.testing import assert_array_equal
# Regression import 


from sklearn.preprocessing import PolynomialFeatures    # function to generate polynomial and interaction features
from sklearn.linear_model import LinearRegression    # classes providing Linear Regression with ordinary squared error loss and Huber loss, respectively
from sklearn.metrics import mean_squared_error    # function to calculate mean squared error


playerData = pd.read_csv('/Users/pauli/Downloads/ML_icehockey-main/Machine Learning Hockey/ML_10y_playerdata.csv')   #read data fetched from eliteprospects ADD DATAFILE
U16sm_standings = pd.read_csv('Machine Learning Hockey/U16sm_standings10year.csv')

Combined = pd.merge(playerData, U16sm_standings, how='left', left_on=['team', 'season'], right_on=['team', 'season']).dropna()
Better = Combined[Combined.GP != 0]
data = Better.drop(['GP','G','A','P','PIM', '+/-'], inplace = False, axis = 1) #drop unnecessary columns 


data.head(10) 


x = 0

while x < data.shape[0]:
    position = data.iloc[x, 2][1]
    data.iloc[x, 2] = position
    x = x + 1
    
data.head(30)
 
 
data.loc[data['pos'] == 'L', 'pos'] = 1
data.loc[data['pos'] == 'F', 'pos'] = 1
data.loc[data['pos'] == 'R', 'pos'] = 1
data.loc[data['pos'] == 'C', 'pos'] = 1
data.loc[data['pos'] == 'D', 'pos'] = 0

data.head(20)
data.tail(5)

data.to_csv("10y_Combined_dirtyPGPC.csv", index=False)
