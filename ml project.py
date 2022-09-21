


import numpy as np                   # import numpy package under shorthand "np"
import pandas as pd                  # import pandas package under shorthand "pd"
import matplotlib.pyplot as plt
import seaborn as sns
from nose.tools import assert_equal
from numpy.testing import assert_array_equal

# Regression import 
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', 'Completer.use_jedi = False  # enable code auto-completion')
from sklearn.preprocessing import PolynomialFeatures    # function to generate polynomial and interaction features
from sklearn.linear_model import LinearRegression    # classes providing Linear Regression with ordinary squared error loss and Huber loss, respectively
from sklearn.metrics import mean_squared_error    # function to calculate mean squared error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


playerData = pd.read_csv('10y_Combined_GP_incl.csv')   #read data fetched from eliteprospects ADD DATAFILE

data = playerData 

data.head(5)        


data['PPG'] = data['PPG'].astype(float)
data['Position'] = data['Position'].astype(float)




X_sub = data[['PPG','Standing','GP','Position']]
y_sub = data['Success']

X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size = 0.3)




sns.pairplot(data = data,
            x_vars = ['GP', 'PPG', 'Standing', 'Position'],
            y_vars = ['Success'])



poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X_sub)

model = LogisticRegression()
model.fit(X_poly, y_sub)

y_pred = model.predict(X_poly)


print(classification_report(y_sub, y_pred))





