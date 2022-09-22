
import numpy as np                   # import numpy package under shorthand "np"
import pandas as pd                  # import pandas package under shorthand "pd"
import matplotlib.pyplot as plt
import seaborn as sns
#from nose.tools import assert_equal
from numpy.testing import assert_array_equal

# Regression import 
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', 'Completer.use_jedi = False  # enable code auto-completion')
from sklearn.preprocessing import PolynomialFeatures    # function to generate polynomial and interaction features
from sklearn.linear_model import LinearRegression    # classes providing Linear Regression with ordinary squared error loss and Huber loss, respectively
from sklearn.metrics import mean_squared_error    # function to calculate mean squared error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss

playerData = pd.read_csv('10y_Combined_GP_incl.csv')   #read data fetched from eliteprospects ADD DATAFILE

data = playerData 

data.head(5)        


data['PPG'] = data['PPG'].astype(float)
data['Position'] = data['Position'].astype(float)




X_sub = data[['PPG','Standing','GP','Position']] #.to_numpy().reshape([-1,4])
y_sub = data['Success'] #.to_numpy

X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size = 0.4)




sns.pairplot(data = data,
            x_vars = ['GP', 'PPG', 'Standing', 'Position'],
            y_vars = ['Success'])



poly = PolynomialFeatures(degree=5, interaction_only=True)
X_poly = poly.fit_transform(X_train)
print(X_sub.to_numpy()[0])


model = LogisticRegression()
model.fit(X_poly, y_train)

y_pred = model.predict(poly.fit_transform(X_test))

print(classification_report(y_test, y_pred))
