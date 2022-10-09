
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
from sklearn.metrics import classification_report, f1_score, confusion_matrix, fbeta_score
from sklearn.metrics import log_loss, accuracy_score

playerData = pd.read_csv('U1610yPos_GP_PPG_PIM_GAP.csv')   #read data fetched from eliteprospects ADD DATAFILE

data = playerData 
data.head(5)        

data['PPG'] = data['PPG'].astype(float)
data['Position'] = data['Position'].astype(float)

def ClfOptimizer(max_degree: int = 5):
    best_classifier = { 'f1': 0,
      'features': [],
      'test': 0,
      'degree': 0 } #Initialize a dict for the best classifier
    feature_lists = [['Position','GP','PPG','Standing'],['Position','GP','Standing','G','A']] #pure vs processed features
    y = data['Success']
    for l in range(len(feature_lists)):
        
        training_features = feature_lists[l]
        X = data[training_features]

        for split in range(1, 7): #test every split from 0.1 to 0.6 with a 0.1 increment            
            for i in range(1, max_degree):
                scores = []
                for n in range(3): #Lets run the same model three times with random splits and take the average f1 score
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split * 0.1)
                    transform = PolynomialFeatures(degree=i, interaction_only=True)
                    X_poly = transform.fit_transform(X_train) #transform the training data
                    model = LogisticRegression(random_state = 0)
                    model.fit(X_poly, y_train)
                    y_pred = model.predict(transform.fit_transform(X_test))
                    score = fbeta_score(y_test, y_pred, beta = 0.1)
                    scores.append(score)
                    print(score)
                avg_score = sum(scores)/len(scores)
                if avg_score > best_classifier['f1']:
                    best_classifier['f1'] = avg_score
                    best_classifier['features'] = feature_lists[l]
                    best_classifier['test'] = split * 0.1
                    best_classifier['degree'] = i

    print("Best classifier has an fbeta(0.1 recall) score of about ", best_classifier['f1'],
    "\nClassifier attributes: ",
    '\nfeatures: ', best_classifier['features'],
    "\ntest size: ", best_classifier['test'],
    "\npolynomial degree: ", best_classifier['degree'])

    #Rebuild and return the best classifier:
    X = data[best_classifier['features']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = best_classifier['test'])
    transform = PolynomialFeatures(degree = best_classifier['degree'], interaction_only = True)
    X_poly = transform.fit_transform(X_train)
    model = LogisticRegression(random_state = 0)
    model.fit(X_poly, y_train)
    X_polytest = transform.fit_transform(X_test)
    y_pred = model.predict(X_polytest)
    y_validation = model.predict(X_poly)

    return [model, transform, X_poly, X_polytest, y_test, y_train, y_pred, y_validation]

classifire = ClfOptimizer()
print(classification_report(classifire[4],classifire[6]))

print("Validation error: ", 1 - accuracy_score(classifire[5],classifire[7]))

confmat = confusion_matrix(classifire[4],classifire[6])

ax = plt.subplot()
sns.heatmap(confmat,annot=True, fmt='g', ax=ax)
ax.set_xlabel('Predicted labels',fontsize=15)
ax.set_ylabel('True labels',fontsize=15)
plt.show()