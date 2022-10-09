

import numpy as np                   # import numpy package under shorthand "np"
import pandas as pd                  # import pandas package under shorthand "pd"
import matplotlib.pyplot as plt
import seaborn as sns
#from nose.tools import assert_equal
from numpy.testing import assert_array_equal

# Regression import 
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', 'Completer.use_jedi = False  # enable code auto-completion')
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score
from sklearn.metrics import log_loss, accuracy_score

playerData = pd.read_csv('U1610yPos_GP_PPG_PIM_GAP.csv')   #read data fetched from eliteprospects ADD DATAFILE

def splitbalancer(dataframe: pd.DataFrame, features: list, random: int=0, test = float):
    y_positives = dataframe.loc[dataframe['Success']==1]['Success']
    y_negatives = dataframe.loc[dataframe['Success']==0]['Success']
    X_positives = dataframe.loc[dataframe['Success']==1][features]
    X_negatives = data.loc[data['Success']==0][features]

    X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(X_positives, y_positives, test_size = 0.2, random_state = random)
    X_train_pos, X_val_pos, y_train_pos, y_val_pos = train_test_split(X_train_pos, y_train_pos, test_size = test, random_state = random)
    X_train_neg, X_test_neg, y_train_neg, y_test_neg = train_test_split(X_negatives, y_negatives, test_size = 0.2, random_state = random)
    X_train_neg, X_val_neg, y_train_neg, y_val_neg = train_test_split(X_train_neg, y_train_neg, test_size = test, random_state = random)

    X_train = pd.concat([pd.concat([X_train_neg, X_train_pos]), X_train_pos])
  
    y_train = pd.concat([pd.concat([y_train_neg, y_train_pos]), y_train_pos])
    X_val = pd.concat([X_val_neg, X_val_pos])
    y_val = pd.concat([y_val_neg, y_val_pos])
    X_test = pd.concat([X_test_neg, X_test_pos])
    y_test = pd.concat([y_test_neg, y_test_pos])
    return [X_train, y_train, X_val, y_val, X_test, y_test]

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

    for l in range(len(feature_lists)):
        
        training_features = feature_lists[l]

        for split in range(1, 7): #test every split from 0.1 to 0.6 with a 0.1 increment            
            for i in range(1, max_degree):
                scores = []
                for n in range(3): #Lets run the same model three times with three splits and take the average f-beta score
                    X_train, y_train, X_val, y_val, X_test, y_test = splitbalancer(data, training_features, random=n, test = split*0.1)
                    
                    transform = PolynomialFeatures(degree=i, interaction_only=True)
                    X_poly = transform.fit_transform(X_train) #transform the training data
                    model = LogisticRegression(random_state = 0)
                    model.fit(X_poly, y_train)
                    y_pred = model.predict(transform.fit_transform(X_val))
                    score = fbeta_score(y_val, y_pred, beta = 0.3)
                    scores.append(score)
                    print(score)
                avg_score = sum(scores)/len(scores)
                if avg_score > best_classifier['f1']:
                    best_classifier['f1'] = avg_score
                    best_classifier['features'] = feature_lists[l]
                    best_classifier['test'] = split * 0.1
                    best_classifier['degree'] = i

    print("Best classifier has an fbeta(0.3 recall) score of about ", best_classifier['f1'],
    "\nClassifier attributes: ",
    '\nfeatures: ', best_classifier['features'],
    "\nvalidation set size: ", best_classifier['test'],
    "\npolynomial degree: ", best_classifier['degree'])

    #Rebuild and return the best classifier:
    X_train, y_train, X_val, y_val, X_test, y_test = splitbalancer(data, best_classifier['features'], random = 0, test = best_classifier['test'])                
    transform = PolynomialFeatures(degree = best_classifier['degree'], interaction_only = True)

    X_polytrain = transform.fit_transform(X_train) #Applying poly transform to all sets
    X_polyval = transform.fit_transform(X_val) 
    X_polytest = transform.fit_transform(X_test)

    model = LogisticRegression(random_state = 0)
    model.fit(X_polytrain, y_train)

    y_testpred = model.predict(X_polytest) #Predicting on all sets
    y_trainpred = model.predict(X_polytrain)
    y_valpred = model.predict(X_polyval)
    

    return [model, transform, X_polytrain, X_polyval, X_polytest, y_train, y_val, y_test, y_trainpred, y_valpred, y_testpred]

model, transform, X_polytrain, X_polyval, X_polytest, y_train, y_val, y_test, y_trainpred, y_valpred, y_testpred = ClfOptimizer()

print("Test results :\n", classification_report(y_test, y_testpred))
print("Validation error: ", 1-accuracy_score(y_val, y_valpred))
print("Training error: ", 1 - accuracy_score(y_train, y_trainpred))

confmat = confusion_matrix(y_test,y_testpred)

ax = plt.subplot()
sns.heatmap(confmat,annot=True, fmt='g', ax=ax)
ax.set_xlabel('Predicted labels',fontsize=10)
ax.set_ylabel('True labels',fontsize=10)
plt.title("Logistic Regression: prediction results", fontsize=10)
plt.show()
