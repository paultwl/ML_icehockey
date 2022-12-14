
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import export_text, plot_tree, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score

from sklearn.preprocessing import PolynomialFeatures 

playerData = pd.read_csv('U1610yPos_GP_PPG_PIM_GAP.csv')   #read data fetched from eliteprospects ADD DATAFILE
data = playerData 

print(data.shape)
data.head(5)        

data['PPG'] = data['PPG'].astype(float)
data['Position'] = data['Position'].astype(float)

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




#We want to create a program to test for best minimum leaf size, 
#polynomial degree and features as well as test/train split
#Then we will rank these model by f1 score for positive values and save the best model. 


def ClfOptimizer(max_degree: int = 5, max_min_leaf: int = 6, ):

    best_classifier = { 'f1': 0,
      'features': [],
      'test': 0,
      'degree': 0,
      'min_leaf': 0 } #Initialize a dict for the best classifier
    feature_lists = [['Position','GP','PPG','Standing'],['Position','GP','Standing','G','A']] #pure vs processed features

    
    for l in range(len(feature_lists)):
        
        training_features = feature_lists[l]

        for split in range(1, 7): #test every split from 0.1 to 0.6 with a 0.1 increment            
            for i in range(1, max_degree):
                for j in range(2 , max_min_leaf):
                    scores = []
                    for n in range(3): #Lets run the same model three times with random splits and take the average f1 score
                        X_train, y_train, X_val, y_val, X_test, y_test = splitbalancer(data, training_features, random=n, test = split*0.1)
                    

                        transform = PolynomialFeatures(degree=i, interaction_only=True)
                        X_poly = transform.fit_transform(X_train) #transform the training data
                        model = DecisionTreeClassifier(random_state = 0, min_samples_leaf = j)
                        model.fit(X_poly, y_train)
                        y_pred = model.predict(transform.fit_transform(X_val))
                        score = fbeta_score(y_val, y_pred, beta = 0.3,)

                        scores.append(score)
                        print(score)
                    avg_score = sum(scores)/len(scores)
                    if avg_score > best_classifier['f1']:
                        best_classifier['f1'] = avg_score
                        best_classifier['features'] = feature_lists[l]
                        best_classifier['test'] = split * 0.1
                        best_classifier['degree'] = i
                        best_classifier['min_leaf'] = j
                        
    print("Best classifier has an f-beta(0.3 recall) score of about ", best_classifier['f1'],
    "\nClassifier attributes: ",
    '\nfeatures: ', best_classifier['features'],
    "\nvalidation set size: ", best_classifier['test'],

    "\npolynomial degree: ", best_classifier['degree'],
    "\nmin leaf size: ", best_classifier['min_leaf'])

    #Rebuild and return the best classifier:
    X_train, y_train, X_val, y_val, X_test, y_test = splitbalancer(data, best_classifier['features'], random = 0, test = best_classifier['test'])                
    transform = PolynomialFeatures(degree = best_classifier['degree'], interaction_only = True)

    X_polytrain = transform.fit_transform(X_train) #Applying poly transform to all sets
    X_polyval = transform.fit_transform(X_val) 
    X_polytest = transform.fit_transform(X_test)

    model = DecisionTreeClassifier(random_state = 0, min_samples_leaf = best_classifier['min_leaf'])
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
plt.title("Decision tree: prediction results", fontsize=10)
plt.show()
