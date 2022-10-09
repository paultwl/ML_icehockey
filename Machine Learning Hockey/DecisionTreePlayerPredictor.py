import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import export_text, plot_tree, DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import PolynomialFeatures 

playerData = pd.read_csv('U1610yPos_GP_PPG_PIM_GAP.csv')   #read data fetched from eliteprospects ADD DATAFILE
data = playerData 

print(data.shape)
data.head(5)        

data['PPG'] = data['PPG'].astype(float)
data['Position'] = data['Position'].astype(float)


X = data[['Position','PPG','GP','Standing']] #.to_numpy().reshape([-1,4])
y = data['Success'] #.to_numpy
print(X.head())

sns.countplot(x = "A", data = data, hue = "Success")
plt.xticks(rotation = 90)
plt.title("Assists")
plt.show()
sns.countplot(x = "G", data = data, hue = "Success")
plt.xticks(rotation = 90)
plt.title("Goals")
plt.show()
sns.countplot(x = "GP", data = data, hue = "Success")
plt.xticks(rotation = 90)
plt.title("Games Played")
plt.show()
sns.countplot(x = "PIM", data = data, hue = "Success")
plt.xticks(rotation = 90)
plt.title("Games Played")
plt.show()
sns.countplot(x = "Standing", data = data, hue = "Success")
plt.xticks(rotation = 90)
plt.title("Games Played")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)


sns.pairplot(data = data,
            x_vars = ['GP', 'PPG', 'Standing', 'Position'],
            y_vars = ['Success'])
#plt.show()#
poly = PolynomialFeatures(degree=1, interaction_only=True) #set the degree of polynomial transformation
X_poly = poly.fit_transform(X_train) #Apply the transformation to the training set

model = DecisionTreeClassifier(random_state=0)
model.fit(X_poly, y_train)

y_pred = model.predict(poly.fit_transform(X_test))
acc = accuracy_score(y_test, y_pred)
clfreport = classification_report(y_test, y_pred)
print(clfreport)
successrecall = classification_report
confmat = confusion_matrix(y_test, y_pred)

ax = plt.subplot()
sns.heatmap(confmat,annot=True, fmt='g', ax=ax)
ax.set_xlabel('Predicted labels',fontsize=15)
ax.set_ylabel('True labels',fontsize=15)
plt.show()

#We want to create a program to test for best minimum leaf size, 
#polynomial degree and features as well as test/train split
#Then we will rank these model by f1 score for positive values and save the best model. 

def ClfOptimizer(max_degree: int = 6, max_min_leaf: int = 6, ):
    best_classifier = { 'f1': 0,
      'features': [],
      'test': 0,
      'degree': 0,
      'min_leaf': 0 } #Initialize a dict for the best classifier
    feature_lists = [['Position','GP','PPG','Standing'],['Position','GP','Standing','G','A']] #pure vs processed features
    y = data['Success']
    for l in range(len(feature_lists)):
        
        training_features = feature_lists[l]
        X = data[training_features]

        for split in range(1, 7): #test every split from 0.1 to 0.6 with a 0.1 increment            
            for i in range(1, max_degree):
                for j in range(1 , max_min_leaf):
                    scores = []
                    for n in range(3): #Lets run the same model three times with random splits and take the average f1 score
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split * 0.1)
                        transform = PolynomialFeatures(degree=i, interaction_only=True)
                        X_poly = transform.fit_transform(X_train) #transform the training data
                        model = DecisionTreeClassifier(random_state = 0, min_samples_leaf = j)
                        model.fit(X_poly, y_train)
                        y_pred = model.predict(transform.fit_transform(X_test))
                        score = f1_score(y_test, y_pred, pos_label = 1)
                        scores.append(score)
                        print(score)
                    avg_score = sum(scores)/len(scores)
                    if avg_score > best_classifier['f1']:
                        best_classifier['f1'] = avg_score
                        best_classifier['features'] = feature_lists[l]
                        best_classifier['test'] = split * 0.1
                        best_classifier['degree'] = i
                        best_classifier['min_leaf'] = j

    print("Best classifier has an f1 score of about ", best_classifier['f1'],
    "\nClassifier attributes: ",
    '\nfeatures: ', best_classifier['features'],
    "\ntest size: ", best_classifier['test'],
    "\npolynomial degree: ", best_classifier['degree'],
    "\nmin leaf size: ", best_classifier['min_leaf'])

    #Rebuild and return the best classifier:
    X = data[best_classifier['features']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = best_classifier['test'])
    transform = PolynomialFeatures(degree = best_classifier['degree'], interaction_only = True)
    X_poly = transform.fit_transform(X_train)
    model = DecisionTreeClassifier(random_state = 0, min_samples_leaf = best_classifier['min_leaf'])
    model.fit(X_poly, y_train)
    X_polytest = transform.fit_transform(X_test)
    y_pred = model.predict(X_polytest)
    y_validation = model.predict(X_poly)
    
    return [model, transform, X_poly, X_polytest, y_test, y_train, y_pred, y_validation]

classifire = ClfOptimizer()
print(classification_report(classifire[4],classifire[6]))

print("Training error: ", accuracy_score(classifire[5],classifire[7]))

confmat = confusion_matrix(classifire[4],classifire[6])

ax = plt.subplot()
sns.heatmap(confmat,annot=True, fmt='g', ax=ax)
ax.set_xlabel('Predicted labels',fontsize=15)
ax.set_ylabel('True labels',fontsize=15)
plt.show()