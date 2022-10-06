import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import export_text, plot_tree, DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import PolynomialFeatures 

playerData = pd.read_csv('U1610yPos_GP_PPG_PIM_GAP.csv')   #read data fetched from eliteprospects ADD DATAFILE
data = playerData 

print(data.shape)
data.head(5)        

data['PPG'] = data['PPG'].astype(float)
data['Position'] = data['Position'].astype(float)


X = data[['Position','G','A']] #.to_numpy().reshape([-1,4])
y = data['Success'] #.to_numpy
print(X.head())
'''
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
plt.show()'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)


sns.pairplot(data = data,
            x_vars = ['GP', 'PPG', 'Standing', 'Position'],
            y_vars = ['Success'])
#plt.show()#
poly = PolynomialFeatures(degree=3, interaction_only=True)
X_poly = poly.fit_transform(X_train)

model = DecisionTreeClassifier(random_state=0)
model.fit(X_poly, y_train)
y_pred = model.predict(poly.fit_transform(X_test))

acc = accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred))

confmat = confusion_matrix(y_test, y_pred)

ax = plt.subplot()
sns.heatmap(confmat,annot=True, fmt='g', ax=ax)
ax.set_xlabel('Predicted labels',fontsize=15)
ax.set_ylabel('True labels',fontsize=15)
plt.show()