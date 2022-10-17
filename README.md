# ML_icehockey
A machine learning model designed to predict breakthrough to the professional level based on a players personal performance as well as team performance. The model takes position, ppg and team rankings as factors and outputs a probability to make it to the Finnish Liiga..



Predicting player success 
with performance data from junior leagues


I) Introduction

This report will discuss a machine learning model designed to predict ice hockey players’ success based on data about their performance in the 
U16 Finnish championship league. A model like this could be useful for coaches, scouts and player agents in deciding which players to follow more 
closely or work with.
The report has been divided into the following six sections: I Introduction, II Problem Formulation, III Methods, IV Results, V Conclusions and 
VI Sources and Appendices. Problem Formulation will explain what the aim of this model actually is, i.e. what is the problem we are trying to solve. 
It will also discuss the types of data used to form the models. Section III will discuss how the problem was solved. It will explain the prediction 
methods and data processing and the motivation behind each choice. The results and final model chosen for the predictor are discussed in section IV.
Applicability of the model, possible future improvement and other conclusions are presented in section V. Section VI will include any sources 
referenced in this report as well as a link to the github.com repository containing the databases and code for the models and preprocessing.

II) Problem Formulation

Problem: How (and if) it is possible to predict whether a junior ice hockey player in Finland will proceed to play in the professional league Liiga?

The goal of this model is to help agents and teams evaluate the potential of players based on their junior level performance. This machine learning 
application will use performance statistic of Finnish U16 national league players as well as their respective teams from seasons 2006-2016 as its 
datapoints.  In total we have fetched 2172 datapoints. Each datapoint represents a player and contains the following values: the players name, games 
played (GP), position (POS), team, goals (G), assists (A), points (P), points per game average (PPG), penalties in minutes (PIM), success, season and 
standing. Of these the following ones could be used (and could be relevant) for the model: PPG, GP, team standing at the end of the season, G, A, P and 
PIM  all as continuous data and POS, i.e., whether they play forward or defense as binary data. The success value of the datapoint is binary. If success 
equals 1 the player in question has made it to Liiga and if 0 they haven’t. Success is the label our model aims to predict. Goaltenders aren’t 
considered in this model, because their performance is measured with an entirely different set of statistics. 


III) Methods

The dataset is collected from eliteprospects.com, an ice hockey statistic database, using a paid pro level account. As the data was not readily available
in a single table and rather spanned across multiple tables from different seasons, it was scraped and formatted using a python script. The script and 
database can both be found in the ML_icehockey GitHub repository. In short, the script fetched all the player statistics mentioned in section II, converted
the string signifying position into a binary value and stored the final data in a csv file. The margin of error for the dataset is not known precisely but
the accuracy of the database has been found reliable through years of personal experience from working with the sport and the website. The model links the
features in the training data to labels that can also be found directly from the training data. Hence, our model can use supervised learning methods
to predict career outcomes for future U16 players.

 
Image 1. Example rows for the data used for predictions

Our model contains 2172 datapoints, as previously stated. Out of these datapoint 524 or 24% are those of successful players, meaning they have a
‘Success’-value of 1. Based on the distribution we can calculate the gini impurity of the data. Gini impurity describes how often samples would
be labeled incorrectly if they were randomly assigned a label based on the overall distribution of labels. The dataset used has a gini impurity of 37%.
This means that if any of the classifiers have a total accuracy of more than 63%, they are better than random guessing. We can further divide
this into label-wise precisions and recall of 76% for the negative values and 24% for the positive ones. Because the data is not very old, 
we believe it accurately represents the current distribution of successful players. Therefore, these values are used as the baseline for comparing
prediction accuracy and precisions. 

While something like success in sports isn’t strictly binary, the predictors presented only focus on one measure of success for the sake of 
simplicity, namely, playing in Liiga or not. Because of this choice the output labels are binary. For training features, two sets of features 
were created based on expertise of hockey player agents consulted for this report. Both sets include POS, team standing and GP. In addition, 
the first set includes PPG and second one includes goals and assists. PPG and goals and assists fundamentally carry the same information. 
PPG is simply (G+A)/GP. The two sets were constructed to see through trial and error if one form of data suits predictions better than the other. 
We would have also wanted to include +/- and age statistics in our model. +/- is a good indicator of a player’s impact on the ice: a player gets +1 
if their team scores a goal and -1 if their team allows a goal while the player is on the ice. Unfortunately, this statistic was not available for 
such young players. Another useful feature would have been the age of the player at the time of the season, as younger players who do well are obviously
more likely to succeed. 
Using the collected data and features discussed earlier, two independent classifiers were created. One using logistic regression and one utilizing a 
decision tree. Both were constructed using the well-known sklearn python package (scikit-learn, no date). Logistic regression was firstly chosen as 
the desired output of the classifier was binary: 0 or 1. This is because first and foremost we wanted this model to help categorize the players to 
promising and less promising ones leaving only the prospectively successful ones for the scouts’ assessment. After applying logistic regression 
to our data, we noticed that there was no clear correlation between many of the features and the label. Therefore, we wanted to utilize a 
decision tree as it is also an efficient machine learning method for categorizing datapoints. It could also be able to disregard values that are given
to much weight in logistic regression.
For the logistic regression model, error is measured with logistic loss during training since it goes well hand in hand with the logistic regression 
model and is also what the LogisticRegression classifier offered by the sklearn python package (scikit-learn, no date) uses by default. The decision 
tree classifier offered by the sklearn package on the other hand is trained using the gini impurity of a branch by default, so this will be used. 
The default metrics were chosen because of ease of use and no obvious weaknesses. 1/0 loss is used to measure performance of both models after 
training. This is the obvious choice because of the binary nature of the labels in both predicted and true data. It also allows for the calculation
of several different metrics, such as recall, precision, f1 and f-beta score.
A ‘ClfOptimizer’ function was defined and used to automatically choose the optimal parameters for each model. The optimizer is essentially a 
collection of nested for loops testing different feature sets, degrees of polynomial transformation and splits for training and validation data
and then choosing the best combination of parameters based on the average f-beta score of three runs for the same parameters. In total, the 
optimizer compares 48 logistic regression models and 192 decision trees and chooses the best model for both. We believe this is a large enough 
comparison to find a model with reasonable accuracy. 
To combat the imbalance of labels in the data, some oversampling was applied. Every datapoint with label 1 in the training set was duplicated. 
Oversampling should help reduce the overpowering effects of one large label. As the data used for training and testing was still imbalanced with a ratio
of roughly 3:2, the models were then ranked based on a weighted f-beta score of the successful prediction. If we were to simply use the accuracy of the
model, we could achieve a good score by labeling datapoints as 0 regardless of input features.  The f-beta score is an f-1 score with a bias towards
either precision or recall. Considering the real-world applications of this model, the argument can be made that precision, i.e., betting on the right
players is more important than recall. Therefore, we weighted the f-beta score towards precision with a weight ratio of 10:3.  

IV) Results

The best models achieved with the optimizer function are presented in images 3 to 5. Both models were chosen based on maximum f-beta score for
the validation set as discussed in section III. The test set has been chosen by dividing the entire dataset two by label value, then 20% of both
labels have been added used to form the test set. This was done to represent the real-world distribution of data as well as possible. The test set
was set aside before training and validation to ensure testing is done with completely new data.
The best logistic regression model was achieved with the following parameters: Feature list containing PPG, validation size of 0.1 and a polynomial
degree of 2. The model beat the performance baseline in all aspects with the test set. The validation accuracy for the logistic classifier was 80%,
while the training accuracy was 70%. This means the model isn’t overfitting too much to the training data and can handle new inputs well. The big 
difference in validation and training accuracy might also hint at too small datasets.
The best decision tree classifier was achieved with the following parameters: Feature list containing PPG, validation size of 0.2, polynomial degree 
of 2 and a minimum leaf size of 4. This model also beat the baseline but achieved lower scores than the logistic classifier on all metrics with the
test set. The decision tree had a training accuracy of 89% which hints to some overfitting, especially with the validation accuracy being 16 percentage
points lower at 73%.

V) Conclusions

Based on the validation accuracy of our models, we can conclude that the logistic regression classifier holds more predictive power and should be
the final choice for future predictions. This conclusion is also backed by the test results as seen in images 3 and 5. Logistic regression
classification was better than the decision tree better on all metrics, achieving a test accuracy of 76% (test error 24%). While the results may
not be considered great for most machine learning applications, we believe this model to be quite accurate given the arbitrary nature of ice hockey,
especially at the junior level.
There are some limitations to our models that we believe are mainly brought on by lack and nature of data rather than our choice of prediction models. 
For example, not all Finnish players at the top level have played in Liiga but have for example moved to play in the United States at a young age. 
Increasing the number of leagues considered “going pro” would explain some of the current outliers in our dataset. However, the player path we have
chosen is by far the most common in Finland. Some limitations of data also came up in discussions with professional hockey agents: features that might
potentially have huge predictive value, such as +/- and player dimensions are missing from the obtained data. Collecting these features for each 
datapoint could potentially improve accuracy.


VI) References

scikit-learn, no date [Online] 
Available at: https://scikit-learn.org/stable/

VI) Code

-	ML_icehockey Github repository: https://github.com/paultwl/ML_icehockey 
