# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.ensemble import AdaBoostClassifier

# training data set
col_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11']
data_train = pd.read_csv("poker-hand-training-true.data", header=None, names=col_names)

data_train.x11.replace(to_replace = range(2,10), 
                 value = 1, 
                  inplace = True)

data_train.head()

feature_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
X_train = data_train[feature_cols] # Features
y_train = data_train.x11 # Target variable

# data testing set
data_test = pd.read_csv("poker-hand-testing.data", header=None, names=col_names)
# change last column to binary
data_test.x11.replace(to_replace = range(2,10), 
                 value = 1, 
                  inplace = True)
X_test = data_test[feature_cols] # Features
y_test = data_test.x11 # Target variable

# Create an AdaBoost classifier with 15 weak classifiers
ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=15, random_state=42)

# Fit the model to the training data
ada.fit(X_train, y_train)

# Evaluate the accuracy of the model on the testing data
accuracy = ada.score(X_test, y_test)
print("Accuracy: {:.2f}%".format(accuracy * 100))
