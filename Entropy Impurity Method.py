# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

col_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11']
# load dataset
data_train = pd.read_csv("poker-hand-training-true.data", header=None, names=col_names)

# change last column to binary
data_train.x11.replace(to_replace = range(2,10), 
                 value = 1, 
                  inplace = True)

data_train.head()

 #split dataset in features and target variable
feature_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
X_train = data_train[feature_cols] # Features
y_train = data_train.x11 # Target variable

# Split dataset into training set and test set
data_test = pd.read_csv("poker-hand-testing.data", header=None, names=col_names)
# change last column to binary
data_test.x11.replace(to_replace = range(2,10), 
                 value = 1, 
                  inplace = True)
X_test = data_test[feature_cols] # Features
y_test = data_test.x11 # Target variable

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy: {:.2f}%".format(metrics.accuracy_score(y_test, y_pred)*100))
