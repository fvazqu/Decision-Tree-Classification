# Decision-Tree-Classification
Built and optimized a Decision Tree Classifier using the Python Scikit-Learn libraries. The entropy impurity measurement method was used to guide the tree generation for classifying the poker hand data set archived at the UCI Machine Learning Repository. The accuracy output was then compared with the output of the decision tree generated using AdaBoost.


Notes:
The 11th Column of the Poker Hand data set was set as the target variable when splitting the data. This column was changed to binary to show two different classes: 0 meaning nothing in hand, and nonzero being 1 to 9.

When using the AdaBoost classifier to generate the decision tree, the settings were set to 15 iterations and 42 random states.


References:                       
The training and testing data sets:
https://archive.ics.uci.edu/ml/machine-learning-databases/poker/

Using Scikit to generate decision trees tutorial:
https://www.datacamp.com/tutorial/decision-tree-classification-python
