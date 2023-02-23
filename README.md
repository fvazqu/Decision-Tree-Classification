# Decision-Tree-Classification
In this project, I built and optimized a Decision Tree Classifier using the Python Scikit-Learn libraries using two different generation methods with the same training set (Poker Hand Data Set archived at the UCI Machine Learning Repository). The entropy impurity measurement method was first used to guide tree generation to classify the training set. The second method involved generating a decision tree using AdaBoost. The accuracy of each was then compared using the testing set.


Notes:                             
The 11th Column of the Poker Hand data set was set as the target variable when splitting the data. This column was changed to binary to show two different classes: 0 meaning nothing in hand, and nonzero being 1 to 9.
When using the AdaBoost classifier to generate the decision tree, the settings were set to 15 iterations and 42 random states.
This project was done to gain a better understanding of Machine Learning, Data Mining, and using Python.

Results:             
![image](https://user-images.githubusercontent.com/63169963/221048571-9c03e5f1-c7dc-4915-a68e-80cd296cbb69.png)



References:                       
The training and testing data sets:
https://archive.ics.uci.edu/ml/machine-learning-databases/poker/

Using Scikit to generate decision trees tutorial:
https://www.datacamp.com/tutorial/decision-tree-classification-python
