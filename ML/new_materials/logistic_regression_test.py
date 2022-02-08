import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


from logistic_regression import LogisticRegression
#from regression import LogisticRegression

# def accuracy(y_true, y_pred):
#     accuracy = np.sum(y_true == y_pred) / len(y_true)
#     return accuracy

# def accuracy(y_true, y_pred):
#     accuracy=np.sum(y_true==y_pred)/len(y_true)
#     return accuracy
bc = datasets.load_breast_cancer()
# print(bc)
# print(datasets.load_breast_cancer())
X, y = bc.data, bc.target
# # print(X.shape)
# # #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
# # print(X_train.shape)
# # print(X_test.shape)
# # # #
regressor = LogisticRegression(learning_rate=0.0001, n_iters=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
print(predictions[90:100])
print(y_test[90:100])
# # # # print(f">>>>>>>>{len(y_test)}>>>>>>>")
# # #
# # #
print("LR classification accuracy:", accuracy(y_test, predictions))
# cm=confusion_matrix(predictions, y_test)
# print(cm)